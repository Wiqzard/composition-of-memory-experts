from typing import Any, Dict, Optional, Union, Sequence, Tuple, Callable, Literal
from functools import partial
import time
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities import grad_norm
from einops import rearrange, repeat, reduce


from src.models.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from src.models.common import BaseLightningTrainer
from src.models.components.diffusion import DiscreteDiffusion, ContinuousDiffusion
from src.models.components.autoencoder.vae.video_vae import VideoVAE
from src.models.components.autoencoder.vae.image_vae import ImageVAE

from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video
from utils.print_utils import cyan
from utils.torch_utils import freeze_model, bernoulli_tensor


class DiffusionModelTrainer(BaseLightningTrainer):
    def __init__(
        self,
        # model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        compile: bool = False,
        num_gen_steps: int = 10,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        is_latent_diffusion: bool = False,
        is_latent_online: bool = False,
        latent_downsampling_factor: Tuple[int, int, int] = (1, 1),
        x_shape: Tuple[int, int, int] = (3, 64, 64),
        diffusion_model: Optional[Union[DiscreteDiffusion, ContinuousDiffusion]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the diffusion model trainer.

        Args:
            model: The neural network (a UNet3D) that predicts the noise given a noisy input and timestep.
            optimizer: The optimizer class (e.g., torch.optim.Adam).
            num_train_timesteps: Number of diffusion timesteps for training.
            beta_schedule: Type of beta schedule ("linear", "cosine", etc.).
            compile: Whether to compile the model (requires PyTorch 2.x).
            lr: Learning rate.
            num_inference_steps: Number of steps to use during sampling in validation.
            kwargs: Additional hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=(
                "model",
                "diffusion_model",
                "optimizer",
                "lr_scheduler",
                "scheduler",
                "adapter.adapter_model",
            )
        )
        self.should_stop = False
        # self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.compile = compile
        self.diffusion_model = diffusion_model

        self.temporal_downsampling_factor = latent_downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1

        self.x_shape = self.hparams.x_shape
        if self.hparams.is_latent_diffusion:
            self.x_shape = [self.hparams.latent_num_channels] + [
                d // latent_downsampling_factor[1] for d in self.x_shape[1:]
            ]
        self.external_cond_dim = self.hparams.external_cond_dim * (
            self.hparams.frame_skip if self.hparams.external_cond_stack else 1
        )
        self.is_full_sequence = (
            self.hparams.noise_level == "random_uniform"
            and not self.hparams.fixed_context.enabled
            and not self.hparams.variable_context.enabled
            and not self.hparams.causal_context.enabled
        )

        self.tasks = [task for task in self.hparams.tasks]
        self.generator = None

        self.num_logged_videos = 0

    # ---------------------------------------------------------------------
    # Prepare Model, Optimizer, and Metrics
    # ---------------------------------------------------------------------

    def _load_vae(self) -> None:
        """
        PUT THIS IN THE CONFIG
        Load the pretrained VAE model.

        """
        vae_cls = VideoVAE if self.is_latent_video_vae else ImageVAE
        self.vae = vae_cls.from_pretrained(
            path=self.hparams.vae_pretrained_path,
            torch_dtype=(
                torch.float16 if self.hparams.vae_use_fp16 else torch.float32
            ),  # only for Diffuser's ImageVAE
            **self.hparams.vae_pretrained_kwargs,
        ).to(self.device)
        # fp16 if self.hparams.vae_use_fp16 else torch.float32
        freeze_model(self.vae)

    def _metrics(
        self,
        task: Literal["prediction", "interpolation"],
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        return getattr(self, f"metrics_{task}", None)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        params = self.trainer.model.parameters()  # correct?
        optimizer = self.optimizer(params=params)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        self.diffusion_model = self.diffusion_model(
            x_shape=self.x_shape,
            max_tokens=self.hparams.chunk_size,  # self.max_tokens,
            external_cond_dim=self.external_cond_dim,
        )

        if self.hparams.ckpt_path:
            state_dict = torch.load(
                self.hparams.ckpt_path, weights_only=False, map_location="cpu"
            )["state_dict"]
            if self.hparams.strict_load:
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
            else:
                self.load_checkpoint_lenient(state_dict)

        if self.compile and stage == "fit":
            if False:  # self.compile == "true_without_ddp_optimizer":
                # NOTE: `cfg.compile` should be set to this value when using `torch.compile` with DDP & Gradient Checkpointing
                # Otherwise, torch.compile will raise an error.
                # Reference: https://github.com/pytorch/pytorch/issues/104674
                # pylint: disable=protected-access
                torch._dynamo.config.optimize_ddp = False

                # print("torch._dynamo.config.optimize_ddp", torch._dynamo.config.optimize_ddp)
                # torch._dynamo.config.compiled_autograd = True

        self.diffusion_model = torch.compile(
            self.diffusion_model,
            disable=not self.compile,
            # fullgraph=self.compile,
        )

        self.register_data_mean_std(self.hparams.data_mean, self.hparams.data_std)

        # 2. VAE model
        if self.hparams.is_latent_diffusion and self.hparams.is_latent_online:
            self._load_vae()
        else:
            self.vae = None

        # 3. Metrics
        registry = SharedVideoMetricModelRegistry()
        metric_types = self.hparams.metrics
        for task in self.tasks:
            match task:
                case "prediction":
                    self.metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.hparams.metrics_batch_size,
                    )
                case "interpolation":
                    assert (
                        not self.hparams.use_causal_mask
                        # and not self.hparams.is_full_sequence
                        and self.max_tokens > 2
                    ), "To execute interpolation, the model must be non-causal, not full sequence, and be able to process more than 2 tokens."
                    self.metrics_interpolation = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.hparams.metrics_batch_size,
                    )

    # ---------------------------------------------------------------------
    # Length-related Properties and Utils
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # The two differ when using a VAE for latent diffusion.
    # ---------------------------------------------------------------------

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        return (n_frames - 1) // self.temporal_downsampling_factor + 1

    def _n_tokens_to_n_frames(self, n_tokens: int) -> int:
        """
        Converts the number of tokens to the number of frames.
        """
        return (n_tokens - 1) * self.temporal_downsampling_factor + 1

    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------

    @property
    def max_frames(self) -> int:
        return self.hparams.max_frames

    @property
    def max_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.max_frames)

    # @property
    # def total_frames(self) -> int:
    # return self.hparams.max_frames
    # @property
    # def total_tokens(self) -> int:
    #    return self._n_frames_to_n_tokens(self.total_frames)

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.max_frames if self.trainer.training else self.hparams.n_frames

    @property
    def n_context_frames(self) -> int:
        return self.hparams.context_frames

    @property
    def n_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_frames)

    @property
    def n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_context_frames)

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.

        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        gt_videos = None
        if self.hparams.is_latent_diffusion:
            xs = (
                self._encode(batch["videos"])
                if self.hparams.is_latent_online
                else batch["latents"]
            )
            if "videos" in batch:
                gt_videos = batch["videos"]
        else:
            xs = batch["videos"]
        xs = self._normalize_x(xs)

        # 2. Prepare external conditions
        conditions = batch.get("conds", None)

        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE."
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(self.device)

        return (
            xs,
            conditions,
            masks,
            gt_videos,
        )

    # ---------------------------------------------------------------------
    # Logging (Metrics, Videos)
    # ---------------------------------------------------------------------

    def _update_metrics(self, all_videos: Dict[str, Tensor]) -> None:
        """Update all metrics during validation/test step."""
        if (
            self.hparams.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {k: v[:, : self.hparams.n_metrics_frames] for k, v in all_videos.items()}

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self._metrics(task)
            videos = all_videos[task]
            context_mask = torch.zeros(self.n_frames).bool().to(self.device)
            match task:
                case "prediction":
                    context_mask[: self.n_context_frames] = True

                    if self.hparams.exclude_context:
                        videos = videos[:, self.n_context_frames :]
                        gt_videos = gt_videos[:, self.n_context_frames :]
                        context_mask = context_mask[self.n_context_frames :]

                case "interpolation":
                    context_mask[[0, -1]] = True
            if self.hparams.n_metrics_frames is not None:
                context_mask = context_mask[: self.hparams.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)

    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str) -> None:
        """Log videos during validation/test step."""
        all_videos = self.gather_data(all_videos)

        batch_size, n_frames = all_videos["gt"].shape[:2]
        raw_dir = self.trainer.log_dir

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.hparams.log_max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.hparams.log_max_num_videos - self.num_logged_videos,
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]

        for task in self.tasks:
            log_video(
                cut_videos(all_videos[task]),
                cut_videos(all_videos["gt"]),
                step=None if namespace == "test" else self.global_step,
                namespace=f"{task}_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                raw_dir=raw_dir,  # self.trainer.log_dir,
                context_frames=(
                    self.n_context_frames
                    if task == "prediction"
                    else torch.tensor([0, n_frames - 1], device=self.device, dtype=torch.long)
                ),
                captions=f"{task} | gt",
                fps=self.hparams.log_fps,
            )

        self.num_logged_videos += batch_size

    # ---------------------------------------------------------------------
    # Data Preprocessing Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _process_conditions(
        self,
        conditions: Optional[Tensor],
        noise_levels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Post-process the conditions before feeding them to the model.
        For example, conditions that should be computed relatively (e.g. relative poses)
        should be processed here instead of the dataset.

        Args:
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            noise_levels (Optional[Tensor], "B T"): Current noise levels for each token during sampling
        """

        if conditions is None:
            return conditions
        match self.hparams.external_cond_processing:
            case "mask_first":
                mask = torch.ones_like(conditions)
                mask[:, :1, : self.external_cond_dim] = 0
                return conditions * mask
            case "none":
                return conditions
            case _:
                raise NotImplementedError(
                    f"External condition processing {self.hparams.external_cond_processing} is not implemented."
                )

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        namespace: str = "training",
    ) -> STEP_OUTPUT:
        """Training step"""
        xs, conditions, masks, *_ = batch

        noise_levels, masks = self._get_training_noise_levels(xs, masks)

        xs_pred, loss, aux_output = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True,
            )
            if aux_output is not None:
                for key, value in aux_output.items():
                    self.log(
                        f"{namespace}/{key}",
                        value,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                        prog_bar=True,
                    )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def loss(self, pred, target, weight=None):
        if weight is None:
            weight = torch.ones_like(pred)
        return torch.mean((weight * (pred - target) ** 2).reshape(pred.shape[0], -1))

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.hparams.log_grad_norm_freq
            and self.global_step % self.hparams.log_grad_norm_freq == 0
        ):
            norms = grad_norm(self.diffusion_model, norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices
            self.log_dict(norms)

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.

        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        if self.trainer.state.fn == "fit" and self.hparams.log_denoising:
            self._eval_denoising(batch, batch_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        if not (self.trainer.sanity_checking and not self.hparams.log_sanity_generation):
            all_videos = self._sample_all_videos(batch, batch_idx, namespace)

        with torch.no_grad():
            self._update_metrics(all_videos)
            self._log_videos(all_videos, namespace)

    def on_validation_epoch_start(self) -> None:
        if self.hparams.log_deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank + self.trainer.world_size * self.hparams.log_deterministic
            )
        if self.hparams.is_latent_diffusion and not self.hparams.is_latent_online:
            self._load_vae()

    def on_validation_epoch_end(self, namespace="validation") -> None:
        self.generator = None
        if self.hparams.is_latent_diffusion and not self.hparams.is_latent_online:
            self.vae = None
            torch.cuda.empty_cache()
        self.num_logged_videos = 0

        if self.trainer.sanity_checking and not self.hparams.log_sanity_generation:
            return

        for task in self.tasks:
            self.log_dict(
                self._metrics(task).log(task),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    # ---------------------------------------------------------------------
    # Denoising Evaluation
    # ---------------------------------------------------------------------

    def _eval_denoising(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        xs, conditions, masks, gt_videos, *_ = batch

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_tokens]

        batch = (xs, conditions, masks, gt_videos)
        output = self.training_step(batch, batch_idx, namespace=namespace)

        gt_videos = gt_videos if self.hparams.is_latent_diffusion else output["xs"]
        recons = output["xs_pred"]
        if self.hparams.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos, recons = self.gather_data((gt_videos, recons))

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.hparams.log_max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.hparams.log_max_num_videos - self.num_logged_videos,
            gt_videos.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos[:num_videos_to_log],
            step=self.global_step,
            namespace="denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
            fps=self.hparams.log_fps,
        )

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation", tag: Optional[str] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        xs, conditions, masks, gt_videos = batch
        all_videos: Dict[str, torch.Tensor] = {"gt": xs}

        for task in self.tasks:
            sample_fn = self._predict_videos if task == "prediction" else self._interpolate_videos
            all_videos[task] = sample_fn(xs, conditions=conditions, tag=tag)

        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # unnormalize/detach the videos
        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.hparams.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos for k, v in all_videos.items()
            }

        # replace the context frames of video predictions with the ground truth
        if "prediction" in all_videos:
            all_videos["prediction"][:, : self.n_context_frames] = all_videos["gt"][
                :, : self.n_context_frames
            ]
        return all_videos

    def _predict_videos(
        self, xs: torch.Tensor, conditions: Optional[torch.Tensor] = None, tag=None
    ) -> torch.Tensor:
        """
        Predict the videos with the given context.
        Optionally do rolling/sliding windows if the sequence is large.
        """
        xs_pred = xs.clone()  # [b, t, c, h, w]

        xs_pred, _ = self._predict_sequence(
            context=xs_pred[:, : self.n_context_tokens],
            length=xs_pred.shape[1],
            conditions=conditions,
            reconstruction_guidance=self.hparams.reconstruction_guidance,
            sliding_context_len=self.hparams.sliding_context_len,
            return_all=False,
            future_frames=xs_pred[:, self.n_context_tokens :],
            # return_all=True,
        )
        return xs_pred

    # ---------------------------------------------------------------------
    # Training Utils
    # ---------------------------------------------------------------------

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Generate random noise levels for training."""
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if not self.diffusion_model.is_discrete  # self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.diffusion_model.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None
        if self.hparams.variable_context.enabled:
            assert (
                not self.hparams.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask = bernoulli_tensor(
                (batch_size, n_tokens),
                self.hparams.variable_context.prob,
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.hparams.fixed_context.enabled:
            context_indices = self.hparams.fixed_context.indices or list(
                range(self.n_context_tokens)
            )
            if self.n_context_tokens >= self.max_tokens:
                context_indices = range(self.hparams.sliding_context_len)

            context_mask = torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=xs.device)
            context_mask[:, context_indices] = True

        elif self.hparams.causal_context.enabled:
            num_context_frames = torch.randint(
                low=(
                    self.hparams.sliding_context_len
                    if self.hparams.causal_context.min in [None, 0]
                    else self.hparams.causal_context.min
                ),
                high=(
                    self.hparams.chunk_size - 1
                    if self.hparams.causal_context.max in [None, 0]
                    else self.hparams.causal_context.max
                ),  # last frame is never context
                size=(batch_size,),
                device=xs.device,
            )
            context_mask = torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=xs.device)
            for i in range(batch_size):
                context_mask[i, : num_context_frames[i]] = True

        match self.hparams.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)

        if self.hparams.uniform_future:  # simplified training (Appendix A.5)
            if self.n_context_tokens >= self.max_tokens:
                noise_levels[:, self.hparams.sliding_context_len :] = rand_fn(
                    (batch_size, 1)
                ).repeat(1, n_tokens - self.hparams.sliding_context_len)
            else:
                noise_levels[:, self.n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                    1, n_tokens - self.n_context_tokens
                )

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                (
                    1
                    if not self.diffusion_model.is_discrete
                    else self.diffusion_model.timesteps - 1
                ),
            ),
        )

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                (
                    self.hparams.variable_context
                    if self.hparams.variable_context.enabled
                    else self.hparams.fixed_context
                ).dropout
                if self.trainer.training
                else 0.0
            )
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if self.diffusion_model.is_discrete:
                context_noise_levels = context_noise_levels.long() * (
                    self.diffusion_model.timesteps - 1
                )

            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            if not self.hparams.cat_context_in_c_dim:
                noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(context_mask, "b t -> b t" + " 1" * len(masks.shape[2:]))
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks

    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    # ---------------------------------------------------------------------
    # Sampling Utilities
    # ---------------------------------------------------------------------

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        """
        Generates a scheduling matrix based on the self.hparams.scheduling_matrix parameter.
        Each row represents a noise-level index (or “timestep”).
        """

        match self.hparams.scheduling_matrix:
            case "full_sequence":
                # Each column has the same countdown from sampling_timesteps -> 0
                scheduling_matrix = np.arange(self.hparams.sampling_timesteps, -1, -1)[:, None]
                scheduling_matrix = np.repeat(scheduling_matrix, horizon, axis=1)

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)

                # If we want to pad the extra tokens as pure noise, do so here:
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (0, padding, 0, 0),
                    value=self.diffusion_model.timesteps - 1,  # or your “max noise index”
                )
            case "autoregressive":
                # Example pyramid / autoregressive schedule
                horizon = self.hparams.chunk_size - self.hparams.sliding_context_len
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon=horizon,
                    timesteps=self.hparams.sampling_timesteps,
                )
                padding = self.hparams.sliding_context_len

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (padding, 0, 0, 0),
                    value=-1,  # or your “max noise index”
                )

            case "causal":
                horizon = self.hparams.chunk_size - self.hparams.sliding_context_len
                scheduling_matrix = self._generate_staircase_scheduling_matrix(
                    horizon=horizon,
                    timesteps=self.hparams.sampling_timesteps,
                    group_size=self.hparams.group_size,
                )
                padding = self.hparams.sliding_context_len

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (padding, 0, 0, 0),
                    value=-1,  # or your “max noise index”
                )

            case other:
                raise ValueError(f"Unknown scheduling_matrix type: {other}")

        return scheduling_matrix

    def _generate_scheduling_matrix2(
        self,
        horizon: int,
        padding: int = 0,
    ):
        """
        Generates a scheduling matrix based on the self.hparams.scheduling_matrix parameter.
        Each row represents a noise-level index (or “timestep”).
        """

        match self.hparams.scheduling_matrix:
            case "full_sequence":
                # Each column has the same countdown from sampling_timesteps -> 0
                scheduling_matrix = np.arange(self.hparams.sampling_timesteps, -1, -1)[:, None]
                scheduling_matrix = np.repeat(scheduling_matrix, horizon, axis=1)

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)

                # If we want to pad the extra tokens as pure noise, do so here:
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (0, padding, 0, 0),
                    value=self.diffusion_model.timesteps - 1,  # or your “max noise index”
                )
            case "autoregressive":
                # Example pyramid / autoregressive schedule
                horizon = self.hparams.chunk_size - self.hparams.sliding_context_len
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon=horizon,
                    timesteps=self.hparams.sampling_timesteps,
                )
                padding = self.hparams.sliding_context_len

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (padding, 0, 0, 0),
                    value=-1,  # or your “max noise index”
                )

            case "causal":
                horizon = self.hparams.chunk_size - self.hparams.sliding_context_len
                scheduling_matrix = self._generate_staircase_scheduling_matrix(
                    horizon=horizon,
                    timesteps=self.hparams.sampling_timesteps,
                    group_size=self.hparams.group_size,
                )
                padding = self.hparams.sliding_context_len

                scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
                # Optionally convert from “timestep index” to actual noise levels:
                scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(scheduling_matrix)
                scheduling_matrix = F.pad(
                    scheduling_matrix,
                    (padding, 0, 0, 0),
                    value=-1,  # or your “max noise index”
                )

            case other:
                raise ValueError(f"Unknown scheduling_matrix type: {other}")

        return scheduling_matrix

    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
        future_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.max_tokens, and
            If bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use max_tokens - 1.
            Has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        """
        if length is None:
            length = self.max_tokens
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape

        chunk_size = (
            self.hparams.chunk_size
        )  # if self.hparams.use_causal_mask else self.max_tokens

        curr_token = gt_len
        xs_pred = context
        x_shape = self.x_shape
        record = None
        pbar = tqdm(
            total=self.hparams.sampling_timesteps
            * (1 + (length - gt_len - 1) // (chunk_size - sliding_context_len)),
            initial=0,
            desc="Dreaming Dreams...",
            leave=False,
        )
        max_horizon = chunk_size - sliding_context_len
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            c = min(sliding_context_len, curr_token)
            # try biggest prediction chunk size
            h = min(length - curr_token, chunk_size - c)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = min(h, max_horizon) if chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((batch_size, h, *x_shape))
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1) if c > 0 else None
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long) if c > 0 else None
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = (
                torch.cat([context_mask, pad.long()], 1).to(context.device) if c > 0 else None
            )

            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c : curr_token - c + chunk_size]

            new_pred, aux_output, record = self._sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                return_all=return_all,
                pbar=pbar,
            )

            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]

        pbar.close()

        return xs_pred, record

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        start_idx: int = 0,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(f"length is expected to <={self.max_tokens}, got {length}.")

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(f"context shape not compatible with x_stacked_shape {x_shape}.")

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if False:  # conditions is not None:
            if self.hparams.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.hparams.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
                )

        horizon = (
            self.hparams.chunk_size
        )  # length if self.hparams.use_causal_mask else self.hparams.chunk_size #self.max_tokens
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros((batch_size, horizon), dtype=torch.long, device=self.device)
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros((batch_size, padding, *x_shape), device=self.device)
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)
            conditions = torch.cat(
                [
                    conditions,
                    torch.zeros((batch_size, padding, *conditions.shape[2:]), device=self.device),
                ],
                1,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            # horizon,
            horizon - padding,
            0,  # padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(context_mask[None] >= 1, -1, scheduling_matrix)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            if reconstruction_guidance > 0:

                def composed_guidance_fn(
                    xk: torch.Tensor,
                    pred_x0: torch.Tensor,
                    alpha_cumprod: torch.Tensor,
                ) -> torch.Tensor:
                    loss = F.mse_loss(pred_x0, context, reduction="none") * alpha_cumprod.sqrt()
                    _context_mask = rearrange(
                        context_mask.bool(),
                        "b t -> b t" + " 1" * len(x_shape),
                    )
                    # scale inversely proportional to the number of context frames
                    loss = torch.sum(
                        loss * _context_mask / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                    )
                    likelihood = -reconstruction_guidance * 0.5 * loss
                    return likelihood

            else:
                composed_guidance_fn = guidance_fn

            # update xs_pred by DDIM or DDPM sampling
            xs_pred, aux_output = self.diffusion_model.sample_step(
                xs_pred,
                from_noise_levels,
                to_noise_levels,
                self._process_conditions(
                    conditions.clone() if conditions is not None else None,
                    from_noise_levels,
                ),
                conditions_mask,
                guidance_fn=composed_guidance_fn,
            )
            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev)
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, aux_output, record

    def _predict_sequence2(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.max_tokens, and
            If bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use max_tokens - 1.
            Has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        """
        if length is None:
            length = self.max_tokens
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape

        chunk_size = (
            self.hparams.chunk_size
        )  # if self.hparams.use_causal_mask else self.max_tokens

        curr_token = gt_len
        xs_pred = context
        x_shape = self.x_shape
        record = None
        pbar = tqdm(
            total=self.hparams.sampling_timesteps
            * (1 + (length - gt_len - 1) // (chunk_size - sliding_context_len)),
            initial=0,
            desc="Dreaming Dreams...",
            leave=False,
        )
        max_horizon = chunk_size - sliding_context_len
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            c = min(sliding_context_len, curr_token)
            # try biggest prediction chunk size
            h = min(length - curr_token, chunk_size - c)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = min(h, max_horizon) if chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((batch_size, h, *x_shape))
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1) if c > 0 else None
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long) if c > 0 else None
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = (
                torch.cat([context_mask, pad.long()], 1).to(context.device) if c > 0 else None
            )

            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c : curr_token - c + chunk_size]

            new_pred, aux_output, record = self._sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                return_all=return_all,
                pbar=pbar,
            )

            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]

        pbar.close()

        return xs_pred, record

    def _sample_sequence2(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        start_idx: int = 0,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(f"length is expected to <={self.max_tokens}, got {length}.")

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(f"context shape not compatible with x_stacked_shape {x_shape}.")

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if False:  # conditions is not None:
            if self.hparams.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.hparams.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
                )

        horizon = (
            self.hparams.chunk_size
        )  # length if self.hparams.use_causal_mask else self.hparams.chunk_size #self.max_tokens
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros((batch_size, horizon), dtype=torch.long, device=self.device)
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros((batch_size, padding, *x_shape), device=self.device)
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)
            conditions = torch.cat(
                [
                    conditions,
                    torch.zeros((batch_size, padding, *conditions.shape[2:]), device=self.device),
                ],
                1,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            # horizon,
            horizon - padding,
            0,  # padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(context_mask[None] >= 1, -1, scheduling_matrix)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            if reconstruction_guidance > 0:

                def composed_guidance_fn(
                    xk: torch.Tensor,
                    pred_x0: torch.Tensor,
                    alpha_cumprod: torch.Tensor,
                ) -> torch.Tensor:
                    loss = F.mse_loss(pred_x0, context, reduction="none") * alpha_cumprod.sqrt()
                    _context_mask = rearrange(
                        context_mask.bool(),
                        "b t -> b t" + " 1" * len(x_shape),
                    )
                    # scale inversely proportional to the number of context frames
                    loss = torch.sum(
                        loss * _context_mask / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                    )
                    likelihood = -reconstruction_guidance * 0.5 * loss
                    return likelihood

            else:
                composed_guidance_fn = guidance_fn

            # update xs_pred by DDIM or DDPM sampling
            xs_pred, aux_output = self.diffusion_model.sample_step(
                xs_pred,
                from_noise_levels,
                to_noise_levels,
                self._process_conditions(
                    conditions.clone() if conditions is not None else None,
                    from_noise_levels,
                ),
                conditions_mask,
                guidance_fn=composed_guidance_fn,
            )
            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev)
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, aux_output, record

    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extend the 2D [batch, tokens] mask to match the shape of your x_shape (i.e., [batch, tokens, c, h, w]).
        """
        return rearrange(
            x,
            "... -> ... " + "1 " * len(self.x_shape),  # e.g. " -> ... 1 1 1" if x_shape=[3,64,64]
        )

    # ---------------------------------------------------------------------
    # Example pyramid scheduling (if your "autoregressive" needs it)
    # ---------------------------------------------------------------------
    def _generate_pyramid_scheduling_matrix2(self, horizon: int, timesteps: int) -> np.ndarray:
        """
        Example “autoregressive” or “pyramid” scheduling logic:
        Decreasing timesteps on smaller chunks, etc.
        Feel free to replace with your own logic.
        """
        # Just a toy example: each column has a line from timesteps->0,
        # but we skip more steps as we move right, forming a “pyramid” shape.
        matrix = []
        for col in range(horizon):
            steps = np.linspace(timesteps, 0, timesteps - col if (timesteps - col) > 0 else 1)
            matrix_col = np.round(steps).astype(int)
            # pad the top so all columns have the same #rows:
            row_pad = timesteps + 1 - len(matrix_col)
            matrix_col = np.pad(matrix_col, (row_pad, 0), constant_values=-1)
            if len(matrix) == 0:
                matrix = matrix_col[:, None]
            else:
                matrix = np.concatenate([matrix, matrix_col[:, None]], axis=1)
        # Replace negative with 0 at the top:
        # matrix = np.where(matrix < 0, 0, matrix)
        return matrix

    def _generate_staircase_scheduling_matrix(
        self,
        horizon: int,
        timesteps: int,
        group_size: int = 1,
        extend_to_zero: bool = True,
    ) -> np.ndarray:
        """
        Grouped-staircase schedule (wider stairs).

        Columns are partitioned into groups of width `group_size`.
        All columns in the same group share the same noise level and start denoising together.
        Group 0 denoises first, then group 1, etc.

        Example with T=4, group_size=2, H=5:
        step 0:  [4,4, 4,4, 4]
        step 1:  [3,3, 4,4, 4]
        ...
        step 4:  [0,0, 4,4, 4]   # group 0 done
        step 5:  [0,0, 3,3, 4]
        ...
        step 9:  [0,0, 0,0, 4]   # group 1 done
        step 10: [0,0, 0,0, 3]
        ...
        step 14: [0,0, 0,0, 0]   # group 2 done

        Args:
            horizon: number of columns (e.g., frames), H.
            timesteps: maximum noise index T.
            group_size: number of columns per group (>=1).
            extend_to_zero: if True, run until the last (possibly partial) group reaches 0:
                rows = T * ceil(H / group_size) + 1.
                If False, only return T+1 rows (i.e., only the first group fully denoised).

        Returns:
            np.ndarray of shape:
                - (T * ceil(H / group_size) + 1, H) if extend_to_zero=True
                - (T + 1, H) if extend_to_zero=False
        """
        if group_size < 1:
            raise ValueError("group_size must be >= 1")

        T, H = timesteps, horizon
        n_groups = math.ceil(H / group_size)

        # rows (steps)
        S = (T * n_groups + 1) if extend_to_zero else (T + 1)

        # step and column indices
        s = np.arange(S, dtype=np.int32)[:, None]  # (S, 1)
        j = np.arange(H, dtype=np.int32)[None, :]  # (1, H)

        # group id per column
        g = j // group_size  # (1, H) in [0 .. n_groups-1]

        # each group g starts decreasing after g*T steps; clip to [0, T]
        dec = np.clip(s - g * T, 0, T)  # (S, H)

        # remaining noise level
        sched = (T - dec).astype(np.int32)  # (S, H)
        return sched

    def _generate_pyramid_scheduling_matrix(
        self,
        horizon: int,
        timesteps: int,
        extend_to_zero: bool = True,
    ) -> np.ndarray:
        """
        Pyramid schedule (prefix expansion):

        step 0: [T, T, T, T, ...]
        step 1: [T-1, T, T, T, ...]
        step 2: [T-2, T-1, T, T, ...]
        step 3: [T-3, T-2, T-1, T, ...]
        ...
        Once step >= horizon-1, all columns get denoised every step.

        Args:
            horizon: number of columns (e.g., frames).
            timesteps: maximum noise index T.
            extend_to_zero: if True, return enough rows so that *every* column
                reaches 0 by the last row (rows = timesteps + horizon).
                If False, return (timesteps+1, horizon) like your full_sequence case.

        Returns:
            scheduling_matrix: int array of shape
                - (timesteps+1, horizon) if extend_to_zero=False
                - (timesteps + horizon, horizon) if extend_to_zero=True
        """
        T = timesteps
        H = horizon

        # number of rows (steps) to return
        S = (T + H) if extend_to_zero else (T + 1)

        # s: step index (rows), j: column index
        s = np.arange(S, dtype=np.int32)[:, None]  # (S, 1)
        j = np.arange(H, dtype=np.int32)[None, :]  # (1, H)

        # how many times column j has been denoised by step s
        dec = np.clip(s - j, 0, T)  # (S, H)

        # remaining noise level per (s, j)
        sched = (T - dec).astype(np.int32)  # (S, H)

        # If not extending, keep exactly T+1 rows (T..0)
        if not extend_to_zero:
            sched = sched[: T + 1]

        return sched

    # ---------------------------------------------------------------------
    # Latent & Normalization Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _run_vae(
        self,
        x: torch.Tensor,
        shape: str,
        vae_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Helper function to run the vae, either for encoding or decoding.
        - Requires `shape` to be a permutation of b, t, c, h, w.
        - Rearranges the input tensor to the required shape for the vae, and
        reshapes the output back to `shape`.
            - x: `shape` shape.
            - Latent video VAE requires (b, c, t, h, w).
            - Image VAE requires (b, c, h, w).
        - Splits the input tensor into chunks to avoid memory errors.
        """

        # x = rearrange(x, f"{shape} -> b c t h w")
        # batch_size = x.shape[0]
        # vae_batch_size = self.hparams.vae_batch_size
        ## chunk the input tensor by vae_batch_size
        # chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, 0)
        # outputs = []
        # for chunk in chunks:
        #    b = chunk.shape[0]
        #    if not self.is_latent_video_vae:
        #        chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
        #    output = vae_fn(chunk)
        #    if not self.is_latent_video_vae:
        #        output = rearrange(output, "(b t) c h w -> b c t h w", b=b)
        #    outputs.append(output)
        # return rearrange(torch.cat(outputs, 0), f"b c t h w -> {shape}")
        # Bring the input to shape: (b, c, t, h, w)

        x = rearrange(x, f"{shape} -> b c t h w")
        if self.is_latent_video_vae:
            batch_size = x.shape[0]
            vae_batch_size = self.hparams.vae_batch_size
            # Chunk the input tensor by vae_batch_size along dim=0 (the batch).
            chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, dim=0)

            outputs = []
            for chunk in chunks:
                # chunk: (b_chunk, c, t, h, w)
                out_chunk = vae_fn(chunk)  # Must expect shape (b, c, t, h, w)
                outputs.append(out_chunk)

            out = torch.cat(outputs, dim=0)  # Concatenate along batch dimension

        else:
            # If we’re NOT using latent video VAE, we’ll chunk along the time dimension:
            #   (b, c, t, h, w) --> chunk in `t`.
            time_chunk_size = self.hparams.vae_time_chunk_size
            t_total = x.shape[2]

            # Number of time-chunks
            n_t_chunks = (t_total + time_chunk_size - 1) // time_chunk_size

            # Split along the time dimension (dim=2)
            tchunks = torch.chunk(x, n_t_chunks, dim=2)

            outputs = []
            for tchunk in tchunks:
                # tchunk: (b, c, t_chunk, h, w)
                b, c, t_chunk, h, w = tchunk.shape

                # Rearrange so that we can feed it into an image VAE which expects (batch, c, h, w).
                # We fold (b, t_chunk) together into one batch dimension.
                tchunk = rearrange(tchunk, "b c t h w -> (b t) c h w")

                out_chunk = vae_fn(tchunk)
                # out_chunk will be (b*t_chunk, c, h, w)

                # Unfold back from (b*t_chunk) to (b, t_chunk)
                out_chunk = rearrange(out_chunk, "(b t) c h w -> b c t h w", b=b, t=t_chunk)
                outputs.append(out_chunk)

            # Now concatenate over the time dimension
            out = torch.cat(outputs, dim=2)

        # Finally, reshape output back to the original "shape" ordering
        return rearrange(out, "b c t h w -> " + shape)

    def _encode(self, x: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(x, shape, lambda y: self.vae.encode(2.0 * y - 1.0).sample())

    def _decode(self, latents: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(
            latents,
            shape,
            lambda y: (
                self.vae.decode(y, self._n_tokens_to_n_frames(latents.shape[1]))
                if self.is_latent_video_vae
                else self.vae.decode(y)
            )
            * 0.5
            + 0.5,
        )

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------

    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.diffusion_model is compiled, to uncompiled."""
        if self.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model._orig_mod.", "diffusion_model."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model.", "diffusion_model._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return key.startswith("diffusion_model.model") or key.startswith(
            "diffusion_model._orig_mod.model"
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if self.hparams.reset_optimizer:
            checkpoint["optimizer_states"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if self._should_include_in_checkpoint(key) and key in checkpoint["state_dict"]:
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key) and key not in checkpoint["state_dict"]
        ]
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            if self.hparams.strict_load:
                raise ValueError(
                    "Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        checkpoint["state_dict"] = new_state_dict

    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if checkpoint.get("pretrained_ema", False) and len(checkpoint["optimizer_states"]) == 0:
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            "diffusion_model." + k for k, _ in self.diffusion_model.named_parameters()
        ]
        assert len(parameter_keys) == len(
            ema_weights
        ), "Number of original weights and EMA weights do not match."
        for key, weight in zip(parameter_keys, ema_weights):
            checkpoint["state_dict"][key] = weight

    def load_checkpoint_lenient(self, state_dict):
        """
        Load a checkpoint while ignoring parameters whose tensor shapes
        don’t match the current model.

        Behaviour:
        ----------
        • If self.hparams.strcit_load is truthy  → fall back to the usual strict load.
        • Else                                  → strip out shape-mismatched keys first,
                                                then load with strict=False so that
                                                *only* genuinely missing / unexpected
                                                keys are reported.

        Parameters
        ----------
        state_dict : dict
            The checkpoint state-dict you got from torch.load(...)

        Returns
        -------
        None  (prints diagnostic information instead)
        """
        current_state = self.state_dict()
        filtered_state = {}
        shape_mismatched = []

        for k, v in state_dict.items():
            if k not in current_state:
                # Key doesn’t exist in the model – let strict=False deal with it.
                continue
            if v.shape != current_state[k].shape:
                # Shape mismatch – drop it (and remember for logging).
                shape_mismatched.append((k, tuple(v.shape), tuple(current_state[k].shape)))
                continue
            filtered_state[k] = v

        # 3. Load what’s left.
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)

        # 4. Diagnostics – no hiding.
        if shape_mismatched:
            print("Shape-mismatched keys (skipped):")
            for name, ckpt_shape, model_shape in shape_mismatched:
                print(f"  {name}: checkpoint {ckpt_shape}  ↔  model {model_shape}")

        if missing_keys:
            print(f"Missing keys (not in checkpoint): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (in checkpoint only): {unexpected_keys}")
