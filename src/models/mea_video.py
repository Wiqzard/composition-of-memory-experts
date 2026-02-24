import contextlib
import copy
import io
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.models.components.optim.muon import Muon, SingleDeviceMuon
from src.models.metrics.video import SharedVideoMetricModelRegistry, VideoMetric

from .diffusion_video import DiffusionModelTrainer


def remap_state_dict(
    sd,
    drop_prefixes=("diffusion_model.model.", "model.", "module."),
    drop_subtrees=("ema_", "optimizer", "sched", "global_step"),
):
    out = {}
    for k, v in sd.items():
        # drop non-model stuff
        if any(k.startswith(p) for p in drop_subtrees):
            continue
        # strip known wrappers
        k2 = k
        for p in drop_prefixes:
            if k2.startswith(p):
                k2 = k2[len(p) :]
        # optionally drop lightning "._forward_module." wrapper (rare)
        if k2.startswith("_forward_module."):
            k2 = k2[len("_forward_module.") :]
        out[k2] = v
    return out


def _iter_kv_mem_blocks(module: nn.Module):
    for m in module.modules():
        if (
            hasattr(m, "k_mem")
            and hasattr(m, "v_mem")
            and isinstance(getattr(m, "k_mem"), torch.nn.Parameter)
        ):
            # only yield if both exist and are Parameters (not None)
            if m.k_mem is not None and m.v_mem is not None:
                yield m


def reset_all_kv_mem_(module: nn.Module):
    with torch.no_grad():
        for blk in _iter_kv_mem_blocks(module):
            blk.k_mem.zero_()
            blk.v_mem.zero_()


def set_kv_mem_trainable(module: nn.Module, trainable: bool = True):
    for blk in _iter_kv_mem_blocks(module):
        blk.k_mem.requires_grad = trainable
        blk.v_mem.requires_grad = trainable


def get_kv_mem_params(module: nn.Module):
    params = []
    for blk in _iter_kv_mem_blocks(module):
        params.extend([blk.k_mem, blk.v_mem])
    return params


class DiffusionModelTrainerMemoryAdaptation(DiffusionModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = self.hparams.adapter
        self.adapter_variant = adapter.variant
        if self.hparams.adapter.enabled:
            self.adapter_model = adapter.adapter_model

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

        if self.hparams.both_actions:
            mem_conditions = conditions[:, :, self.external_cond_dim :]
            conditions = conditions[:, :, : self.external_cond_dim]

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
            mem_conditions if self.hparams.both_actions else None,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        if self.hparams.adapter.enabled:
            params = self.adapter_model.parameters()

        else:
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
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
        )

        self.adapter_model_stm = None

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

        if self.adapter_variant is not None and self.hparams.adapter.enabled:
            #            raise ValueError("Adapter variant is not provided.")
            if self.adapter_variant == "full":
                del self.adapter_model
                self.adapter_model = copy.deepcopy(self.diffusion_model.model)
                state_dict = torch.load(self.hparams.ckpt_path, weights_only=False)["state_dict"]
                state_dict_ = {}
                for k, v in state_dict.items():
                    state_dict_[k] = v
                    k = k.replace("diffusion_model.model", "adapter_model.adapter.0")
                    state_dict_[k] = v
                    k = k.replace("adapter_model.adapter.0", "adapter_model")
                    state_dict_[k] = v

                state_dict = state_dict_
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")

                self.diffusion_model.eval()
                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

            elif self.adapter_variant == "custom":
                del self.adapter_model
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=self.hparams.adapter.lora_r,
                    lora_alpha=self.hparams.adapter.lora_alpha,
                    target_modules=self.hparams.adapter.lora_target_modules,
                    lora_dropout=0.00,
                )
                adapter_model_ltm = get_peft_model(
                    copy.deepcopy(self.diffusion_model.model), lora_config
                )

                for name, param in adapter_model_ltm.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                self.diffusion_model.eval()
                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

                # adapter_model_stm = self.hparams.adapter.stm_model.model(
                #    x_shape=self.x_shape,
                #    max_tokens=self.hparams.adapter.stm_model.max_tokens,
                #    external_cond_dim=self.external_cond_dim
                # )
                # state_dict = torch.load(
                #    self.hparams.adapter.stm_model.ckpt_path, weights_only=False
                # )["state_dict"]
                # missing_keys, unexpected_keys = adapter_model_stm.load_state_dict(state_dict, strict=False)
                # print(f"STM Missing keys: {missing_keys}")
                # if unexpected_keys:
                #    print(f"STM Unexpected keys: {unexpected_keys}")

                ########
                # ---- your code, patched ----
                adapter_model_stm = self.hparams.adapter.stm_model.model(
                    x_shape=self.x_shape,
                    max_tokens=self.hparams.adapter.stm_model.max_tokens,
                    external_cond_dim=self.external_cond_dim,
                )

                if self.hparams.adapter.stm_model.enabled:
                    ckpt = torch.load(
                        self.hparams.adapter.stm_model.ckpt_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                    sd = remap_state_dict(raw_sd)

                    missing, unexpected = adapter_model_stm.load_state_dict(sd, strict=False)
                    print("STM Missing keys:", missing)
                    print("STM Unexpected keys:", unexpected)

                self.adapter_model = adapter_model_ltm
                self.adapter_model_stm = adapter_model_stm

                for param in self.adapter_model_stm.parameters():
                    param.requires_grad = False

                self.adapter_model_stm.eval()

            elif self.adapter_variant == "custom_adapter_lora":
                ltm_ckpt = torch.load(
                    self.hparams.adapter.ckpt_path, map_location="cpu", weights_only=False
                )
                ltm_raw_sd = ltm_ckpt["state_dict"] if "state_dict" in ltm_ckpt else ltm_ckpt
                ltm_sd = remap_state_dict(ltm_raw_sd)

                missing, unexpected = self.adapter_model.load_state_dict(ltm_sd, strict=False)
                print("LTM Missing keys:", missing)
                print("LTM Unexpected keys:", unexpected)

                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

                for param in self.adapter_model.parameters():
                    param.requires_grad = False

                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=self.hparams.adapter.lora_r,
                    lora_alpha=self.hparams.adapter.lora_alpha,
                    target_modules=self.hparams.adapter.lora_target_modules,
                    lora_dropout=0.00,
                )
                self.adapter_model_uncond = copy.deepcopy(self.adapter_model)
                adapter_model_ltm = get_peft_model(copy.deepcopy(self.adapter_model), lora_config)

                for name, param in adapter_model_ltm.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                del self.adapter_model
                self.adapter_model = adapter_model_ltm

                if self.hparams.adapter.train_adapter:
                    self.diffusion_model.eval()
                    for param in self.diffusion_model.model.parameters():
                        param.requires_grad = False
                else:
                    self.diffusion_model.train()

                # ---- your code, patched ----
                adapter_model_stm = self.hparams.adapter.stm_model.model(
                    x_shape=self.x_shape,
                    max_tokens=self.hparams.adapter.stm_model.max_tokens,
                    external_cond_dim=self.external_cond_dim,
                )

                ckpt = torch.load(
                    self.hparams.adapter.stm_model.ckpt_path,
                    map_location="cpu",
                    weights_only=False,
                )
                raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                sd = remap_state_dict(raw_sd)

                missing, unexpected = adapter_model_stm.load_state_dict(sd, strict=False)
                print("STM Missing keys:", missing)
                print("STM Unexpected keys:", unexpected)

                self.adapter_model_stm = adapter_model_stm

                for param in self.adapter_model_stm.parameters():
                    param.requires_grad = False

                self.adapter_model_stm.eval()

            elif self.adapter_variant == "lora":
                del self.adapter_model
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=self.hparams.adapter.lora_r,
                    lora_alpha=self.hparams.adapter.lora_alpha,
                    target_modules=self.hparams.adapter.lora_target_modules,
                    lora_dropout=0.00,
                )
                self.adapter_model = get_peft_model(
                    copy.deepcopy(self.diffusion_model.model), lora_config
                )

                for name, param in self.adapter_model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                self.diffusion_model.eval()
                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

            elif self.adapter_variant == "mem_kv":
                # No separate adapter model: adapt via memory tokens inside diffusion_model
                # if sum(1 for _ in _iter_kv_mem_blocks(self.diffusion_model.model)) == 0:
                #    raise ValueError(
                #        "adapter_variant='mem_kv' requires Transformer blocks constructed with num_mem_lora_tokens > 0."
                #    )

                # Freeze everything
                for p in self.diffusion_model.model.parameters():
                    p.requires_grad = False

                # Enable only k_mem/v_mem and reset them to zero

                # Optional: put them in their own attribute for optimizer convenience
                # self.mem_kv_params = nn.ParameterList(
                #    get_kv_mem_params(self.diffusion_model.model)
                # )
                # self.adapter_model = copy.deepcopy(self.diffusion_model.model)
                ####

                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=self.hparams.adapter.lora_r,
                    lora_alpha=self.hparams.adapter.lora_alpha,
                    target_modules=self.hparams.adapter.lora_target_modules,
                    lora_dropout=0.00,
                )
                # self.adapter_model = get_peft_model(
                #    copy.deepcopy(self.diffusion_model.model), lora_config
                # )

                self.adapter_model = copy.deepcopy(self.diffusion_model.model)
                for name, param in self.diffusion_model.model.named_parameters():
                    if "lora" in name or "mem" in name:
                        del param

                for name, param in self.adapter_model.named_parameters():
                    if "lora" in name or "mem" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                ####
                ####set_kv_mem_trainable(self.adapter_model, True)
                # reset_all_kv_mem_(self.adapter_model)

                self.diffusion_model.eval()
                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

            elif self.adapter_variant == "adapter":
                adapter_ckpt = self.hparams.adapter.ckpt_path
                if adapter_ckpt is not None:
                    state_dict = torch.load(adapter_ckpt, weights_only=False)["state_dict"]
                    self.adapter_model.load_state_dict(state_dict, strict=False)

                if self.adapter_model is None:
                    raise ValueError("Adapter model is not provided.")

                self.diffusion_model.eval()
                for param in self.diffusion_model.model.parameters():
                    param.requires_grad = False

        if self.compile and stage == "fit":
            if False:  # self.compile == "True_without_ddp_optimizer":
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
    # Training
    # ---------------------------------------------------------------------

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        namespace: str = "training",
        batched_weights: bool = False,
        noise_levels: Optional[Tensor] = None,
        log=True,
    ) -> STEP_OUTPUT:
        """Training step"""
        xs, conditions, masks, *_ = batch
        if (
            self.hparams.both_actions and batch[-1] is not None
        ):  # and self.hparams.adapter.train_adapter :
            conditions = batch[-1]

        if noise_levels is None:
            noise_levels, masks = self._get_training_noise_levels(xs, masks)

        xs_pred, loss, aux_output = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            adapter=(  # self.adapter_model #
                self.adapter_model
                if self.hparams.adapter.enabled
                # if (self.hparams.adapter.enabled and self.hparams.adapter.train_adapter)
                or self.hparams.adapter.stm_only or self.hparams.both_actions
                else None
            ),
            stm_model=self.adapter_model_stm if hasattr(self, "adapter_model_stm") else None,
            combine=self.hparams.adapter.combine,
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.trainer.log_every_n_steps == 0 and log:
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

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

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

    # @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.

        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        with torch.no_grad():
            if self.trainer.state.fn == "fit" and self.hparams.log_denoising:
                self._eval_denoising(batch, batch_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        if not (self.trainer.sanity_checking and not self.hparams.log_sanity_generation):
            all_videos = {}
            if self.hparams.adapter.generate_unbatched and self.hparams.adapter.enabled:
                self.adapter_model.train()  # set adapter to train mode (for batchnorm, dropout etc.)
                original_adapter_state = copy.deepcopy(self.adapter_model.state_dict())
                batch_size = batch[0].shape[0]
                for i in range(batch_size):
                    # print("optimizer")
                    if (
                        self.hparams.adapter.optimizer == "adamw"
                        or self.hparams.adapter.optimizer is None
                    ):
                        self.local_optimizer = torch.optim.AdamW(
                            self.adapter_model.parameters(),
                            lr=self.hparams.adapter.adapter_lr,
                        )
                    elif self.hparams.adapter.optimizer == "muon":
                        muon_params = [
                            p
                            for n, p in self.adapter_model.named_parameters()
                            if p.requires_grad and p.dim() >= 2
                        ]

                        self.local_optimizer = Muon(  # SingleDeviceMuon(
                            muon_params,  # self.adapter_model.parameters(),
                            lr=self.hparams.adapter.adapter_lr,
                            weight_decay=0,
                            momentum=0.95,
                        )

                    elif self.hparams.adapter.optimizer == "sgd":
                        self.local_optimizer = torch.optim.SGD(
                            self.adapter_model.parameters(),
                            lr=self.hparams.adapter.adapter_lr,
                            momentum=0.9,  # typical value
                            weight_decay=0,  # 1e-4,  # l2 regularization (not decoupled like adamw)
                        )

                    batch_i = [
                        batch[j][i : i + 1] for j in range(len(batch)) if batch[j] is not None
                    ]
                    batch_i = batch_i + [None] * (len(batch) - len(batch_i))
                    videos = self._sample_all_videos(
                        batch_i, batch_idx, namespace, tag=f"unbatched_{i}"
                    )
                    for key, vid in videos.items():
                        if key not in all_videos:
                            all_videos[key] = [vid]  # start a list of Tensors
                        else:
                            all_videos[key].append(vid)

                    self.adapter_model.load_state_dict(original_adapter_state)
                    del self.local_optimizer

                for key in all_videos:
                    all_videos[key] = torch.cat(
                        all_videos[key], dim=0
                    )  # now shape: (b, t, c, h, w)

            else:
                with torch.no_grad():
                    all_videos = self._sample_all_videos(batch, batch_idx, namespace)

            with torch.no_grad():
                self._update_metrics(all_videos)
                self._log_videos(all_videos, namespace)

    def _predict_videos(
        self,
        xs: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        tag=None,
        mem_conditions=None,
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
            mem_conditions=mem_conditions,
            # return_all=True,
        )
        return xs_pred

    def _sample_all_videos(
        self,
        batch,
        batch_idx,
        namespace="validation",
        tag: Optional[str] = None,
        mem_conditions: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if len(batch) == 4:
            xs, conditions, masks, gt_videos = batch
        else:
            xs, conditions, masks, gt_videos, mem_conditions = batch
        all_videos: Dict[str, torch.Tensor] = {"gt": xs}

        for task in self.tasks:
            sample_fn = self._predict_videos if task == "prediction" else self._interpolate_videos
            all_videos[task] = sample_fn(
                xs, conditions=conditions, tag=tag, mem_conditions=mem_conditions
            )

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

    def on_validation_epoch_start(self) -> None:
        if self.hparams.log_deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank + self.trainer.world_size * self.hparams.log_deterministic
            )
        if self.hparams.is_latent_diffusion and not self.hparams.is_latent_online:
            self._load_vae()

        if self.hparams.adapter.enabled:
            self.adapter_model.train()

        self._mem_curves = {}

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

        if self.hparams.adapter.enabled:
            self.adapter_model.train()

    def _no_sync_cm(self):
        strat = self.trainer.strategy
        if getattr(strat, "is_distributed", False) and hasattr(strat, "no_backward_sync"):
            return strat.no_backward_sync(self)  # works for DDPStrategy
        return nullcontext()

    def memorize_sequence2(
        self,
        context: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        namespace: str = "validation",
        sample_tag: Optional[str] = None,
    ) -> None:
        """
        perform local adaptation on the adapter_model for a single validation batch.
        weights are restored afterwards.
        """
        # rank_zero_print(f"starting local memorization for batch on rank {self.global_rank}")
        torch.set_grad_enabled(True)
        self.automatic_optimization = False

        self.adapter_model.train()
        model_to_sync_control = self.trainer.model if hasattr(self.trainer, "model") else self
        is_ddp = isinstance(model_to_sync_control, DDP)
        per_step_losses: list[float] = []

        for module in self.adapter_model.modules():
            setattr(module, "is_training", True)  # set training mode for all modules
            setattr(module, "local_loss", None)  # reset local loss for all modules

        # if True:
        maybe_no_sync = model_to_sync_control.no_sync() if is_ddp else contextlib.nullcontext()
        # precision = self.trainer.strategy.precision_plugin
        # maybe_no_sync =  contextlib.nullcontext()
        total_loss = 0.0
        layer_losses = []
        # with maybe_no_sync:
        with self._no_sync_cm():
            # mem_start = time.perf_counter()
            # for i in tqdm(range(self.hparams.adapter.num_memorization_steps)):
            iterator = range(self.hparams.adapter.num_memorization_steps)
            if self.global_rank == 0:
                iterator = tqdm(iterator)
            for i in iterator:
                if context.shape[1] > self.hparams.chunk_size:
                    # randomly sample a chunk of the context
                    strategy = self.hparams.adapter.memorization_strategy
                    if strategy == "random":
                        start_idx = torch.randint(
                            0,
                            context.shape[1] - self.hparams.chunk_size + 1,
                            (1,),
                            device=self.device,
                        ).item()

                    elif strategy == "sequential_block":
                        n_blocks = context.shape[1] // self.hparams.chunk_size
                        n_steps_per_block = self.hparams.adapter.num_memorization_steps // n_blocks
                        curr_block = i // n_steps_per_block
                        curr_block = min(curr_block, n_blocks - 1)  # avoid overflow
                        start_idx = torch.randint(
                            curr_block * self.hparams.chunk_size,
                            min(
                                (curr_block + 1) * self.hparams.chunk_size,
                                context.shape[1] - self.hparams.chunk_size + 1,
                            ),
                            (1,),
                            device=self.device,
                        )

                    elif strategy == "random_max_2_draws":
                        # draws each index maximum 2 times
                        a = 2

                    elif strategy == "sequential":
                        # take last 100 frames
                        # start_idx = - i % min(100, context.shape[1] - self.hparams.chunk_size + 1)
                        max_start = context.shape[1] - self.hparams.chunk_size
                        start_idx = min(i, max_start)

                        # 1 for first 50, 2 for 50-75, 4 for 75-100
                    elif strategy == "sequential_last":
                        t = context.shape[1]
                        chunk_size = self.hparams.chunk_size
                        start_min = max(0, t - 2 * chunk_size)
                        start_max = t - chunk_size  # inclusive
                        start_idx = torch.randint(
                            start_min, start_max + 1, (1,), device=self.device
                        ).item()

                    context_to_memorize = context[
                        :, start_idx : start_idx + self.hparams.chunk_size
                    ]
                    conditions_to_memorize = (
                        conditions[:, start_idx : start_idx + self.hparams.chunk_size]
                        if conditions is not None
                        else None
                    )

                    # batched training
                    if False:
                        t = context.shape[1]
                        chunk_size = self.hparams.chunk_size
                        num_chunks = 16  # number of random chunks to sample
                        assert t >= chunk_size, "context too short to sample chunks"
                        start_indices = torch.randint(
                            0,
                            t - chunk_size + 1,
                            (num_chunks,),
                            device=context.device,
                        )

                        # collect b chunks efficiently
                        context_to_memorize = torch.stack(
                            [context[0, start : start + chunk_size] for start in start_indices],
                            dim=0,
                        )  # [b, chunk_size, d]
                        conditions_to_memorize = (
                            torch.stack(
                                [
                                    conditions[0, start : start + chunk_size]
                                    for start in start_indices
                                ],
                                dim=0,
                            )
                            if conditions is not None
                            else None
                        )

                else:
                    # use the whole context
                    context_to_memorize = context
                    conditions_to_memorize = conditions[:,]

                # adaptation_batch = context, conditions, masks.bool(), None
                adaptation_batch = (
                    context_to_memorize,
                    conditions_to_memorize,
                    masks.bool(),
                    None,
                )
                # self.local_optimizer.zero_grad()

                noise_levels = None
                # if i < 200:
                #    batch_size, n_tokens, *_ = context_to_memorize.shape
                #    # noise_levels = torch.ones(batch_size, n_tokens, device=self.device) * 0
                #    noise_levels = (
                #        torch.rand(batch_size, n_tokens, device=self.device) * 0.1
                #    )
                out_dict = self.training_step(
                    adaptation_batch,
                    i,
                    namespace="memorization_val",
                    batched_weights=False,
                    noise_levels=noise_levels,
                )
                loss = out_dict["loss"]
                loss_item = float(loss.detach().item())

                if True:
                    losses = []
                    for module in self.adapter_model.modules():
                        if hasattr(module, "local_loss"):
                            val = getattr(module, "local_loss")
                            if val is not None:
                                losses.append(val)

                    if len(losses) > 0:
                        total_local_loss = torch.stack(
                            losses
                        ).sum()  # mean()  # or torch.stack(losses).mean() if they’re tensors
                        if i < 200:
                            loss += total_local_loss * self.hparams.adapter.local_lr
                        # for each layer append the loss to the layer_losses
                        if len(layer_losses) == 0:
                            layer_losses = [
                                [losses[i].mean().detach().item()] for i in range(len(losses))
                            ]
                        else:
                            layer_losses = [
                                (layer_losses[i] + [losses[i].mean().detach().item()])
                                for i in range(len(losses))
                            ]

                        # layer_losses = torch.stack(losses).detach().mean(-1).mean(-1).squeeze()
                    else:
                        total_local_loss = None

                # loss.backward()  # standard backward call

                self.manual_backward(loss)  # not loss.backward()
                self.local_optimizer.zero_grad(set_to_none=True)
                self.local_optimizer.step()

                total_grad = sum(
                    p.grad.abs().sum()
                    for p in self.adapter_model.parameters()
                    if p.grad is not None
                )
                if total_grad == 0:
                    print("no gradients to optimize")
                if torch.isnan(loss):
                    print("loss is nan")
                self.local_optimizer.step()
                per_step_losses.append(loss_item)
                total_loss += loss_item

        total_loss /= self.hparams.adapter.num_memorization_steps
        # print(f"rank: {self.global_rank}, loss: {total_loss:.4f}")

        # self.timings["memorization"] += time.perf_counter() - mem_start

        torch.set_grad_enabled(False)  # disable gradients after adaptation
        self.automatic_optimization = True
        self.adapter_model.eval()  # set adapter back to eval mode
        # pick a stable tag if none provided
        if sample_tag is None:
            sample_tag = f"{sample_tag}_rank{getattr(self, 'global_rank', 0)}"
            # sample_tag = f"rank{self.global_rank}_t{int(time.time())}"

        for module in self.adapter_model.modules():
            setattr(module, "is_training", False)  # set training mode for all modules
            setattr(module, "local_loss", None)  # reset local loss for all modules

        # store raw + log curve
        self._mem_curves.setdefault(sample_tag, []).extend(per_step_losses)
        self._log_mem_curve(sample_tag, per_step_losses, namespace=namespace)

        if len(layer_losses) > 0:
            self._log_layer_mem_curves(sample_tag, layer_losses, namespace=namespace)

    def memorize_sequence(
        self,
        context: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        namespace: str = "validation",
        sample_tag: Optional[str] = None,
    ) -> None:
        """
        perform local adaptation on the adapter_model for a single validation batch.
        weights are restored afterwards.
        """
        # rank_zero_print(f"starting local memorization for batch on rank {self.global_rank}")
        torch.set_grad_enabled(True)
        self.adapter_model.train()
        model_to_sync_control = self.trainer.model if hasattr(self.trainer, "model") else self
        is_ddp = isinstance(model_to_sync_control, DDP)
        per_step_losses: list[float] = []
        # self.trainer.training = True

        for module in self.adapter_model.modules():
            setattr(module, "is_training", True)  # set training mode for all modules
            setattr(module, "local_loss", None)  # reset local loss for all modules

        # try:
        # if True:
        maybe_no_sync = model_to_sync_control.no_sync() if is_ddp else contextlib.nullcontext()
        # precision = self.trainer.strategy.precision_plugin
        # maybe_no_sync =  contextlib.nullcontext()
        total_loss = 0.0
        layer_losses = []
        with maybe_no_sync:
            # mem_start = time.perf_counter()
            # for i in tqdm(range(self.hparams.adapter.num_memorization_steps)):
            iterator = range(self.hparams.adapter.num_memorization_steps)
            if self.global_rank == 0:
                iterator = tqdm(iterator)
            for i in iterator:
                if context.shape[1] > self.hparams.chunk_size:
                    # randomly sample a chunk of the context
                    strategy = self.hparams.adapter.memorization_strategy
                    if strategy == "random":
                        start_idx = torch.randint(
                            0,
                            context.shape[1] - self.hparams.chunk_size + 1,
                            (1,),
                            device=self.device,
                        ).item()

                    elif strategy == "sequential_block":
                        n_blocks = context.shape[1] // self.hparams.chunk_size
                        n_steps_per_block = self.hparams.adapter.num_memorization_steps // n_blocks
                        curr_block = i // n_steps_per_block
                        curr_block = min(curr_block, n_blocks - 1)  # avoid overflow
                        start_idx = torch.randint(
                            curr_block * self.hparams.chunk_size,
                            min(
                                (curr_block + 1) * self.hparams.chunk_size,
                                context.shape[1] - self.hparams.chunk_size + 1,
                            ),
                            (1,),
                            device=self.device,
                        )

                    elif strategy == "random_max_2_draws":
                        # draws each index maximum 2 times
                        a = 2

                    elif strategy == "sequential":
                        # take last 100 frames
                        # start_idx = - i % min(100, context.shape[1] - self.hparams.chunk_size + 1)
                        max_start = context.shape[1] - self.hparams.chunk_size
                        start_idx = min(i, max_start)

                        # 1 for first 50, 2 for 50-75, 4 for 75-100
                    elif strategy == "sequential_last":
                        t = context.shape[1]
                        chunk_size = self.hparams.chunk_size
                        start_min = max(0, t - 2 * chunk_size)
                        start_max = t - chunk_size  # inclusive
                        start_idx = torch.randint(
                            start_min, start_max + 1, (1,), device=self.device
                        ).item()

                    context_to_memorize = context[
                        :, start_idx : start_idx + self.hparams.chunk_size
                    ]
                    conditions_to_memorize = (
                        conditions[:, start_idx : start_idx + self.hparams.chunk_size]
                        if conditions is not None
                        else None
                    )

                    # batched training
                    if False:
                        t = context.shape[1]
                        chunk_size = self.hparams.chunk_size
                        num_chunks = 16  # number of random chunks to sample
                        assert t >= chunk_size, "context too short to sample chunks"
                        start_indices = torch.randint(
                            0,
                            t - chunk_size + 1,
                            (num_chunks,),
                            device=context.device,
                        )

                        # collect b chunks efficiently
                        context_to_memorize = torch.stack(
                            [context[0, start : start + chunk_size] for start in start_indices],
                            dim=0,
                        )  # [b, chunk_size, d]
                        conditions_to_memorize = (
                            torch.stack(
                                [
                                    conditions[0, start : start + chunk_size]
                                    for start in start_indices
                                ],
                                dim=0,
                            )
                            if conditions is not None
                            else None
                        )

                else:
                    # use the whole context
                    context_to_memorize = context
                    conditions_to_memorize = conditions[:,]

                # adaptation_batch = context, conditions, masks.bool(), None
                adaptation_batch = (
                    context_to_memorize,
                    conditions_to_memorize,
                    masks.bool(),
                    None,
                )
                self.local_optimizer.zero_grad()

                noise_levels = None
                # if i < 200:
                #    batch_size, n_tokens, *_ = context_to_memorize.shape
                #    # noise_levels = torch.ones(batch_size, n_tokens, device=self.device) * 0
                #    noise_levels = (
                #        torch.rand(batch_size, n_tokens, device=self.device) * 0.1
                #    )
                out_dict = self.training_step(
                    adaptation_batch,
                    i,
                    namespace="memorization_val",
                    batched_weights=False,
                    noise_levels=noise_levels,
                    log=False,
                )
                loss = out_dict["loss"]
                loss_item = float(loss.detach().item())

                if False:  # True:
                    losses = []
                    for module in self.adapter_model.modules():
                        if hasattr(module, "local_loss"):
                            val = getattr(module, "local_loss")
                            if val is not None:
                                losses.append(val)

                    if len(losses) > 0:
                        total_local_loss = torch.stack(
                            losses
                        ).sum()  # mean()  # or torch.stack(losses).mean() if they’re tensors
                        # if i < 200:
                        loss += total_local_loss * self.hparams.adapter.local_lr
                        # for each layer append the loss to the layer_losses
                        if len(layer_losses) == 0:
                            layer_losses = [
                                [losses[i].mean().detach().item()] for i in range(len(losses))
                            ]
                        else:
                            layer_losses = [
                                (layer_losses[i] + [losses[i].mean().detach().item()])
                                for i in range(len(losses))
                            ]

                        # layer_losses = torch.stack(losses).detach().mean(-1).mean(-1).squeeze()
                    else:
                        total_local_loss = None

                loss.backward()  # standard backward call
                total_grad = sum(
                    p.grad.abs().sum()
                    for p in self.adapter_model.parameters()
                    if p.grad is not None
                )
                if total_grad == 0:
                    print("no gradients to optimize")
                if torch.isnan(loss):
                    print("loss is nan")
                self.local_optimizer.step()
                per_step_losses.append(loss_item)
                total_loss += loss_item

        total_loss /= self.hparams.adapter.num_memorization_steps
        # print(f"rank: {self.global_rank}, loss: {total_loss:.4f}")

        # self.timings["memorization"] += time.perf_counter() - mem_start

        # except Exception as e:
        #    print(f"error during memorization: {e}")

        # finally:
        torch.set_grad_enabled(False)  # disable gradients after adaptation
        self.adapter_model.eval()  # set adapter back to eval mode

        # self.trainer.training = False
        # pick a stable tag if none provided
        if sample_tag is None:
            sample_tag = f"{sample_tag}_rank{getattr(self, 'global_rank', 0)}"
            # sample_tag = f"rank{self.global_rank}_t{int(time.time())}"

        for module in self.adapter_model.modules():
            setattr(module, "is_training", False)  # set training mode for all modules
            setattr(module, "local_loss", None)  # reset local loss for all modules

        # store raw + log curve
        self._mem_curves.setdefault(sample_tag, []).extend(per_step_losses)
        self._log_mem_curve(sample_tag, per_step_losses, namespace=namespace)

        if len(layer_losses) > 0:
            self._log_layer_mem_curves(sample_tag, layer_losses, namespace=namespace)

    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[callable] = None,
        reconstruction_guidance: float = 0.0,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
        tag: Optional[str] = None,
        future_frames: Optional[int] = None,
        mem_conditions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.tensor]]:
        """
        predict a sequence given context tokens at the beginning, using sliding window if necessary.
        args
        ----
        context: torch.Tensor, shape (batch_size, init_context_len, *self.x_shape)
            initial context tokens to condition on
        length: Optional[int]
            desired number of tokens in sampled sequence.
            if None, fall back to to self.max_tokens, and
            if bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], shape (batch_size, conditions_len, ...)
            unprocessed external conditions for sampling, e.g. action or text, Optional
        guidance_fn: Optional[callable]
            guidance function for sampling
        reconstruction_guidance: float
            scale of reconstruction guidance (from video diffusion models ho. et al.)
        sliding_context_len: Optional[int]
            max context length when using sliding window. -1 to use max_tokens - 1.
            has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            whether to return all steps of the sampling process.

        returns
        -------
        xs_pred: torch.Tensor, shape (batch_size, length, *self.x_shape)
            predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], shape (num_steps, batch_size, length, *self.x_shape)
            record of all steps of the sampling process
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
            desc="dreaming dreams...",
            leave=False,
        )
        max_horizon = chunk_size - sliding_context_len
        mem_iter = 0
        sample_tag_base = f"ep{self.current_epoch}_{tag}_rank{self.global_rank}"
        context_copy = context.clone()
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")

            ################################################################
            # <-------------------- start memorization -------------------->
            ################################################################
            if self.hparams.adapter.enabled and self.hparams.adapter.train_adapter:
                # extract the latest context
                if False:
                    context_to_memorize = xs_pred[:, -chunk_size:]
                    external_cond = conditions[:, -chunk_size:] if conditions is not None else None

                    # pad context if it's shorter than chunk_size
                    pad_len = max(0, chunk_size - context_to_memorize.shape[1])
                    if pad_len > 0:
                        pad_mem = torch.zeros((batch_size, pad_len, *x_shape), device=self.device)
                        context_to_memorize = torch.cat([context_to_memorize, pad_mem], dim=1)

                    # create memory mask
                    memory_mask = torch.ones(
                        (batch_size, chunk_size), dtype=torch.long, device=self.device
                    )

                    memory_mask[:, :sliding_context_len] = 0
                    if pad_len > 0:
                        memory_mask[:, -pad_len:] = 0
                    # else:
                    #    memory_mask[:, sliding_context_len:] = 1
                else:
                    context_to_memorize = xs_pred
                    pad_len = max(0, chunk_size - context_to_memorize.shape[1])

                    external_cond = conditions[:, : xs_pred.shape[1]]

                    if mem_conditions is not None:
                        external_cond = mem_conditions[:, : xs_pred.shape[1]]

                    memory_mask = torch.ones(
                        (batch_size, chunk_size), dtype=torch.long, device=self.device
                    )

                if self.hparams.adapter.only_memorize_context:
                    context_to_memorize = context_to_memorize[:, : self.n_context_tokens]

                    external_cond = external_cond[:, : self.n_context_tokens]

                    memory_mask = memory_mask[:, : self.n_context_tokens]
                    assert context_to_memorize.shape[1] >= self.hparams.chunk_size
                    if mem_iter == 0:
                        self.memorize_sequence(
                            context_to_memorize,
                            conditions=external_cond,
                            masks=memory_mask,
                            sample_tag=f"{sample_tag_base}_onlyctx",
                            namespace="validation",
                        )
                        mem_iter += 1
                else:
                    if pad_len == 0:
                        # memorize
                        self.memorize_sequence(
                            context_to_memorize,
                            conditions=external_cond,
                            masks=memory_mask,
                            sample_tag=f"{sample_tag_base}_iter{mem_iter}",
                            namespace="validation",
                        )
            self.eval()
            ##############################################################
            # <-------------------- end memorization -------------------->
            ##############################################################

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
            # calculate number of model generated tokens (not gt context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long) if c > 0 else None

            full_context_mask = torch.zeros(
                (batch_size, curr_token + max_horizon), dtype=torch.long
            ).to(xs_pred.device)
            # full_context_mask[:, curr_token - c : curr_token] = 1
            full_context_mask[:, :curr_token] = 1
            if self.hparams.use_gt_context_only:
                full_context = torch.cat(
                    [context_copy, future_frames[:, : curr_token - context_copy.shape[1]]], dim=1
                )
                context = (
                    torch.cat([full_context[:, -c:], pad.to(self.device)], 1) if c > 0 else None
                )
                full_conditions = conditions[:, : curr_token - c + chunk_size]
            else:
                full_context = xs_pred
                full_conditions = conditions[:, : curr_token - c + chunk_size]

            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
                full_context_mask[:, curr_token - generated_len : curr_token] = 2

            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = (
                torch.cat([context_mask, pad.long()], 1).to(context.device) if c > 0 else None
            )

            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c : curr_token - c + chunk_size]

            if mem_conditions is not None:
                mem_cond_slice = mem_conditions[:, curr_token - c : curr_token - c + chunk_size]

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
                full_context=full_context,
                full_context_mask=full_context_mask,
                full_conditions=full_conditions,
                future_frames=future_frames,
                mem_conditions=mem_cond_slice if mem_conditions is not None else None,
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
        guidance_fn: Optional[callable] = None,
        reconstruction_guidance: float = 0.0,
        start_idx: int = 0,
        pbar: Optional[tqdm] = None,
        full_context: Optional[torch.Tensor] = None,
        full_context_mask: Optional[torch.Tensor] = None,
        full_conditions: Optional[torch.Tensor] = None,
        future_frames: Optional[int] = None,
        mem_conditions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.tensor]]:
        """
        the unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        args
        ----
        batch_size: int
            batch size of the sampling process
        length: Optional[int]
            number of frames in sampled sequence
            if None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], shape (batch_size, length, *self.x_shape)
            context tokens to condition on. assumed to be same across batch.
            tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], shape (batch_size, length)
            mask for context
            0 = to be generated, 1 = ground truth context, 2 = generated context
            some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            unprocessed external conditions for sampling
        guidance_fn: Optional[callable]
            guidance function for sampling
        history_guidance: Optional[historyguidance]
            history guidance object that handles compositional generation
        return_all: bool
            whether to return all steps of the sampling process
        returns
        -------
        xs_pred: torch.Tensor, shape (batch_size, length, *self.x_shape)
            complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], shape (num_steps, batch_size, length, *self.x_shape)
            all recorded intermediate results during the sampling process
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
        orig_noise = xs_pred

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros((batch_size, horizon), dtype=torch.long, device=self.device)
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros((batch_size, padding, *x_shape), device=self.device)
            # note: in context mask, -1 = padding, 0 = to be generated, 1 = gt context, 2 = generated context
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
        if self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled:
            full_context_mask = full_context_mask[:, -self.hparams.adapter.stm_model.max_tokens :]
            full_conditions = full_conditions[:, -self.hparams.adapter.stm_model.max_tokens :]
            # full_context = full_context[:, -(self.hparams.adapter.stm_model.max_tokens-horizon):]
            full_context = full_context[:, -(self.hparams.adapter.stm_model.max_tokens) :]
            # stm_padding = max(0, self.hparams.adapter.stm_model.max_tokens - full_context_mask.shape[1])
            # stm_padding_context = max(0, self.hparams.adapter.stm_model.max_tokens - horizon - full_context.shape[1])
            # if stm_padding > 0:
            #    full_context_mask = torch.cat([torch.ones((batch_size, stm_padding), device=self.device)*-1, full_context_mask], 1)
            #    full_conditions = torch.cat([torch.zeros((batch_size, stm_padding, *full_conditions.shape[2:]), device=self.device), full_conditions], 1)

            # if stm_padding_context > 0:
            #    full_context = torch.cat([torch.zeros((batch_size, stm_padding_context, *full_context.shape[2:]), device=self.device), full_context], 1)

            stm_scheduling_matrix = self._generate_scheduling_matrix(
                full_context_mask.shape[1],  # + stm_padding,
                0,
            )
            stm_scheduling_matrix = stm_scheduling_matrix.to(self.device)
            stm_scheduling_matrix = repeat(stm_scheduling_matrix, "m t -> m b t", b=batch_size)
            if not self.is_full_sequence:
                stm_scheduling_matrix = torch.where(
                    full_context_mask[None] >= 1, -1, stm_scheduling_matrix
                )
            stm_diff = stm_scheduling_matrix[1:] - stm_scheduling_matrix[:-1]
            stm_skip = torch.argmax((~reduce(stm_diff == 0, "m b t -> m", torch.all)).float())
            stm_scheduling_matrix = stm_scheduling_matrix[stm_skip:]
            full_xs_pred = torch.randn(
                # (batch_size, full_context_mask.shape[1]-xs_pred.shape[1], *x_shape),
                # (batch_size, full_context_mask.shape[1], *x_shape),
                (batch_size, self.hparams.adapter.stm_model.max_tokens, *x_shape),
                device=self.device,
                generator=self.generator,
            )
            full_xs_pred = torch.clamp(
                full_xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise
            )
            full_xs_pred_init_noise = full_xs_pred[:, : self.hparams.adapter.stm_model.n_context]
            # full_xs_pred = torch.where(self._extend_x_dim(full_context_mask[:,full_xs_pred.shape[1]]) >= 1, full_context, full_xs_pred)
            # full_xs_pred = torch.cat([full_xs_pred[:, :self.hparams.adapter.stm_model.n_context], xs_pred[:, self.hparams.sliding_context_len:]], 1)
            # full_xs_pred = torch.where(self._extend_x_dim(full_context_mask) >= 1, full_context, full_xs_pred)
            # if full_xs_pred.shape[1] != self.hparams.adapter.stm_model.max_tokens:
            #    raise ValueError("full_xs_pred length incorrect")

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        if self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled:
            # append extra context to xs_pred for stm adapter
            stm_max_tokens = self.hparams.adapter.stm_model.max_tokens
            n_context = self.hparams.sliding_context_len
            # want the last
            # full_xs_pred = torch.cat([full_context[:, -stm_max_tokens:-horizon], xs_pred], dim=1)
            full_xs_pred = torch.cat(
                [
                    full_context[:, -stm_max_tokens + xs_pred.shape[1] - n_context : -n_context],
                    xs_pred,
                ],
                dim=1,
            )
            if self.hparams.adapter.stm_model.limit_stm_context > 0:
                mask_end = (
                    self.hparams.adapter.stm_model.n_context
                    - self.hparams.adapter.stm_model.limit_stm_context
                )
                full_xs_pred[:, :mask_end] = full_xs_pred_init_noise[:, :mask_end]
                full_context_mask[:, :mask_end] = 0  # -1
                stm_scheduling_matrix[:, :, :mask_end] = 999
            assert full_xs_pred.shape[1] == stm_max_tokens

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
                desc="sampling with dfot",
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
            if self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled:
                from_noise_levels_stm = stm_scheduling_matrix[m]
                to_noise_levels_stm = stm_scheduling_matrix[m + 1]
                full_context_mask = torch.where(
                    torch.logical_and(full_context_mask == 0, from_noise_levels_stm == -1),
                    2,
                    full_context_mask,
                ).squeeze(1)
                full_xs_pred_prev = full_xs_pred.clone()

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
                    loss = F.mse_loss(pred_x0, context, reduction="None") * alpha_cumprod.sqrt()
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

            # update xs_pred by ddim or ddpm sampling
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
                adapter=self.adapter_model if self.hparams.adapter.enabled else None,
                cfg_scale=(
                    self.hparams.adapter.memory_strength if self.hparams.adapter.enabled else 1
                ),
                n_ula=self.hparams.adapter.n_ula if self.hparams.adapter.enabled else 0,
                adapter_stm=(
                    self.adapter_model_stm if self.hparams.adapter.stm_model.enabled else None
                ),
                full_xs_pred=(
                    full_xs_pred
                    if (self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled)
                    else None
                ),
                full_conditions=self._process_conditions(
                    (
                        full_conditions.clone()
                        if (
                            full_conditions is not None
                            and (
                                self.hparams.adapter.stm_model.enabled
                                and self.hparams.adapter.enabled
                            )
                        )
                        else None
                    ),
                    from_noise_levels,
                ),
                full_context_mask=(
                    full_context_mask
                    if (self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled)
                    else None
                ),
                from_noise_levels_stm=(
                    from_noise_levels_stm
                    if (self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled)
                    else None
                ),
                to_noise_levels_stm=(
                    to_noise_levels_stm
                    if (self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled)
                    else None
                ),
                stm_only=self.hparams.adapter.stm_only,
                full_xs_pred_init_noise=(
                    full_xs_pred_init_noise
                    if (self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled)
                    else None
                ),
                orig_noise=orig_noise,
                context_mask=context_mask,
                mem_conditions=self._process_conditions(
                    mem_conditions.clone() if mem_conditions is not None else None,
                    from_noise_levels,
                ),
                adapter_model_uncond=(
                    self.adapter_model_uncond if hasattr(self, "adapter_model_uncond") else None
                ),
            )
            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev)
            if self.hparams.adapter.stm_model.enabled and self.hparams.adapter.enabled:
                # full_xs_pred = torch.where(
                #    self._extend_x_dim(full_context_mask) == 0, full_xs_pred, full_xs_pred_prev
                # )
                full_xs_pred = torch.cat([full_xs_pred[:, : -xs_pred.shape[1]], xs_pred], 1)
                # full_xs_pred = torch.cat([full_xs_pred[:,:-xs_pred.shape[1]], xs_pred], 1)

            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, aux_output, record

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return (
            key.startswith("diffusion_model.model")
            or key.startswith("diffusion_model._orig_mod.model")
            or key.startswith("adapter_model.model")
        )

    def _log_mem_curve(self, sample_tag: str, losses: list[float], namespace: str = "validation"):
        """
        Log a per-sample curve of memorization loss vs. step.
        Supports TensorBoard and WandB; falls back silently otherwise.
        """
        # --- removed the rank==0 early return so ALL ranks log ---

        logger = getattr(self, "logger", None)
        exp = getattr(logger, "experiment", None)

        # compute EMA
        alpha = 0.05
        ema_losses = []
        ema_val = losses[0]
        for loss in losses:
            ema_val = alpha * loss + (1 - alpha) * ema_val
            ema_losses.append(ema_val)

        # make figure
        steps = list(range(1, len(losses) + 1))
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111, facecolor="white")
        ax.plot(steps, losses, linewidth=1.5, label="Raw Loss")
        ax.plot(steps, ema_losses, linewidth=2, label=f"EMA (α={alpha})")
        ax.set_xlabel("Memorization step")
        ax.set_ylabel("Loss")
        ax.grid(False)
        ax.legend()

        # tag includes namespace + sample_tag (which already has rank suffix)
        tag = f"{namespace}/mem_curve/{sample_tag}"

        try:
            if hasattr(exp, "add_figure"):  # TensorBoard
                exp.add_figure(tag, fig, global_step=self.global_step)
            elif hasattr(exp, "log"):  # Weights & Biases
                exp.log({tag: fig, "global_step": self.global_step})
            else:
                # optionally print once per rank if your setup needs it
                pass
        finally:
            plt.close(fig)

    def _log_layer_mem_curves(
        self,
        sample_tag: str,
        layer_losses: list[list[float]],
        namespace: str = "validation",
    ):
        """
        Log per-layer memorization loss curves.
        Input: list of lists, each inner list are the losses for one layer.
        Example:
            [
            [0.2, 0.3, 0.4],  # layer 1
            [0.2, 0.2, 0.4],  # layer 2
            ...
            ]
        """

        logger = getattr(self, "logger", None)
        exp = getattr(logger, "experiment", None)

        alpha = 0.05  # EMA smoothing factor

        for layer_idx, losses in enumerate(layer_losses, start=1):
            if not losses:  # skip empty layers
                continue

            # compute EMA
            ema_losses = []
            ema_val = losses[0]
            for loss in losses:
                ema_val = alpha * loss + (1 - alpha) * ema_val
                ema_losses.append(ema_val)

            # make figure
            steps = list(range(1, len(losses) + 1))
            fig = plt.figure(facecolor="white")
            ax = fig.add_subplot(111, facecolor="white")
            ax.plot(steps, losses, linewidth=1.5, label="Raw Loss")
            ax.plot(steps, ema_losses, linewidth=2, label=f"EMA (α={alpha})")
            ax.set_xlabel("Memorization step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Layer {layer_idx}")
            ax.grid(False)
            ax.legend()

            # tag includes namespace + sample_tag + layer index
            tag = f"{namespace}/mem_curve/{sample_tag}/layer{layer_idx}"

            try:
                if hasattr(exp, "add_figure"):  # TensorBoard
                    exp.add_figure(tag, fig, global_step=self.global_step)
                elif hasattr(exp, "log"):  # Weights & Biases
                    exp.log({tag: fig, "global_step": self.global_step})
                else:
                    pass
            finally:
                plt.close(fig)

    # def _log_mem_curve(self, sample_tag: str, losses: list[float], namespace: str = "validation"):
    # """
    # Log a per-sample curve of memorization loss vs. step.
    # Supports TensorBoard and WandB; falls back silently otherwise.
    # """

    ##if getattr(self, "global_rank", 0) != 0:
    ##    return

    # logger = getattr(self, "logger", None)
    # alpha = 0.05
    # ema_losses = []
    # ema_val = losses[0]  # start with the first loss
    # for loss in losses:
    # ema_val = alpha * loss + (1 - alpha) * ema_val
    # ema_losses.append(ema_val)

    # steps = list(range(1, len(losses) + 1))
    # fig = plt.figure(facecolor="white")
    # ax = fig.add_subplot(111, facecolor="white")

    # ax.plot(steps, losses, linewidth=1.5, color="tab:blue", label="Raw Loss")
    # ax.plot(steps, ema_losses, linewidth=2, color="tab:orange", label=f"EMA (α={alpha})")

    # ax.set_xlabel("Memorization step")
    # ax.set_ylabel("Loss")
    ##ax.set_title(f"{namespace}/mem_curve: {sample_tag}")
    ##ax.grid(True, alpha=0.3)
    # ax.grid(False)

    # ax.legend()

    # exp = getattr(logger, "experiment", None)
    # try:
    # if hasattr(exp, "log"):
    # exp.log(
    # {f"{namespace}/mem_curve/{sample_tag}": fig, "global_step": self.global_step}
    # )
    # return
    # except Exception:
    # pass

    # plt.close(fig)
