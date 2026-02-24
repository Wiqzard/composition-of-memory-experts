import math
from collections import namedtuple
from typing import Any, Callable, Dict, Literal, Optional

import torch
from einops import rearrange, reduce
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from src.models.components.diffusion.noise_schedule import make_beta_schedule


def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class CosineNoiseSchedule(nn.Module):
    """
    A minimal cosine noise schedule for continuous-time diffusion,
    parameterized by logSNR-min, logSNR-max, and a shift factor.
    """

    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        # shift is applied as logSNR += log(shift^2)
        self.register_buffer(
            "shift",
            2.0 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t in [0, 1], output logSNR(t).
        """
        return -2.0 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min))) + self.shift


class ContinuousDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Dict = {},
        objective: str = "v_pred",
        loss_weighting: Dict = {},
        ddim_sampling_eta: float = 0.0,
        clip_noise: float = 20.0,
        use_causal_mask: bool = False,
        reconstruction_guidance: Optional[Callable] = None,
        training_schedule: str = "cosine",
        schedule_logsnr_min: float = -15.0,
        schedule_logsnr_max: float = 15.0,
        training_schedule_shift: float = 1.0,
        precond_scale: float = 1.0,
    ):
        super().__init__()
        self.x_shape = x_shape
        self.model = model(
            x_shape=x_shape, max_tokens=max_tokens, external_cond_dim=external_cond_dim
        )
        self.max_tokens = max_tokens
        self.external_cond_dim = external_cond_dim
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.schedule_fn_kwargs = schedule_fn_kwargs
        self.objective = objective
        self.loss_weighting = loss_weighting
        self.ddim_sampling_eta = ddim_sampling_eta
        self.clip_noise = clip_noise
        self.use_causal_mask = use_causal_mask
        self.reconstruction_guidance = reconstruction_guidance
        if loss_weighting == {}:
            self.loss_weighting = {
                "strategy": "fused_min_snr",
                "snr_clip": 5,
                "cum_snr_decay": 0.9,
            }
        if schedule_fn_kwargs == {}:
            self.schedule_fn_kwargs = {
                "shift": 1.0,
            }

        self.is_discrete = False

        # Additional continuous-diffusion hyperparams
        self.precond_scale = precond_scale
        self.sigmoid_bias = loss_weighting.get("sigmoid_bias", 0.0)

        # Build the chosen continuous schedule
        # You can make it swappable if you need more than just "cosine"
        if training_schedule == "cosine":
            self.training_schedule = CosineNoiseSchedule(
                logsnr_min=schedule_logsnr_min,
                logsnr_max=schedule_logsnr_max,
                shift=training_schedule_shift,
            )
        else:
            raise ValueError(f"Unknown continuous schedule '{training_schedule}'.")
        self._build_buffer()

    def _build_buffer(self):
        betas = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr.clamp_(max=self.loss_weighting.snr_clip)
            register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            register_buffer("logsnr", torch.log(snr))

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        uncond_cond=None,
        uncond_cond_mask=None,
        cfg_scale=1.0,
        adapter=None,
        adapter_only=False,
        only_model=False,
        adapter_stm=None,
        **kwargs,
    ):
        """
        x:             The current noisy sample (batch, channels, height, width).
        k:             The diffusion timestep (or index).
        external_cond: The conditional embedding for your model (e.g. text embedding).
        external_cond_mask: Any mask needed for external_cond (optional).
        uncond_cond:   The 'null' or unconditional embedding.
        uncond_cond_mask: Any mask needed for uncond_cond (optional).
        cfg_scale:     The guidance scale. cfg_scale=1 means no guidance
                    (only a single forward pass), and >1 amplifies
                    the difference between conditional and unconditional outputs.
        """

        if adapter_only and adapter is None:
            raise ValueError("adapter_only is True but adapter is None.")

        model_output = None
        if not adapter_only:
            model_output = self.model(
                x,
                self.precond_scale * self.logsnr[k],
                external_cond,
                external_cond_mask,
                **kwargs,
            )

        adapter_output = None
        if adapter is not None:  # and not only_model:
            adapter_output = adapter(
                x,
                self.precond_scale * self.logsnr[k],
                external_cond,
                external_cond_mask,
                **kwargs,
            )
            pred_adapter_noise = torch.clamp(adapter_output, -self.clip_noise, self.clip_noise)

        if self.objective == "pred_noise":
            raise ValueError("wrong")
            pred_noise = None
            if model_output is not None:
                pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            if pred_noise is not None and adapter_output is not None:
                # pred_noise = -2 * pred_noise + 3 * pred_adapter_noise
                # gamma' = 2 * gamma - 1 -> gamma = 3 -> gamma' =  5
                # 1- gamma' + gamma'
                # pred_noise = 0.3 * pred_noise +  pred_adapter_noise
                pred_noise = -0.3 * pred_noise + 1.3 * pred_adapter_noise

                # pred_noise = -4 * pred_noise + 5 * pred_adapter_noise
            if adapter_only:
                pred_noise = pred_adapter_noise
                model_output = adapter_output

            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":

            pred_noise = None
            if model_output is not None:
                pred_noise = self.predict_noise_from_v(x, k, model_output)
            if adapter_output is not None:
                pred_adapter_noise = self.predict_noise_from_v(x, k, adapter_output)

            if pred_noise is not None and adapter_output is not None:
                # pred_noise = -2 * pred_noise + 3 * pred_adapter_noise
                # pred_noise = -0.3 * pred_noise + 1.3 * pred_adapter_noise
                # pred_noise = pred_adapter_noise + cfg_scale * ( pred_adapter_noise - pred_noise)
                pred_noise = pred_noise + cfg_scale * (pred_adapter_noise - pred_noise)
                # pred_noise = pred_noise + 1.5 * ( pred_adapter_noise - pred_noise)
                # pred_noise = pred_adapter_noise + 1.5 * ( pred_adapter_noise - pred_noise)
                # pred_noise =  0.3 * pred_noise + pred_adapter_noise

            if adapter_only:
                pred_noise = pred_adapter_noise
                model_output = adapter_output

            x_start = self.predict_start_from_noise(x, k, pred_noise)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)
        return model_pred, None

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    def predict_v(self, x_start, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, k, x_k.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def p_mean_variance(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        adapter=None,
        adapter_only=False,
        **kwargs,
    ):
        model_pred, _ = self.model_predictions(
            x=x,
            k=k,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            adapter=adapter,
            adapter_only=adapter_only,
            **kwargs,
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def compute_loss_weights(
        self,
        k: torch.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> torch.Tensor:
        if strategy == "uniform":
            return torch.ones_like(k)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = torch.sigmoid(self.loss_weighting.sigmoid_bias - logsnr)
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1) if reverse else normalized_clipped_snr
                    )
                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.objective:
            case "pred_noise":
                return epsilon_weighting
            case "pred_x0":
                return epsilon_weighting * snr
            case "pred_v":
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    def _reshape_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c h w -> b (t h w) c")
        return x

    def _reshape_to_original(self, x: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        x = rearrange(x, "b (t h w) c -> b t c h w", t=x_shape[1], h=x_shape[2], w=x_shape[3])
        return x

    def forward_patchwise(self, x, external_cond, k, adapter=None, **kwargs):
        x_shape = x.shape
        logsnr = self.training_schedule(k)

        x = self._reshape_to_sequence(x)

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        alpha_t_seq = self._reshape_to_sequence(alpha_t)
        sigma_t_seq = self._reshape_to_sequence(sigma_t)

        x_t = alpha_t_seq * x + sigma_t_seq * noise
        x_t = self._reshape_to_original(x_t, x_shape)

        model = self.model if adapter is None else adapter
        v_pred = model(
            x_t,
            self.precond_scale * logsnr,
            external_cond,
            **kwargs,
        )

        v_pred_seq = self._reshape_to_sequence(v_pred)

        noise_pred = alpha_t_seq * v_pred_seq + sigma_t_seq * x
        x_pred_seq = alpha_t_seq * x - sigma_t_seq * v_pred_seq

        x_pred = self._reshape_to_original(x_pred_seq, x_shape)

        loss = F.mse_loss(noise_pred, noise, reduction="none")
        bias = self.sigmoid_bias
        loss_weight = torch.sigmoid(bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss, None

    def forward(self, x, external_cond, k, adapter=None, patch_wise_noie_level=False, **kwargs):

        if patch_wise_noie_level:
            return self.forward_patchwise(x, external_cond, k, adapter=adapter, **kwargs)

        logsnr = self.training_schedule(k)

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        x_t = alpha_t * x + sigma_t * noise

        model = self.model if adapter is None else adapter
        v_pred = model(
            x_t,
            self.precond_scale * logsnr,
            external_cond,
            **kwargs,
        )

        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        loss = F.mse_loss(noise_pred, noise, reduction="none")
        bias = self.sigmoid_bias
        loss_weight = torch.sigmoid(bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss, None

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        adapter=None,
        n_ula: int = 0,
        adapter_stm=None,
        **kwargs,
    ):
        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
                cfg_scale=cfg_scale,
                adapter=adapter,
                n_ula=n_ula,
                adapter_stm=adapter_stm,
                **kwargs,
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            guidance_fn=guidance_fn,
            adapter=adapter,
            **kwargs,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        adapter: Optional[nn.Module] = None,
        n_ula: int = 5,
        **kwargs,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implemented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            adapter=adapter,
            adapter_only=False,  # True, #False, # True
            **kwargs,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        if clipped_curr_noise_level[0, -1].item() > 100:
            for _ in range(n_ula):
                x_pred = self._ula_step(
                    x_k=x_pred,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    adapter=adapter,
                    **kwargs,
                )

        # only update frames where the noise level decreases
        return (
            torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred),
            None,
        )

    def _ula_step(
        self,
        x_k,
        k,
        external_cond=None,
        external_cond_mask=None,
        adapter=None,
        adapter_only=False,
        **kwargs,
    ):
        model_pred, _ = self.model_predictions(
            x=x_k,
            k=k,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            adapter=adapter,
            adapter_only=adapter_only,
            **kwargs,
        )
        # x_start = model_pred.pred_x_start
        eps = model_pred.pred_noise
        noise = torch.randn_like(x_k)

        # factor_1 = 2 * extract(self.betas, k, x_k.shape) / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
        # factor_2 = torch.sqrt(2 * 2 * extract(self.betas, k, x_k.shape))
        factor_1 = (
            1
            / 2
            * extract(self.betas, k, x_k.shape)
            / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
        )
        factor_2 = torch.sqrt(extract(self.betas, k, x_k.shape))
        # factor_1 = extract(self.betas, k, x_k.shape) / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
        # factor_2 = torch.sqrt(2 * extract(self.betas, k, x_k.shape))
        x_k = x_k - eps * factor_1 + noise * factor_2
        return x_k

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] = None,
        n_ula: int = 2,  # 10,
        adapter_stm: Optional[nn.Module] = None,
        **kwargs,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()
        c_curr = (1 - alpha).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    adapter=adapter,
                    adapter_only=True,  # False,  # True,
                    adapter_stm=adapter_stm,
                    **kwargs,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise),
                    model_pred.pred_x_start,
                )

        else:
            model_pred, *aux_output = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                cfg_scale=cfg_scale,
                adapter=adapter,
                # adapter_only=True, #True,  # True, #False,# True, # False
                adapter_only=False,  # True, #False,# True, # False
                **kwargs,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise
        # dterministic sampling
        # x_pred = x_start * alpha.sqrt() + pred_noise * c_curr

        # x_pred = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

        # only update frames where the noise level decreases

        # alpha_next[:, 0]  = alpha_next[:, 1]
        # sigma[:, 0] = sigma[:, 1]
        # x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )
        if n_ula > 0 and next_noise_level[0, 2].item() > 20:
            for _ in range(n_ula):
                model_pred_ula, _ = self.model_predictions(
                    x=x_pred,
                    k=next_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    # adapter=adapter,
                    # adapter_only=False,  # True, #False,
                    **kwargs,
                )
                eps = model_pred_ula.pred_noise
                eta = torch.randn_like(x_pred)
                beta_t = extract(self.betas, next_noise_level, x_pred.shape)
                factor = 1  # /8
                if False:
                    g = eps
                    r = 0.02  # 4
                    eps = 2 * (1 - beta_t) * (r * torch.norm(eta, p=2) / torch.norm(g, p=2)) ** 2
                    x_pred = x_pred + eps * g + (2 * eps).sqrt() * eta
                x_pred = (
                    x_pred
                    - beta_t / (2 * (1 - alpha_next).sqrt()) * eps
                    + beta_t.sqrt() * eta
                    #    #x_pred
                    #    #- beta_t / ((1 - alpha_next).sqrt()) * eps
                    #    #+ (2 * beta_t).sqrt() * eta
                    #    #x_pred - factor * beta_t * eps + ( 2 * factor * beta_t).sqrt() * eta
                    #    #x_pred - 0.001 * eps + math.sqrt(2 * 0.001) * eta
                    #    #x_pred - 0.0001 * eps + (2 * torch.tensor([0.0001], device=x_pred.device)).sqrt() * eta
                )

                mask = curr_noise_level == next_noise_level
                x_pred = torch.where(
                    self.add_shape_channels(mask),
                    x,
                    x_pred,
                )
                # if nan or inf in x_pred raise error
                if torch.isnan(x_pred).any() or torch.isinf(x_pred).any():
                    print("nan or inf in x_pred")
                # x_pred = self._ula_step(
                #    x_k=x_pred,
                #    k=next_noise_level,
                #    external_cond=external_cond,
                #    external_cond_mask=external_cond_mask,
                #    adapter=adapter,
                #    adapter_only=True, #False,
                #    **kwargs,
                # )

        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred, aux_output

    def estimate_noise_level(self, x, mu=None):
        # x ~ ( B, T, C, ...)
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
            ..., None
        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
        k = torch.argmax(ll_except_c, -1)
        return k


# from typing import Optional, Callable, Literal, Dict, Any
# from collections import namedtuple
# from omegaconf import DictConfig
# import torch
# import math
# from torch import nn
# from torch.nn import functional as F
# from einops import rearrange, reduce
#
# from src.models.components.diffusion.noise_schedule import make_beta_schedule
#
#
# def extract(a, t, x_shape):
#    shape = t.shape
#    out = a[t]
#    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))
#
#
# ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])
#
#
# class CosineNoiseSchedule(nn.Module):
#    """
#    A minimal cosine noise schedule for continuous-time diffusion,
#    parameterized by logSNR-min, logSNR-max, and a shift factor.
#    """
#
#    def __init__(
#        self,
#        logsnr_min: float = -15.0,
#        logsnr_max: float = 15.0,
#        shift: float = 1.0,
#    ):
#        super().__init__()
#        self.register_buffer(
#            "t_min",
#            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
#            persistent=False,
#        )
#        self.register_buffer(
#            "t_max",
#            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
#            persistent=False,
#        )
#        # shift is applied as logSNR += log(shift^2)
#        self.register_buffer(
#            "shift",
#            2.0 * torch.log(torch.tensor(shift, dtype=torch.float32)),
#            persistent=False,
#        )
#
#    def forward(self, t: torch.Tensor) -> torch.Tensor:
#        """
#        t in [0, 1], output logSNR(t).
#        """
#        return -2.0 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min))) + self.shift
#
#
# class ContinuousDiffusion(nn.Module):
#    def __init__(
#        self,
#        model: nn.Module,
#        x_shape: torch.Size,
#        max_tokens: int,
#        external_cond_dim: int,
#        timesteps: int = 1000,
#        sampling_timesteps: int = 50,
#        beta_schedule: str = "cosine",
#        schedule_fn_kwargs: Dict = {},
#        objective: str = "v_pred",
#        loss_weighting: Dict = {},
#        ddim_sampling_eta: float = 0.0,
#        clip_noise: float = 20.0,
#        use_causal_mask: bool = False,
#        reconstruction_guidance: Optional[Callable] = None,
#        training_schedule: str = "cosine",
#        schedule_logsnr_min: float = -15.0,
#        schedule_logsnr_max: float = 15.0,
#        training_schedule_shift: float = 1.0,
#        precond_scale: float = 1.0,
#    ):
#        super().__init__()
#        self.x_shape = x_shape
#        self.model = model(
#            x_shape=x_shape, max_tokens=max_tokens, external_cond_dim=external_cond_dim
#        )
#        self.max_tokens = max_tokens
#        self.external_cond_dim = external_cond_dim
#        self.timesteps = timesteps
#        self.sampling_timesteps = sampling_timesteps
#        self.beta_schedule = beta_schedule
#        self.schedule_fn_kwargs = schedule_fn_kwargs
#        self.objective = objective
#        self.loss_weighting = loss_weighting
#        self.ddim_sampling_eta = ddim_sampling_eta
#        self.clip_noise = clip_noise
#        self.use_causal_mask = use_causal_mask
#        self.reconstruction_guidance = reconstruction_guidance
#        if loss_weighting == {}:
#            self.loss_weighting = {
#                "strategy": "fused_min_snr",
#                "snr_clip": 5,
#                "cum_snr_decay": 0.9,
#            }
#        if schedule_fn_kwargs == {}:
#            self.schedule_fn_kwargs = {
#                "shift": 1.0,
#            }
#
#        self.is_discrete = False
#
#        # Additional continuous-diffusion hyperparams
#        self.precond_scale = precond_scale
#        self.sigmoid_bias = loss_weighting.get("sigmoid_bias", 0.0)
#
#        # Build the chosen continuous schedule
#        # You can make it swappable if you need more than just "cosine"
#        if training_schedule == "cosine":
#            self.training_schedule = CosineNoiseSchedule(
#                logsnr_min=schedule_logsnr_min,
#                logsnr_max=schedule_logsnr_max,
#                shift=training_schedule_shift,
#            )
#        else:
#            raise ValueError(f"Unknown continuous schedule '{training_schedule}'.")
#        self._build_buffer()
#
#    def _build_buffer(self):
#        betas = make_beta_schedule(
#            schedule=self.beta_schedule,
#            timesteps=self.timesteps,
#            zero_terminal_snr=self.objective != "pred_noise",
#            **self.schedule_fn_kwargs,
#        )
#
#        alphas = 1.0 - betas
#        alphas_cumprod = torch.cumprod(alphas, dim=0)
#        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#
#        # sampling related parameters
#        assert self.sampling_timesteps <= self.timesteps
#        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps
#
#        # helper function to register buffer from float64 to float32
#        register_buffer = lambda name, val: self.register_buffer(
#            name, val.to(torch.float32), persistent=False
#        )
#
#        register_buffer("betas", betas)
#        register_buffer("alphas_cumprod", alphas_cumprod)
#        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
#
#        # calculations for diffusion q(x_t | x_{t-1}) and others
#
#        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
#        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
#        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
#        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
#        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
#
#        # calculations for posterior q(x_{t-1} | x_t, x_0)
#        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#
#        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
#
#        register_buffer("posterior_variance", posterior_variance)
#        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
#
#        register_buffer(
#            "posterior_log_variance_clipped",
#            torch.log(posterior_variance.clamp(min=1e-20)),
#        )
#        register_buffer(
#            "posterior_mean_coef1",
#            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
#        )
#        register_buffer(
#            "posterior_mean_coef2",
#            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
#        )
#
#        # snr: signal noise ratio
#        snr = alphas_cumprod / (1 - alphas_cumprod)
#        register_buffer("snr", snr)
#        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
#            clipped_snr = snr.clone()
#            clipped_snr.clamp_(max=self.loss_weighting.snr_clip)
#            register_buffer("clipped_snr", clipped_snr)
#        elif self.loss_weighting.strategy == "sigmoid":
#            register_buffer("logsnr", torch.log(snr))
#
#    def add_shape_channels(self, x):
#        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")
#
#    def model_predictions(
#        self,
#        x,
#        k,
#        external_cond=None,
#        external_cond_mask=None,
#        uncond_cond=None,
#        uncond_cond_mask=None,
#        cfg_scale=1.0,
#        adapter=None,
#        adapter_only=False,
#        only_model=False,
#        **kwargs,
#    ):
#        """
#        x:             The current noisy sample (batch, channels, height, width).
#        k:             The diffusion timestep (or index).
#        external_cond: The conditional embedding for your model (e.g. text embedding).
#        external_cond_mask: Any mask needed for external_cond (optional).
#        uncond_cond:   The 'null' or unconditional embedding.
#        uncond_cond_mask: Any mask needed for uncond_cond (optional).
#        cfg_scale:     The guidance scale. cfg_scale=1 means no guidance
#                    (only a single forward pass), and >1 amplifies
#                    the difference between conditional and unconditional outputs.
#        """
#
#        if adapter_only and adapter is None:
#            raise ValueError("adapter_only is True but adapter is None.")
#
#        model_output = None
#        if not adapter_only:
#            model_output = self.model(
#                x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask, **kwargs
#            )
#
#        adapter_output = None
#        if adapter is not None:  # and not only_model:
#            adapter_output = adapter(
#                x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask, **kwargs
#            )
#            pred_adapter_noise = torch.clamp(adapter_output, -self.clip_noise, self.clip_noise)
#
#        if False: #True:
#            noise_levels = k
#            noise_levels[:, :3] = 999
#            uncond_model_output = self.model(
#                x,
#                self.precond_scale * self.logsnr[noise_levels],
#                external_cond,
#                external_cond_mask,
#                **kwargs,
#            )
#
#        if self.objective == "pred_noise":
#            raise ValueError("wrong")
#            pred_noise = None
#            if model_output is not None:
#                pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
#            if pred_noise is not None and adapter_output is not None:
#                # pred_noise = -2 * pred_noise + 3 * pred_adapter_noise
#                # gamma' = 2 * gamma - 1 -> gamma = 3 -> gamma' =  5
#                # 1- gamma' + gamma'
#                # pred_noise = 0.3 * pred_noise +  pred_adapter_noise
#                pred_noise = -0.3 * pred_noise + 1.3 * pred_adapter_noise
#
#                # pred_noise = -4 * pred_noise + 5 * pred_adapter_noise
#            if adapter_only:
#                pred_noise = pred_adapter_noise
#                model_output = adapter_output
#
#            x_start = self.predict_start_from_noise(x, k, pred_noise)
#
#        elif self.objective == "pred_x0":
#            x_start = model_output
#            pred_noise = self.predict_noise_from_start(x, k, x_start)
#
#        elif self.objective == "pred_v":
#
#            pred_noise = None
#            if model_output is not None:
#                pred_noise = self.predict_noise_from_v(x, k, model_output)
#            if adapter_output is not None:
#                pred_adapter_noise = self.predict_noise_from_v(x, k, adapter_output)
#
#            if False: #uncond_model_output is not None:
#                pred_uncond_noise = self.predict_noise_from_v(x, noise_levels, uncond_model_output)
#
#            if pred_noise is not None and adapter_output is not None:
#                # pred_noise = -2 * pred_noise + 3 * pred_adapter_noise
#                # pred_noise = -0.3 * pred_noise + 1.3 * pred_adapter_noise
#                #pred_noise = pred_adapter_noise + cfg_scale * ( pred_adapter_noise - pred_noise)
#                x = pred_noise + cfg_scale * (pred_adapter_noise - pred_noise)
#                #x = 0.8 * x + 0.2 * pred_uncond_noise
#                #pred_noise = (
#                #    pred_noise
#                #    + cfg_scale * (pred_adapter_noise - pred_noise)
#                #    + 1.5 * (pred_noise - pred_uncond_noise)
#                #)
#                #y = pred_uncond_noise
#                #pred_noise = torch.empty_like(x)
#                #pred_noise[:, 3:] = 0.5 * x[:, 3:] + 0.5 * y[:, 3:]
#                #pred_noise[:, :3] = x[:, :3]
#                #pred_noise = (
#                #    0.5 * (pred_noise
#                #    + cfg_scale * (pred_adapter_noise - pred_noise))
#                #    + 0.5 * pred_uncond_noise
#                #)
#               #`` if torch.rand(1).item() < 0.02:
#               #``     print("Using adapter noise")
#
#                # pred_noise = pred_noise + 1.5 * ( pred_adapter_noise - pred_noise)
#                # pred_noise = pred_adapter_noise + 1.5 * ( pred_adapter_noise - pred_noise)
#                # pred_noise =  0.3 * pred_noise + pred_adapter_noise
#
#            if adapter_only:
#                pred_noise = pred_adapter_noise
#                model_output = adapter_output
#
#            x_start = self.predict_start_from_noise(x, k, pred_noise)
#
#        model_pred = ModelPrediction(pred_noise, x_start, model_output)
#        return model_pred, None
#
#    def predict_start_from_noise(self, x_k, k, noise):
#        return (
#            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
#            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
#        )
#
#    def predict_noise_from_start(self, x_k, k, x0):
#        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
#            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
#        )
#
#    def predict_v(self, x_start, k, noise):
#        return (
#            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
#            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
#        )
#
#    def predict_start_from_v(self, x_k, k, v):
#        return (
#            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
#            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
#        )
#
#    def predict_noise_from_v(self, x_k, k, v):
#        return (
#            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
#            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
#        )
#
#    def q_mean_variance(self, x_start, k):
#        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
#        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
#        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
#        return mean, variance, log_variance
#
#    def q_posterior(self, x_start, x_k, k):
#        posterior_mean = (
#            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
#            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
#        )
#        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
#        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, k, x_k.shape)
#        return posterior_mean, posterior_variance, posterior_log_variance_clipped
#
#    def q_sample(self, x_start, k, noise=None):
#        if noise is None:
#            noise = torch.randn_like(x_start)
#            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
#
#        return (
#            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
#            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
#        )
#
#    def p_mean_variance(
#        self,
#        x,
#        k,
#        external_cond=None,
#        external_cond_mask=None,
#        adapter=None,
#        adapter_only=False,
#        **kwargs,
#    ):
#        model_pred, _ = self.model_predictions(
#            x=x,
#            k=k,
#            external_cond=external_cond,
#            external_cond_mask=external_cond_mask,
#            adapter=adapter,
#            adapter_only=adapter_only,
#            **kwargs,
#        )
#        x_start = model_pred.pred_x_start
#        return self.q_posterior(x_start=x_start, x_k=x, k=k)
#
#    def compute_loss_weights(
#        self,
#        k: torch.Tensor,
#        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
#    ) -> torch.Tensor:
#        if strategy == "uniform":
#            return torch.ones_like(k)
#        snr = self.snr[k]
#        epsilon_weighting = None
#        match strategy:
#            case "sigmoid":
#                logsnr = self.logsnr[k]
#                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
#                # and adopted by https://arxiv.org/abs/2410.19324
#                epsilon_weighting = torch.sigmoid(self.loss_weighting.sigmoid_bias - logsnr)
#            case "min_snr":
#                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
#                clipped_snr = self.clipped_snr[k]
#                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
#            case "fused_min_snr":
#                # fused min-SNR reweighting proposed by Diffusion Forcing v1
#                # with an additional support for bi-directional Fused min-SNR for non-causal models
#                snr_clip, cum_snr_decay = (
#                    self.loss_weighting.snr_clip,
#                    self.loss_weighting.cum_snr_decay,
#                )
#                clipped_snr = self.clipped_snr[k]
#                normalized_clipped_snr = clipped_snr / snr_clip
#                normalized_snr = snr / snr_clip
#
#                def compute_cum_snr(reverse: bool = False):
#                    new_normalized_clipped_snr = (
#                        normalized_clipped_snr.flip(1) if reverse else normalized_clipped_snr
#                    )
#                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
#                    for t in range(0, k.shape[1]):
#                        if t == 0:
#                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
#                        else:
#                            cum_snr[:, t] = (
#                                cum_snr_decay * cum_snr[:, t - 1]
#                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
#                            )
#                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
#                    return cum_snr.flip(1) if reverse else cum_snr
#
#                if self.use_causal_mask:
#                    cum_snr = compute_cum_snr()
#                else:
#                    # bi-directional cum_snr when not using causal mask
#                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
#                    cum_snr *= 0.5
#                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
#                    1 - normalized_clipped_snr
#                )
#                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
#                clipped_snr = clipped_fused_snr * snr_clip
#                snr = fused_snr * snr_clip
#                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
#            case _:
#                raise ValueError(f"unknown loss weighting strategy {strategy}")
#
#        match self.objective:
#            case "pred_noise":
#                return epsilon_weighting
#            case "pred_x0":
#                return epsilon_weighting * snr
#            case "pred_v":
#                return epsilon_weighting * snr / (snr + 1)
#            case _:
#                raise ValueError(f"unknown objective {self.objective}")
#
#    def _reshape_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
#        x = rearrange(x, "b t c h w -> b (t h w) c")
#        return x
#
#    def _reshape_to_original(self, x: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
#        x = rearrange(x, "b (t h w) c -> b t c h w", t=x_shape[1], h=x_shape[2], w=x_shape[3])
#        return x
#
#    def forward_patchwise(self, x, external_cond, k, adapter=None, **kwargs):
#        x_shape = x.shape
#        logsnr = self.training_schedule(k)
#
#        x = self._reshape_to_sequence(x)
#
#        noise = torch.randn_like(x)
#        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
#
#        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
#        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
#
#        alpha_t_seq = self._reshape_to_sequence(alpha_t)
#        sigma_t_seq = self._reshape_to_sequence(sigma_t)
#
#        x_t = alpha_t_seq * x + sigma_t_seq * noise
#        x_t = self._reshape_to_original(x_t, x_shape)
#
#        model = self.model if adapter is None else adapter
#        v_pred = model(
#            x_t,
#            self.precond_scale * logsnr,
#            external_cond,
#            **kwargs,
#        )
#
#        v_pred_seq = self._reshape_to_sequence(v_pred)
#
#        noise_pred = alpha_t_seq * v_pred_seq + sigma_t_seq * x
#        x_pred_seq = alpha_t_seq * x - sigma_t_seq * v_pred_seq
#
#        x_pred = self._reshape_to_original(x_pred_seq, x_shape)
#
#        loss = F.mse_loss(noise_pred, noise, reduction="none")
#        bias = self.sigmoid_bias
#        loss_weight = torch.sigmoid(bias - logsnr)
#        loss_weight = self.add_shape_channels(loss_weight)
#        loss = loss * loss_weight
#
#        return x_pred, loss, None
#
#    def forward(self, x, external_cond, k, adapter=None, patch_wise_noie_level=False, **kwargs):
#
#        if patch_wise_noie_level:
#            return self.forward_patchwise(x, external_cond, k, adapter=adapter, **kwargs)
#
#        logsnr = self.training_schedule(k)
#
#        noise = torch.randn_like(x)
#        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
#
#        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
#        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
#
#        x_t = alpha_t * x + sigma_t * noise
#
#        model = self.model if adapter is None else adapter
#        v_pred = model(
#            x_t,
#            self.precond_scale * logsnr,
#            external_cond,
#            **kwargs,
#        )
#
#        noise_pred = alpha_t * v_pred + sigma_t * x_t
#        x_pred = alpha_t * x_t - sigma_t * v_pred
#
#        loss = F.mse_loss(noise_pred, noise, reduction="none")
#        bias = self.sigmoid_bias
#        loss_weight = torch.sigmoid(bias - logsnr)
#        loss_weight = self.add_shape_channels(loss_weight)
#        loss = loss * loss_weight
#
#        return x_pred, loss, None
#
#    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
#        shape = indices.shape
#        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
#        real_steps = real_steps.long().to(indices.device)
#        k = real_steps[indices.flatten()]
#        return k.view(shape)
#
#    def sample_step(
#        self,
#        x: torch.Tensor,
#        curr_noise_level: torch.Tensor,
#        next_noise_level: torch.Tensor,
#        external_cond: Optional[torch.Tensor],
#        external_cond_mask: Optional[torch.Tensor] = None,
#        guidance_fn: Optional[Callable] = None,
#        cfg_scale: float = 1.0,
#        adapter=None,
#        n_ula: int = 0,
#        **kwargs,
#    ):
#        if self.is_ddim_sampling:
#            return self.ddim_sample_step(
#                x=x,
#                curr_noise_level=curr_noise_level,
#                next_noise_level=next_noise_level,
#                external_cond=external_cond,
#                external_cond_mask=external_cond_mask,
#                guidance_fn=guidance_fn,
#                cfg_scale=cfg_scale,
#                adapter=adapter,
#                n_ula=n_ula,
#                **kwargs,
#            )
#
#        # FIXME: temporary code for checking ddpm sampling
#        assert torch.all(
#            (curr_noise_level - 1 == next_noise_level)
#            | ((curr_noise_level == -1) & (next_noise_level == -1))
#        ), "Wrong noise level given for ddpm sampling."
#
#        assert (
#            self.sampling_timesteps == self.timesteps
#        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."
#
#        return self.ddpm_sample_step(
#            x=x,
#            curr_noise_level=curr_noise_level,
#            external_cond=external_cond,
#            external_cond_mask=external_cond_mask,
#            guidance_fn=guidance_fn,
#            adapter=adapter,
#            **kwargs,
#        )
#
#    def ddpm_sample_step(
#        self,
#        x: torch.Tensor,
#        curr_noise_level: torch.Tensor,
#        external_cond: Optional[torch.Tensor],
#        external_cond_mask: Optional[torch.Tensor] = None,
#        guidance_fn: Optional[Callable] = None,
#        adapter: Optional[nn.Module] = None,
#        n_ula: int = 5,
#        **kwargs,
#    ):
#        if guidance_fn is not None:
#            raise NotImplementedError("guidance_fn is not yet implemented for ddpm.")
#
#        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)
#
#        model_mean, _, model_log_variance = self.p_mean_variance(
#            x=x,
#            k=clipped_curr_noise_level,
#            external_cond=external_cond,
#            external_cond_mask=external_cond_mask,
#            adapter=adapter,
#            adapter_only=False,  # True, #False, # True
#            **kwargs,
#        )
#
#        noise = torch.where(
#            self.add_shape_channels(clipped_curr_noise_level > 0),
#            torch.randn_like(x),
#            0,
#        )
#        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
#        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise
#
#        if clipped_curr_noise_level[0, -1].item() > 100:
#            for _ in range(n_ula):
#                x_pred = self._ula_step(
#                    x_k=x_pred,
#                    k=clipped_curr_noise_level,
#                    external_cond=external_cond,
#                    external_cond_mask=external_cond_mask,
#                    adapter=adapter,
#                    **kwargs,
#                )
#
#        # only update frames where the noise level decreases
#        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred), None
#
#    def _ula_step(
#        self,
#        x_k,
#        k,
#        external_cond=None,
#        external_cond_mask=None,
#        adapter=None,
#        adapter_only=False,
#        **kwargs,
#    ):
#        model_pred, _ = self.model_predictions(
#            x=x_k,
#            k=k,
#            external_cond=external_cond,
#            external_cond_mask=external_cond_mask,
#            adapter=adapter,
#            adapter_only=adapter_only,
#            **kwargs,
#        )
#        # x_start = model_pred.pred_x_start
#        eps = model_pred.pred_noise
#        noise = torch.randn_like(x_k)
#
#        # factor_1 = 2 * extract(self.betas, k, x_k.shape) / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
#        # factor_2 = torch.sqrt(2 * 2 * extract(self.betas, k, x_k.shape))
#        factor_1 = (
#            1
#            / 2
#            * extract(self.betas, k, x_k.shape)
#            / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
#        )
#        factor_2 = torch.sqrt(extract(self.betas, k, x_k.shape))
#        # factor_1 = extract(self.betas, k, x_k.shape) / extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)
#        # factor_2 = torch.sqrt(2 * extract(self.betas, k, x_k.shape))
#        x_k = x_k - eps * factor_1 + noise * factor_2
#        return x_k
#
#    def ddim_sample_step(
#        self,
#        x: torch.Tensor,
#        curr_noise_level: torch.Tensor,
#        next_noise_level: torch.Tensor,
#        external_cond: Optional[torch.Tensor],
#        external_cond_mask: Optional[torch.Tensor] = None,
#        guidance_fn: Optional[Callable] = None,
#        cfg_scale: float = 1.0,
#        adapter: Optional[nn.Module] = None,
#        n_ula: int = 2,  # 10,
#        **kwargs,
#    ):
#
#        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)
#
#        alpha = self.alphas_cumprod[clipped_curr_noise_level]
#        alpha_next = torch.where(
#            next_noise_level < 0,
#            torch.ones_like(next_noise_level),
#            self.alphas_cumprod[next_noise_level],
#        )
#        sigma = torch.where(
#            next_noise_level < 0,
#            torch.zeros_like(next_noise_level),
#            self.ddim_sampling_eta
#            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
#        )
#        c = (1 - alpha_next - sigma**2).sqrt()
#        c_curr = (1 - alpha).sqrt()
#
#        alpha = self.add_shape_channels(alpha)
#        alpha_next = self.add_shape_channels(alpha_next)
#        c = self.add_shape_channels(c)
#        sigma = self.add_shape_channels(sigma)
#
#        if guidance_fn is not None:
#            with torch.enable_grad():
#                x = x.detach().requires_grad_()
#
#                model_pred = self.model_predictions(
#                    x=x,
#                    k=clipped_curr_noise_level,
#                    external_cond=external_cond,
#                    external_cond_mask=external_cond_mask,
#                    adapter=adapter,
#                    adapter_only=True,  # False,  # True,
#                    **kwargs,
#                )
#
#                guidance_loss = guidance_fn(
#                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
#                )
#
#                grad = -torch.autograd.grad(
#                    guidance_loss,
#                    x,
#                )[0]
#                grad = torch.nan_to_num(grad, nan=0.0)
#
#                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
#                x_start = torch.where(
#                    alpha > 0,  # to avoid NaN from zero terminal SNR
#                    self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise),
#                    model_pred.pred_x_start,
#                )
#
#        else:
#            model_pred, *aux_output = self.model_predictions(
#                x=x,
#                k=clipped_curr_noise_level,
#                external_cond=external_cond,
#                external_cond_mask=external_cond_mask,
#                cfg_scale=cfg_scale,
#                adapter=adapter,
#                # adapter_only=True, #True,  # True, #False,# True, # False
#                adapter_only=False,  # True, #False,# True, # False
#                **kwargs,
#            )
#            x_start = model_pred.pred_x_start
#            pred_noise = model_pred.pred_noise
#
#        noise = torch.randn_like(x)
#        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
#
#        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise
#        # deterministic sampling
#        # x_pred = x_start * alpha.sqrt() + pred_noise * c_curr
#
#        # x_pred = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()
#
#        # only update frames where the noise level decreases
#
#        # alpha_next[:, 0]  = alpha_next[:, 1]
#        # sigma[:, 0] = sigma[:, 1]
#        # x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise
#        mask = curr_noise_level == next_noise_level
#        x_pred = torch.where(
#            self.add_shape_channels(mask),
#            x,
#            x_pred,
#        )
#        if n_ula > 0 and next_noise_level[0, 2].item() > 20:
#            for _ in range(n_ula):
#                model_pred_ula, _ = self.model_predictions(
#                    x=x_pred,
#                    k=next_noise_level,
#                    external_cond=external_cond,
#                    external_cond_mask=external_cond_mask,
#                    # adapter=adapter,
#                    # adapter_only=False,  # True, #False,
#                    **kwargs,
#                )
#                eps = model_pred_ula.pred_noise
#                eta = torch.randn_like(x_pred)
#                beta_t = extract(self.betas, next_noise_level, x_pred.shape)
#                factor = 1  # /8
#                if False:
#                    g = eps
#                    r = 0.02  # 4
#                    eps = 2 * (1 - beta_t) * ((r * torch.norm(eta, p=2) / torch.norm(g, p=2))) ** 2
#                    x_pred = x_pred + eps * g + (2 * eps).sqrt() * eta
#                x_pred = (
#                    x_pred
#                    - beta_t / (2 * (1 - alpha_next).sqrt()) * eps
#                    + beta_t.sqrt() * eta
#                    #    #x_pred
#                    #    #- beta_t / ((1 - alpha_next).sqrt()) * eps
#                    #    #+ (2 * beta_t).sqrt() * eta
#                    #    #x_pred - factor * beta_t * eps + ( 2 * factor * beta_t).sqrt() * eta
#                    #    #x_pred - 0.001 * eps + math.sqrt(2 * 0.001) * eta
#                    #    #x_pred - 0.0001 * eps + (2 * torch.tensor([0.0001], device=x_pred.device)).sqrt() * eta
#                )
#
#                mask = curr_noise_level == next_noise_level
#                x_pred = torch.where(
#                    self.add_shape_channels(mask),
#                    x,
#                    x_pred,
#                )
#                # if nan or inf in x_pred raise error
#                if torch.isnan(x_pred).any() or torch.isinf(x_pred).any():
#                    print("nan or inf in x_pred")
#                # x_pred = self._ula_step(
#                #    x_k=x_pred,
#                #    k=next_noise_level,
#                #    external_cond=external_cond,
#                #    external_cond_mask=external_cond_mask,
#                #    adapter=adapter,
#                #    adapter_only=True, #False,
#                #    **kwargs,
#                # )
#
#        mask = curr_noise_level == next_noise_level
#        x_pred = torch.where(
#            self.add_shape_channels(mask),
#            x,
#            x_pred,
#        )
#
#        return x_pred, aux_output
#
#    def estimate_noise_level(self, x, mu=None):
#        # x ~ ( B, T, C, ...)
#        if mu is None:
#            mu = torch.zeros_like(x)
#        x = x - mu
#        mse = reduce(x**2, "b t ... -> b t", "mean")
#        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
#            ..., None
#        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
#        k = torch.argmax(ll_except_c, -1)
#        return k
#
