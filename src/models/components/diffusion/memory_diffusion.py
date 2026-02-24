from collections import namedtuple
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F

# from .discrete_diffusion import DiscreteDiffusion, ModelPrediction  # or wherever you keep these
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
    """
    A minimal continuous-time diffusion class that reuses the DiscreteDiffusion
    interface but overrides scheduling/buffering logic. By design:

    - Only supports objective = 'pred_v'
    - Only supports loss_weighting.strategy = 'sigmoid'
    """

    def __init__(
        self,
        model: nn.Module,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        beta_schedule: str = "cosine",
        # Noise schedule (continuous) parameters:
        training_schedule: str = "cosine",
        schedule_logsnr_min: float = -15.0,
        schedule_logsnr_max: float = 15.0,
        training_schedule_shift: float = 1.0,
        # Diffusion objective & weighting:
        objective: str = "pred_v",
        loss_weighting: Dict = None,  # e.g. {"strategy": "sigmoid", "sigmoid_bias": 0.0}
        # Other:
        precond_scale: float = 1.0,
        clip_noise: float = 20.0,
        ddim_sampling_eta: float = 0.0,
        use_causal_mask: bool = False,
        schedule_fn_kwargs: Dict = {},
    ):
        """
        Args:
            model: The neural network model used for predicting noise/v/etc.
            x_shape: The shape of each data sample (e.g., (channels, height, width)).
            max_tokens: The maximum chunk size or tokens processed at once.
            external_cond_dim: Dimension of external conditioning (if used).
            timesteps: Total number of discrete “timesteps” (still used for sampling).
            sampling_timesteps: Number of timesteps to use at sampling (DDIM or DDPM).
            schedule_name: Which continuous schedule to use, e.g., 'cosine'.
            schedule_logsnr_min: Minimum logSNR for the schedule.
            schedule_logsnr_max: Maximum logSNR for the schedule.
            schedule_shift: Shift factor in log-space for the schedule.
            objective: Must be 'pred_v' for this continuous version.
            loss_weighting: Must have strategy='sigmoid'. E.g. {"strategy": "sigmoid", "sigmoid_bias": 0.0}
            precond_scale: Scale factor applied to logSNR inside the model for v-pred.
            clip_noise: Clamping range for random noise.
            ddim_sampling_eta: DDIM sampling hyper-parameter (as in DiscreteDiffusion).
            use_causal_mask: Whether to use a causal strategy in weighting (not typically used here).
        """

        super().__init__()
        self.model = model
        self.x_shape = x_shape
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
        self._build_buffer()

        if loss_weighting is None:
            loss_weighting = {"strategy": "sigmoid", "sigmoid_bias": 0.0}

        # Validate required settings for continuous
        if objective != "pred_v":
            raise ValueError("ContinuousDiffusion only supports objective='pred_v'.")
        if loss_weighting.get("strategy", "") != "sigmoid":
            raise ValueError(
                "ContinuousDiffusion only supports loss_weighting.strategy='sigmoid'."
            )
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
        # if (
        #     self.objective == "pred_noise"
        #     or self.reconstruction_guidance is not None
        # ):
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

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        # return (
        #     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        # ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
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

    def p_mean_variance(self, x, k, external_cond=None, external_cond_mask=None):
        model_pred = self.model_predictions(
            x=x, k=k, external_cond=external_cond, external_cond_mask=external_cond_mask
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

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        adapter: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Forward pass for training. Interprets `k` as a continuous time t ∈ [0,1].
        1. We compute the logSNR(t).
        2. Create a noised input x_t = α(t)*x + σ(t)*noise.
        3. Model predicts v(x_t).
        4. We derive the noise prediction and x_0 prediction from v.
        5. MSE loss vs. the real noise (with a sigmoid weighting).
        """
        # Convert continuous time t in [0,1] to logSNR
        logsnr = self.training_schedule(k)  # shape = [B, T, ...]

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        # Noised input
        x_t = alpha_t * x + sigma_t * noise

        # Model does v-pred
        # (If the model needs shape [B, T, D], ensure it matches your architecture.)
        # print(0)
        model = self.model if adapter is None else adapter
        v_pred = model(
            x_t,
            self.precond_scale * logsnr,  # e.g. condition on scaled logSNR
            external_cond,
            **kwargs,
        )

        # Post-process to noise_pred & x_0 pred:
        noise_pred = alpha_t * v_pred + sigma_t * x_t  # => predicted noise
        x_pred = alpha_t * x_t - sigma_t * v_pred  # => predicted x_0

        # MSE loss wrt real noise
        loss = F.mse_loss(noise_pred, noise, reduction="none")

        # Sigmoid weighting: shape-match with add_shape_channels
        bias = self.sigmoid_bias
        loss_weight = torch.sigmoid(bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss, None

    def model_predictions(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        # uncond_cond=None,
        # uncond_cond_mask=None,
        cfg_scale=1,
        adapter=None,
        **kwargs,
    ):

        other_output = None
        model = self.model if adapter is None else adapter
        model_output = model(
            x,
            self.precond_scale * self.logsnr[k],
            external_cond,
            external_cond_mask,
            **kwargs,
        )
        if type(model_output) is tuple:
            model_output = model_output[0]

        if adapter is not None:
            adapter_output, *_ = adapter(
                x,
                self.precond_scale * self.logsnr[k],
                external_cond,
                external_cond_mask,
                **kwargs,
            )
            if type(adapter_output) is tuple:
                adapter_output = adapter_output[0]

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            pred_noise = self.predict_noise_from_v(x, k, v)
            if True:
                adapter_noise = self.predict_noise_from_v(x, k, adapter_output)
                pred_noise = pred_noise + 2 * adapter_noise
            # x_start = self.predict_start_from_v(x, k, v)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred, other_output

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] = None,
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
            cfg_scale=cfg_scale,
        )
        # else:
        # raise NotImplementedError(
        # "Only DDIM sampling is implemented for continuous diffusion."
        # )

    def _predict_noise_and_xstart(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor],
        guidance_fn: Optional[Callable],
        cfg_scale: float,
        adapter: Optional[nn.Module],
        alpha: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Returns
        -------
        pred_noise : ε̂_θ  (same shape as `x`)
        x_start    : predicted clean sample ẋ₀
        aux_output : anything extra returned by self.model_predictions
        """
        # ------------------------------------------------------------------ guidance
        if guidance_fn is not None:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_()
                model_pred = self.model_predictions(
                    x=x_in,
                    k=k,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    **kwargs,
                )

                guidance_loss = guidance_fn(
                    xk=x_in, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(guidance_loss, x_in, retain_graph=False)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # avoid NaN at terminal SNR
                    self.predict_start_from_noise(x_in, k, pred_noise),
                    model_pred.pred_x_start,
                )
            return pred_noise, x_start, ()

        # -------------------------------------------------------- plain / adapter fn
        # (no guidance_fn branch)
        aux_output = ()
        model_pred, *aux_output = self.model_predictions(
            x=x,
            k=k,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            **kwargs,
        )
        pred_noise = model_pred.pred_noise
        x_start = model_pred.pred_x_start

        if adapter is not None:
            adapter_pred, *_ = self.model_predictions(
                x=x,
                k=k,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                adapter=adapter,
                **kwargs,
            )
            # classifier‑free–style interpolation
            # pred_noise = pred_noise + cfg_scale * (adapter_pred.pred_noise - pred_noise)
            pred_noise = adapter_pred.pred_noise
            # pred_noise = 0.5 * ( pred_noise + adapter_pred.pred_noise)
            x_start = torch.where(
                alpha > 0,
                self.predict_start_from_noise(x, k, pred_noise),
                # model_pred.pred_x_start,
                adapter_pred.pred_x_start,
            )

        return pred_noise, x_start, tuple(aux_output)

    def p_mean_variance(self, x, k, external_cond=None, external_cond_mask=None, **kwargs):
        model_pred, _ = self.model_predictions(
            x=x,
            k=k,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            **kwargs,
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def ddpm_sample_step(
        self,
        x,
        curr_noise_level,
        external_cond,
        external_cond_mask=None,
        guidance_fn=None,
        adapter=None,
        cfg_scale=1.0,
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
            cfg_scale=cfg_scale,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return (
            torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred),
            None,
        )

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
        ula_steps: int = 0,  # ← NEW: # Langevin steps at current σ
        ula_eta_sqrt2: bool = True,  # ← NEW: exact ULA (True) vs. tempered (False)
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

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        # ------------------------------------------------------- ε̂ and x̂₀ (initial)
        pred_noise, x_start, aux_output = self._predict_noise_and_xstart(
            x,
            clipped_curr_noise_level,
            external_cond,
            external_cond_mask,
            guidance_fn,
            cfg_scale,
            adapter,
            alpha,
            **kwargs,
        )

        # ------------------------------------------------------- ULA inner loop
        if ula_steps > 0:
            for _ in range(ula_steps):
                score = -pred_noise / torch.sqrt(1.0 - alpha)
                # score = -pred_noise / torch.sqrt(sigma)
                # x = self._ula_substep(x, score, beta_t, eta_sqrt2=ula_eta_sqrt2)
                x = self._ula_substep(x, score, sigma, eta_sqrt2=ula_eta_sqrt2)
                # Re‑predict ε̂ and x̂₀ after the hop
                pred_noise, x_start, _ = self._predict_noise_and_xstart(
                    x,
                    clipped_curr_noise_level,
                    external_cond,
                    external_cond_mask,
                    guidance_fn,
                    cfg_scale,
                    adapter,
                    alpha,
                    **kwargs,
                )

        # ------------------------------------------------------- standard DDIM jump
        noise = torch.randn_like(x).clamp(-self.clip_noise, self.clip_noise)
        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # keep frames where noise level does not change (video‑batch case)
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(self.add_shape_channels(mask), x, x_pred)

        return x_pred, aux_output

    def _ula_substep(self, x, score, beta_t, eta_sqrt2=True):
        """
        Perform a single ULA update at fixed noise level t.

        Args
        ----
        x        : current sample (requires_grad=False)
        score    : ∇_x log p_t(x) ≈ -eps / sqrt(1-α_t)
        beta_t   : forward‑process variance at level t
        eta_sqrt2: if True use √2 β_t noise (exact ULA);
                if False use β_t noise (temperature 1/√2)
        """
        step = beta_t  # η = β_t
        noise_coeff = (2.0 if eta_sqrt2 else 1.0) * step
        x = x + step * score + noise_coeff.sqrt() * torch.randn_like(x)
        return torch.clamp(x, -self.clip_noise, self.clip_noise)

    def ddim_sample_step_leg(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] = None,
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
                # adapter=adapter,
                **kwargs,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

            if adapter is not None:
                adapter_pred, *aux_output = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                    adapter=adapter,
                    **kwargs,
                )

                # pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * adapter_pred.pred_noise
                # pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * adapter_pred.pred_noise
                # pred_noise = adapter_pred.pred_noise
                # pred_noise = 0.4 * model_pred.pred_noise + adapter_pred.pred_noise
                pred_noise = model_pred.pred_noise + cfg_scale * (
                    adapter_pred.pred_noise - model_pred.pred_noise
                )
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise),
                    model_pred.pred_x_start,
                    # adapter_pred.pred_x_start,
                )

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred, aux_output

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)
