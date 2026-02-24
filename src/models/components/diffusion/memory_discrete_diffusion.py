from collections import namedtuple
from typing import Any, Callable, Dict, Literal, Optional

import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F

# local import – unchanged
from src.models.components.diffusion.noise_schedule import make_beta_schedule

__all__ = [
    "DiscreteDiffusion",
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather 1‑D buffer values at indices *t* and reshape for broadcasting."""
    out = a[t]
    return out.view(*t.shape, *((1,) * (len(x_shape) - t.ndim)))


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])

# -----------------------------------------------------------------------------
# DiscreteDiffusion – DDPM‑only sampler with Video‑Adapter & CFG
# -----------------------------------------------------------------------------


class DiscreteDiffusion(nn.Module):
    """A *discrete* diffusion model with:
    • Video‑Adapter product‑of‑experts fusion (γ)
    • Classifier‑free guidance (α)
    • Canonical DDPM ancestral sampler (Ho et al., 2020)

    Public signatures remain identical to the original implementation so this is
    a drop‑in replacement.
    """

    # ------------------------------------------------------------------
    # INITIALISATION ----------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: nn.Module,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Dict[str, Any] | None = None,
        objective: str = "v_pred",
        loss_weighting: Dict[str, Any] | None = None,
        clip_noise: float = 20.0,
        use_causal_mask: bool = False,
        reconstruction_guidance: Optional[Callable] | None = None,
    ) -> None:
        super().__init__()

        # save args ------------------------------------------------------
        self.model = model  # task‑specific (small) diffusion model
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        self.external_cond_dim = external_cond_dim
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.schedule_fn_kwargs = schedule_fn_kwargs or {"shift": 1.0}
        self.objective = objective
        self.loss_weighting = loss_weighting or {
            "strategy": "fused_min_snr",
            "snr_clip": 5.0,
            "cum_snr_decay": 0.9,
        }
        self.clip_noise = clip_noise
        self.use_causal_mask = use_causal_mask
        self.reconstruction_guidance = reconstruction_guidance
        self.is_discrete = True

        # build constant buffers ----------------------------------------
        self._build_buffer()

    # ------------------------------------------------------------------
    # BUFFER PRE‑COMPUTATION -------------------------------------------
    # ------------------------------------------------------------------

    def _build_buffer(self) -> None:
        betas: torch.Tensor = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def reg(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.float(), persistent=False)

        # diffusion schedule --------------------------------------------
        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)

        # derived terms --------------------------------------------------
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        reg("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))

        # posterior q(x_{t‑1}|x_t,x₀) ----------------------------------
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        reg("posterior_variance", posterior_variance)
        reg(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        reg(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod),
        )
        reg(
            "posterior_mean_coef2",
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod),
        )

        # SNR -----------------------------------------------------------
        snr = alphas_cumprod / (1 - alphas_cumprod)
        reg("snr", snr)
        if self.loss_weighting["strategy"] in {"min_snr", "fused_min_snr"}:
            reg("clipped_snr", snr.clamp(max=self.loss_weighting["snr_clip"]))
        elif self.loss_weighting["strategy"] == "sigmoid":
            reg("logsnr", torch.log(snr))

    # ------------------------------------------------------------------
    # SHAPE UTIL -------------------------------------------------------
    # ------------------------------------------------------------------

    def add_shape_channels(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    # ------------------------------------------------------------------
    # MODEL PREDICTIONS (Video‑Adapter + CFG) --------------------------
    # ------------------------------------------------------------------

    def _run_single(self, net: nn.Module, x, k, cond, cond_mask, **kwargs):
        eps = net(x, k, cond, cond_mask, **kwargs)
        return eps.clamp(-self.clip_noise, self.clip_noise), []

    def model_predictions(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor] | None = None,
        external_cond_mask: Optional[torch.Tensor] | None = None,
        uncond_cond: Optional[torch.Tensor] | None = None,
        uncond_cond_mask: Optional[torch.Tensor] | None = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] | None = None,
        prior_scale: float = 1.0,
        **kwargs,
    ) -> tuple[ModelPrediction, list[Any]]:
        aux: list[Any] = []

        def run_combined(cond, cond_mask):
            eps_small, rest = self._run_single(self.model, x, k, cond, cond_mask, **kwargs)
            aux.extend(rest)
            if adapter is not None:
                eps_prior, rest = self._run_single(adapter, x, k, cond, cond_mask, **kwargs)
                aux.extend(rest)
                return eps_small + prior_scale * eps_prior
            return eps_small

        # no CFG path ---------------------------------------------------
        if cfg_scale == 1.0 and uncond_cond is None:
            eps = run_combined(external_cond, external_cond_mask)
            x_start = self.predict_start_from_noise(x, k, eps)
            return ModelPrediction(eps, x_start, eps), aux

        # CFG path ------------------------------------------------------
        eps_text = run_combined(external_cond, external_cond_mask)
        eps_uncond = run_combined(uncond_cond, uncond_cond_mask)
        eps_final = eps_uncond + cfg_scale * (eps_text - eps_uncond)
        x_start = self.predict_start_from_noise(x, k, eps_final)
        return ModelPrediction(eps_final, x_start, eps_final), aux

    # ------------------------------------------------------------------
    # PREDICTION HELPERS ----------------------------------------------
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Q(x_t) UTILITY ---------------------------------------------------
    # ------------------------------------------------------------------

    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        var = extract(1 - self.alphas_cumprod, k, x_start.shape)
        log_var = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, var, log_var

    def q_posterior(self, x_start, x_k, k):
        mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        var = extract(self.posterior_variance, k, x_k.shape)
        log_var = extract(self.posterior_log_variance_clipped, k, x_k.shape)
        return mean, var, log_var

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).clamp(-self.clip_noise, self.clip_noise)
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    # ------------------------------------------------------------------
    # LOSS WEIGHTING ---------------------------------------------------
    # ------------------------------------------------------------------

    def compute_loss_weights(
        self,
        k: torch.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> torch.Tensor:
        if strategy == "uniform":
            return torch.ones_like(k)

        snr = self.snr[k]
        match strategy:
            case "sigmoid":
                logsnr = self.logsnr[k]
                eps_w = torch.sigmoid(self.loss_weighting.get("sigmoid_bias", 0.0) - logsnr)
            case "min_snr":
                clipped = self.clipped_snr[k]
                eps_w = clipped / snr.clamp(min=1e-8)
            case "fused_min_snr":
                snr_clip = self.loss_weighting["snr_clip"]
                decay = self.loss_weighting["cum_snr_decay"]
                clipped = self.clipped_snr[k]
                norm_clip = clipped / snr_clip
                norm_snr = snr / snr_clip

                def cum(reverse=False):
                    nclip = norm_clip.flip(1) if reverse else norm_clip
                    out = torch.zeros_like(nclip)
                    for t in range(k.shape[1]):
                        if t == 0:
                            out[:, t] = nclip[:, t]
                        else:
                            out[:, t] = decay * out[:, t - 1] + (1 - decay) * nclip[:, t]
                    out = F.pad(out[:, :-1], (1, 0, 0, 0), value=0.0)
                    return out.flip(1) if reverse else out

                cum_snr = cum() if self.use_causal_mask else 0.5 * (cum() + cum(True))
                fused_clip = 1 - (1 - cum_snr * decay) * (1 - norm_clip)
                fused = 1 - (1 - cum_snr * decay) * (1 - norm_snr)
                eps_w = (fused_clip * snr_clip) / fused.clamp(min=1e-8)
            case _:
                raise ValueError(f"unknown loss strategy {strategy}")

        match self.objective:
            case "pred_noise":
                return eps_w
            case "pred_x0":
                return eps_w * snr
            case "pred_v":
                return eps_w * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    # ------------------------------------------------------------------
    # TRAIN FORWARD (unchanged logic) ----------------------------------
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        adapter: Optional[nn.Module] = None,
        prior_scale: float = 1.0,
    ):
        noise = torch.randn_like(x).clamp(-self.clip_noise, self.clip_noise)
        noised_x = self.q_sample(x_start=x, k=k, noise=noise)

        model_pred, _ = self.model_predictions(
            x=noised_x,
            k=k,
            external_cond=external_cond,
            adapter=adapter,
            prior_scale=prior_scale,
        )

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        target: torch.Tensor
        match self.objective:
            case "pred_noise":
                target = noise
            case "pred_x0":
                target = x
            case "pred_v":
                target = self.predict_v(x, k, noise)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")
        weight = self.compute_loss_weights(k, self.loss_weighting["strategy"])
        loss = loss * self.add_shape_channels(weight)

        return x_pred, loss, None

    # ------------------------------------------------------------------
    # SAMPLING – ancestral DDPM + CFG ----------------------------------
    # ------------------------------------------------------------------

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] | None = None,
        guidance_fn: Optional[Callable] | None = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] | None = None,
        prior_scale: float = 1.0,
        **kwargs,
    ):
        return self.ddpm_sample_step(
            x,
            curr_noise_level,
            external_cond,
            external_cond_mask,
            guidance_fn,
            cfg_scale,
            adapter,
            prior_scale,
            **kwargs,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] | None = None,
        guidance_fn: Optional[Callable] | None = None,
        cfg_scale: float = 1.0,
        adapter: Optional[nn.Module] | None = None,
        prior_scale: float = 1.0,
        **kwargs,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("grad guidance not supported in DDPM mode")

        t = curr_noise_level.clamp(min=0)
        # unconditional cond == None -> assumes classifier‑free training
        model_pred, aux = self.model_predictions(
            x,
            t,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            uncond_cond=None,
            uncond_cond_mask=None,
            cfg_scale=cfg_scale,
            adapter=adapter,
            prior_scale=prior_scale,
            **kwargs,
        )
        eps_theta = model_pred.pred_noise

        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphabar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = torch.rsqrt(1.0 - beta_t)
        sigma_t = torch.sqrt(beta_t)

        model_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphabar_t * eps_theta)

        noise = torch.randn_like(x).clamp(-self.clip_noise, self.clip_noise)
        nonzero_mask = self.add_shape_channels(t != 0)
        x_prev = model_mean + nonzero_mask * sigma_t * noise

        x_prev = torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_prev)
        return x_prev, aux

    # ------------------------------------------------------------------
    # ESTIMATE NOISE LEVEL (unchanged) ---------------------------------
    # ------------------------------------------------------------------

    def estimate_noise_level(self, x: torch.Tensor, mu: Optional[torch.Tensor] | None = None):
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll = -self.log_one_minus_alphas_cumprod[None, None] - mse[..., None] * self.alphas_cumprod[
            None, None
        ] / (1 - self.alphas_cumprod[None, None])
        return torch.argmax(ll, -1)

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)
