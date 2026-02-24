from collections import namedtuple
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .discrete_diffusion import (  # or wherever you keep these
    DiscreteDiffusion,
    ModelPrediction,
)


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


class ContinuousDiffusion(DiscreteDiffusion):
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
        if loss_weighting is None:
            loss_weighting = {"strategy": "sigmoid", "sigmoid_bias": 0.0}

        # Validate required settings for continuous
        if objective != "pred_v":
            raise ValueError("ContinuousDiffusion only supports objective='pred_v'.")
        if loss_weighting.get("strategy", "") != "sigmoid":
            raise ValueError(
                "ContinuousDiffusion only supports loss_weighting.strategy='sigmoid'."
            )

        # We call DiscreteDiffusion.__init__ to reuse its chunking, sampling, etc.
        super().__init__(
            model=model,
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            beta_schedule=beta_schedule,  # This won't be used for training, but is used for sampling steps
            schedule_fn_kwargs=schedule_fn_kwargs,  # We won't rely on them, but keep to satisfy DiscreteDiffusion's init
            objective=objective,
            loss_weighting=loss_weighting,
            ddim_sampling_eta=ddim_sampling_eta,
            clip_noise=clip_noise,
            use_causal_mask=use_causal_mask,
        )
        # Overwrite discrete flag:
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

    def model_predictions(
        self,
        x,
        k,
        external_cond=None,
        external_cond_mask=None,
        uncond_cond=None,
        uncond_cond_mask=None,
        cfg_scale=1,
        **kwargs,
    ):

        other_output = None
        if cfg_scale == 1.0 or uncond_cond is None:
            model_output = self.model(
                x,
                self.precond_scale * self.logsnr[k],
                external_cond,
                external_cond_mask,
                **kwargs,
            )
            if type(model_output) is tuple:
                model_output_cond = model_output
        else:
            model_output_cond = self.model(
                x,
                self.precond_scale * self.logsnr[k],
                external_cond,
                external_cond_mask,
                **kwargs,
            )
            if type(model_output_cond) is tuple:
                model_output_cond, *other_output_cond = model_output_cond

            if uncond_cond is None:
                uncond_cond = torch.zeros_like(external_cond)
                uncond_cond_mask = torch.zeros_like(external_cond_mask)

            model_output_uncond = self.model(
                x,
                self.precond_scale * self.logsnr[k],
                uncond_cond,
                uncond_cond_mask,
                **kwargs,
            )
            if type(model_output_uncond) is tuple:
                model_output_uncond, *other_output_uncond = model_output_uncond

            model_output = model_output_uncond + cfg_scale * (
                model_output_cond - model_output_uncond
            )
            other_output = other_output_cond

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred, other_output

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
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
        v_pred = self.model(
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
