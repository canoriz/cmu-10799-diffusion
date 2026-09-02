"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    """An epsilon-prediction DDPM with optional respaced sampling."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        # TODO: Implement your own init
        if self.num_timesteps < 1:
            raise ValueError("num_timesteps must be positive")
        if not (0.0 < self.beta_start <= self.beta_end < 1.0):
            raise ValueError("beta_start and beta_end must satisfy 0 < start <= end < 1")

        # Configuration values validated; begin diffusion schedule construction.
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log((1.0 - alphas_cumprod).clamp(min=1e-20)))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Added implementation: precompute and register all diffusion schedule
        # and posterior coefficients needed by training and sampling.

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================

    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that

    # =========================================================================
    # Forward process
    # =========================================================================

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Gather schedule values at t and broadcast over image dimensions."""
        out = values.to(device=t.device)[t.long()]
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def _validate_t(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=self.betas.device)
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim != 1 or t.shape[0] not in (1, batch_size):
            raise ValueError(f"Expected timestep tensor with shape ({batch_size},), got {tuple(t.shape)}")
        elif t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size)
        t = t.to(device=self.betas.device, dtype=torch.long)
        if torch.any(t < 0) or torch.any(t >= self.num_timesteps):
            raise ValueError(f"Timesteps must be in [0, {self.num_timesteps - 1}]")
        # Timestep type, shape, and range validation complete.
        return t

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t from the closed-form forward process q(x_t | x_0)."""
        # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        if x_0.ndim < 2:
            raise ValueError("x_0 must have a batch dimension and image/features dimensions")
        if t is None:
            t = torch.randint(self.num_timesteps, (x_0.shape[0],), device=x_0.device)
        t = self._validate_t(t, x_0.shape[0]).to(x_0.device)
        if noise is None:
            noise = torch.randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError("noise must have the same shape as x_0")
        else:
            noise = noise.to(device=x_0.device, dtype=x_0.dtype)

        # Input types, shapes, and values validated; begin the forward process.
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape).to(dtype=x_0.dtype)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(dtype=x_0.dtype)
        # Added implementation: sample x_t directly from x_0 using the closed-form
        # forward process instead of iterating through every earlier timestep.
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Train the denoiser to predict the Gaussian noise used by q.
        """
        batch_size = x_0.shape[0]
        t = kwargs.pop("t", None)
        if t is None:
            t = torch.randint(self.num_timesteps, (batch_size,), device=x_0.device)
        t = self._validate_t(t, batch_size).to(x_0.device)
        noise = kwargs.pop("noise", None)
        if noise is None:
            noise = torch.randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError("noise must have the same shape as x_0")
        else:
            noise = noise.to(device=x_0.device, dtype=x_0.dtype)

        # Training inputs validated; begin the core noise-prediction logic.
        x_t = self.forward_process(x_0, t=t, noise=noise)
        predicted_noise = self.model(x_t, t)
        if predicted_noise.shape != noise.shape:
            raise ValueError(
                f"Model output shape {tuple(predicted_noise.shape)} does not match noise shape {tuple(noise.shape)}"
            )
        loss = F.mse_loss(predicted_noise, noise)
        detached_loss = loss.detach()
        # Added implementation: train an epsilon-prediction model with MSE and
        # expose both names used by the training logger.
        return loss, {"loss": detached_loss, "mse": detached_loss}

    @torch.no_grad()
    def reverse_process(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)

        Take one reverse step, optionally jumping from t to t_prev.
        """
        if "step" in kwargs and t_prev is None:
            t_prev = kwargs["step"]
        if x_t.ndim < 2:
            raise ValueError("x_t must have a batch dimension")
        t = self._validate_t(t, x_t.shape[0]).to(x_t.device)
        if t_prev is None:
            t_prev = t - 1
        elif not torch.is_tensor(t_prev):
            t_prev = torch.full_like(t, int(t_prev))
        else:
            t_prev = t_prev.to(device=x_t.device, dtype=torch.long)
            if t_prev.ndim == 0:
                t_prev = t_prev.expand(x_t.shape[0])
            elif t_prev.shape[0] == 1 and x_t.shape[0] != 1:
                t_prev = t_prev.expand(x_t.shape[0])
        if t_prev.shape != t.shape:
            raise ValueError("t_prev must be scalar or have the same batch shape as t")
        if torch.any(t_prev < -1) or torch.any(t_prev >= t):
            raise ValueError("t_prev must satisfy -1 <= t_prev < t")

        # Reverse-step inputs validated; begin the core denoising logic.
        predicted_noise = self.model(x_t, t)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape).to(x_t.dtype)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape).to(x_t.dtype)
        x0_pred = (x_t - sqrt_one_minus_t * predicted_noise) / sqrt_alpha_bar_t.clamp(min=1e-20)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        final_mask = t_prev < 0
        if torch.all(final_mask):
            return x0_pred

        safe_prev = t_prev.clamp(min=0)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape).to(x_t.dtype)
        alpha_bar_prev = self._extract(self.alphas_cumprod, safe_prev, x_t.shape).to(x_t.dtype)
        alpha_ratio = (alpha_bar_t / alpha_bar_prev).clamp(min=1e-20, max=1.0)
        beta_ratio = 1.0 - alpha_ratio
        denom = (1.0 - alpha_bar_t).clamp(min=1e-20)
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_ratio / denom
        coef_xt = torch.sqrt(alpha_ratio) * (1.0 - alpha_bar_prev) / denom
        mean = coef_x0 * x0_pred + coef_xt * x_t
        variance = (beta_ratio * (1.0 - alpha_bar_prev) / denom).clamp(min=0.0)
        sample = mean + torch.sqrt(variance) * torch.randn_like(x_t)
        if torch.any(final_mask):
            mask = final_mask.reshape(-1, *((1,) * (x_t.ndim - 1)))
            sample = torch.where(mask, x0_pred, sample)
        # Added implementation: support both single-step and respaced posterior
        # transitions while avoiding noise on the final x_0 output.
        return sample

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs,
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)

        Generate samples from Gaussian noise using full or respaced DDPM.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(image_shape) != 3 or any(int(dim) < 1 for dim in image_shape):
            raise ValueError("image_shape must be a positive (channels, height, width) tuple")
        num_steps = kwargs.pop("num_steps", None)
        num_steps = self.num_timesteps if num_steps is None else int(num_steps)
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f"num_steps must be in [1, {self.num_timesteps}]")
        clip_denoised = bool(kwargs.pop("clip_denoised", True))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected sampling arguments: {unknown}")

        # Sampling arguments validated; begin the reverse diffusion loop.
        device = self.device
        x = torch.randn((batch_size, *tuple(image_shape)), device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        self.eval_mode()
        for index, current in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < timesteps.numel() else -1
            t = torch.full((batch_size,), int(current.item()), device=device, dtype=torch.long)
            x = self.reverse_process(x, t, t_prev=previous, clip_denoised=clip_denoised)
        # Added implementation: start from Gaussian noise and run the configured
        # full or respaced reverse-time schedule.
        return x.clamp(-1.0, 1.0)

    def to(self, device: torch.device) -> "DDPM":
        nn.Module.to(self, device)
        self.device = torch.device(device)
        return self

    def get_config(self) -> Dict[str, object]:
        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    def state_dict(self, *args, **kwargs) -> Dict:
        # TODO: add other things you want to save
        # Added implementation: include the schedule configuration and registered
        # buffers alongside the wrapped model state.
        state = super().state_dict(*args, **kwargs)
        state.update(self.get_config())
        state.update({name: value.detach().clone() for name, value in self.named_buffers()})
        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True, **kwargs):
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
