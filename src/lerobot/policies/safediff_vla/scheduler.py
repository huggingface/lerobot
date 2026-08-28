import math

import torch
from torch import Tensor, nn


class DDPMScheduler(nn.Module):
    """Small dependency-free DDPM scheduler for normalized action chunks."""

    def __init__(self, num_steps: int, schedule: str = "cosine") -> None:
        super().__init__()
        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, num_steps)
        elif schedule == "cosine":
            x = torch.linspace(0, num_steps, num_steps + 1)
            alpha_bar = torch.cos(((x / num_steps + 0.008) / 1.008) * math.pi / 2).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def step(self, predicted_noise: Tensor, timestep: int, sample: Tensor, add_noise: bool = True) -> Tensor:
        beta = self.betas[timestep]
        alpha = self.alphas[timestep]
        alpha_bar = self.alpha_bars[timestep]
        mean = (sample - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if timestep > 0 and add_noise:
            previous_alpha_bar = self.alpha_bars[timestep - 1]
            variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
            mean = mean + variance.clamp_min(1e-20).sqrt() * torch.randn_like(sample)
        return mean
