#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Objective implementations for Multi-Task DiT policy.

- DiffusionObjective: Standard DDPM/DDIM diffusion
- FlowMatchingObjective: Flow matching with ODE integration
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor


class BaseObjective(ABC):
    """Base class for objectives used in Multi-Task DiT policy."""

    def __init__(self, config, action_dim: int, horizon: int):
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        """Compute training loss."""
        pass

    @abstractmethod
    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        """Generate action samples conditioned on observations."""
        pass


class DiffusionObjective(BaseObjective):
    """Standard diffusion (DDPM/DDIM) objective implementation."""

    def __init__(self, config, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__(config, action_dim, horizon)
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        # Build noise scheduler
        scheduler_kwargs = {
            "num_train_timesteps": config.num_train_timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "clip_sample_range": config.clip_sample_range,
            "prediction_type": config.prediction_type,
        }

        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler: DDPMScheduler | DDIMScheduler = DDPMScheduler(**scheduler_kwargs)
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")

        # Inference steps default to training steps if not provided
        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        clean_actions = batch["action"]
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(clean_actions.shape[0],),
            device=clean_actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

        # Target depends on prediction type
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = clean_actions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        predicted = model(noisy_actions, timesteps, conditioning_vec=conditioning_vec)
        loss = F.mse_loss(predicted, target, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_actions = ~batch["action_is_pad"]  # (B, T)
            loss = loss * valid_actions.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sample = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning_vec,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample


class FlowMatchingObjective(BaseObjective):
    """Flow matching objective: trains a model to predict velocity fields."""

    def __init__(self, config, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__(config, action_dim, horizon)
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps according to configured strategy."""
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.timestep_sampling_strategy == "beta":
            # Sample u ~ Beta(α, β) then transform: t = s(1-u)
            # This emphasizes t near 0 (high noise) when α > β
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha, self.config.timestep_sampling_beta
            )
            u = beta_dist.sample((batch_size,)).to(device)
            return self.config.timestep_sampling_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.config.timestep_sampling_strategy}")

    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        """Compute flow matching training loss."""
        data = batch["action"]  # Clean action sequences (B, T, D)
        batch_size = data.shape[0]
        device = data.device

        noise = torch.randn_like(data)
        t = self._sample_timesteps(batch_size, device)
        t_expanded = t.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise

        # The velocity we want the model to learn: v = data - (1-σ)·noise
        target_velocity = data - (1 - self.config.sigma_min) * noise
        predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        # Optionally mask padded actions
        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"]  # (B, T)
            loss = loss * valid_mask.unsqueeze(-1)  # (B, T, D)

        return loss.mean()

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        """Generate actions by integrating the learned velocity field via ODE."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Start from random noise at t=0
        x = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device)

        # Time grid from 0 to 1
        num_steps = self.config.num_integration_steps
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        # Integrate ODE using chosen method
        if self.config.integration_method == "euler":
            x = self._euler_integrate(model, x, time_grid, conditioning_vec)
        elif self.config.integration_method == "rk4":
            x = self._rk4_integrate(model, x, time_grid, conditioning_vec)
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

        return x

    def _euler_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        """Euler integration: x_{n+1} = x_n + dt * v_θ(x_n, t_n)"""
        x = x_init

        for i in range(len(time_grid) - 1):
            t_scalar = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            # Create time tensor for batch
            t_batch = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)

            # Get velocity at current point
            with torch.no_grad():
                velocity = model(x, t_batch, conditioning_vec=conditioning_vec)

            # Euler step
            x = x + dt * velocity

        return x

    def _rk4_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        """4th-order Runge-Kutta integration."""
        x = x_init

        def dynamics(x_val: Tensor, t_scalar: float) -> Tensor:
            t_batch = torch.full((x_val.shape[0],), t_scalar, dtype=x_val.dtype, device=x_val.device)
            with torch.no_grad():
                return model(x_val, t_batch, conditioning_vec=conditioning_vec)

        for i in range(len(time_grid) - 1):
            t = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            # RK4 stages
            k1 = dynamics(x, t)
            k2 = dynamics(x + dt * k1 / 2, t + dt / 2)
            k3 = dynamics(x + dt * k2 / 2, t + dt / 2)
            k4 = dynamics(x + dt * k3, t + dt)

            # Weighted combination
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
