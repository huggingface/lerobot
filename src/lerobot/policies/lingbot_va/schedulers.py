# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Flow-matching scheduler for LingBot-VA.

Vendored verbatim from the upstream LingBot-VA repository
(https://github.com/Robbyant/lingbot-va, ``wan_va/utils/scheduler.py``). LingBot-VA uses
two independent instances of this scheduler at inference time — one for the video-latent
stream and one for the action stream — each with its own ``shift`` (signal-to-noise ratio
shift) and number of denoising steps.
"""

import math

import torch

__all__ = ["FlowMatchScheduler"]


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len)
                if dynamic_shift_len is not None
                else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep, t_dim=2):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0
        )
        weights = self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
