#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .manifest import IterativeConfig


def _compute_betas(
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
) -> NDArray[np.floating]:
    if beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    elif beta_schedule == "scaled_linear":
        return np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
    elif beta_schedule == "squaredcos_cap_v2":
        return _betas_for_alpha_bar(num_train_timesteps)
    else:
        raise ValueError(f"Unknown beta_schedule: {beta_schedule}")


def _betas_for_alpha_bar(num_train_timesteps: int, max_beta: float = 0.999) -> NDArray[np.floating]:
    def alpha_bar(t: float) -> float:
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class DDPMScheduler:
    def __init__(self, config: IterativeConfig):
        self.num_train_timesteps = config.num_train_timesteps
        self.prediction_type = config.prediction_type
        self.clip_sample = config.clip_sample
        self.clip_sample_range = config.clip_sample_range

        self.betas = _compute_betas(
            config.num_train_timesteps,
            config.beta_start,
            config.beta_end,
            config.beta_schedule,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.timesteps: NDArray[np.int64] | None = None
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> NDArray[np.int64]:
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = timesteps
        return timesteps

    def step(
        self,
        model_output: NDArray[np.floating],
        timestep: int,
        sample: NDArray[np.floating],
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step")

        t = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else np.float32(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = np.clip(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )

        pred_original_sample_coeff = (np.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        if t > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
            variance = np.clip(variance, 1e-20, None)
            if generator is not None:
                noise = generator.standard_normal(model_output.shape).astype(np.float32)
            else:
                noise = np.random.randn(*model_output.shape).astype(np.float32)
            pred_prev_sample = pred_prev_sample + np.sqrt(variance) * noise

        return pred_prev_sample


class DDIMScheduler:
    def __init__(self, config: IterativeConfig, eta: float = 0.0):
        self.num_train_timesteps = config.num_train_timesteps
        self.prediction_type = config.prediction_type
        self.clip_sample = config.clip_sample
        self.clip_sample_range = config.clip_sample_range
        self.eta = eta

        self.betas = _compute_betas(
            config.num_train_timesteps,
            config.beta_start,
            config.beta_end,
            config.beta_schedule,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.final_alpha_cumprod = np.float32(1.0)

        self.timesteps: NDArray[np.int64] | None = None
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> NDArray[np.int64]:
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = timesteps
        return timesteps

    def step(
        self,
        model_output: NDArray[np.floating],
        timestep: int,
        sample: NDArray[np.floating],
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step")

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - np.sqrt(alpha_prod_t) * pred_original_sample) / np.sqrt(beta_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
            pred_epsilon = np.sqrt(alpha_prod_t) * model_output + np.sqrt(beta_prod_t) * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = np.clip(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )

        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * np.sqrt(variance)

        pred_sample_direction = np.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        prev_sample = np.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        if self.eta > 0:
            if generator is not None:
                noise = generator.standard_normal(model_output.shape).astype(np.float32)
            else:
                noise = np.random.randn(*model_output.shape).astype(np.float32)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample


def create_scheduler(config: IterativeConfig, eta: float = 0.0):
    scheduler_type = config.scheduler.lower()
    if scheduler_type == "ddpm":
        return DDPMScheduler(config)
    elif scheduler_type == "ddim":
        return DDIMScheduler(config, eta=eta)
    elif scheduler_type == "euler":
        return None
    else:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. Supported schedulers: 'ddpm', 'ddim', 'euler'."
        )
