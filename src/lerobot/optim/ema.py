#!/usr/bin/env python

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
"""Exponential moving average (EMA) of the trainable policy weights."""

import abc
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from lerobot.utils.constants import EMA_STATE


@dataclass
class EMAConfig(draccus.ChoiceRegistry, abc.ABC):
    """Exponential moving average (EMA) of the trainable policy weights, enabled by choosing a decay with `--ema.type`.

    The first averaging update copies the weights, every later one blends them in with the decay of the selected type.
    """

    # Number of optimizer updates during which the average stays a copy of the live weights.
    update_after_step: int = 0

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "warmup"

    def __post_init__(self) -> None:
        if self.update_after_step < 0:
            raise ValueError(f"ema.update_after_step must be >= 0, got {self.update_after_step}.")

    @abc.abstractmethod
    def get_decay(self, step: int) -> float:
        """Decay of the `step`-th averaging update, for `step` >= 2."""


@EMAConfig.register_subclass("warmup")
@dataclass
class WarmupEMAConfig(EMAConfig):
    """`1 - (1 + (step - 1) / inv_gamma) ** -power`, clamped to `[min_decay, max_decay]`.

    The warmup schedule of diffusers' `EMAModel`, with the defaults of the Diffusion Policy reference.
    """

    inv_gamma: float = 1.0
    power: float = 0.75
    min_decay: float = 0.0
    max_decay: float = 0.9999

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= self.min_decay <= self.max_decay <= 1.0):
            raise ValueError(
                "Expected 0 <= ema.min_decay <= ema.max_decay <= 1, got "
                f"min_decay={self.min_decay} and max_decay={self.max_decay}."
            )
        if self.inv_gamma <= 0:
            raise ValueError(f"ema.inv_gamma must be positive, got {self.inv_gamma}.")
        if self.power <= 0:
            raise ValueError(f"ema.power must be positive, got {self.power}.")

    def get_decay(self, step: int) -> float:
        return max(min(1 - (1 + (step - 1) / self.inv_gamma) ** -self.power, self.max_decay), self.min_decay)


@EMAConfig.register_subclass("constant")
@dataclass
class ConstantEMAConfig(EMAConfig):
    """The same decay at every update, like `torch.optim.swa_utils.get_ema_multi_avg_fn` and openpi's `ema_decay`."""

    decay: float = 0.999

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"ema.decay must be in [0, 1], got {self.decay}.")

    def get_decay(self, step: int) -> float:
        return self.decay


@EMAConfig.register_subclass("power")
@dataclass
class PowerEMAConfig(EMAConfig):
    """`(1 - 1 / step) ** (gamma + 1)`, the power function EMA of EDM2 (Karras et al. 2024).

    `gamma` gives the averaging profile a standard deviation of `sigma_rel` (EDM2's relative standard deviation, `std`
    in its code) times the number of updates so far, so the averaged window stays the same fraction of training at any
    length.
    """

    sigma_rel: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.sigma_rel <= 12**-0.5:
            raise ValueError(f"ema.sigma_rel must be in (0, 12 ** -0.5], got {self.sigma_rel}.")

    @property
    def gamma(self) -> float:
        return np.roots([1, 7, 16 - self.sigma_rel**-2, 12 - self.sigma_rel**-2]).real.max()

    def get_decay(self, step: int) -> float:
        return float((1 - 1 / step) ** (self.gamma + 1))


class EMA:
    """Float32 exponential moving average of the trainable parameters of a model.

    Each average has the layout of its parameter, a DTensor with the same placements when the model is sharded, so
    every update stays local to the rank. Build it after the model is prepared for training.
    """

    def __init__(self, model: nn.Module, config: EMAConfig):
        self.config = config
        self.num_updates = 0
        self.decay: float | None = None
        self.averages = {
            name: param.detach().to(torch.float32, copy=True)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """Blend the trainable parameters of `model` into the averages, once per optimizer update."""
        self.num_updates += 1
        step = self.num_updates - self.config.update_after_step
        self.decay = self.config.get_decay(step) if step > 1 else 0.0
        parameters = dict(model.named_parameters())
        torch._foreach_lerp_(
            list(self.averages.values()),
            [parameters[name].detach().to(torch.float32) for name in self.averages],
            1.0 - self.decay,
        )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Load the averages into `model`, restoring its trainable parameters on exit."""
        from torch.distributed.fsdp import FSDPModule

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        parameters = dict(model.named_parameters())
        live = {name: parameters[name].detach().clone() for name in self.averages}
        for name, average in self.averages.items():
            parameters[name].detach().copy_(average)
        try:
            yield
        finally:
            for name, value in live.items():
                parameters[name].detach().copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"num_updates": self.num_updates, "averages": self.averages}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["averages"].keys() != self.averages.keys():
            raise ValueError(
                "The saved EMA averages a different set of parameters: "
                f"{sorted(state_dict['averages'].keys() ^ self.averages.keys())[:5]}..."
            )
        self.num_updates = state_dict["num_updates"]
        for name, average in self.averages.items():
            average.copy_(state_dict["averages"][name])


def save_ema_state(ema: EMA, save_dir: Path) -> None:
    """Write the averages and the update count (non-sharded runs; sharded runs use the DCP channel)."""
    save_file(ema.averages, save_dir / EMA_STATE, metadata={"num_updates": str(ema.num_updates)})


def load_ema_state(ema: EMA, save_dir: Path) -> None:
    """Restore the averages written by `save_ema_state`, restarting the EMA when the checkpoint carries none."""
    if not (save_dir / EMA_STATE).is_file():
        logging.warning("No %s in the checkpoint: the EMA restarts from the resumed weights.", EMA_STATE)
        return
    with safe_open(save_dir / EMA_STATE, framework="pt") as file:
        ema.load_state_dict(
            {
                "num_updates": int(file.metadata()["num_updates"]),
                "averages": {name: file.get_tensor(name) for name in file.keys()},  # noqa: SIM118
            }
        )
