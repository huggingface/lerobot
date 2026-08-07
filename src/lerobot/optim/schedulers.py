#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package
from lerobot.utils.io_utils import deserialize_json_into_object, write_json

if TYPE_CHECKING or _diffusers_available:
    from diffusers.optimization import get_scheduler
else:
    get_scheduler = None


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration shared by every learning rate scheduler.

    Concrete schedulers subclass this and register themselves with
    `@LRSchedulerConfig.register_subclass("name")`, which is what makes `--scheduler.type=name` work on the
    command line.

    Args:
        num_warmup_steps (`int | None`):
            Number of steps over which the learning rate ramps up from 0 before the scheduler's own
            behavior takes over. `None` disables warmup.
    """

    num_warmup_steps: int | None

    @property
    def type(self) -> str:
        """Return the registered name this config was registered under.

        Returns:
            `str`: The name passed to `@LRSchedulerConfig.register_subclass`, e.g. `"diffuser"`.
        """
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        """Build the scheduler for a given optimizer and training length.

        Args:
            optimizer (`Optimizer`):
                The optimizer whose learning rate the scheduler will adjust.
            num_training_steps (`int`):
                Total number of training steps, used to compute decay/annealing schedules.

        Returns:
            `LRScheduler | None`: The built scheduler.
        """
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    """A [`diffusers`](https://huggingface.co/docs/diffusers) learning rate schedule.

    Args:
        num_warmup_steps (`int`, *optional*):
            Number of steps over which the learning rate ramps up from 0. `None` disables warmup.
        name (`str`, *optional*, defaults to `"cosine"`):
            Name of the `diffusers` schedule to build, e.g. `"cosine"`, `"linear"`, `"constant"`. See
            [`diffusers.optimization.get_scheduler`](https://huggingface.co/docs/diffusers/api/schedulers/overview)
            for the full list.
    """

    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """See [`~optim.schedulers.LRSchedulerConfig.build`]. Delegates to `diffusers.optimization.get_scheduler`."""
        require_package("diffusers", extra="diffusion")

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    """Used to train VQ-BeT: constant LR during VQ-VAE pretraining, then warmup and cosine decay.

    Args:
        num_warmup_steps (`int`):
            Number of steps over which the learning rate ramps up from 0, counted from the end of VQ-VAE
            pretraining.
        num_vqvae_training_steps (`int`):
            Number of initial steps spent pretraining the VQ-VAE, during which the LR stays at its peak.
        num_cycles (`float`, *optional*, defaults to 0.5):
            Number of cosine cycles in the decay phase; 0.5 decays smoothly to 0 by the end of training.
    """

    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """See [`~optim.schedulers.LRSchedulerConfig.build`].

        Holds the LR at its peak during VQ-VAE pretraining, then applies linear warmup followed by cosine
        decay for the remaining steps.
        """

        def lr_lambda(current_step):
            """Return the LR multiplier for `current_step`, per the VQ-BeT schedule."""
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("constant_with_warmup")
@dataclass
class ConstantWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Linear warmup followed by a constant learning rate.

    Mirrors the ``warmup_constant_lambda`` used by LingBot-VA (upstream ``wan_va/train.py``):
    the LR ramps linearly from 0 to the peak over ``num_warmup_steps`` steps, then stays flat.

    Args:
        num_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of steps over which the learning rate ramps up from 0 to its peak.
    """

    num_warmup_steps: int = 1000

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """See [`~optim.schedulers.LRSchedulerConfig.build`]."""
        warmup_steps = self.num_warmup_steps or 0

        def lr_lambda(current_step):
            """Return the LR multiplier for `current_step`: linear ramp, then constant `1.0`."""
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_annealing_with_warmup")
@dataclass
class CosineAnnealingWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Linear warmup followed by cosine annealing from the peak LR to zero.

    Used by EVO1; the annealing phase always spans the remaining training steps.

    Args:
        num_warmup_steps (`int`):
            Number of steps over which the learning rate ramps up from 0 to its peak.
    """

    num_warmup_steps: int

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """See [`~optim.schedulers.LRSchedulerConfig.build`]."""

        def lr_lambda(current_step: int) -> float:
            """Return the LR multiplier for `current_step`: linear warmup, then cosine annealing to 0."""
            if current_step < self.num_warmup_steps:
                return current_step / max(1, self.num_warmup_steps)
            progress = (current_step - self.num_warmup_steps) / max(
                1, num_training_steps - self.num_warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.

    Args:
        num_warmup_steps (`int`):
            Number of steps over which the learning rate ramps up from `peak_lr / (num_warmup_steps + 1)`
            to `peak_lr`.
        num_decay_steps (`int`):
            Number of steps over which the learning rate decays from `peak_lr` to `decay_lr`. Scaled down
            automatically if `num_training_steps` is shorter than this.
        peak_lr (`float`):
            Learning rate reached at the end of warmup.
        decay_lr (`float`):
            Learning rate reached at the end of decay.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        """See [`~optim.schedulers.LRSchedulerConfig.build`].

        If `num_training_steps` is shorter than `num_decay_steps`, scales `num_warmup_steps` and
        `num_decay_steps` down proportionally so the schedule still completes.
        """
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            """Return the LR multiplier for `current_step`: linear warmup, then cosine decay."""

            def linear_warmup_schedule(current_step):
                """Return the LR multiplier during warmup, ramping from `1 / (warmup + 1)` to 1."""
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                """Return the LR multiplier during decay, from 1 down to `decay_lr / peak_lr`."""
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    """Save a scheduler's state to disk.

    Args:
        scheduler (`LRScheduler`):
            The scheduler whose state to save.
        save_dir (`Path`):
            Directory to save the scheduler state.
    """
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    """Load a scheduler's state from disk.

    Args:
        scheduler (`LRScheduler`):
            The scheduler to load state into.
        save_dir (`Path`):
            Directory to load the scheduler state from.

    Returns:
        `LRScheduler`: The same scheduler, with its state loaded.
    """
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
