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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import ChoicesFromPolicies, _diffusers_available, require_package
from lerobot.utils.io_utils import deserialize_json_into_object, write_json

if TYPE_CHECKING or _diffusers_available:
    from diffusers.optimization import get_scheduler
else:
    get_scheduler = None


@dataclass
class LRSchedulerConfig(ChoicesFromPolicies, draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int | None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        require_package("diffusers", extra="diffusion")

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
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
    """

    num_warmup_steps: int = 1000

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup_steps = self.num_warmup_steps or 0

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_annealing_with_warmup")
@dataclass
class CosineAnnealingWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Linear warmup followed by cosine annealing from the peak LR to zero.

    Used by EVO1; the annealing phase always spans the remaining training steps.
    """

    num_warmup_steps: int

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
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
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
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
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
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
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler


@LRSchedulerConfig.register_subclass("frozen_warmup_constant")
@dataclass
class FrozenWarmupConstantSchedulerConfig(LRSchedulerConfig):
    """Two-group schedule of the FLUX 3 Action finetune recipe.

    Param group 0 (the trunk) has LR 0 for the first ``freeze_steps`` (the fresh heads train alone), then
    warms up linearly over ``num_warmup_steps`` and stays constant. Param group 1 (the fresh action heads) warms
    up from step 0 over ``warmup_steps_heads``. ``decay_steps`` (linear decay counted from the end of the
    frozen phase, 0 = constant) and ``cooldown_steps`` (linear decay to zero over the final steps, 0 = off)
    are optional. With a single param group only the trunk schedule is used; extra groups follow the heads.

    All phase lengths use training-loop microsteps, as does ``num_training_steps``.
    With gradient accumulation, multiply desired optimizer-update counts by the
    accumulation factor. The trainer advances the scheduler after every microbatch.
    """

    num_warmup_steps: int = 2000  # trunk warmup, after the frozen phase
    freeze_steps: int = 1000
    warmup_steps_heads: int = 1000
    decay_steps: int = 0
    cooldown_steps: int = 0

    @staticmethod
    def lr_lambda(
        warmup: int,
        decay_steps: int,
        frozen: int = 0,
        cooldown: int = 0,
        total_steps: int = 0,
        *,
        heads: bool = False,
    ) -> Callable[[int], float]:
        """Factor for the zero-based training-loop microstep ``step``.

        Trunk (``heads=False``): zero through microstep ``frozen`` inclusive, then ``(step - frozen) / warmup``
        (the first non-zero factor is ``1 / warmup``), then one. Heads (``heads=True``): ``(step + 1) /
        (warmup + 1)`` for the first ``warmup`` microsteps (never zero), then one. ``decay_steps`` decays
        linearly from the end of the frozen phase; ``cooldown`` decays linearly to zero over the last
        ``cooldown`` of ``total_steps`` microsteps. Zero-length phases are skipped.
        """

        def f(step: int) -> float:
            if heads:
                s = step
                warm = min(1.0, (s + 1) / (warmup + 1)) if warmup > 0 else 1.0
            else:
                if (frozen > 0 or warmup > 0) and step <= frozen:
                    return 0.0
                s = step - frozen
                warm = min(1.0, s / warmup) if warmup > 0 else 1.0
            decay = max(0.0, 1.0 - s / decay_steps) if decay_steps > 0 else 1.0
            cool = 1.0
            if cooldown > 0 and total_steps > 0 and step >= total_steps - cooldown:
                cool = max(0.0, 1.0 - (step - (total_steps - cooldown)) / cooldown)
            return warm * decay * cool

        return f

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        trunk = self.lr_lambda(
            self.num_warmup_steps,
            self.decay_steps,
            frozen=self.freeze_steps,
            cooldown=self.cooldown_steps,
            total_steps=num_training_steps,
        )
        heads = self.lr_lambda(
            self.warmup_steps_heads,
            self.decay_steps,
            cooldown=self.cooldown_steps,
            total_steps=num_training_steps,
            heads=True,
        )
        n_groups = len(optimizer.param_groups)
        lambdas = [trunk] + [heads] * (n_groups - 1) if n_groups > 1 else trunk
        return LambdaLR(optimizer, lambdas, -1)
