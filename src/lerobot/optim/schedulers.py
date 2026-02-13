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

from accelerate import optimizer
import draccus
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.datasets.utils import write_json
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
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
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


from lerobot.optim.flower.tri_stage_scheduler import TriStageLRScheduler, TriStageLRSchedulerPt
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
@LRSchedulerConfig.register_subclass("TriStage")
@dataclass
class TriStageLRSchedulerConfig(LRSchedulerConfig):
    configs: Dict[str, Any] = field(default_factory=dict)
    num_warmup_steps: int | None = None
    
    def build(self, optimizer, num_training_steps):
        configs = OmegaConf.create(self.configs)
        scheduler = TriStageLRScheduler(
            optimizer,
            configs
        )
        return scheduler

@LRSchedulerConfig.register_subclass("MultiTriStagePt")
@dataclass
class MultiTriStageLRSchedulerPtConfig(LRSchedulerConfig):
    # configs: Dict[str, Any] = field(default_factory=dict)
    num_warmup_steps: int | None = None
    
    init_lr_scale=0.1
    final_lr_scale=0.5
    total_steps=50000  
    phase_ratio="(0.05, 0.1, 0.85)"

    scheduler_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def build(self, optimizers, num_training_steps):
        schedulers = {} 
        for name, optimizer in optimizers.items():
            # Get group-specific hyperparameters or use defaults
            group_config = self.scheduler_groups.get(name, {})

            # Create scheduler with merged parameters (defaults + group-specific)
 
            scheduler_kwargs = {
                "total_steps": num_training_steps or group_config.get("total_steps", self.total_steps),
                "phase_ratio": group_config.get("phase_ratio", self.phase_ratio),
                "init_lr_scale": group_config.get("init_lr_scale", self.init_lr_scale),
                "final_lr_scale": group_config.get("final_lr_scale", self.final_lr_scale),
            }

            schedulers[name] = TriStageLRSchedulerPt(optimizer, **scheduler_kwargs)
        return schedulers
 
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

def save_scheduler_state(
    scheduler: LRScheduler | dict[str, LRScheduler], save_dir: Path
) -> None:
    """Save scheduler state to disk.

    Args:
        scheduler: Either a single scheduler or a dictionary of schedulers.
        save_dir: Directory to save the scheduler state.
    """
    if isinstance(scheduler, dict):
        # Handle dictionary of schedulers
        for name, sched in scheduler.items():
            scheduler_dir = save_dir / name
            scheduler_dir.mkdir(exist_ok=True, parents=True)
            save_single_scheduler_state(sched, scheduler_dir)
    else:
        # Handle single scheduler
        save_single_scheduler_state(scheduler, save_dir)

def save_single_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)

def load_scheduler_state(scheduler: LRScheduler | dict[str, LRScheduler], save_dir: Path) -> LRScheduler | dict[str, LRScheduler]:
    if isinstance(scheduler, dict):
        # Handle dictionary of schedulers
        loaded_schedulers = {}
        for name, sched in scheduler.items():
            scheduler_dir = save_dir / name
            if scheduler_dir.exists():
                loaded_schedulers[name] = load_single_scheduler_state(sched, scheduler_dir)
            else:
                loaded_schedulers[name] = sched
        return loaded_schedulers
    else:
        # Handle single scheduler 
        return load_single_scheduler_state(scheduler, save_dir)

def load_single_scheduler_state(scheduler: LRScheduler | dict[str, LRScheduler], save_dir: Path) -> LRScheduler | dict[str, LRScheduler]:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
