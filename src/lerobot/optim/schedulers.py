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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict
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


@LRSchedulerConfig.register_subclass("TriStage")
@dataclass
class TriStageLRSchedulerConfig(LRSchedulerConfig):
    configs: Dict[str, Any] = field(default_factory=dict)
    num_warmup_steps: int | None = None
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        configs = OmegaConf.create(self.configs)
        pr = configs.lr_scheduler.phase_ratio
        if isinstance(pr, str):
            phase_ratio = eval(pr)
        else:
            phase_ratio = pr
        total_configured_steps = configs.lr_scheduler.total_steps
        actual_warmup_steps = int(total_configured_steps * phase_ratio[0])
        actual_hold_steps = int(total_configured_steps * phase_ratio[1])
        actual_decay_steps = int(total_configured_steps * phase_ratio[2])
        # peak_lr = configs.lr_scheduler.lr
        init_lr_scale = configs.lr_scheduler.init_lr_scale 
        final_lr_scale = configs.lr_scheduler.final_lr_scale

        if num_training_steps < total_configured_steps:
            scale_factor = num_training_steps / total_configured_steps
            actual_warmup_steps = int(actual_warmup_steps * scale_factor)
            actual_hold_steps = int(actual_hold_steps * scale_factor)
            actual_decay_steps = num_training_steps - actual_warmup_steps - actual_hold_steps
            
            logging.info(
                f"Auto-scaling TriStage LR: {total_configured_steps} -> {num_training_steps} steps. "
                f"Scale factor: {scale_factor:.3f}"
            )

        def lr_lambda(current_step: int):
            # 1. Warmup Stage
            if current_step < actual_warmup_steps:
                if actual_warmup_steps == 0: return 1.0
                # Linearly increase from init_lr_scale to 1.0
                pct = current_step / actual_warmup_steps
                return init_lr_scale + (1.0 - init_lr_scale) * pct

            # 2. Hold Stage
            offset = actual_warmup_steps
            if current_step < offset + actual_hold_steps:
                return 1.0

            # 3. Decay Stage (Cosine)
            offset += actual_hold_steps
            if current_step < offset + actual_decay_steps:
                if actual_decay_steps == 0: return final_lr_scale
                steps_in_decay = current_step - offset
                # Decay from 1.0 to final_lr_scale
                cosine_decay = 0.5 * (1 + math.cos(math.pi * steps_in_decay / actual_decay_steps))
                return final_lr_scale + (1.0 - final_lr_scale) * cosine_decay

            # 4. Final Stage
            return final_lr_scale

        # LambdaLR multiplies the base_lr set in the optimizer by the return value of lr_lambda
        # Therefore, when defining the optimizer, lr should be set to peak_lr
        return LambdaLR(optimizer, lr_lambda, -1)

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
    
    def build(self, optimizers: Dict[str, Optimizer], num_training_steps: int) -> Dict[str, LambdaLR]:
        schedulers = {}
        
        for name, optimizer in optimizers.items():
            # 1. Priority: group_config > self (global default)
            group_config = self.scheduler_groups.get(name, {})
            
            # parse parameters
            actual_total_steps = num_training_steps or group_config.get("total_steps", self.total_steps)
            p_ratio = group_config.get("phase_ratio", self.phase_ratio)
            i_scale = group_config.get("init_lr_scale", self.init_lr_scale)
            f_scale = group_config.get("final_lr_scale", self.final_lr_scale)

            # 2. parse phase_ratio
            if isinstance(p_ratio, str):
                p_ratio = eval(p_ratio)

            # 3. Calculate the number of steps for each phase
            w_steps = int(actual_total_steps * p_ratio[0])
            h_steps = int(actual_total_steps * p_ratio[1])
            d_steps = int(actual_total_steps * p_ratio[2])

            # 4. Define closure function (Lambda logic)
            def get_lr_lambda(current_step: int, ws=w_steps, hs=h_steps, ds=d_steps, iscl=i_scale, fscl=f_scale):
                # Warmup
                if current_step < ws:
                    if ws <= 0: return 1.0
                    return iscl + (1.0 - iscl) * (current_step / ws)
                
                # Hold
                step = current_step - ws
                if step < hs:
                    return 1.0
                
                # Decay (Cosine)
                step = step - hs
                if step < ds:
                    if ds <= 0: return fscl
                    progress = step / ds
                    return fscl + 0.5 * (1.0 - fscl) * (1.0 + math.cos(progress * math.pi))
                
                # Final
                return fscl

            # 5. Create and store the scheduler
            # Note: The -1 here is the default value for last_epoch
            schedulers[name] = LambdaLR(optimizer, get_lr_lambda, -1)
            
            logging.info(f"Created MultiTriStage Scheduler for [{name}]: steps={actual_total_steps}, ratios={p_ratio}")

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
