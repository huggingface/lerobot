# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from tacorl.utils.lr_schedulers import register_scheduler, LearningRateSchedulerConfigs
from tacorl.utils.lr_schedulers.lr_scheduler import LearningRateScheduler


@dataclass
class WarmupLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="warmup", metadata={"help": "Name of learning rate scheduler."})
    peak_lr: float = field(default=1e-04, metadata={"help": "Maximum learning rate."})
    init_lr: float = field(default=1e-7, metadata={"help": "Initial learning rate."})
    warmup_steps: int = field(
        default=4000, metadata={"help": "Warmup the learning rate linearly for the first N updates"}
    )
    total_steps: int = field(default=200000, metadata={"help": "Total training steps."})


@register_scheduler("warmup", dataclass=WarmupLRSchedulerConfigs)
class WarmupLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, configs.lr_scheduler.init_lr)
        if configs.lr_scheduler.warmup_steps != 0:
            warmup_rate = configs.lr_scheduler.peak_lr - configs.lr_scheduler.init_lr
            self.warmup_rate = warmup_rate / configs.lr_scheduler.warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = configs.lr_scheduler.init_lr
        self.warmup_steps = configs.lr_scheduler.warmup_steps

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr