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

from .ema import EMA, EMAConfig, load_ema_state, save_ema_state
from .factory import make_optimizer_and_scheduler
from .grad_scaler import (
    load_scaler_state,
    save_scaler_state,
)
from .optimizers import (
    AdamConfig as AdamConfig,
    AdamWConfig as AdamWConfig,
    MultiAdamConfig as MultiAdamConfig,
    OptimizerConfig as OptimizerConfig,
    OptimizerParams,
    SGDConfig as SGDConfig,
    XVLAAdamWConfig as XVLAAdamWConfig,
    load_optimizer_state,
    save_optimizer_state,
)
from .schedulers import (
    ConstantWithWarmupSchedulerConfig,
    CosineAnnealingWithWarmupSchedulerConfig,
    CosineDecayWithWarmupSchedulerConfig as CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig as DiffuserSchedulerConfig,
    FrozenWarmupConstantSchedulerConfig,
    LRSchedulerConfig as LRSchedulerConfig,
    VQBeTSchedulerConfig as VQBeTSchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)

__all__ = [
    # Optimizer configs
    "AdamConfig",
    "AdamWConfig",
    "MultiAdamConfig",
    "OptimizerConfig",
    "OptimizerParams",
    "SGDConfig",
    "XVLAAdamWConfig",
    # Scheduler configs
    "ConstantWithWarmupSchedulerConfig",
    "CosineAnnealingWithWarmupSchedulerConfig",
    "CosineDecayWithWarmupSchedulerConfig",
    "DiffuserSchedulerConfig",
    "FrozenWarmupConstantSchedulerConfig",
    "LRSchedulerConfig",
    "VQBeTSchedulerConfig",
    # EMA
    "EMA",
    "EMAConfig",
    # State management
    "load_ema_state",
    "load_optimizer_state",
    "load_scaler_state",
    "load_scheduler_state",
    "save_ema_state",
    "save_optimizer_state",
    "save_scaler_state",
    "save_scheduler_state",
    # Factory functions
    "make_optimizer_and_scheduler",
]
