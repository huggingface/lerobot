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
import pytest
import torch

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import VQBeTSchedulerConfig


@pytest.fixture
def model_params():
    return [torch.nn.Parameter(torch.randn(10, 10))]


@pytest.fixture
def optimizer(model_params):
    optimizer = AdamConfig().build(model_params)
    # Dummy step to populate state
    loss = sum(param.sum() for param in model_params)
    loss.backward()
    optimizer.step()
    return optimizer


@pytest.fixture
def scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    return config.build(optimizer, num_training_steps=100)
