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

from lerobot.common.wandb_utils import WandBLogger


class _FakeWandB:
    def __init__(self):
        self.logged = []

    def log(self, data=None, step=None):
        self.logged.append((data, step))


def test_log_dict_expands_list_metrics_with_step():
    wandb = _FakeWandB()
    logger = object.__new__(WandBLogger)
    logger._wandb = wandb
    logger._wandb_custom_step_key = None

    logger.log_dict({"loss_per_dim": [1.0, 2.5], "loss": 3.0}, step=7)

    assert wandb.logged == [
        ({"train/loss_per_dim_0": 1.0}, 7),
        ({"train/loss_per_dim_1": 2.5}, 7),
        ({"train/loss": 3.0}, 7),
    ]


def test_log_dict_expands_list_metrics_with_custom_step():
    wandb = _FakeWandB()
    logger = object.__new__(WandBLogger)
    logger._wandb = wandb
    logger._wandb_custom_step_key = {"train/env_step"}

    logger.log_dict({"env_step": 42, "loss_per_dim": (1.0, 2.5)}, custom_step_key="env_step")

    assert wandb.logged == [
        ({"train/env_step": 42}, None),
        ({"train/loss_per_dim_0": 1.0, "train/env_step": 42}, None),
        ({"train/loss_per_dim_1": 2.5, "train/env_step": 42}, None),
    ]
