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

import logging
from unittest.mock import MagicMock

import pytest

from lerobot.common.wandb_utils import WandBLogger


@pytest.fixture
def logger():
    instance = WandBLogger.__new__(WandBLogger)
    instance._wandb = MagicMock()
    instance._wandb_custom_step_key = None
    return instance


def test_log_dict_expands_list_of_floats(logger):
    logger.log_dict({"loss_per_dim": [0.1, 0.2, 0.3]}, step=0, mode="train")

    calls = logger._wandb.log.call_args_list
    logged = {next(iter(c.kwargs["data"])): next(iter(c.kwargs["data"].values())) for c in calls}
    assert logged == {
        "train/loss_per_dim_0": 0.1,
        "train/loss_per_dim_1": 0.2,
        "train/loss_per_dim_2": 0.3,
    }


def test_log_dict_unsupported_value_still_warned(logger, caplog):
    with caplog.at_level(logging.WARNING):
        logger.log_dict({"weird": [0.1, "string", 0.3]}, step=0, mode="train")
    assert logger._wandb.log.call_count == 0
    assert any("was ignored" in record.message for record in caplog.records)
