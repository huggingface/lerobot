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
"""Fine-tuning from a pretrained checkpoint silently replaces the checkpoint's processor
stats with the fine-tune dataset's stats (`lerobot_train.py`, the `not cfg.resume` branch of
the processor-overrides block). That contract is deliberate (see #4006 for the resume rule),
but it is the number-one suspect when a fine-tuned policy's actions collapse toward the
dataset mean, so the replacement must at least be loud. These pins keep it loud.
"""

import logging

import pytest

# Importing the training script pulls in the dataset stack; the fast-test tiers that do not
# install `lerobot[dataset]` skip this module rather than fail collection.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import _warn_pretrained_stats_replaced  # noqa: E402


def test_finetune_stats_replacement_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_pretrained_stats_replaced("lerobot/smolvla_base")

    assert any(
        "lerobot/smolvla_base" in rec.message
        and "overridden with the fine-tune dataset's stats" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]


def test_no_pretrained_path_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_pretrained_stats_replaced(None)

    assert caplog.records == []
