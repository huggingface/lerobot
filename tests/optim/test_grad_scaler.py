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
"""fp16 loss-scaler state persistence."""

import json

import pytest
import torch
from torch.amp import GradScaler

from lerobot.optim import load_scaler_state, save_scaler_state
from lerobot.utils.constants import SCALER_STATE


def _calibrated_scaler() -> GradScaler:
    """A scaler whose state differs from every constructor default, so a silent
    "restored the defaults" pass cannot be mistaken for a real round trip."""
    scaler = GradScaler("cpu", init_scale=1024.0, growth_factor=4.0, backoff_factor=0.25)
    state = scaler.state_dict()
    state["_growth_tracker"] = 7
    scaler.load_state_dict(state)
    return scaler


def test_round_trip_restores_every_field(tmp_path):
    saved = _calibrated_scaler()
    save_scaler_state(saved, tmp_path)

    restored = GradScaler("cpu")  # stock defaults: nothing here matches the saved state
    load_scaler_state(restored, tmp_path)

    assert restored.state_dict() == saved.state_dict()


def test_state_is_plain_readable_json(tmp_path):
    save_scaler_state(_calibrated_scaler(), tmp_path)

    payload = json.loads((tmp_path / SCALER_STATE).read_text())
    assert payload == {
        "scale": 1024.0,
        "growth_factor": 4.0,
        "backoff_factor": 0.25,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }


def test_missing_state_warns_and_keeps_the_configured_scale(tmp_path, caplog):
    scaler = GradScaler("cpu", init_scale=512.0)

    with caplog.at_level("WARNING"):
        assert load_scaler_state(scaler, tmp_path) is scaler

    # A checkpoint from before fp16 support, or from a bf16 run, must not fail the resume.
    assert SCALER_STATE in caplog.text
    assert scaler.get_scale() == 512.0


def test_load_does_not_materialize_the_scale(tmp_path):
    """The restore must leave the scaler lazy.

    `_scale`/`_growth_tracker` are created by the first `scaler.scale(loss)` inside
    `accelerator.backward()`, on the loss's own device. Forcing them here would both pick the
    device by guesswork and re-open the door for anything running before that first backward
    to advance the scale that was just restored.
    """
    save_scaler_state(_calibrated_scaler(), tmp_path)

    restored = GradScaler("cpu")
    load_scaler_state(restored, tmp_path)

    assert restored._scale is None
    assert restored._growth_tracker is None
    # The values are still live: get_scale() reads the init fields while lazy, and the first
    # scale() call materializes tensors carrying them.
    assert restored.get_scale() == 1024.0
    restored.scale(torch.zeros(()))
    assert restored.state_dict()["scale"] == 1024.0
    assert restored.state_dict()["_growth_tracker"] == 7


def test_corrupt_state_fails_loudly(tmp_path):
    (tmp_path / SCALER_STATE).write_text(json.dumps({"scale": 1024.0}))

    with pytest.raises((ValueError, TypeError)):
        load_scaler_state(GradScaler("cpu"), tmp_path)
