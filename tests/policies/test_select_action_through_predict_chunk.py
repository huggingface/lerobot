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

"""X-VLA's select_action must compute a fresh chunk through predict_action_chunk.

lerobot-rollout's --use_torch_compile wraps predict_action_chunk, and X-VLA has no compile of its own,
so a select_action that computes its chunk some other way runs uncompiled.
"""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

CHUNK = 4
ACTION_DIM = 3


class TinyModel(nn.Module):
    """Stands in for X-VLA's network, so the test needs no pretrained backbone."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def test_xvla_select_action_computes_chunks_through_predict_action_chunk(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.xvla import modeling_xvla
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    monkeypatch.setattr(modeling_xvla, "XVLAModel", TinyModel)
    monkeypatch.setattr(XVLAConfig, "validate_features", lambda self: None)
    monkeypatch.setattr(XVLAConfig, "get_florence_config", lambda self: None)
    policy = modeling_xvla.XVLAPolicy(
        XVLAConfig(
            chunk_size=CHUNK,
            n_action_steps=CHUNK,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        )
    )
    calls = []

    def predict_action_chunk(batch, noise=None):
        calls.append(noise)
        return torch.arange(CHUNK * ACTION_DIM, dtype=torch.float32).reshape(1, CHUNK, ACTION_DIM)

    policy.predict_action_chunk = predict_action_chunk
    noise = torch.zeros(1, CHUNK, ACTION_DIM)
    batch = {OBS_STATE: torch.zeros(1, 2)}

    first = policy.select_action(batch, noise=noise)
    for _ in range(CHUNK):
        policy.select_action(batch, noise=noise)

    assert len(calls) == 2, "select_action did not compute its chunks through predict_action_chunk"
    assert calls[0] is noise, "select_action did not pass its noise to predict_action_chunk"
    torch.testing.assert_close(first, torch.arange(ACTION_DIM, dtype=torch.float32).reshape(1, ACTION_DIM))
