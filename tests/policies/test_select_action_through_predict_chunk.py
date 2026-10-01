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

"""select_action must compute a fresh chunk through predict_action_chunk.

lerobot-rollout's --use_torch_compile wraps predict_action_chunk, so a policy whose select_action
computes its chunk some other way silently runs uncompiled.
"""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

CHUNK = 4
ACTION_DIM = 3


class TinyModel(nn.Module):
    """Stands in for the policy's network, so the test needs no pretrained backbone."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _chunk_through_predict(policy):
    """Counts predict_action_chunk calls; each returns a chunk of distinct actions."""
    calls = []

    def predict_action_chunk(batch, noise=None, **kwargs):
        calls.append(noise)
        return torch.arange(CHUNK * ACTION_DIM, dtype=torch.float32).reshape(1, CHUNK, ACTION_DIM)

    policy.predict_action_chunk = predict_action_chunk
    return calls


def _check(policy):
    calls = _chunk_through_predict(policy)
    noise = torch.zeros(1, CHUNK, ACTION_DIM)
    batch = {OBS_STATE: torch.zeros(1, 2)}

    first = policy.select_action(batch, noise=noise)
    for _ in range(CHUNK - 1):
        policy.select_action(batch, noise=noise)
    policy.select_action(batch, noise=noise)

    assert len(calls) == 2, "select_action did not compute its chunks through predict_action_chunk"
    assert calls[0] is noise, "select_action did not pass its noise to predict_action_chunk"
    torch.testing.assert_close(first, torch.arange(ACTION_DIM, dtype=torch.float32).reshape(1, ACTION_DIM))


@pytest.mark.parametrize("policy_type", ["smolvla", "xvla"])
def test_select_action_computes_chunks_through_predict_action_chunk(policy_type, monkeypatch):
    pytest.importorskip("transformers")
    features = {
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    }
    if policy_type == "smolvla":
        from lerobot.policies.smolvla import modeling_smolvla
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        monkeypatch.setattr(modeling_smolvla, "VLAFlowMatching", TinyModel)
        monkeypatch.setattr(SmolVLAConfig, "validate_features", lambda self: None)
        policy = modeling_smolvla.SmolVLAPolicy(
            SmolVLAConfig(chunk_size=CHUNK, n_action_steps=CHUNK, **features)
        )
    else:
        from lerobot.policies.xvla import modeling_xvla
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig

        monkeypatch.setattr(modeling_xvla, "XVLAModel", TinyModel)
        monkeypatch.setattr(XVLAConfig, "validate_features", lambda self: None)
        monkeypatch.setattr(XVLAConfig, "get_florence_config", lambda self: None)
        policy = modeling_xvla.XVLAPolicy(XVLAConfig(chunk_size=CHUNK, n_action_steps=CHUNK, **features))

    _check(policy)
