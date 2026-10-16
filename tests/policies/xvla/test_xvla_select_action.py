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
N_ACTION_STEPS = 3
ACTION_DIM = 2


def test_xvla_select_action_computes_chunks_through_predict_action_chunk(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.xvla import modeling_xvla
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    monkeypatch.setattr(modeling_xvla, "XVLAModel", lambda **_: nn.Module())
    monkeypatch.setattr(XVLAConfig, "validate_features", lambda self: None)
    monkeypatch.setattr(XVLAConfig, "get_florence_config", lambda self: None)
    policy = modeling_xvla.XVLAPolicy(
        XVLAConfig(
            chunk_size=CHUNK,
            n_action_steps=N_ACTION_STEPS,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        )
    )
    chunks = []

    def get_action_chunk(batch):
        chunks.append(torch.randn(1, CHUNK, ACTION_DIM))
        return chunks[-1]

    policy._get_action_chunk = get_action_chunk
    # Wrapped the way the rollout installs torch.compile, so the real predict_action_chunk still runs.
    real_predict_action_chunk = policy.predict_action_chunk
    calls = []

    def recording_predict_action_chunk(batch, noise=None):
        calls.append((batch, noise))
        return real_predict_action_chunk(batch, noise=noise)

    policy.predict_action_chunk = recording_predict_action_chunk
    noise = torch.zeros(1, CHUNK, ACTION_DIM)
    batch = {OBS_STATE: torch.zeros(1, 2)}

    actions = [policy.select_action(batch, noise=noise) for _ in range(N_ACTION_STEPS + 1)]

    assert len(calls) == 2, f"select_action computed {len(calls)} chunks through predict_action_chunk, not 2"
    assert all(b is batch and n is noise for b, n in calls), "predict_action_chunk got another batch or noise"
    first, second = chunks
    torch.testing.assert_close(torch.cat(actions), torch.cat([first[0, :N_ACTION_STEPS], second[0, :1]]))
