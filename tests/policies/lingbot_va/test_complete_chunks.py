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
"""LingBot-VA served as complete chunks, reading back the keyframes they led to.

The model calls are stubbed: the contract under test is which observations and
executed actions reach the KV cache, and when.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference import SyncInferenceEngine
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.policies.lingbot_va.processor_lingbot_va import make_lingbot_va_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

CAMS = ["observation.images.image", "observation.images.image2"]


def stubbed_policy() -> tuple[LingBotVAPolicy, list, list]:
    config = LingBotVAConfig(
        device="cpu",
        action_per_frame=8,
        frame_chunk_size=3,
        input_features={
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)) for key in CAMS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = LingBotVAPolicy.__new__(LingBotVAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy._frozen = {}  # the frozen VAE and text encoder are never loaded here
    policy.reset()
    predicted: list = []
    fed_back: list = []

    def infer(init_latent, frame_st_id=0, *, noise=None, video_noise=None):
        shape = (1, config.action_dim, config.frame_chunk_size, config.action_per_frame, 1)
        actions = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) / 1000
        predicted.append(actions + len(predicted))
        return predicted[-1], torch.zeros(1)

    def compute_kv_cache(keyframes, executed_actions):
        fed_back.append(
            ([{key: frame[key].clone() for key in CAMS} for frame in keyframes], executed_actions)
        )

    policy._ensure_frozen_modules = lambda: None
    policy._maybe_init_prompt = lambda batch: None
    policy._encode_frames = lambda frames: torch.zeros(1)
    policy._init_streaming_cache = lambda latent: None
    policy._infer = infer
    policy._compute_kv_cache = compute_kv_cache
    return policy, predicted, fed_back


def frame(t: int) -> dict:
    return {
        **{key: torch.full((1, 3, 2, 2), float(t + 100 * i)) for i, key in enumerate(CAMS)},
        OBS_STATE: torch.zeros(1, 7),
        "task": ["stack"],
    }


def test_declares_complete_chunks_with_the_upstream_keyframe_pattern():
    policy, _, _ = stubbed_policy()
    spec = policy.chunk_inference_spec()
    assert spec.complete_execution
    assert spec.prediction_steps == spec.execution_steps == 24
    # The observation after executed actions 1, 3, 5, ... of a chunk, the last one current.
    for key in CAMS:
        assert list(policy.config.inference_deltas.of(key, FeatureType.VISUAL)) == list(range(-22, 1, 2))
    assert policy.config.inference_deltas.of(OBS_STATE, FeatureType.STATE) is None


def test_executed_chunks_feed_back_the_upstream_keyframes():
    """A shorter first chunk, then every observation after an odd sub-step feeds the KV cache."""
    policy, predicted, fed_back = stubbed_policy()
    pre, post = make_lingbot_va_pre_post_processors(policy.config)
    executor = SyncInferenceEngine(policy, pre, post, observation_keys=[*CAMS, OBS_STATE])
    executor.reset()
    played = []
    for t in range(16 + 24 + 1):  # a short first chunk, a full one, then the next call
        played.append(executor.step(frame(t), float(t)))
        if t == 0:
            assert executor.queued_actions == 15  # the conditioning frame's actions are dropped
        if t == 16:
            assert executor.queued_actions == 23

    assert len(predicted) == 3 and len(fed_back) == 2
    # Keyframes: the observation after executed actions 1, 3, 5, ... of each chunk.
    assert [[f[CAMS[0]].flatten()[0].item() for f in frames] for frames, _ in fed_back] == [
        [float(t) for t in range(2, 17, 2)],
        [float(t) for t in range(18, 41, 2)],
    ]
    for frames, _ in fed_back:
        assert all(torch.equal(f[CAMS[1]], f[CAMS[0]] + 100) for f in frames)
    # Each call feeds back the whole previous chunk the model predicted, conditioning frame included.
    for (_, executed), chunk in zip(fed_back, predicted, strict=False):
        assert executed is chunk
    # The first chunk starts at frame 1 of its prediction, the next at frame 0.
    used = policy.config.used_action_channel_ids
    torch.testing.assert_close(played[0][0], predicted[0][0, used, 1, 0, 0])
    torch.testing.assert_close(played[16][0], predicted[1][0, used, 0, 0, 0])
