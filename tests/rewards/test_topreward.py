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

"""End-to-end TOPReward smoke test with the real Qwen3-VL model."""

import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig  # noqa: E402
from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel  # noqa: E402
from lerobot.rewards.topreward.processor_topreward import (  # noqa: E402
    TOPREWARD_FEATURE_PREFIX,
    TOPREWARD_INPUT_KEYS,
    make_topreward_pre_post_processors,
)
from tests.utils import require_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires downloading and loading Qwen3-VL and is not meant for CI",
)


def _make_dummy_topreward_batch(image_key: str, task_key: str) -> dict[str, object]:
    num_frames = 4
    image_size = 64
    frames = torch.zeros(1, num_frames, 3, image_size, image_size, dtype=torch.uint8)
    for frame_idx in range(num_frames):
        frames[0, frame_idx, 0].fill_(min(frame_idx * 48, 255))
        frames[0, frame_idx, 1].fill_(96)
        frames[0, frame_idx, 2].fill_(192)

    return {
        image_key: frames,
        task_key: ["pick up the red cube"],
    }


@require_cuda
def test_topreward_full_qwen3vl_preprocessor_to_compute_reward():
    cfg = TOPRewardConfig(
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        max_frames=4,
        fps=2.0,
        max_input_length=4096,
    )

    preprocessor, _ = make_topreward_pre_post_processors(cfg)
    encoded_batch = preprocessor(_make_dummy_topreward_batch(cfg.image_key, cfg.task_key))
    for key in TOPREWARD_INPUT_KEYS:
        assert f"{TOPREWARD_FEATURE_PREFIX}{key}" in encoded_batch

    model = TOPRewardModel(cfg)
    try:
        model.to(cfg.device)
        model.eval()
        rewards = model.compute_reward(encoded_batch)
    finally:
        del model
        torch.cuda.empty_cache()

    assert rewards.shape == (1,)
    assert rewards.dtype == torch.float32
    assert torch.isfinite(rewards).all()
