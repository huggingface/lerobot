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

"""Opt-in end-to-end smoke test for the released RynnValue-4B checkpoint."""

import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.rewards.rynnvalue import RynnValueConfig, RynnValueRewardModel  # noqa: E402
from lerobot.rewards.rynnvalue.processor_rynnvalue import (  # noqa: E402
    make_rynnvalue_pre_post_processors,
)
from tests.utils import require_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_RYNNVALUE_SMOKE") != "1",
    reason="Set LEROBOT_RUN_RYNNVALUE_SMOKE=1 to download and test RynnValue-4B",
)


@require_cuda
def test_released_rynnvalue_4b_preprocessor_to_remaining_time():
    config = RynnValueConfig(
        device="cuda",
        max_frames=2,
        robot_description="a robot arm",
        camera_description="a third-person camera",
    )
    preprocessor, _ = make_rynnvalue_pre_post_processors(config)
    batch = {
        config.image_key: torch.zeros(2, 3, 64, 64, dtype=torch.uint8),
        config.task_key: "pick up the cube",
    }
    encoded = preprocessor(batch)
    input_length = encoded["observation.rynnvalue.input_ids"].shape[-1]
    assert encoded["observation.rynnvalue.attention_mask"].shape[-1] == input_length
    assert encoded["observation.rynnvalue.mm_token_type_ids"].shape[-1] == input_length
    model = RynnValueRewardModel(config).to("cuda").eval()
    prediction = model.predict_remaining_time(encoded)
    assert prediction.remaining_time_s.shape == (1,)
    assert prediction.remaining_time_s.dtype == torch.float32
    assert torch.isfinite(prediction.remaining_time_s).all()
