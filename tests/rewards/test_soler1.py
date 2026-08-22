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

"""End-to-end SOLE-R1 smoke test using the published checkpoint."""

import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config  # noqa: E402
from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel  # noqa: E402
from lerobot.rewards.soler1.processor_soler1 import (  # noqa: E402
    make_soler1_pre_post_processors,
)
from tests.utils import require_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Downloads the full SOLE-R1 checkpoint; not intended for CI",
)


@require_cuda
def test_soler1_transformers_end_to_end():
    config = SOLER1Config(
        device="cuda",
        external_image_key="observation.images.front",
        max_new_tokens=600,
    )

    preprocessor, _ = make_soler1_pre_post_processors(config)
    model = SOLER1RewardModel(config).to(config.device).eval()

    try:
        trajectory = torch.stack(
            [
                torch.zeros(64, 64, 3),
                torch.ones(64, 64, 3),
            ]
        )  # shape: (2, 64, 64, 3)

        batch = preprocessor(
            {
                config.external_image_key: trajectory,
                config.task_key: "pick up the cube",
            }
        )
        reward = model.compute_reward(batch)

        assert reward.shape == (1,)
        assert reward.dtype == torch.float32
        assert torch.isfinite(reward).all()
        assert -1.0 <= reward.item() <= 1.0
    finally:
        del model
        torch.cuda.empty_cache()
