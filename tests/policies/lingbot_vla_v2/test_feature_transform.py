# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""End-to-end test for the LingBot-VLA 2.0 feature transform (robot-config slot
mapping + Qwen3-VL image/language processing).

Requires a local Qwen3-VL processor (config + tokenizer + image processor, not the
weights). Point ``LINGBOT_VLA_V2_QWEN3VL`` at it, or place it at the default path
below. Skipped otherwise (mirrors the weight-guarded tests of the v1 policy).
"""

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

DEFAULT_QWEN3VL = os.path.expanduser("~/lingbot/Qwen3-VL-4B-Instruct-proc")
QWEN3VL_PATH = os.environ.get("LINGBOT_VLA_V2_QWEN3VL", DEFAULT_QWEN3VL)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(QWEN3VL_PATH),
    reason=f"Qwen3-VL processor not found at {QWEN3VL_PATH}; set LINGBOT_VLA_V2_QWEN3VL.",
)

SO101_ROBOT_CONFIG = """
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state:
            start: 0
            end: 6
actions:
  - action.arm.position:
      origin_keys:
        - action:
            start: 0
            end: 6
      subtract_state: False
images:
  - observation.images.camera_top:
      origin_keys: observation.images.front
norm_stats: {norm_stats_path}
"""


@pytest.fixture
def so101_feature_transform(tmp_path):
    from transformers import AutoProcessor

    from lerobot.policies.lingbot_vla_v2.feature_transform import FeatureTransform
    from lerobot.policies.lingbot_vla_v2.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    norm_stats_path = tmp_path / "norm_stats.json"
    norm_stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "observation.state.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                    "action.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                }
            }
        )
    )
    robot_config_path = tmp_path / "so101_robot.yaml"
    robot_config_path.write_text(SO101_ROBOT_CONFIG.format(norm_stats_path=norm_stats_path))

    apply_lingbot_qwen3_vl_patch()
    processor = AutoProcessor.from_pretrained(QWEN3VL_PATH, padding_side="right", trust_remote_code=True)
    data_config = SimpleNamespace(
        joints=["{'arm.position': 6}"],
        norm_type=["{'arm.position': 'meanstd'}"],
        cameras=["camera_top"],
        img_size=224,
        chat_template="default",
        text_keys="task",
    )
    model_config = SimpleNamespace(
        max_state_dim=55,
        max_action_dim=55,
        chunk_size=50,
        tokenizer_max_length=72,
        use_qwen3_chat_template=True,
        return_image_grid_thw=True,
        qwen3vl_use_vision_boundaries=True,
        resize_imgs_with_padding=(224, 224),
    )
    return FeatureTransform(
        robot_config_path=str(robot_config_path),
        data_config=data_config,
        model_config=model_config,
        processor=processor,
        chunk_size=50,
    )


def _so101_item():
    return {
        "observation.state": torch.randn(6),
        "action": torch.randn(50, 6),
        "observation.images.front": torch.randint(0, 255, (3, 480, 640)).float(),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "task": "pick up the red cube",
    }


def test_feature_transform_apply_produces_model_ready_tensors(so101_feature_transform):
    """apply() maps a raw SO101 item into the padded, Qwen3-VL-ready model inputs."""
    ft = so101_feature_transform
    assert ft.states == ["observation.state.arm.position"]
    assert ft.actions == ["action.arm.position"]
    assert ft.images == ["observation.images.camera_top"]

    out = ft.apply(_so101_item(), policy_eval=False)

    # Native-resolution Qwen3-VL image tokens + grid.
    assert out["images"].ndim == 3  # (n_views, num_patches, patch_dim)
    assert out["image_grid_thw"].shape[-1] == 3
    assert out["img_masks"].dtype == torch.bool
    # State / action padded to the 55-D canonical slots.
    assert out["state"].shape[-1] == 55
    assert out["actions"].shape == (50, 55)
    # Joint masks mark the 6 real arm dims valid, the rest padding.
    assert out["state_joint_mask"].shape[-1] == 55
    assert out["action_joint_mask"].shape[-1] == 55
    assert int(out["state_joint_mask"].sum()) == 6
    assert int(out["action_joint_mask"].sum()) == 6
    # Language.
    assert out["lang_tokens"].shape[-1] == 72
    assert out["lang_masks"].shape[-1] == 72


def test_feature_transform_roundtrip(so101_feature_transform):
    """unapply() inverts the padding/normalization back to the real action dim."""
    ft = so101_feature_transform
    out = ft.apply(_so101_item(), policy_eval=False)
    # Emulate a model action chunk in the padded canonical space.
    out["actions"] = torch.randn(50, 55)
    recovered = ft.unapply(dict(out))
    # unapply maps the canonical slots back to the original dataset keys.
    assert "action" in recovered
    assert recovered["action"].shape == (50, 6)
