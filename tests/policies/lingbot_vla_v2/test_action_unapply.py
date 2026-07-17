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

"""Processor-free tests for the LingBot-VLA 2.0 inference action de-normalization.

These exercise the ``unapply`` path used by ``LingbotVLAV2Policy._postprocess_actions``
(which builds a ``processor=None`` FeatureTransform), so they run on plain CPU without a
Qwen3-VL processor or checkpoint.
"""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

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


def _unapply_transform(tmp_path, mean, std):
    from lerobot.policies.lingbot_vla_v2.feature_transform import FeatureTransform

    norm_stats_path = tmp_path / "norm_stats.json"
    norm_stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "observation.state.arm.position": {"mean": mean, "std": std},
                    "action.arm.position": {"mean": mean, "std": std},
                }
            }
        )
    )
    robot_config_path = tmp_path / "so101_robot.yaml"
    robot_config_path.write_text(SO101_ROBOT_CONFIG.format(norm_stats_path=norm_stats_path))

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
    # processor=None -> unapply-only transform (no Qwen3-VL processor needed).
    return FeatureTransform(
        robot_config_path=str(robot_config_path),
        data_config=data_config,
        model_config=model_config,
        processor=None,
        chunk_size=50,
        norm_stats_path=str(norm_stats_path),
    )


def test_action_unapply_denormalizes(tmp_path):
    """unapply inverts meanstd normalization + the canonical slot mapping on actions."""
    mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    std = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ft = _unapply_transform(tmp_path, mean, std)

    mean_t = torch.tensor(mean)
    std_t = torch.tensor(std)
    raw = torch.arange(18, dtype=torch.float32).reshape(3, 6)  # known raw action chunk
    normalized = (raw - mean_t) / (std_t + 1e-6)
    # 55-D canonical chunk: the 6 real joints occupy the first slots, the rest is padding.
    chunk = torch.zeros(3, 55)
    chunk[:, :6] = normalized

    action_joint_mask = torch.zeros(55, dtype=torch.bool)
    action_joint_mask[:6] = True
    state_joint_mask = torch.zeros(55, dtype=torch.bool)
    state_joint_mask[:6] = True
    state = torch.zeros(55)  # non-subtract_state -> state does not affect the action

    recovered = ft.unapply(
        {
            "actions": chunk,
            "action_joint_mask": action_joint_mask,
            "state": state,
            "state_joint_mask": state_joint_mask,
        }
    )
    assert "action" in recovered
    assert recovered["action"].shape == (3, 6)
    torch.testing.assert_close(recovered["action"], raw, atol=1e-4, rtol=1e-4)
