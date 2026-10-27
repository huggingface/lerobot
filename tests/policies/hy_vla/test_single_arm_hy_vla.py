# Copyright 2026 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.processor_hy_vla import (
    HyVLADecodeStep,
    HyVLAEncodeStep,
    make_hy_vla_pre_post_processors,
    make_hy_vla_pre_post_processors_from_pretrained,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    single_arm_relative_actions,
)


def _config(**kwargs):
    return HyVLAConfig(
        embodiment="single_arm",
        model_action_dim=11,
        relative_convention="current",
        chunk_size=3,
        n_action_steps=3,
        recipe=None,
        device="cpu",
        **kwargs,
    )


def _stats():
    return {
        "qpos_mean": torch.tensor([0.4, 0.3]),
        "qpos_std": torch.tensor([0.2, 0.1]),
        "action_mean": torch.arange(33).reshape(3, 11).float() / 100,
        "action_std": torch.arange(33).reshape(3, 11).float() / 100 + 0.5,
    }


def _poses():
    reference = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    reference[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    reference[:, :3, 3] = torch.tensor([[0.4, -0.2, 0.1], [-0.3, 0.5, 0.8]])
    delta = torch.eye(4).repeat(2, 3, 1, 1)
    delta[:, :, 0, 3] = torch.tensor([0.0, 0.01, 0.02])
    delta[:, 1:, :3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    absolute = reference[:, None] @ delta
    gripper = torch.tensor([[[0.1, 0.2], [0.3, 1.0], [0.5, 1.0]]]).expand(2, -1, -1)
    action = torch.cat((absolute[..., :3, 3], rotation_matrix_to_6d(absolute[..., :3, :3]), gripper), -1)
    return reference, delta, action


def test_single_arm_features_and_required_stats():
    config = _config()
    assert config.input_features["observation.state"].shape == (2,)
    assert config.output_features["action"].shape == (11,)
    with pytest.raises(ValueError, match="action_relative"):
        make_hy_vla_pre_post_processors(config)


def test_relative_pose_uses_current_frame_and_preserves_two_gripper_targets():
    reference, expected, action = _poses()
    result = single_arm_relative_actions(action, reference)
    torch.testing.assert_close(result[..., :3], expected[..., :3, 3])
    torch.testing.assert_close(rotation_6d_to_matrix(result[..., 3:9]), expected[..., :3, :3])
    torch.testing.assert_close(result[..., 9:], action[..., 9:])


def test_single_arm_mask_and_fixed_planning_reference():
    reference, _, action = _poses()
    encoder = HyVLAEncodeStep(
        embodiment="single_arm", action_validity_key="observation.target_valid", **_stats()
    )
    decoder = HyVLADecodeStep(embodiment="single_arm", execution_horizon=3, encoder=encoder, **_stats())
    transition = encoder(
        {
            TransitionKey.OBSERVATION: {
                "observation.state": torch.tensor([[0.4, 0.3], [0.6, 0.5]]),
                "observation.gripper_pose": reference[:, None],
                "observation.target_valid": torch.tensor([[True, True, False], [True, True, True]]),
            },
            TransitionKey.ACTION: action,
            TransitionKey.COMPLEMENTARY_DATA: {
                "action_is_pad": torch.tensor([[False, True, False], [False, False, False]])
            },
        }
    )
    mask = transition[TransitionKey.COMPLEMENTARY_DATA]["action.mask"]
    assert not mask[0, 1:].any() and not mask[..., 11:].any()
    assert mask[1, :, :11].all()
    torch.testing.assert_close(
        transition[TransitionKey.OBSERVATION]["observation.state"][0, :2], torch.zeros(2)
    )
    for index in range(3):
        decoded = decoder({TransitionKey.ACTION: transition[TransitionKey.ACTION][:, index]})[
            TransitionKey.ACTION
        ]
        torch.testing.assert_close(decoded[1], action[1, index], atol=1e-6, rtol=1e-6)
        # New measured poses must not change a chunk already in execution.
        encoder.latest_native_reference = reference + 100
    encoder.latest_native_reference = reference
    decoded = decoder({TransitionKey.ACTION: transition[TransitionKey.ACTION][:, 0]})[TransitionKey.ACTION]
    torch.testing.assert_close(decoded, action[:, 0], atol=1e-6, rtol=1e-6)


def test_single_arm_pipeline_reload_reconnects_pose_and_stats(tmp_path: Path):
    config = _config()
    pre, post = make_hy_vla_pre_post_processors(config, norm_stats=_stats())
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    loaded_pre, loaded_post = make_hy_vla_pre_post_processors_from_pretrained(config, tmp_path)
    encoder = next(step for step in loaded_pre.steps if isinstance(step, HyVLAEncodeStep))
    decoder = next(step for step in loaded_post.steps if isinstance(step, HyVLADecodeStep))
    assert decoder.encoder is encoder and encoder.embodiment == "single_arm"
    torch.testing.assert_close(encoder.qpos_mean, _stats()["qpos_mean"])
    reference, _, action = _poses()
    encoded = encoder(
        {
            TransitionKey.OBSERVATION: {
                "observation.state": torch.zeros(2, 2),
                "observation.gripper_pose": reference,
            },
            TransitionKey.ACTION: action,
        }
    )
    decoded = loaded_post(encoded[TransitionKey.ACTION][:, 0])
    torch.testing.assert_close(decoded, action[:, 0], atol=1e-6, rtol=1e-6)
