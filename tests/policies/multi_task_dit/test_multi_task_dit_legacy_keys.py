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

"""Checkpoints saved with the pre-flattening CLIP key layout still load every weight."""

import re

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import OBS_STATE

# Current key prefix -> the level a checkpoint saved with the pre-flattening CLIP layout carries after it.
LEGACY = (
    (re.compile(r"^(observation_encoder\.text_encoder\.text_encoder\.)"), r"\1text_model."),
    (
        re.compile(r"^(observation_encoder\.(?:vision_encoder|vision_encoders\.\d+)\.model\.)"),
        r"\1vision_model.",
    ),
)
CAMERAS = ("observation.images.front", "observation.images.wrist")


@pytest.fixture(autouse=True)
def tiny_clip(monkeypatch):
    monkeypatch.setattr(
        CLIPVisionModel,
        "from_pretrained",
        lambda *a, **kw: CLIPVisionModel(
            CLIPVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=16,
                patch_size=8,
            )
        ),
    )
    monkeypatch.setattr(
        CLIPTextModel,
        "from_pretrained",
        lambda *a, **kw: CLIPTextModel(
            CLIPTextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
                bos_token_id=0,
                eos_token_id=2,
                pad_token_id=1,
            )
        ),
    )


def make_policy(seed: int, separate: bool) -> MultiTaskDiTPolicy:
    torch.manual_seed(seed)
    return MultiTaskDiTPolicy(
        MultiTaskDiTConfig(
            use_separate_rgb_encoder_per_camera=separate,
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
            image_resize_shape=(16, 16),
            image_crop_shape=(16, 16),
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
                **{camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for camera in CAMERAS},
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
            device="cpu",
        )
    )


def legacy_state_dict(policy: MultiTaskDiTPolicy) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in policy.state_dict().items():
        for pattern, legacy in LEGACY:
            key, count = pattern.subn(legacy, key, count=1)
            if count:
                break
        state[key] = value.detach().contiguous().clone()
    return state


def assert_same_weights(actual: MultiTaskDiTPolicy, expected: MultiTaskDiTPolicy) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        torch.testing.assert_close(actual_state[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("separate", [False, True], ids=["shared-encoder", "encoder-per-camera"])
def test_legacy_clip_keys_load_every_weight(separate):
    source = make_policy(0, separate)
    legacy = legacy_state_dict(source)
    vision = "vision_encoders.1.model.vision_model." if separate else "vision_encoder.model.vision_model."
    assert any("text_encoder.text_model." in key for key in legacy) and any(vision in key for key in legacy)
    target = make_policy(1, separate)
    target.load_state_dict(legacy, strict=True)
    assert_same_weights(target, source)


@pytest.mark.parametrize("separate", [False, True], ids=["shared-encoder", "encoder-per-camera"])
def test_legacy_checkpoint_loads_through_from_pretrained(tmp_path, separate):
    source = make_policy(0, separate)
    source.save_pretrained(tmp_path)
    save_file(legacy_state_dict(source), str(tmp_path / "model.safetensors"))
    assert_same_weights(MultiTaskDiTPolicy.from_pretrained(tmp_path, strict=True), source)


@pytest.mark.parametrize("separate", [False, True], ids=["shared-encoder", "encoder-per-camera"])
def test_current_keys_are_left_alone(separate):
    source = make_policy(0, separate)
    target = make_policy(1, separate)
    target.load_state_dict(source.state_dict(), strict=True)
    assert_same_weights(target, source)
