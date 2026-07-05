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

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.policies.groot.action_head.cross_attention_dit import AlternateVLDiT
from lerobot.policies.groot.groot_n1_7 import GR00TN17
from lerobot.policies.groot.processor_groot import (
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE

OSS_REFERENCE_COMMIT = "ab88b50c718f6528e1df9dcbaf75865d1b604760"


def _fixture_path(filename: str) -> Path:
    fixture_dir = os.environ.get("GROOT_N17_OSS_PARITY_FIXTURE_DIR")
    if fixture_dir is None:
        pytest.skip("Set GROOT_N17_OSS_PARITY_FIXTURE_DIR to run external OSS parity fixtures.")
    path = Path(fixture_dir) / filename
    if not path.is_file():
        pytest.skip(f"External OSS parity fixture not found: {path}")
    return path


def test_groot_n1_7_eval_image_transform_matches_oss_reference():
    """Match the native N1.7 eval transform for a non-square SO-101 frame."""

    y, x = np.indices((480, 640), dtype=np.uint16)
    image = np.stack(
        ((x + 3 * y) % 256, (2 * x + y) % 256, (x + 5 * y) % 256),
        axis=-1,
    ).astype(np.uint8)
    actual = _transform_n1_7_image_for_vlm_albumentations(
        image,
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
    )

    assert actual.shape == (256, 340, 3)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == (
        "c17e47af68a812aa79db3bb7b64b549ddf10148ac1b204a9686095018561ae9e"
    )


def test_groot_n1_7_vlm_chat_content_order_matches_oss_reference():
    """Native OSS places all image items before the language item."""

    class RecordingProcessor:
        def __init__(self):
            self.content_types = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            self.content_types = [item["type"] for item in conversation[0]["content"]]
            return "rendered"

        def __call__(self, **kwargs):
            return {}

    processor = RecordingProcessor()
    step = GrootN17VLMEncodeStep(
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        device="cpu",
    )
    step._proc = processor
    transition = {
        TransitionKey.OBSERVATION: {
            "video": np.zeros((1, 1, 2, 480, 640, 3), dtype=np.uint8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"language": ["pick up the vial"]},
    }

    step(transition)

    assert processor.content_types == ["image", "image", "text"]


def test_groot_n1_7_alternate_vl_dit_matches_oss_reference():
    """Run the LeRobot DiT with native OSS weights and identical inputs."""

    pytest.importorskip("diffusers")

    fixture = torch.load(_fixture_path("alternate_vl_dit_small.pt"), map_location="cpu", weights_only=True)
    model = AlternateVLDiT(
        output_dim=8,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        max_num_positional_embeddings=16,
        compute_dtype=torch.float32,
        interleave_self_attention=True,
        cross_attention_dim=6,
    ).eval()
    model.load_state_dict(fixture["state_dict"], strict=True)

    actual = model(
        hidden_states=fixture["hidden_states"],
        encoder_hidden_states=fixture["encoder_hidden_states"],
        timestep=fixture["timestep"],
        image_mask=fixture["image_mask"],
        backbone_attention_mask=fixture["backbone_attention_mask"],
    )

    torch.testing.assert_close(actual, fixture["output"], atol=1e-6, rtol=1e-6)


def _state_decode_reference():
    fixture = np.load(_fixture_path("state_and_action_decode.npz"))
    raw_stats = {
        "state": {
            "single_arm": {"q01": fixture["state_single_arm_q01"], "q99": fixture["state_single_arm_q99"]},
            "gripper": {"q01": fixture["state_gripper_q01"], "q99": fixture["state_gripper_q99"]},
        },
        "action": {
            "single_arm": {"q01": fixture["action_single_arm_q01"], "q99": fixture["action_single_arm_q99"]},
            "gripper": {"q01": fixture["action_gripper_q01"], "q99": fixture["action_gripper_q99"]},
        },
        "relative_action": {
            "single_arm": {
                "min": fixture["relative_single_arm_min"],
                "max": fixture["relative_single_arm_max"],
            },
        },
    }
    for modality_stats in raw_stats.values():
        for entry in modality_stats.values():
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    entry[key] = value.tolist()
    modality_config = {
        "state": {"modality_keys": ["single_arm", "gripper"]},
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
            ],
        },
    }
    state_min = np.concatenate((fixture["state_single_arm_q01"], fixture["state_gripper_q01"]))
    state_max = np.concatenate((fixture["state_single_arm_q99"], fixture["state_gripper_q99"]))
    pack_step = GrootN17PackInputsStep(
        normalize_min_max=True,
        stats={OBS_STATE: {"min": state_min, "max": state_max}},
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=True,
    )
    raw_state = np.concatenate((fixture["state_single_arm"], fixture["state_gripper"]), axis=-1)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.from_numpy(raw_state)},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    packed = pack_step(transition)
    return fixture, raw_stats, modality_config, pack_step, packed


def test_groot_n1_7_state_normalization_matches_oss_checkpoint_reference():
    fixture, _raw_stats, _modality_config, _pack_step, packed = _state_decode_reference()
    expected = np.concatenate(
        (fixture["normalized_state_single_arm"], fixture["normalized_state_gripper"]), axis=-1
    )

    actual = packed[TransitionKey.OBSERVATION]["state"][:, 0, :6]

    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-6, rtol=1e-6)


def test_groot_n1_7_relative_action_decode_matches_oss_checkpoint_reference():
    fixture, raw_stats, modality_config, pack_step, _packed = _state_decode_reference()
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=6,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=True,
        use_relative_action=True,
        pack_step=pack_step,
    )
    decoded = decode_step({TransitionKey.ACTION: torch.from_numpy(fixture["normalized_action"])})[
        TransitionKey.ACTION
    ]
    expected = np.concatenate((fixture["decoded_single_arm"], fixture["decoded_gripper"]), axis=-1).astype(
        np.float32
    )

    torch.testing.assert_close(decoded, torch.from_numpy(expected), atol=1e-5, rtol=1e-5)


def test_groot_n1_7_qwen_backbone_matches_oss_checkpoint_reference():
    """Compare the actual 3B checkpoint backbone when explicitly enabled."""

    checkpoint = os.environ.get("GROOT_N17_PARITY_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set GROOT_N17_PARITY_CHECKPOINT to run the 3B OSS Qwen parity test.")
    if not torch.cuda.is_available():
        pytest.skip("The 3B OSS Qwen parity test requires CUDA.")

    pytest.importorskip("transformers")

    from transformers.feature_extraction_utils import BatchFeature

    fixture = torch.load(_fixture_path("qwen_backbone_so101.pt"), map_location="cpu", weights_only=True)
    model = GR00TN17.from_pretrained(checkpoint).to(device="cuda", dtype=torch.bfloat16).eval()
    backbone_input = BatchFeature(
        data={
            key.removeprefix("input."): value.to("cuda")
            for key, value in fixture.items()
            if key.startswith("input.")
        }
    )

    with torch.inference_mode():
        actual = model.backbone(backbone_input)

    feature_error = (
        actual.backbone_features.cpu().float() - fixture["output.backbone_features"].float()
    ).abs()
    # Native OSS and LeRobot use different Torch/Transformers/Flash-Attention releases.
    # Require the measured BF16 accumulation envelope while rejecting structural drift.
    assert feature_error.mean().item() <= 0.04
    assert feature_error.max().item() <= 2.0
    torch.testing.assert_close(
        actual.backbone_attention_mask.cpu(), fixture["output.backbone_attention_mask"]
    )
    torch.testing.assert_close(actual.image_mask.cpu(), fixture["output.image_mask"])
