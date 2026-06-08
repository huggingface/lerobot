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

from __future__ import annotations

import dataclasses

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.cosmos3 import modeling_cosmos3 as cosmos3_modeling
from lerobot.policies.cosmos3.configuration_cosmos3 import (
    COSMOS3_LEFT_IMAGE,
    COSMOS3_RIGHT_IMAGE,
    COSMOS3_WRIST_IMAGE,
    Cosmos3Config,
)
from lerobot.policies.cosmos3.modeling_cosmos3 import (
    Cosmos3Policy,
    preprocess_action_video_batch,
)
from lerobot.policies.cosmos3.processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_ACTION_DOMAIN_ID,
    COSMOS3_CLEAN_ACTION,
    COSMOS3_COND_INPUT_IDS,
    COSMOS3_CONDITIONING_FPS,
    COSMOS3_PROMPT,
    COSMOS3_RAW_ACTION_DIM,
    COSMOS3_TRAINING_SIGMA,
    COSMOS3_VIDEO,
    format_cosmos3_action_prompt,
    make_cosmos3_pre_post_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config(**overrides) -> Cosmos3Config:
    config = Cosmos3Config(
        device="cpu",
        transformer_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "patch_latent_dim": 4,
            "latent_channel": 4,
            "latent_patch_size": 1,
            "action_dim": 64,
            "action_gen": True,
            "vocab_size": 128,
            "rope_scaling": {"mrope_section": [2, 1, 1]},
        },
        vae_config={
            "base_dim": 4,
            "decoder_base_dim": 4,
            "z_dim": 4,
            "dim_mult": [1],
            "num_res_blocks": 1,
            "attn_scales": [],
            "temperal_downsample": [],  # spellchecker:disable-line
            "scale_factor_temporal": 1,
            "scale_factor_spatial": 1,
        },
        scheduler_config={
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "use_karras_sigmas": False,
            "flow_shift": 5.0,
        },
        input_features={
            COSMOS3_LEFT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            COSMOS3_RIGHT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            COSMOS3_WRIST_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )
    return dataclasses.replace(config, **overrides) if overrides else config


def constant_image(value: int) -> torch.Tensor:
    return torch.full((3, 360, 640), value / 255.0, dtype=torch.float32)


def constant_image_sequence(value: int, num_frames: int) -> torch.Tensor:
    return torch.full((1, num_frames, 3, 360, 640), value / 255.0, dtype=torch.float32)


def test_cosmos3_factory_registration():
    cfg = make_policy_config("cosmos3", device="cpu")

    assert isinstance(cfg, Cosmos3Config)
    assert get_policy_class("cosmos3") is Cosmos3Policy

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_cosmos3_processor_packs_multi_view_inputs():
    cfg = make_config()
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)
    state = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25], dtype=torch.float32)

    batch = {
        COSMOS3_LEFT_IMAGE: constant_image(10),
        COSMOS3_RIGHT_IMAGE: constant_image(20),
        COSMOS3_WRIST_IMAGE: constant_image(30),
        OBS_STATE: state,
        "task": "Pick up the banana and place it in the bowl.",
    }

    processed = preprocessor(batch)
    video = processed[COSMOS3_VIDEO]
    action_condition = processed[COSMOS3_ACTION_CONDITION]
    action_condition_mask = processed[COSMOS3_ACTION_CONDITION_MASK]

    # Default layout: primary view (wrist) on top, the remaining views tile the bottom.
    assert video.shape == (1, 3, 33, 540, 640)
    assert torch.all(video[0, :, 0, :360] == 30)
    assert torch.all(video[0, :, 0, 360:, :320] == 10)
    assert torch.all(video[0, :, 0, 360:, 320:] == 20)
    assert torch.count_nonzero(video[:, :, 1:]) == 0

    assert action_condition.shape == (1, 33, 8)
    expected_state = state.clone()
    expected_state[-1] = 1.0 - expected_state[-1]
    torch.testing.assert_close(action_condition[0, 0], expected_state)
    assert torch.count_nonzero(action_condition[0, 1:]) == 0
    torch.testing.assert_close(action_condition_mask[0, 0], torch.ones(1))
    assert torch.count_nonzero(action_condition_mask[0, 1:]) == 0
    assert processed[COSMOS3_PROMPT] == ["Pick up the banana and place it in the bowl."]


def test_cosmos3_processor_pads_missing_views():
    cfg = make_config(image_keys=[COSMOS3_WRIST_IMAGE], num_views=3)
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)

    batch = {
        COSMOS3_WRIST_IMAGE: constant_image(30),
        OBS_STATE: torch.zeros(8, dtype=torch.float32),
        "task": "Do something useful.",
    }

    video = preprocessor(batch)[COSMOS3_VIDEO]
    assert video.shape == (1, 3, 33, 540, 640)
    # Single supplied view stays primary; the two padded views are blank.
    assert torch.all(video[0, :, 0, :360] == 30)
    assert torch.all(video[0, :, 0, 360:] == 0)


def test_cosmos3_processor_pads_narrow_state_and_caps_wide_state():
    cfg = make_config(max_state_dim=8)
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)
    images = {
        COSMOS3_LEFT_IMAGE: constant_image(10),
        COSMOS3_RIGHT_IMAGE: constant_image(20),
        COSMOS3_WRIST_IMAGE: constant_image(30),
        "task": "Do something useful.",
    }

    narrow = preprocessor({**images, OBS_STATE: torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)})
    action_condition = narrow[COSMOS3_ACTION_CONDITION]
    assert action_condition.shape == (1, 33, 8)
    torch.testing.assert_close(action_condition[0, 0, :4], torch.tensor([0.1, 0.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        preprocessor({**images, OBS_STATE: torch.zeros(16, dtype=torch.float32)})


def test_cosmos3_processor_packs_training_video_and_clean_actions():
    cfg = make_config()
    preprocessor, _ = make_cosmos3_pre_post_processors(cfg)
    state = torch.zeros(1, cfg.chunk_size + 1, 8, dtype=torch.float32)
    state[:, :, -1] = 0.25
    actions = torch.zeros(1, cfg.chunk_size, 8, dtype=torch.float32)
    actions[:, :, -1] = 0.75

    batch = {
        COSMOS3_LEFT_IMAGE: constant_image_sequence(10, cfg.chunk_size + 1),
        COSMOS3_RIGHT_IMAGE: constant_image_sequence(20, cfg.chunk_size + 1),
        COSMOS3_WRIST_IMAGE: constant_image_sequence(30, cfg.chunk_size + 1),
        OBS_STATE: state,
        ACTION: actions,
        "task": ["Pick up the banana and place it in the bowl."],
    }

    processed = preprocessor(batch)

    assert processed[COSMOS3_VIDEO].shape == (1, 3, cfg.chunk_size + 1, 540, 640)

    clean_action = processed[COSMOS3_CLEAN_ACTION]
    assert clean_action.shape == (1, cfg.chunk_size + 1, cfg.max_action_dim)
    torch.testing.assert_close(clean_action[0, 0, :8], torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.75]).float())
    torch.testing.assert_close(clean_action[0, 1, :8], torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.25]).float())
    assert torch.count_nonzero(clean_action[..., 8:]) == 0


def test_cosmos3_formats_action_prompt():
    prompt = format_cosmos3_action_prompt(
        "Pick up the banana and place it in the bowl.",
        viewpoint="concat_view",
        additional_view_description=(
            "The top row is from the wrist-mounted camera. "
            "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
            "opposite sides, with the robot visible."
        ),
        num_frames=33,
        height=544,
        width=736,
        fps=15.0,
    )

    assert prompt == (
        "Pick up the banana and place it in the bowl. "
        "This video contains concatenated views from multiple camera perspectives. "
        "The top row is from the wrist-mounted camera. "
        "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
        "opposite sides, with the robot visible. "
        "The video is 2.0 seconds long and is of 15 FPS. "
        "This video is of 544x736 resolution."
    )


def test_cosmos3_preprocess_action_video_batch_resizes_and_normalizes():
    video = torch.zeros(2, 3, 33, 540, 640, dtype=torch.uint8)
    video[:, :, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8).view(1, 3, 1, 1)

    frames, image_size, height, width = preprocess_action_video_batch(
        video,
        resolution_tier=480,
        num_frames=33,
        device="cpu",
        dtype=torch.float32,
    )

    assert frames.shape == (2, 3, 33, 544, 736)
    torch.testing.assert_close(image_size, torch.tensor([544.0, 736.0, 540.0, 640.0]))
    assert (height, width) == (544, 736)
    torch.testing.assert_close(frames[0, :, 0, 0, 0], torch.tensor([10, 20, 30]) / 127.5 - 1.0)


def test_cosmos3_select_action_uses_chunk_queue(monkeypatch):
    policy = Cosmos3Policy(make_config())
    fixed_chunk = torch.arange(32 * 8, dtype=torch.float32).view(1, 32, 8)
    sample_calls = {"count": 0}

    def fake_sample_actions(batch, **kwargs):
        sample_calls["count"] += 1
        return fixed_chunk

    monkeypatch.setattr(policy.model, "sample_actions", fake_sample_actions)

    action_0 = policy.select_action({})
    action_1 = policy.select_action({})

    torch.testing.assert_close(action_0, fixed_chunk[:, 0])
    torch.testing.assert_close(action_1, fixed_chunk[:, 1])
    assert sample_calls["count"] == 1


def test_cosmos3_masked_flow_matching_mse_uses_per_sample_batch_denominator():
    policy = Cosmos3Policy(make_config())
    pred = torch.tensor([[10.0, 0.0], [2.0, 4.0]])
    target = torch.zeros_like(pred)
    noisy_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    torch.testing.assert_close(
        policy.model._masked_flow_matching_mse_by_sample(pred, target, noisy_mask),
        torch.tensor(30.0),
    )

    policy.config.normalize_loss_by_active = True
    torch.testing.assert_close(
        policy.model._masked_flow_matching_mse_by_sample(pred, target, noisy_mask),
        torch.tensor(55.0),
    )


def test_cosmos3_forward_uses_batched_training_loss(monkeypatch):
    cfg = make_config()
    cfg.dtype = "float32"
    cfg.eos_token_id = 5
    cfg.start_of_generation_token_id = 6
    policy = Cosmos3Policy(cfg)
    batch_size = 2
    sequence_length = 3
    clean_vision = torch.zeros(batch_size, 4, 2, 2, 2)
    calls = {"batch": 0}

    def fake_video_conditioning(videos, **kwargs):
        return clean_vision, torch.tensor([2.0, 2.0, 2.0, 2.0]), 2, 2

    def fake_predict_velocity_batch(**kwargs):
        calls["batch"] += 1
        assert len(kwargs["packed_samples"]) == batch_size
        assert kwargs["vision_tokens"].shape[0] == batch_size
        assert kwargs["action_tokens"].shape[0] == batch_size
        return torch.zeros_like(kwargs["vision_tokens"]), torch.zeros_like(kwargs["action_tokens"])

    monkeypatch.setattr(cosmos3_modeling, "preprocess_action_video_batch", fake_video_conditioning)
    monkeypatch.setattr(policy.model, "_encode_video", lambda video: video)
    monkeypatch.setattr(policy.model, "_predict_velocity_batch", fake_predict_velocity_batch)

    batch = {
        COSMOS3_VIDEO: torch.zeros(batch_size, 3, sequence_length, 4, 4, dtype=torch.uint8),
        COSMOS3_ACTION_CONDITION: torch.zeros(batch_size, sequence_length, cfg.raw_action_dim),
        COSMOS3_ACTION_CONDITION_MASK: torch.tensor([[[1.0], [0.0], [0.0]]] * batch_size),
        COSMOS3_ACTION_DOMAIN_ID: torch.full((batch_size,), cfg.domain_id),
        COSMOS3_CONDITIONING_FPS: torch.full((batch_size,), 15.0),
        COSMOS3_RAW_ACTION_DIM: torch.full((batch_size,), cfg.raw_action_dim),
        COSMOS3_COND_INPUT_IDS: [torch.tensor([1, 2]), torch.tensor([1, 2, 3])],
        COSMOS3_CLEAN_ACTION: torch.zeros(batch_size, sequence_length, cfg.max_action_dim),
        COSMOS3_TRAINING_SIGMA: torch.full((batch_size, 1), 0.5),
    }

    loss, metrics = policy.model(batch)

    assert calls["batch"] == 1
    assert not hasattr(policy.model, "_predict_velocity")
    assert loss.ndim == 0
    assert set(metrics) == {"loss", "flow_matching_loss_vision", "flow_matching_loss_action"}


def test_cosmos3_batched_velocity_treats_single_sample_as_batch_case():
    cfg = make_config()
    cfg.dtype = "float32"
    cfg.eos_token_id = 5
    cfg.start_of_generation_token_id = 6
    policy = Cosmos3Policy(cfg)
    policy.eval()

    vision_tokens = torch.randn(2, 4, 2, 2, 2)
    action_tokens = torch.randn(2, 3, cfg.max_action_dim)
    cond_input_ids = [torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)]

    single_packed_samples = policy.model._pack_batch_static(
        cond_input_ids=[cond_input_ids[0]],
        vision_tokens=vision_tokens[:1],
        action_tokens=action_tokens[:1],
        conditioning_fps=torch.tensor([15.0]),
    )
    batch_packed_samples = policy.model._pack_batch_static(
        cond_input_ids=cond_input_ids,
        vision_tokens=vision_tokens,
        action_tokens=action_tokens,
        conditioning_fps=torch.tensor([15.0, 12.0]),
    )
    single_vision_timesteps = [
        torch.full((sample["num_noisy_vision_tokens"],), 3.0) for sample in single_packed_samples
    ]
    single_action_timesteps = [
        torch.full((sample["num_noisy_action_tokens"],), 3.0) for sample in single_packed_samples
    ]
    batch_vision_timesteps = [
        torch.full((sample["num_noisy_vision_tokens"],), 3.0) for sample in batch_packed_samples
    ]
    batch_action_timesteps = [
        torch.full((sample["num_noisy_action_tokens"],), 3.0) for sample in batch_packed_samples
    ]
    vision_condition_mask = torch.tensor([[[[[1.0]], [[0.0]]]]])
    action_condition_mask = torch.tensor([[[1.0], [0.0], [0.0]], [[1.0], [0.0], [0.0]]])

    with torch.no_grad():
        single_vision, single_action = policy.model._predict_velocity_batch(
            packed_samples=single_packed_samples,
            vision_tokens=vision_tokens[:1],
            action_tokens=action_tokens[:1],
            vision_timesteps=single_vision_timesteps,
            action_timesteps=single_action_timesteps,
            action_domain_ids=torch.tensor([cfg.domain_id]),
            vision_condition_mask=vision_condition_mask,
            action_condition_mask=action_condition_mask[:1],
            raw_action_dims=torch.tensor([cfg.raw_action_dim]),
        )
        batch_vision, batch_action = policy.model._predict_velocity_batch(
            packed_samples=batch_packed_samples,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens,
            vision_timesteps=batch_vision_timesteps,
            action_timesteps=batch_action_timesteps,
            action_domain_ids=torch.tensor([cfg.domain_id, cfg.domain_id]),
            vision_condition_mask=vision_condition_mask,
            action_condition_mask=action_condition_mask,
            raw_action_dims=torch.tensor([cfg.raw_action_dim, cfg.raw_action_dim]),
        )

    torch.testing.assert_close(batch_vision[:1], single_vision)
    torch.testing.assert_close(batch_action[:1], single_action)


def test_cosmos3_batched_velocity_supports_variable_text_lengths():
    cfg = make_config()
    cfg.dtype = "float32"
    cfg.eos_token_id = 5
    cfg.start_of_generation_token_id = 6
    policy = Cosmos3Policy(cfg)
    policy.eval()

    vision_tokens = torch.randn(2, 4, 2, 2, 2)
    action_tokens = torch.randn(2, 3, cfg.max_action_dim)
    cond_input_ids = [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4])]
    packed_samples = policy.model._pack_batch_static(
        cond_input_ids=cond_input_ids,
        vision_tokens=vision_tokens,
        action_tokens=action_tokens,
        conditioning_fps=torch.tensor([15.0, 15.0]),
    )
    vision_timesteps = [torch.full((sample["num_noisy_vision_tokens"],), 3.0) for sample in packed_samples]
    action_timesteps = [torch.full((sample["num_noisy_action_tokens"],), 3.0) for sample in packed_samples]

    with torch.no_grad():
        batch_vision, batch_action = policy.model._predict_velocity_batch(
            packed_samples=packed_samples,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens,
            vision_timesteps=vision_timesteps,
            action_timesteps=action_timesteps,
            action_domain_ids=torch.tensor([cfg.domain_id, cfg.domain_id]),
            vision_condition_mask=torch.tensor([[[[[1.0]], [[0.0]]]]]),
            action_condition_mask=torch.tensor([[[1.0], [0.0], [0.0]], [[1.0], [0.0], [0.0]]]),
            raw_action_dims=torch.tensor([cfg.raw_action_dim, cfg.raw_action_dim]),
        )

    assert batch_vision.shape == vision_tokens.shape
    assert batch_action.shape == action_tokens.shape
