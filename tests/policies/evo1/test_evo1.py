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

import pytest
import torch
from torch import nn

import lerobot.policies.evo1.modeling_evo1 as modeling_evo1
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.flow_matching import FlowmatchingActionHead
from lerobot.policies.evo1.internvl3_embedder import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _batched_pixel_values,
)
from lerobot.policies.evo1.processor_evo1 import (
    Evo1ActionProcessorStep,
    Evo1PadActionProcessorStep,
    Evo1PadStateProcessorStep,
    ensure_evo1_processor_steps,
    make_evo1_pre_post_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

STATE_DIM = 4
ACTION_DIM = 3
MAX_STATE_DIM = 6
MAX_ACTION_DIM = 5
CHUNK_SIZE = 2
EMBED_DIM = 8


class DummyEVO1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = nn.Dropout(p=0.0)
        self.action_head = nn.Linear(1, 1)
        self.get_vl_embeddings_calls = 0
        self.grad_enabled_calls = []
        self.embedder_training_calls = []

    def set_finetune_flags(self):
        return None

    def get_vl_embeddings(self, images, image_mask, prompt=None, return_cls_only=False):
        self.get_vl_embeddings_calls += 1
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        self.embedder_training_calls.append(self.embedder.training)
        # images is a list of per-camera (B, C, H, W) tensors, so the batch dim is images[0].shape[0].
        batch_size = images[0].shape[0]
        return torch.ones(batch_size, 4, EMBED_DIM, requires_grad=torch.is_grad_enabled())

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
    ):
        batch_size = fused_tokens.shape[0]
        if actions_gt is None:
            return torch.ones(batch_size, CHUNK_SIZE * MAX_ACTION_DIM)
        pred_velocity = torch.zeros(batch_size, CHUNK_SIZE * MAX_ACTION_DIM)
        noise = torch.zeros_like(actions_gt)
        return pred_velocity, noise


def make_config(training_stage="stage1", **kwargs):
    config_kwargs = {
        "device": "cpu",
        "vlm_model_name": "dummy-internvl3",
        "training_stage": training_stage,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": 1,
        "max_state_dim": MAX_STATE_DIM,
        "max_action_dim": MAX_ACTION_DIM,
        "max_views": 2,
        "embed_dim": EMBED_DIM,
        "hidden_dim": 16,
        "state_hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_inference_timesteps": 2,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
    }
    config_kwargs.update(kwargs)
    return Evo1Config(**config_kwargs)


def make_batch(include_action=True):
    batch = {
        "task": ["pick the block", "place the block"],
        OBS_STATE: torch.randn(2, STATE_DIM),
        f"{OBS_IMAGES}.front": torch.rand(2, 3, 16, 16),
    }
    if include_action:
        batch[ACTION] = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
    return batch


def make_stats(state_dim=STATE_DIM, action_dim=ACTION_DIM):
    return {
        OBS_STATE: {
            "min": torch.full((state_dim,), -2.0),
            "max": torch.full((state_dim,), 2.0),
        },
        ACTION: {
            "min": torch.full((action_dim,), -1.0),
            "max": torch.full((action_dim,), 1.0),
        },
    }


def test_evo1_factory_registration():
    cfg = make_policy_config(
        "evo1",
        device="cpu",
        vlm_model_name="dummy-internvl3",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )

    assert isinstance(cfg, Evo1Config)
    assert get_policy_class("evo1") is modeling_evo1.EVO1Policy


def test_evo1_stage_defaults_and_consistency():
    stage1 = make_config(training_stage="stage1")
    assert (stage1.finetune_vlm, stage1.finetune_language_model, stage1.finetune_vision_model) == (
        False,
        False,
        False,
    )
    assert stage1.finetune_action_head is True

    stage2 = make_config(training_stage="stage2")
    assert (stage2.finetune_vlm, stage2.finetune_language_model, stage2.finetune_vision_model) == (
        True,
        True,
        True,
    )
    assert stage2.finetune_action_head is True

    stage2_from_stage1_checkpoint_flags = make_config(
        training_stage="stage2",
        finetune_vlm=False,
        finetune_language_model=False,
        finetune_vision_model=False,
        finetune_action_head=False,
    )
    assert (
        stage2_from_stage1_checkpoint_flags.finetune_vlm,
        stage2_from_stage1_checkpoint_flags.finetune_language_model,
        stage2_from_stage1_checkpoint_flags.finetune_vision_model,
    ) == (
        True,
        True,
        True,
    )
    assert stage2_from_stage1_checkpoint_flags.finetune_action_head is True

    explicit_off = make_config(
        training_stage="stage2",
        apply_training_stage_defaults=False,
        finetune_vlm=False,
        finetune_language_model=False,
        finetune_vision_model=False,
        finetune_action_head=False,
    )
    assert (
        explicit_off.finetune_vlm,
        explicit_off.finetune_language_model,
        explicit_off.finetune_vision_model,
    ) == (
        False,
        False,
        False,
    )
    assert explicit_off.finetune_action_head is False

    try:
        make_config(
            training_stage="stage2",
            apply_training_stage_defaults=False,
            finetune_vlm=True,
            finetune_language_model=False,
        )
    except ValueError as exc:
        assert "Inconsistent EVO1 finetune config" in str(exc)
    else:
        raise AssertionError("Expected inconsistent finetune config to raise ValueError")


def test_evo1_rejects_non_square_image_resolution():
    with pytest.raises(ValueError, match="square image_resolution"):
        make_config(image_resolution=(448, 320))


def test_evo1_build_model_config_uses_image_resolution_and_trainable_checkpointing():
    stage1 = make_config(training_stage="stage1", image_resolution=(224, 224))
    stage1_model_config = modeling_evo1.EVO1Policy._build_model_config(stage1)

    assert stage1_model_config["image_size"] == 224
    assert stage1_model_config["enable_gradient_checkpointing"] is False

    stage2 = make_config(training_stage="stage2", image_resolution=(224, 224))
    stage2_model_config = modeling_evo1.EVO1Policy._build_model_config(stage2)

    assert stage2_model_config["enable_gradient_checkpointing"] is True


def test_evo1_policy_processors_pad_state_crop_action_and_binarize_gripper():
    libero_action_dim = 7
    config = make_config(
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=8,
        postprocess_action_dim=libero_action_dim,
        binarize_gripper=True,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(libero_action_dim,))},
    )
    stats = make_stats(action_dim=libero_action_dim)

    preprocessor, postprocessor = make_evo1_pre_post_processors(config, dataset_stats=stats)

    assert isinstance(preprocessor.steps[2], Evo1PadStateProcessorStep)
    assert isinstance(preprocessor.steps[3], Evo1PadActionProcessorStep)
    assert isinstance(preprocessor.steps[4], NormalizerProcessorStep)
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], Evo1ActionProcessorStep)

    normalizer = preprocessor.steps[4]
    assert normalizer.features[OBS_STATE].shape == (MAX_STATE_DIM,)
    assert normalizer.features[ACTION].shape == (8,)
    assert normalizer._tensor_stats[OBS_STATE]["min"].shape == (MAX_STATE_DIM,)
    assert normalizer._tensor_stats[ACTION]["min"].shape == (8,)

    processed_batch = preprocessor(
        {
            "task": "pick the block",
            OBS_STATE: torch.zeros(STATE_DIM),
            ACTION: torch.zeros(libero_action_dim),
            f"{OBS_IMAGES}.front": torch.rand(3, 16, 16),
        }
    )
    processed_state = processed_batch[OBS_STATE]
    assert processed_state.shape == (1, MAX_STATE_DIM)
    assert torch.allclose(processed_state, torch.zeros_like(processed_state))
    assert processed_batch[ACTION].shape == (1, 8)
    assert torch.allclose(processed_batch[ACTION], torch.zeros_like(processed_batch[ACTION]))
    assert processed_batch["action_mask"].shape == (1, 8)
    assert processed_batch["action_mask"][:, :libero_action_dim].all()
    assert not processed_batch["action_mask"][:, libero_action_dim:].any()

    action = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ],
        dtype=torch.float32,
    )
    processed = postprocessor(action)

    assert processed.shape == (2, 7)
    assert torch.allclose(processed[:, :6], action[:, :6])
    assert torch.equal(processed[:, 6], torch.tensor([1.0, -1.0]))


def test_evo1_legacy_processors_are_completed_before_normalization():
    config = make_config(
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=8,
        postprocess_action_dim=7,
        binarize_gripper=True,
    )
    stats = make_stats(action_dim=7)
    legacy_pre = PolicyProcessorPipeline(
        steps=[
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=stats,
            )
        ]
    )
    legacy_post = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=stats,
            )
        ]
    )

    preprocessor, postprocessor = ensure_evo1_processor_steps(config, legacy_pre, legacy_post)

    assert isinstance(preprocessor.steps[0], Evo1PadStateProcessorStep)
    assert isinstance(preprocessor.steps[1], Evo1PadActionProcessorStep)
    assert isinstance(preprocessor.steps[2], NormalizerProcessorStep)
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], Evo1ActionProcessorStep)
    assert postprocessor.steps[1].action_dim == 7
    assert postprocessor.steps[1].binarize_gripper is True
    assert preprocessor.steps[2].features[OBS_STATE].shape == (MAX_STATE_DIM,)
    assert preprocessor.steps[2]._tensor_stats[OBS_STATE]["min"].shape == (MAX_STATE_DIM,)
    assert preprocessor.steps[2]._tensor_stats[ACTION]["min"].shape == (8,)
    assert postprocessor.steps[0].features[ACTION].shape == (8,)
    assert postprocessor.steps[0]._tensor_stats[ACTION]["min"].shape == (8,)

    preprocessor, postprocessor = ensure_evo1_processor_steps(config, preprocessor, postprocessor)
    assert sum(isinstance(step, Evo1PadStateProcessorStep) for step in preprocessor.steps) == 1
    assert sum(isinstance(step, Evo1PadActionProcessorStep) for step in preprocessor.steps) == 1
    assert sum(isinstance(step, Evo1ActionProcessorStep) for step in postprocessor.steps) == 1


def test_evo1_policy_forward_and_inference_use_batched_embedding(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config())
    preprocessor, _postprocessor = make_evo1_pre_post_processors(policy.config, dataset_stats=make_stats())
    training_batch = preprocessor(make_batch(include_action=True))

    assert training_batch[ACTION].shape == (2, CHUNK_SIZE, MAX_ACTION_DIM)
    assert training_batch["action_mask"].shape == (2, CHUNK_SIZE, MAX_ACTION_DIM)
    assert training_batch["action_mask"][:, :, :ACTION_DIM].all()
    assert not training_batch["action_mask"][:, :, ACTION_DIM:].any()

    loss, metrics = policy.forward(training_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["active_action_dims"] == ACTION_DIM * CHUNK_SIZE
    assert policy.model.get_vl_embeddings_calls == 1

    action_chunk = policy.predict_action_chunk(make_batch(include_action=False))
    assert action_chunk.shape == (2, CHUNK_SIZE, MAX_ACTION_DIM)

    policy.reset()
    selected = policy.select_action(make_batch(include_action=False))
    assert selected.shape == (2, MAX_ACTION_DIM)


def test_stage1_frozen_vlm_embeddings_do_not_track_gradients(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config(training_stage="stage1"))
    policy.train()

    image_batches, image_masks = policy._collect_image_batches(make_batch(include_action=False))
    fused_tokens = policy._compute_fused_tokens(["pick", "place"], image_batches, image_masks)

    assert policy.model.grad_enabled_calls == [False]
    assert policy.model.embedder_training_calls == [False]
    assert not fused_tokens.requires_grad
    assert policy.model.embedder.training is False


def test_stage2_vlm_embeddings_track_gradients(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config(training_stage="stage2"))
    policy.train()

    image_batches, image_masks = policy._collect_image_batches(make_batch(include_action=False))
    fused_tokens = policy._compute_fused_tokens(["pick", "place"], image_batches, image_masks)

    assert policy.model.grad_enabled_calls == [True]
    assert policy.model.embedder_training_calls == [True]
    assert fused_tokens.requires_grad


def test_collect_image_batches_handles_unbatched_chw(monkeypatch):
    # Regression for an issue where batch_size was read from shape[0] before normalizing
    # per-camera tensor dims, so an unbatched (C, H, W) input was treated as batch_size=C.
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config())
    batch = {
        OBS_STATE: torch.randn(1, STATE_DIM),
        f"{OBS_IMAGES}.front": torch.rand(3, 16, 16),
    }

    camera_images, image_masks = policy._collect_image_batches(batch)

    # One present camera, returned as a batched (B, C, H, W) tensor with the unbatched CHW frame
    # promoted to batch_size=1 (not read as batch_size=C).
    assert len(camera_images) == 1
    assert camera_images[0].shape == (1, 3, 16, 16)
    assert image_masks.tolist() == [[True, False]]


def test_evo1_action_mask_accepts_chunk_size_one(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    config = make_config(chunk_size=1, n_action_steps=1)
    policy = modeling_evo1.EVO1Policy(config)
    batch = make_batch(include_action=True)
    batch[ACTION] = torch.randn(2, ACTION_DIM)
    batch["action_mask"] = torch.ones(2, ACTION_DIM, dtype=torch.bool)

    actions, action_mask = policy._prepare_actions(batch)

    assert actions.shape == (2, 1, MAX_ACTION_DIM)
    assert action_mask.shape == (2, 1, MAX_ACTION_DIM)
    assert action_mask[:, :, :ACTION_DIM].all()
    assert not action_mask[:, :, ACTION_DIM:].any()


def test_flowmatching_dict_config_enables_state_encoder_for_horizon_one():
    head = FlowmatchingActionHead(
        config={
            "embed_dim": EMBED_DIM,
            "hidden_dim": 16,
            "action_dim": ACTION_DIM,
            "horizon": 1,
            "per_action_dim": ACTION_DIM,
            "num_heads": 2,
            "num_layers": 1,
            "num_inference_timesteps": 2,
            "state_dim": STATE_DIM,
            "state_hidden_dim": 16,
            "num_categories": 1,
        }
    )

    assert head.state_encoder is not None
    pred_velocity, noise = head(
        torch.randn(2, 4, EMBED_DIM),
        state=torch.randn(2, STATE_DIM),
        actions_gt=torch.randn(2, 1, ACTION_DIM),
        action_mask=torch.ones(2, 1, ACTION_DIM, dtype=torch.bool),
    )

    assert pred_velocity.shape == (2, ACTION_DIM)
    assert noise.shape == (2, 1, ACTION_DIM)


def test_evo1_batched_pixel_values_shape_and_zero_padding():
    torch.manual_seed(0)
    batch_size, image_size, max_views = 2, 448, 3
    camera_images = [torch.rand(batch_size, 3, 40, 50)]  # a single present camera
    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)

    pixel_values = _batched_pixel_values(
        camera_images, max_views, image_size, mean, std, torch.float32, torch.device("cpu")
    )

    assert pixel_values.shape == (batch_size * max_views, 3, image_size, image_size)
    grouped = pixel_values.reshape(batch_size, max_views, 3, image_size, image_size)
    # Absent views (indices 1, 2) are zero images normalized to -mean/std, matching the old padding.
    expected_pad = (-mean / std).view(1, 3, 1, 1)
    for view in (1, 2):
        assert torch.allclose(
            grouped[:, view], expected_pad.expand(batch_size, 3, image_size, image_size), atol=1e-5
        )
    # The present view is genuinely different from the constant pad value.
    assert not torch.allclose(
        grouped[:, 0], expected_pad.expand(batch_size, 3, image_size, image_size), atol=1e-3
    )
