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

import torch
from torch import nn

import lerobot.policies.evo1.modeling_evo1 as modeling_evo1
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.flow_matching import FlowmatchingActionHead
from lerobot.policies.factory import get_policy_class, make_policy_config
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
        return torch.ones(len(images), 4, EMBED_DIM, requires_grad=torch.is_grad_enabled())

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


def test_evo1_policy_forward_and_inference_use_batched_embedding(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config())

    loss, metrics = policy.forward(make_batch(include_action=True))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["active_action_dims"] == ACTION_DIM * CHUNK_SIZE
    assert policy.model.get_vl_embeddings_calls == 1

    action_chunk = policy.predict_action_chunk(make_batch(include_action=False))
    assert action_chunk.shape == (2, CHUNK_SIZE, ACTION_DIM)

    policy.reset()
    selected = policy.select_action(make_batch(include_action=False))
    assert selected.shape == (2, ACTION_DIM)


def test_stage1_frozen_vlm_embeddings_do_not_track_gradients(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "EVO1", DummyEVO1)
    policy = modeling_evo1.EVO1Policy(make_config(training_stage="stage1"))
    policy.train()

    image_batches, image_masks = policy._collect_image_batches(make_batch(include_action=False))
    fused_tokens = policy._compute_fused_tokens(["pick", "place"], image_batches, image_masks)

    assert policy.model.grad_enabled_calls == [False]
    assert policy.model.embedder_training_calls == [False]
    assert not fused_tokens.requires_grad
    assert policy.model.embedder.training is True


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

    image_batches, image_masks = policy._collect_image_batches(batch)

    assert len(image_batches) == 1
    assert len(image_batches[0]) == policy.config.max_views
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
