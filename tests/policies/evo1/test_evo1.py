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

import lerobot.policies.evo1.evo1_model as evo1_model
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
    evo1_batch_to_transition,
    make_evo1_pre_post_processors,
    reconcile_evo1_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

STATE_DIM = 4
ACTION_DIM = 3
MAX_STATE_DIM = 6
MAX_ACTION_DIM = 5
CHUNK_SIZE = 2
EMBED_DIM = 8


class DummyEvo1Model(nn.Module):
    def __init__(self, config, vlm_hub_kwargs=None):
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
        tokens = torch.ones(batch_size, 4, EMBED_DIM, requires_grad=torch.is_grad_enabled())
        valid_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        return tokens, valid_mask

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
        context_mask=None,
        **kwargs,
    ):
        batch_size = fused_tokens.shape[0]
        if actions_gt is None:
            return torch.ones(batch_size, CHUNK_SIZE * MAX_ACTION_DIM)
        pred_velocity = torch.zeros(batch_size, CHUNK_SIZE * MAX_ACTION_DIM)
        noise = torch.zeros_like(actions_gt)
        return pred_velocity, noise


class ChunkCountingDummyModel(DummyEvo1Model):
    """Emits per-step distinguishable actions so queue ordering and re-prediction are observable."""

    def __init__(self, config, vlm_hub_kwargs=None):
        super().__init__(config, vlm_hub_kwargs)
        self.chunks_predicted = 0

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
        context_mask=None,
        **kwargs,
    ):
        if actions_gt is not None:
            return super().forward(fused_tokens, state, actions_gt, action_mask, embodiment_ids, context_mask)
        self.chunks_predicted += 1
        batch_size = fused_tokens.shape[0]
        step_values = torch.arange(CHUNK_SIZE, dtype=torch.float32) + 10.0 * self.chunks_predicted
        chunk = step_values.repeat_interleave(MAX_ACTION_DIM).unsqueeze(0).repeat(batch_size, 1)
        return chunk


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


def make_flowmatching_head(**overrides):
    kwargs = {
        "embed_dim": EMBED_DIM,
        "hidden_dim": 16,
        "action_dim": CHUNK_SIZE * ACTION_DIM,
        "horizon": CHUNK_SIZE,
        "per_action_dim": ACTION_DIM,
        "num_heads": 2,
        "num_layers": 1,
        "num_inference_timesteps": 2,
        "state_dim": STATE_DIM,
        "state_hidden_dim": 16,
        "num_categories": 1,
    }
    kwargs.update(overrides)
    return FlowmatchingActionHead(**kwargs)


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
    assert get_policy_class("evo1") is modeling_evo1.Evo1Policy


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

    # An explicit finetune_vlm=False without branch-level flags freezes both branches instead of
    # raising an inconsistency error.
    frozen_vlm = make_config(
        training_stage="stage2",
        apply_training_stage_defaults=False,
        finetune_vlm=False,
    )
    assert (
        frozen_vlm.finetune_vlm,
        frozen_vlm.finetune_language_model,
        frozen_vlm.finetune_vision_model,
    ) == (False, False, False)

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


def test_evo1_rejects_out_of_range_default_embodiment_id():
    with pytest.raises(ValueError, match="default_embodiment_id"):
        make_config(default_embodiment_id=3, num_categories=2)


def test_evo1_model_uses_image_resolution_and_trainable_checkpointing(monkeypatch):
    captured: dict = {}

    class SpyEmbedder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(evo1_model, "InternVL3Embedder", SpyEmbedder)

    stage1 = make_config(training_stage="stage1", image_resolution=(224, 224))
    evo1_model.Evo1Model(stage1)
    assert captured["image_size"] == 224
    # VLM is frozen in stage1, so gradient checkpointing is gated off.
    assert captured["enable_gradient_checkpointing"] is False

    stage2 = make_config(training_stage="stage2", image_resolution=(224, 224))
    evo1_model.Evo1Model(stage2)
    assert captured["enable_gradient_checkpointing"] is True


class FakeInternVLModel(nn.Module):
    """Minimal stand-in with the native HF InternVL submodule layout."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)


class FakeEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = FakeInternVLModel()


def test_set_finetune_flags_targets_native_hf_internvl_submodules(monkeypatch):
    monkeypatch.setattr(evo1_model, "InternVL3Embedder", FakeEmbedder)

    stage2_model = evo1_model.Evo1Model(make_config(training_stage="stage2"))
    stage2_model.set_finetune_flags()
    vlm = stage2_model.embedder.model
    assert all(p.requires_grad for p in vlm.language_model.parameters())
    assert all(p.requires_grad for p in vlm.vision_tower.parameters())
    assert all(p.requires_grad for p in vlm.multi_modal_projector.parameters())
    assert all(p.requires_grad for p in stage2_model.action_head.parameters())

    stage1_model = evo1_model.Evo1Model(make_config(training_stage="stage1"))
    stage1_model.set_finetune_flags()
    vlm = stage1_model.embedder.model
    assert not any(p.requires_grad for p in vlm.parameters())
    assert all(p.requires_grad for p in stage1_model.action_head.parameters())


def test_set_finetune_flags_fails_loudly_on_unknown_vlm_layout(monkeypatch):
    class LegacyLayoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Linear(2, 2)
            self.vision_model = nn.Linear(2, 2)  # trust_remote_code-era attribute name
            self.mlp1 = nn.Linear(2, 2)

    class FakeEmbedder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = LegacyLayoutModel()

    monkeypatch.setattr(evo1_model, "InternVL3Embedder", FakeEmbedder)
    model = evo1_model.Evo1Model(make_config(training_stage="stage2"))
    with pytest.raises(AttributeError, match="vision_tower"):
        model.set_finetune_flags()


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
    assert processed.dtype == torch.float32
    assert torch.allclose(processed[:, :6], action[:, :6])
    assert torch.equal(processed[:, 6], torch.tensor([1.0, -1.0]))


def test_evo1_postprocessor_returns_float32_for_bf16_actions():
    config = make_config()
    _preprocessor, postprocessor = make_evo1_pre_post_processors(config, dataset_stats=make_stats())

    processed = postprocessor(torch.zeros(2, MAX_ACTION_DIM, dtype=torch.bfloat16))
    assert processed.dtype == torch.float32


def test_evo1_processor_save_load_round_trip_applies_config_overrides(tmp_path):
    train_config = make_config()
    preprocessor, postprocessor = make_evo1_pre_post_processors(train_config, dataset_stats=make_stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_pre = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Simulate eval-time CLI overrides applied on top of the loaded pipelines.
    eval_config = make_config(binarize_gripper=True, postprocess_action_dim=ACTION_DIM)
    loaded_pre, loaded_post = reconcile_evo1_processors(eval_config, loaded_pre, loaded_post)

    assert loaded_pre.to_transition is evo1_batch_to_transition
    assert sum(isinstance(step, Evo1ActionProcessorStep) for step in loaded_post.steps) == 1
    action_step = next(step for step in loaded_post.steps if isinstance(step, Evo1ActionProcessorStep))
    assert action_step.binarize_gripper is True
    assert action_step.action_dim == ACTION_DIM
    # The float32 output dtype is part of the serialized pipeline itself.
    device_step = next(step for step in loaded_post.steps if isinstance(step, DeviceProcessorStep))
    assert device_step.float_dtype == "float32"

    # Non-observation extras (embodiment_id, ...) must survive the reloaded preprocessor.
    processed = loaded_pre(
        {
            "task": "pick the block",
            OBS_STATE: torch.zeros(STATE_DIM),
            f"{OBS_IMAGES}.front": torch.rand(3, 16, 16),
            "embodiment_id": torch.tensor([0]),
        }
    )
    assert "embodiment_id" in processed


def test_evo1_policy_forward_and_inference_use_batched_embedding(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config())
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
    assert action_chunk.dtype == torch.float32

    policy.reset()
    selected = policy.select_action(make_batch(include_action=False))
    assert selected.shape == (2, MAX_ACTION_DIM)


def test_evo1_forward_masks_padded_action_timesteps(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config())

    batch = make_batch(include_action=True)
    batch[ACTION] = torch.ones(2, CHUNK_SIZE, ACTION_DIM)
    # Give the padded (past-episode-end) timestep a huge value: if it leaked into the loss, the
    # loss would blow up far beyond 1.0.
    batch[ACTION][:, -1, :] = 100.0
    batch["action_is_pad"] = torch.zeros(2, CHUNK_SIZE, dtype=torch.bool)
    batch["action_is_pad"][:, -1] = True

    loss, metrics = policy.forward(batch)

    # DummyEvo1Model predicts zero velocity and zero noise, so each active element contributes
    # (0 - action)^2 = 1.0 for the in-episode ones-valued actions.
    assert metrics["active_action_dims"] == ACTION_DIM * (CHUNK_SIZE - 1)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_evo1_select_action_queue_orders_steps_and_repredicts(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", ChunkCountingDummyModel)
    policy = modeling_evo1.Evo1Policy(make_config(n_action_steps=CHUNK_SIZE))

    batch = make_batch(include_action=False)
    first = policy.select_action(batch)
    second = policy.select_action(batch)
    third = policy.select_action(batch)

    # First chunk provides steps 10, 11 in order; the third call triggers a fresh prediction (20).
    assert torch.all(first == 10.0)
    assert torch.all(second == 11.0)
    assert torch.all(third == 20.0)
    assert policy.model.chunks_predicted == 2


def test_evo1_predict_action_chunk_rejects_rtc_kwargs_without_rtc_config(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config())
    with pytest.raises(RuntimeError, match="RTC"):
        policy.predict_action_chunk(make_batch(include_action=False), inference_delay=2)


def test_evo1_rtc_processor_wiring(monkeypatch):
    monkeypatch.setattr(evo1_model, "InternVL3Embedder", FakeEmbedder)
    policy = modeling_evo1.Evo1Policy(make_config())
    assert policy.rtc_processor is None
    assert policy.model.rtc_processor is None

    # The RTC rollout backend assigns rtc_config after loading and re-inits the processor.
    policy.config.rtc_config = RTCConfig(execution_horizon=CHUNK_SIZE)
    policy.init_rtc_processor()
    assert isinstance(policy.rtc_processor, RTCProcessor)
    assert policy.model.rtc_processor is policy.rtc_processor

    # RTC drives predict_action_chunk directly; the select_action queue path is unsupported.
    with pytest.raises(AssertionError, match="select_action"):
        policy.select_action(make_batch(include_action=False))


def test_flowmatching_rtc_guidance_pulls_prefix_toward_previous_chunk():
    head = make_flowmatching_head(num_inference_timesteps=16)
    processor = RTCProcessor(RTCConfig(execution_horizon=CHUNK_SIZE))
    fused = torch.randn(2, 4, EMBED_DIM)
    state = torch.randn(2, STATE_DIM)
    action_mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)
    prev_chunk = torch.tensor([0.7, -0.4, 0.2]).expand(2, CHUNK_SIZE, ACTION_DIM).contiguous()

    torch.manual_seed(0)
    unguided = head.get_action(fused, state=state, action_mask=action_mask)
    unguided = unguided.view(2, CHUNK_SIZE, ACTION_DIM)
    torch.manual_seed(0)
    guided = head.get_action(
        fused,
        state=state,
        action_mask=action_mask,
        inference_delay=1,
        prev_chunk_left_over=prev_chunk,
        rtc_processor=processor,
    )
    guided = guided.view(2, CHUNK_SIZE, ACTION_DIM)

    # The frozen prefix (first inference_delay steps) must land far closer to the previous chunk
    # than the unguided sample from the same noise does.
    guided_dist = (guided[:, 0] - prev_chunk[:, 0]).abs().mean()
    unguided_dist = (unguided[:, 0] - prev_chunk[:, 0]).abs().mean()
    assert guided_dist < 0.5 * unguided_dist
    assert torch.isfinite(guided).all()


def test_flowmatching_rtc_first_chunk_without_leftover_matches_unguided():
    head = make_flowmatching_head(num_inference_timesteps=4)
    processor = RTCProcessor(RTCConfig(execution_horizon=CHUNK_SIZE))
    fused = torch.randn(2, 4, EMBED_DIM)
    state = torch.randn(2, STATE_DIM)
    action_mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)

    torch.manual_seed(0)
    unguided = head.get_action(fused, state=state, action_mask=action_mask)
    torch.manual_seed(0)
    first_chunk = head.get_action(
        fused,
        state=state,
        action_mask=action_mask,
        inference_delay=2,
        prev_chunk_left_over=None,
        rtc_processor=processor,
    )

    assert torch.allclose(unguided, first_chunk)


def test_evo1_missing_configured_camera_needs_empty_cameras_budget(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    batch = make_batch(include_action=False)  # only provides the front camera

    two_camera_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    strict_policy = modeling_evo1.Evo1Policy(make_config(input_features=dict(two_camera_features)))
    with pytest.raises(ValueError, match="empty_cameras"):
        strict_policy._collect_image_batches(batch)

    # empty_cameras adds placeholder camera features that are never present in the batch; they
    # become masked-out views instead of crashing with a KeyError.
    padded_policy = modeling_evo1.Evo1Policy(make_config(empty_cameras=1))
    assert len(padded_policy.config.image_features) == 2
    camera_images, image_masks = padded_policy._collect_image_batches(batch)
    assert len(camera_images) == 1
    assert image_masks.tolist() == [[True, False], [True, False]]


def test_stage1_frozen_vlm_embeddings_do_not_track_gradients(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config(training_stage="stage1"))
    policy.train()

    image_batches, image_masks = policy._collect_image_batches(make_batch(include_action=False))
    fused_tokens, context_mask = policy._compute_fused_tokens(["pick", "place"], image_batches, image_masks)

    assert policy.model.grad_enabled_calls == [False]
    assert policy.model.embedder_training_calls == [False]
    assert not fused_tokens.requires_grad
    assert context_mask is not None
    assert policy.model.embedder.training is False


def test_stage2_vlm_embeddings_track_gradients(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config(training_stage="stage2"))
    policy.train()

    image_batches, image_masks = policy._collect_image_batches(make_batch(include_action=False))
    fused_tokens, _context_mask = policy._compute_fused_tokens(["pick", "place"], image_batches, image_masks)

    assert policy.model.grad_enabled_calls == [True]
    assert policy.model.embedder_training_calls == [True]
    assert fused_tokens.requires_grad


def test_collect_image_batches_handles_unbatched_chw(monkeypatch):
    # Regression for an issue where batch_size was read from shape[0] before normalizing
    # per-camera tensor dims, so an unbatched (C, H, W) input was treated as batch_size=C.
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config())
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


def test_evo1_state_mask_zeroes_masked_dims(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    policy = modeling_evo1.Evo1Policy(make_config())
    batch = {
        OBS_STATE: torch.ones(2, STATE_DIM),
        "state_mask": torch.tensor([[True, True, False, False]] * 2),
    }

    states, mask = policy._prepare_state(batch)

    assert torch.all(states[:, :2] == 1.0)
    assert torch.all(states[:, 2:] == 0.0)
    assert mask[:, :2].all()
    assert not mask[:, 2:].any()


def test_evo1_action_mask_accepts_chunk_size_one(monkeypatch):
    monkeypatch.setattr(modeling_evo1, "Evo1Model", DummyEvo1Model)
    config = make_config(chunk_size=1, n_action_steps=1)
    policy = modeling_evo1.Evo1Policy(config)
    batch = make_batch(include_action=True)
    batch[ACTION] = torch.randn(2, ACTION_DIM)
    batch["action_mask"] = torch.ones(2, ACTION_DIM, dtype=torch.bool)

    actions, action_mask = policy._prepare_actions(batch)

    assert actions.shape == (2, 1, MAX_ACTION_DIM)
    assert action_mask.shape == (2, 1, MAX_ACTION_DIM)
    assert action_mask[:, :, :ACTION_DIM].all()
    assert not action_mask[:, :, ACTION_DIM:].any()


def test_flowmatching_state_encoder_for_horizon_one():
    head = make_flowmatching_head(action_dim=ACTION_DIM, horizon=1)

    assert head.state_encoder is not None
    pred_velocity, noise = head(
        torch.randn(2, 4, EMBED_DIM),
        state=torch.randn(2, STATE_DIM),
        actions_gt=torch.randn(2, 1, ACTION_DIM),
        action_mask=torch.ones(2, 1, ACTION_DIM, dtype=torch.bool),
    )

    assert pred_velocity.shape == (2, ACTION_DIM)
    assert noise.shape == (2, 1, ACTION_DIM)


def test_flowmatching_get_action_real_path_respects_action_mask():
    torch.manual_seed(0)
    head = make_flowmatching_head()

    action_mask = torch.zeros(2, ACTION_DIM, dtype=torch.bool)
    action_mask[:, :2] = True
    actions = head.get_action(
        torch.randn(2, 4, EMBED_DIM),
        state=torch.randn(2, STATE_DIM),
        action_mask=action_mask,
    )

    assert actions.shape == (2, CHUNK_SIZE * ACTION_DIM)
    assert torch.isfinite(actions).all()
    action_seq = actions.view(2, CHUNK_SIZE, ACTION_DIM)
    assert torch.all(action_seq[..., 2] == 0.0)


def test_flowmatching_context_mask_blocks_masked_context_tokens():
    head = make_flowmatching_head()
    state = torch.randn(2, STATE_DIM)
    action_mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)
    fused = torch.randn(2, 4, EMBED_DIM)
    context_mask = torch.ones(2, 4, dtype=torch.bool)
    context_mask[:, -1] = False
    corrupted = fused.clone()
    corrupted[:, -1] = 1e4

    torch.manual_seed(0)
    reference = head.get_action(fused, state=state, action_mask=action_mask, context_mask=context_mask)
    torch.manual_seed(0)
    with_garbage = head.get_action(corrupted, state=state, action_mask=action_mask, context_mask=context_mask)

    assert torch.allclose(reference, with_garbage)


def test_flowmatching_head_accepts_pooled_2d_context():
    head = make_flowmatching_head()
    pred_velocity, noise = head(
        torch.randn(2, EMBED_DIM),  # pooled (B, E) context from return_cls_only
        state=torch.randn(2, STATE_DIM),
        actions_gt=torch.randn(2, CHUNK_SIZE, ACTION_DIM),
        action_mask=torch.ones(2, CHUNK_SIZE, ACTION_DIM, dtype=torch.bool),
    )
    assert pred_velocity.shape == (2, CHUNK_SIZE * ACTION_DIM)

    actions = head.get_action(
        torch.randn(2, EMBED_DIM),
        state=torch.randn(2, STATE_DIM),
        action_mask=torch.ones(2, ACTION_DIM, dtype=torch.bool),
    )
    assert actions.shape == (2, CHUNK_SIZE * ACTION_DIM)


def test_flowmatching_rejects_out_of_range_embodiment_ids():
    head = make_flowmatching_head(num_categories=2)
    with pytest.raises(ValueError, match="num_categories"):
        head.get_action(
            torch.randn(2, 4, EMBED_DIM),
            state=torch.randn(2, STATE_DIM),
            action_mask=torch.ones(2, ACTION_DIM, dtype=torch.bool),
            embodiment_id=torch.tensor([0, 5]),
        )


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
    # Absent views (indices 1, 2) are zero images, normalized to the constant -mean/std.
    expected_pad = (-mean / std).view(1, 3, 1, 1)
    for view in (1, 2):
        assert torch.allclose(
            grouped[:, view], expected_pad.expand(batch_size, 3, image_size, image_size), atol=1e-5
        )
    # The present view is genuinely different from the constant pad value.
    assert not torch.allclose(
        grouped[:, 0], expected_pad.expand(batch_size, 3, image_size, image_size), atol=1e-3
    )
