# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for MolmoAct2's LeRobot policy interface."""

# ruff: noqa: E402

from __future__ import annotations

import copy
import json
import math
import os
import time
from collections import deque
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.optim import load_optimizer_state, save_optimizer_state
from lerobot.policies import get_policy_class, make_policy_config
from lerobot.policies.molmoact2 import (
    modeling_molmoact2 as molmoact2_modeling,
    processor_molmoact2 as molmoact2_processor,
)
from lerobot.policies.molmoact2.configuration_molmoact2 import (
    MolmoAct2AdamW,
    MolmoAct2AdamWConfig,
    MolmoAct2Config,
    MolmoAct2CosineWithWarmupSchedulerConfig,
)
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    MolmoAct2Policy,
    _apply_action_chunk_padding_mask,
    _apply_action_dim_padding_mask,
    _call_module_without_gradient_checkpointing_layer,
    _combine_rollout_seeds,
    _position_ids_from_attention_mask,
)
from lerobot.policies.molmoact2.molmoact2_hf_model import (
    modeling_molmoact2 as hf_molmoact2_modeling,
)
from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
    ActionExpert,
    ActionExpertModulation,
    ActionExpertRMSNorm,
    MolmoAct2RMSNorm,
    MolmoAct2RotaryEmbedding,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (
    MolmoAct2ActionFrameTransformStep,
    MolmoAct2ClampNormalizedProcessorStep,
    MolmoAct2MaskedNormalizerProcessorStep,
    MolmoAct2MaskedUnnormalizerProcessorStep,
    MolmoAct2PackInputsProcessorStep,
    MolmoAct2StateFrameTransformStep,
    _add_gripper_masks_to_stats,
    _build_discrete_state_string,
    _load_hf_norm_stats_for_tag,
    _normalize_question_text,
    infer_molmoact2_max_sequence_length,
    make_molmoact2_pre_post_processors,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_molmoact2_policy_registration():
    cfg = make_policy_config("molmoact2", checkpoint_path="/tmp/not-a-real-checkpoint")

    assert cfg.type == "molmoact2"
    assert cfg.action_mode == "continuous"
    assert cfg.normalize_gripper is False
    assert cfg.enable_knowledge_insulation is False
    assert cfg.freeze_embedding is True
    assert cfg.per_episode_seed is False
    assert cfg.eval_seed is None
    assert cfg.normalize_language is True
    assert cfg.dtype == "bfloat16"
    assert cfg.llm_residual_dropout == 0.1
    assert not hasattr(cfg, "model_dtype")
    assert cfg.get_scheduler_preset().num_decay_steps == 30_000
    assert cfg.action_delta_indices == list(range(cfg.chunk_size))
    assert get_policy_class("molmoact2") is MolmoAct2Policy


def test_molmoact2_scheduler_warmup_is_continuous_with_post_warmup_cosine():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 5e-5}])
    config = MolmoAct2CosineWithWarmupSchedulerConfig(
        num_warmup_steps=200,
        num_decay_steps=600,
        peak_lr=1e-5,
        decay_lr=1e-6,
    )
    scheduler = config.build(optimizer, num_training_steps=600)

    # LambdaLR index k - 1 is the LR used by optimizer update k. Official
    # MolmoAct2 increments global_step before selecting that update's LR.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 / 200)
    assert scheduler.lr_lambdas[0](0) == pytest.approx(1 / 200)
    assert scheduler.lr_lambdas[0](99) == pytest.approx(0.5)
    assert scheduler.lr_lambdas[0](198) == pytest.approx(0.995)
    assert scheduler.lr_lambdas[0](199) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](399) == pytest.approx(0.55)
    assert scheduler.lr_lambdas[0](599) == pytest.approx(0.1)


def test_molmoact2_scheduler_preserves_configured_clock_when_training_stops_early():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    config = MolmoAct2CosineWithWarmupSchedulerConfig(
        num_warmup_steps=200,
        num_decay_steps=30_000,
        peak_lr=1e-5,
        decay_lr=1e-6,
    )
    scheduler = config.build(optimizer, num_training_steps=3_000)

    # A 3K diagnostic stop must remain at the corresponding point on the final
    # 30K clock; it must not be compressed to the 0.1 decay floor.
    expected_multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (3_000 - 200) / (30_000 - 200)))
    assert scheduler.lr_lambdas[0](2_999) == pytest.approx(expected_multiplier)
    assert scheduler.lr_lambdas[0](2_999) > 0.97
    assert scheduler.lr_lambdas[0](29_999) == pytest.approx(0.1)


def test_molmoact2_scheduler_checkpoint_resume_matches_uninterrupted_updates():
    config = MolmoAct2CosineWithWarmupSchedulerConfig(
        num_warmup_steps=7,
        num_decay_steps=41,
        peak_lr=1e-3,
        decay_lr=1e-4,
    )

    def make_training_state(parameter_value):
        parameter = torch.nn.Parameter(parameter_value.clone())
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.9, 0.95), eps=1e-6)
        scheduler = config.build(optimizer, num_training_steps=41)
        return parameter, optimizer, scheduler

    def run_updates(parameter, optimizer, scheduler, start, stop):
        for update_index in range(start, stop):
            parameter.grad = torch.tensor(
                [0.25 + update_index / 128, -0.5 + (update_index % 5) / 32],
                dtype=parameter.dtype,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    initial = torch.tensor([0.75, -0.125])
    full_parameter, full_optimizer, full_scheduler = make_training_state(initial)
    run_updates(full_parameter, full_optimizer, full_scheduler, 0, 31)

    split_parameter, split_optimizer, split_scheduler = make_training_state(initial)
    run_updates(split_parameter, split_optimizer, split_scheduler, 0, 13)
    optimizer_state = copy.deepcopy(split_optimizer.state_dict())
    scheduler_state = copy.deepcopy(split_scheduler.state_dict())

    resumed_parameter, resumed_optimizer, resumed_scheduler = make_training_state(split_parameter.detach())
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    run_updates(resumed_parameter, resumed_optimizer, resumed_scheduler, 13, 31)

    assert torch.equal(full_parameter, resumed_parameter)
    assert full_scheduler.state_dict() == resumed_scheduler.state_dict()
    full_state = full_optimizer.state[full_parameter]
    resumed_state = resumed_optimizer.state[resumed_parameter]
    assert full_state.keys() == resumed_state.keys()
    for name in full_state:
        assert torch.equal(full_state[name], resumed_state[name])


def test_lerobot_checkpoint_routes_processors_through_molmoact2_factory():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(pretrained_path="/tmp/molmoact2-checkpoint")

    policy._route_pretrained_processors_through_molmoact2_factory()

    assert policy.config.pretrained_path is None
    assert policy.config._molmoact2_processor_pretrained_path == "/tmp/molmoact2-checkpoint"


def test_pretrained_molmoact2_processors_use_masked_override_keys(monkeypatch):
    calls = []
    preprocessor = object()
    postprocessor = object()

    def fake_from_pretrained(cls, **kwargs):
        del cls
        calls.append(kwargs)
        if kwargs["config_filename"] == "policy_preprocessor.json":
            return preprocessor
        return postprocessor

    monkeypatch.setattr(
        molmoact2_processor.PolicyProcessorPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    config = MolmoAct2Config(device="cuda")
    config._molmoact2_processor_pretrained_path = "/tmp/molmoact2-checkpoint"

    loaded_preprocessor, loaded_postprocessor = make_molmoact2_pre_post_processors(config)

    assert loaded_preprocessor is preprocessor
    assert loaded_postprocessor is postprocessor
    assert calls[0]["pretrained_model_name_or_path"] == "/tmp/molmoact2-checkpoint"
    assert set(calls[0]["overrides"]) == {
        "device_processor",
        "molmoact2_masked_normalizer",
    }
    assert set(calls[1]["overrides"]) == {"molmoact2_masked_unnormalizer"}


def test_molmoact2_optimizer_preset_uses_component_clipping():
    cfg = MolmoAct2Config(optimizer_grad_clip_norm=1.25)

    preset = cfg.get_optimizer_preset()

    assert isinstance(preset, MolmoAct2AdamWConfig)
    assert preset.type == "molmoact2_adamw"
    assert preset.grad_clip_norm == 0.0
    assert preset.group_grad_clip_norm == 1.25


def test_molmoact2_policy_optimizer_groups_keep_legacy_checkpoint_schema():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        train_mode_vlm="fft",
        optimizer_lr=1e-5,
        optimizer_vit_lr=2e-5,
        optimizer_connector_lr=3e-5,
        optimizer_action_expert_lr=4e-5,
    )
    policy.llm = torch.nn.Linear(2, 2)
    policy.vision = torch.nn.Linear(2, 2)
    policy.image_projector = torch.nn.Linear(2, 2)
    policy.action_expert = torch.nn.Linear(2, 2)

    groups = policy.get_optim_params()

    assert [group["lr"] for group in groups] == [1e-5, 2e-5, 3e-5, 4e-5]
    assert all("group_name" not in group for group in groups)


def test_molmoact2_additional_vocabulary_uses_official_connector_optimizer_group():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        train_mode_vlm="fft",
        optimizer_lr=1e-5,
        optimizer_vit_lr=2e-5,
        optimizer_connector_lr=3e-5,
        optimizer_action_expert_lr=4e-5,
    )
    policy.llm = torch.nn.Linear(2, 2)
    policy.transformer = torch.nn.Module()
    policy.transformer.wte = torch.nn.Module()
    policy.transformer.wte.register_parameter(
        "new_embedding",
        torch.nn.Parameter(torch.zeros(128, 2)),
    )
    policy.vision = torch.nn.Linear(2, 2)
    policy.image_projector = torch.nn.Linear(2, 2)
    policy.action_expert = torch.nn.Linear(2, 2)

    groups = policy.get_optim_params()
    new_embedding = policy.transformer.wte.new_embedding
    memberships = [any(param is new_embedding for param in group["params"]) for group in groups]

    assert memberships == [False, False, True, False]
    assert groups[2]["lr"] == pytest.approx(policy.config.optimizer_connector_lr)


def test_molmoact2_lora_uses_published_uniform_adapter_learning_rate():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        train_mode_vlm="lora",
        optimizer_lr=1e-5,
        optimizer_vit_lr=5e-6,
        optimizer_connector_lr=5e-6,
        optimizer_action_expert_lr=5e-5,
    )
    policy.llm = torch.nn.Linear(2, 2)
    policy.vision = torch.nn.Linear(2, 2)
    policy.image_projector = torch.nn.Linear(2, 2)
    policy.action_expert = torch.nn.Linear(2, 2)

    groups = policy.get_optim_params()

    assert [group["lr"] for group in groups] == [5e-5, 5e-5, 5e-5, 5e-5]


def test_molmoact2_optimizer_clips_each_component_independently():
    llm = torch.nn.Parameter(torch.zeros(2))
    action_expert = torch.nn.Parameter(torch.zeros(2))
    optimizer = MolmoAct2AdamWConfig(
        lr=0.0,
        group_grad_clip_norm=1.0,
    ).build(
        [
            {"params": [llm], "group_name": "llm"},
            {"params": [action_expert], "group_name": "action_expert"},
        ]
    )
    assert isinstance(optimizer, MolmoAct2AdamW)
    llm.grad = torch.tensor([3.0, 4.0])
    action_expert.grad = torch.tensor([0.0, 5.0])

    optimizer.step()

    assert torch.linalg.vector_norm(llm.grad).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(action_expert.grad).item() == pytest.approx(1.0)
    combined_norm = torch.linalg.vector_norm(torch.cat((llm.grad, action_expert.grad))).item()
    assert combined_norm == pytest.approx(2**0.5)


def test_molmoact2_optimizer_skips_entire_step_for_nonfinite_component():
    finite_component = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    nonfinite_component = torch.nn.Parameter(torch.tensor([0.75, -1.25]))
    optimizer = MolmoAct2AdamWConfig(
        lr=1e-3,
        group_grad_clip_norm=1.0,
    ).build(
        [
            {"params": [finite_component], "group_name": "llm"},
            {"params": [nonfinite_component], "group_name": "action_expert"},
        ]
    )
    original_params = [finite_component.detach().clone(), nonfinite_component.detach().clone()]
    finite_component.grad = torch.tensor([3.0, 4.0])
    nonfinite_component.grad = torch.tensor([float("nan"), 1.0])

    optimizer.step()

    assert torch.equal(finite_component, original_params[0])
    assert torch.equal(nonfinite_component, original_params[1])
    assert finite_component not in optimizer.state
    assert nonfinite_component not in optimizer.state
    assert finite_component.grad is None
    assert nonfinite_component.grad is None


def test_molmoact2_optimizer_nonfinite_step_does_not_advance_existing_state():
    finite_component = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    nonfinite_component = torch.nn.Parameter(torch.tensor([0.75, -1.25]))
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=1.0).build(
        [
            {"params": [finite_component], "group_name": "llm"},
            {"params": [nonfinite_component], "group_name": "action_expert"},
        ]
    )
    for param in (finite_component, nonfinite_component):
        param.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    params_before_skip = [finite_component.detach().clone(), nonfinite_component.detach().clone()]
    state_before_skip = {
        param: {name: value.detach().clone() for name, value in optimizer.state[param].items()}
        for param in (finite_component, nonfinite_component)
    }
    finite_component.grad = torch.tensor([0.5, -0.25])
    nonfinite_component.grad = torch.tensor([float("inf"), 0.0])

    optimizer.step()

    for param, expected_param in zip(
        (finite_component, nonfinite_component), params_before_skip, strict=True
    ):
        assert torch.equal(param, expected_param)
        assert param.grad is None
        assert optimizer.state[param].keys() == state_before_skip[param].keys()
        for state_name, expected_value in state_before_skip[param].items():
            assert torch.equal(optimizer.state[param][state_name], expected_value)


def test_molmoact2_optimizer_state_round_trip_preserves_component_groups(tmp_path):
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    groups = [
        {"params": [params[0]], "group_name": "llm"},
        {"params": [params[1]], "group_name": "action_expert"},
    ]
    preset = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=1.0)
    optimizer = preset.build(groups)
    for param in params:
        param.grad = torch.tensor([3.0, 4.0])
    optimizer.step()
    save_optimizer_state(optimizer, tmp_path)

    restored_params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    restored = preset.build(
        [
            {"params": [restored_params[0]], "group_name": "llm"},
            {"params": [restored_params[1]], "group_name": "action_expert"},
        ]
    )
    load_optimizer_state(restored, tmp_path)

    assert [group["group_name"] for group in restored.param_groups] == ["llm", "action_expert"]
    for old_param, restored_param in zip(params, restored_params, strict=True):
        assert torch.equal(optimizer.state[old_param]["exp_avg"], restored.state[restored_param]["exp_avg"])
        assert torch.equal(
            optimizer.state[old_param]["exp_avg_sq"], restored.state[restored_param]["exp_avg_sq"]
        )
        restored_param.grad = torch.tensor([3.0, 4.0])
    restored.step()
    for restored_param in restored_params:
        assert torch.linalg.vector_norm(restored_param.grad).item() == pytest.approx(1.0)


def _optimizer_effective_parameter(
    optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter
) -> torch.Tensor:
    compensation = optimizer.state.get(parameter, {}).get("compensation")
    if compensation is None:
        return parameter.detach().float()
    return parameter.detach().float() + compensation.detach().float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_molmoact2_optimizer_matches_native_adamw_or_fp32_oracle(dtype):
    params = [
        torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=dtype)),
        torch.nn.Parameter(torch.tensor([0.75, -1.25], dtype=dtype)),
    ]
    native_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    oracle_params = [torch.nn.Parameter(param.detach().float().clone()) for param in params]
    kwargs = {"lr": 3e-3, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.2}
    optimizer = MolmoAct2AdamWConfig(**kwargs, group_grad_clip_norm=100.0).build(params)
    native_optimizer = torch.optim.AdamW(native_params, **kwargs)
    oracle_optimizer = torch.optim.AdamW(oracle_params, **kwargs)

    assert optimizer.param_groups[0]["foreach"] is None
    assert optimizer.param_groups[0]["fused"] is None
    for step in range(8):
        gradients = [
            torch.tensor([0.125 + step / 16, -0.375], dtype=dtype),
            torch.tensor([-0.25, 0.5 - step / 32], dtype=dtype),
        ]
        for param, native_param, oracle_param, gradient in zip(
            params, native_params, oracle_params, gradients, strict=True
        ):
            param.grad = gradient.clone()
            native_param.grad = gradient.clone()
            oracle_param.grad = gradient.float()
        optimizer.step()
        native_optimizer.step()
        oracle_optimizer.step()

    for param, native_param, oracle_param in zip(params, native_params, oracle_params, strict=True):
        if dtype == torch.float32:
            assert torch.equal(param, native_param)
            assert "compensation" not in optimizer.state[param]
        else:
            assert optimizer.state[param]["compensation"].dtype == torch.bfloat16
            compensated_error = torch.linalg.vector_norm(
                _optimizer_effective_parameter(optimizer, param) - oracle_param.detach()
            )
            native_error = torch.linalg.vector_norm(native_param.detach().float() - oracle_param.detach())
            assert compensated_error < native_error
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                optimizer.state[param][state_name],
                native_optimizer.state[native_param][state_name],
            )


def test_molmoact2_optimizer_mixed_dtype_uses_storage_dtype_for_state():
    bfloat16_param = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16))
    float32_param = torch.nn.Parameter(torch.tensor([0.75, -1.25], dtype=torch.float32))
    optimizer = MolmoAct2AdamWConfig(lr=3e-3, group_grad_clip_norm=100.0).build(
        [bfloat16_param, float32_param]
    )
    bfloat16_param.grad = torch.tensor([0.25, -0.75], dtype=torch.bfloat16)
    float32_param.grad = torch.tensor([0.125, -0.375], dtype=torch.float32)

    optimizer.step()

    for state_name in ("exp_avg", "exp_avg_sq"):
        assert optimizer.state[bfloat16_param][state_name].dtype == torch.bfloat16
        assert optimizer.state[float32_param][state_name].dtype == torch.float32
    assert optimizer.state[bfloat16_param]["compensation"].dtype == torch.bfloat16
    assert "compensation" not in optimizer.state[float32_param]


def test_molmoact2_optimizer_mixed_dtype_matches_native_adamw():
    params = [
        torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.tensor([0.75, -1.25], dtype=torch.float32)),
    ]
    native_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    bfloat16_oracle = torch.nn.Parameter(params[0].detach().float())
    kwargs = {"lr": 3e-3, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.2}
    optimizer = MolmoAct2AdamWConfig(**kwargs, group_grad_clip_norm=100.0).build(params)
    native_optimizer = torch.optim.AdamW(native_params, **kwargs)
    oracle_optimizer = torch.optim.AdamW([bfloat16_oracle], **kwargs)

    for step in range(4):
        gradients = [
            torch.tensor([0.25, -0.75], dtype=torch.bfloat16),
            torch.tensor([0.125 + step, -0.375], dtype=torch.float32),
        ]
        for param, native_param, gradient in zip(params, native_params, gradients, strict=True):
            param.grad = gradient.clone()
            native_param.grad = gradient.clone()
        bfloat16_oracle.grad = gradients[0].float()
        optimizer.step()
        native_optimizer.step()
        oracle_optimizer.step()

    compensated_error = torch.linalg.vector_norm(
        _optimizer_effective_parameter(optimizer, params[0]) - bfloat16_oracle.detach()
    )
    native_error = torch.linalg.vector_norm(native_params[0].detach().float() - bfloat16_oracle.detach())
    assert compensated_error < native_error
    assert torch.equal(params[1], native_params[1])
    assert "compensation" in optimizer.state[params[0]]
    assert "compensation" not in optimizer.state[params[1]]
    for param, native_param in zip(params, native_params, strict=True):
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                optimizer.state[param][state_name], native_optimizer.state[native_param][state_name]
            )


def test_molmoact2_optimizer_bfloat16_checkpoint_resume_is_bitwise(tmp_path):
    original_param = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16))
    preset = MolmoAct2AdamWConfig(
        lr=7e-4,
        betas=(0.7, 0.91),
        eps=1e-6,
        weight_decay=0.03,
        group_grad_clip_norm=100.0,
    )
    original_optimizer = preset.build([original_param])
    gradients = [
        torch.tensor([0.25 + index / 32, -0.5 + index / 64], dtype=torch.bfloat16) for index in range(12)
    ]

    for gradient in gradients[:5]:
        original_param.grad = gradient.clone()
        original_optimizer.step()
    save_optimizer_state(original_optimizer, tmp_path)

    resumed_param = torch.nn.Parameter(original_param.detach().clone())
    resumed_optimizer = preset.build([resumed_param])
    load_optimizer_state(resumed_optimizer, tmp_path)

    for gradient in gradients[5:]:
        original_param.grad = gradient.clone()
        resumed_param.grad = gradient.clone()
        original_optimizer.step()
        resumed_optimizer.step()

    assert torch.equal(original_param, resumed_param)
    assert original_optimizer.state[original_param].keys() == resumed_optimizer.state[resumed_param].keys()
    for state_name in original_optimizer.state[original_param]:
        assert torch.equal(
            original_optimizer.state[original_param][state_name],
            resumed_optimizer.state[resumed_param][state_name],
        )


def test_molmoact2_optimizer_loads_legacy_native_adamw_state():
    legacy_param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    legacy_optimizer = torch.optim.AdamW(
        [legacy_param],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-6,
    )
    legacy_param.grad = torch.ones_like(legacy_param)
    legacy_optimizer.step()

    resumed_param = torch.nn.Parameter(legacy_param.detach().clone())
    optimizer = MolmoAct2AdamWConfig(
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-6,
        group_grad_clip_norm=100.0,
    ).build([resumed_param])
    optimizer.load_state_dict(legacy_optimizer.state_dict())

    resumed_param.grad = torch.ones_like(resumed_param)
    optimizer.step()

    assert optimizer.state[resumed_param]["step"].item() == 2
    assert optimizer.state[resumed_param]["compensation"].dtype == torch.bfloat16
    assert optimizer.param_groups[0]["foreach"] is None
    assert optimizer.param_groups[0]["fused"] is None


def test_molmoact2_optimizer_grad_none_does_not_initialize_state():
    updated_param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    untouched_param = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16))
    optimizer = MolmoAct2AdamWConfig(
        lr=1e-3,
        weight_decay=0.1,
        group_grad_clip_norm=100.0,
    ).build([updated_param, untouched_param])
    updated_param.grad = torch.ones_like(updated_param)

    optimizer.step()

    assert updated_param in optimizer.state
    assert untouched_param not in optimizer.state
    assert torch.equal(untouched_param, torch.tensor([2.0], dtype=torch.bfloat16))


@pytest.mark.parametrize(("amsgrad", "maximize"), [(False, False), (True, True)])
def test_molmoact2_optimizer_preserves_native_advanced_semantics(amsgrad, maximize):
    params = [
        torch.nn.Parameter(torch.linspace(-0.75, 0.75, steps=size, dtype=torch.bfloat16))
        for size in (17, 9, 3)
    ]
    native_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    oracle_params = [torch.nn.Parameter(param.detach().float()) for param in params]
    kwargs = {
        "lr": 7e-4,
        "betas": (0.7, 0.91),
        "eps": 1e-6,
        "weight_decay": 0.03,
        "amsgrad": amsgrad,
        "maximize": maximize,
        "foreach": False,
    }
    optimizer = MolmoAct2AdamW(
        params,
        **kwargs,
        group_grad_clip_norm=1e6,
    )
    native_optimizer = torch.optim.AdamW(native_params, **kwargs)
    oracle_optimizer = torch.optim.AdamW(oracle_params, **kwargs)

    for step_index in range(7):
        gradients = [
            torch.linspace(
                -0.5 + step_index / 32,
                0.25 + step_index / 64,
                steps=param.numel(),
                dtype=torch.bfloat16,
            )
            for param in params
        ]
        if step_index % 2:
            gradients[1] = None
        for param, native_param, oracle_param, gradient in zip(
            params, native_params, oracle_params, gradients, strict=True
        ):
            param.grad = None if gradient is None else gradient.clone()
            native_param.grad = None if gradient is None else gradient.clone()
            oracle_param.grad = None if gradient is None else gradient.float()
        optimizer.step()
        native_optimizer.step()
        oracle_optimizer.step()

    for param, native_param, oracle_param in zip(params, native_params, oracle_params, strict=True):
        if param not in optimizer.state:
            assert native_param not in native_optimizer.state
            continue
        compensated_error = torch.linalg.vector_norm(
            _optimizer_effective_parameter(optimizer, param) - oracle_param.detach()
        )
        native_error = torch.linalg.vector_norm(native_param.detach().float() - oracle_param.detach())
        assert compensated_error < native_error
        assert optimizer.state[param]["compensation"].dtype == torch.bfloat16
        native_state_names = ("step", "exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ())
        for state_name in native_state_names:
            assert torch.equal(
                optimizer.state[param][state_name],
                native_optimizer.state[native_param][state_name],
            )


def test_molmoact2_optimizer_leaves_native_fast_path_flags_unset():
    params = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4))]
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=100.0).build(
        [{"params": [params[0]]}, {"params": [params[1]]}]
    )

    assert [group["foreach"] for group in optimizer.param_groups] == [None, None]
    assert [group["fused"] for group in optimizer.param_groups] == [None, None]


def test_molmoact2_optimizer_bfloat16_tracks_native_long_horizon():
    param = torch.nn.Parameter(torch.tensor([0.75, -0.5, 0.125], dtype=torch.bfloat16))
    native_param = torch.nn.Parameter(param.detach().clone())
    oracle_param = torch.nn.Parameter(param.detach().float())
    optimizer = MolmoAct2AdamWConfig(
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.01,
        group_grad_clip_norm=100.0,
    ).build([param])
    native_optimizer = torch.optim.AdamW(
        [native_param],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.01,
    )
    oracle_optimizer = torch.optim.AdamW(
        [oracle_param],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.01,
    )

    with torch.no_grad():
        for step in range(1, 1_001):
            gradient = torch.tensor(
                [
                    0.125 + (step % 17 - 8) / 256,
                    -0.25 + (step % 13 - 6) / 512,
                    0.0625 - (step % 11 - 5) / 1024,
                ],
                dtype=torch.bfloat16,
            )
            param.grad = gradient.clone()
            native_param.grad = gradient.clone()
            oracle_param.grad = gradient.float()
            optimizer.step()
            native_optimizer.step()
            oracle_optimizer.step()

    compensated_error = torch.linalg.vector_norm(
        _optimizer_effective_parameter(optimizer, param) - oracle_param.detach()
    )
    native_error = torch.linalg.vector_norm(native_param.detach().float() - oracle_param.detach())
    assert compensated_error < native_error * 0.1
    assert optimizer.state[param]["compensation"].dtype == torch.bfloat16
    for state_name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            optimizer.state[param][state_name], native_optimizer.state[native_param][state_name]
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required to inspect native foreach dispatch."
)
def test_molmoact2_optimizer_uses_native_cuda_foreach(monkeypatch):
    bfloat16_param = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=torch.bfloat16))
    float32_param = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=torch.float32))
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=100.0).build(
        [bfloat16_param, float32_param]
    )
    observed_foreach = []
    native_step = torch.optim.AdamW.step

    def record_native_step(native_optimizer, *args, **kwargs):
        observed_foreach.extend(group["foreach"] for group in native_optimizer.param_groups)
        return native_step(native_optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_native_step)
    bfloat16_param.grad = torch.ones_like(bfloat16_param)
    float32_param.grad = torch.ones_like(float32_param)

    optimizer.step()

    assert observed_foreach == [None]
    assert optimizer.param_groups[0]["foreach"] is None
    assert optimizer.param_groups[0]["fused"] is None
    assert torch.isfinite(bfloat16_param).all()
    assert torch.isfinite(float32_param).all()
    assert optimizer.state[bfloat16_param]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[bfloat16_param]["compensation"].dtype == torch.bfloat16
    assert optimizer.state[float32_param]["exp_avg"].dtype == torch.float32
    assert "compensation" not in optimizer.state[float32_param]


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_MOLMOACT2_OPTIMIZER_BENCHMARK") != "1",
    reason="Set LEROBOT_RUN_MOLMOACT2_OPTIMIZER_BENCHMARK=1 on a CUDA worker.",
)
def test_molmoact2_optimizer_native_foreach_microbenchmark():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MolmoAct2 optimizer microbenchmark.")
    device = torch.device("cuda")
    tensor_count = 256
    tensor_numel = 16 * 1024
    optimized_params = [
        torch.nn.Parameter(torch.ones(tensor_numel, device=device, dtype=torch.bfloat16))
        for _ in range(tensor_count)
    ]
    native_params = [torch.nn.Parameter(param.detach().clone()) for param in optimized_params]
    optimized = MolmoAct2AdamWConfig(
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-6,
        group_grad_clip_norm=1e6,
    ).build(optimized_params)
    native = torch.optim.AdamW(
        native_params,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.0,
    )
    for param in (*optimized_params, *native_params):
        param.grad = torch.full_like(param, 0.25)

    for _ in range(2):
        optimized.step()
        native.step()
    torch.cuda.synchronize()

    def elapsed_seconds(callable_step):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            callable_step()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    optimized_seconds = elapsed_seconds(optimized.step)
    native_seconds = elapsed_seconds(native.step)
    print(
        "MolmoAct2 AdamW microbenchmark: "
        f"compensated={optimized_seconds:.6f}s native_bf16={native_seconds:.6f}s "
        f"overhead={optimized_seconds / native_seconds:.2f}x"
    )
    assert optimized_seconds < native_seconds * 1.5


def test_molmoact2_checkpoint_download_ignores_remote_python(monkeypatch):
    import huggingface_hub

    download_kwargs = {}

    def fake_snapshot_download(**kwargs):
        download_kwargs.update(kwargs)
        return "/tmp/downloaded-molmoact2"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    checkpoint_location = molmoact2_modeling._resolve_checkpoint_location("allenai/MolmoAct2")

    assert checkpoint_location == "/tmp/downloaded-molmoact2"
    assert download_kwargs["ignore_patterns"] == ["*.py", "*.pyc", "__pycache__/*"]


def test_local_molmoact2_processor_forces_left_padding(monkeypatch, tmp_path):
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    tokenizer = SimpleNamespace(padding_side="right")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return tokenizer

    class FakeComponent:
        def __init__(self, **kwargs):
            del kwargs

    class FakeProcessor:
        def __init__(self, **kwargs):
            assert kwargs["tokenizer"].padding_side == "left"
            self.tokenizer = kwargs["tokenizer"]

    monkeypatch.setattr(molmoact2_processor, "Qwen2Tokenizer", FakeTokenizer)
    monkeypatch.setattr(molmoact2_processor, "MolmoAct2ImageProcessor", FakeComponent)
    monkeypatch.setattr(molmoact2_processor, "MolmoAct2VideoProcessor", FakeComponent)
    monkeypatch.setattr(molmoact2_processor, "MolmoAct2Processor", FakeProcessor)

    processor = molmoact2_processor._load_local_molmoact2_processor(str(tmp_path))

    assert processor.tokenizer is tokenizer
    assert tokenizer.padding_side == "left"


def test_molmoact2_insert_bos_preserves_mixed_left_padding():
    pad_token_id = 0
    bos_token_id = 99
    input_ids = np.array(
        [
            [pad_token_id, pad_token_id, 10, 11],
            [pad_token_id, 20, 21, 22],
        ]
    )
    attention_mask = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ]
    )

    output_ids, output_mask = molmoact2_processor.MolmoAct2Processor.insert_bos(
        None,
        input_ids,
        attention_mask,
        bos_token_id,
        pad_token_id,
    )

    np.testing.assert_array_equal(
        output_ids,
        np.array(
            [
                [pad_token_id, pad_token_id, bos_token_id, 10, 11],
                [pad_token_id, bos_token_id, 20, 21, 22],
            ]
        ),
    )
    np.testing.assert_array_equal(
        output_mask,
        np.array(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ]
        ),
    )


def test_molmoact2_insert_bos_preserves_pre_bos_bucket_alignment():
    input_ids = np.array([[0, 0, 0, 0, 0, 10, 11, 12]])
    attention_mask = np.array([[0, 0, 0, 0, 0, 1, 1, 1]])

    output_ids, output_mask = molmoact2_processor.MolmoAct2Processor.insert_bos(
        None,
        input_ids,
        attention_mask,
        bos_token_id=99,
        pad_token_id=0,
    )

    assert output_ids.shape == (1, 9)
    assert output_ids.shape[1] % 8 == 1
    np.testing.assert_array_equal(output_ids, np.array([[0, 0, 0, 0, 0, 99, 10, 11, 12]]))
    np.testing.assert_array_equal(output_mask, np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1]]))


def test_molmoact2_left_padding_uses_native_mask_relative_positions():
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    position_ids = _position_ids_from_attention_mask(attention_mask)

    assert torch.equal(
        position_ids,
        torch.tensor(
            [
                [0, 0, 0, 1, 2],
                [0, 0, 1, 2, 3],
            ],
            dtype=torch.long,
        ),
    )


def test_molmoact2_pack_inputs_emits_mask_relative_positions():
    class FakeProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def __call__(self, **kwargs):
            assert len(kwargs["text"]) == 2
            assert len(kwargs["images"]) == 2
            assert kwargs["padding"] is True
            assert kwargs["pad_to_multiple_of"] == 8
            return {
                "input_ids": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 10, 11, 12],
                        [0, 0, 0, 0, 0, 20, 21, 22, 23],
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
                "token_type_ids": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
            }

    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.processor = FakeProcessor()
    step.action_processor = None
    step.action_mode = "continuous"
    step.image_keys = [f"{OBS_IMAGES}.image"]
    step.allow_image_key_fallback = False
    step.setup_type = "single-arm tabletop"
    step.control_mode = "delta end effector pose"
    step.normalize_language = True
    step.add_setup_tokens = True
    step.add_control_tokens = True
    step.num_state_tokens = 256
    step.max_sequence_length = 64
    step.chunk_size = 10
    step.max_action_dim = 32
    step.env_action_dim = 7

    packed = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.zeros(2, 7),
                f"{OBS_IMAGES}.image": torch.zeros(2, 3, 4, 4),
            },
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick up block", "open drawer"]},
        }
    )

    complementary = packed[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["input_ids"].shape == (2, 9)
    assert complementary["input_ids"].shape[1] % 8 == 1
    assert complementary["token_type_ids"].shape == (2, 9)
    assert torch.equal(
        complementary["position_ids"],
        _position_ids_from_attention_mask(complementary["attention_mask"]),
    )


def test_joint_training_fallback_positions_ignore_left_padding():
    captured_attention_kwargs = []

    class DummyBackbone:
        def merge_visual_inputs(self, **kwargs):
            del kwargs
            return None, None

        def _build_native_attention_bias(self, **kwargs):
            captured_attention_kwargs.append(kwargs)
            return kwargs["attention_mask"]

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy._backbone = lambda: DummyBackbone()
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    _embeddings, _causal_mask, position_ids, _cache_position = policy._prepare_joint_training_backbone_inputs(
        {
            "inputs_embeds": torch.ones(2, 5, 4),
            "attention_mask": attention_mask,
            "token_type_ids": torch.ones(2, 5, dtype=torch.long),
        }
    )

    assert torch.equal(position_ids, _position_ids_from_attention_mask(attention_mask))
    assert captured_attention_kwargs[0]["token_type_ids"] is None


def test_vendored_text_model_fallback_positions_ignore_left_padding(monkeypatch):
    captured_position_ids = []
    config = hf_molmoact2_modeling.MolmoAct2TextConfig(
        vocab_size=32,
        additional_vocab_size=None,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    model = hf_molmoact2_modeling.MolmoAct2TextModel(config)

    original_rotary_forward = model.rotary_emb.forward

    def capture_rotary_positions(x, position_ids):
        captured_position_ids.append(position_ids.detach().clone())
        return original_rotary_forward(x, position_ids)

    monkeypatch.setattr(model.rotary_emb, "forward", capture_rotary_positions)
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    model(
        input_ids=torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [0, 4, 5, 6, 7],
            ]
        ),
        attention_mask=attention_mask,
        use_cache=False,
    )

    assert len(captured_position_ids) == 1
    assert torch.equal(captured_position_ids[0], _position_ids_from_attention_mask(attention_mask))


def test_continuous_generation_forwards_explicit_positions_to_prefill():
    captured_prefill_kwargs = {}

    class DummyActionExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_embed = torch.nn.Linear(1, 1, bias=False)

        def prepare_context(self, **kwargs):
            del kwargs
            return SimpleNamespace()

        def get_or_prepare_modulation_cache(self, timesteps, *, cache_key=None):
            del cache_key
            return [SimpleNamespace(conditioning=timestep) for timestep in timesteps]

        def forward_with_context(self, actions, timesteps, *, context, modulation=None):
            del timesteps, context, modulation
            return torch.zeros_like(actions)

    class DummyModel(hf_molmoact2_modeling.MolmoAct2Model):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                action_mode="continuous",
                action_expert_config=SimpleNamespace(num_layers=1),
                action_expert_depth_gate=False,
                action_start_token_id=None,
                action_end_token_id=None,
                eos_token_id=None,
                flow_matching_num_steps=1,
                mask_action_dim_padding=False,
                max_action_dim=2,
                max_action_horizon=3,
            )
            self.action_expert = DummyActionExpert()
            self.action_expert_depth_gate = None
            self.action_cuda_graph_manager = None

        def forward(self, **kwargs):
            captured_prefill_kwargs.update(kwargs)
            return SimpleNamespace(past_key_values=object())

        def _extract_kv_states(self, past_key_values):
            del past_key_values
            kv = torch.zeros(2, 5, 1)
            return [(kv, kv)]

    model = DummyModel()
    input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    position_ids = _position_ids_from_attention_mask(attention_mask)

    actions = model.generate_actions_from_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        generator=torch.Generator().manual_seed(0),
    )

    assert actions.shape == (2, 3, 2)
    assert torch.equal(captured_prefill_kwargs["position_ids"], position_ids)

    explicit_encoder_mask = attention_mask.bool()

    def reject_implicit_mask(*args, **kwargs):
        del args, kwargs
        raise AssertionError("explicit encoder mask was overwritten")

    model._get_encoder_attention_mask = reject_implicit_mask
    model.generate_actions_from_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        encoder_attention_mask=explicit_encoder_mask,
        generator=torch.Generator().manual_seed(0),
    )


def test_cached_ar_fallback_separates_physical_cache_and_logical_rope_positions():
    captured = {}

    class DummyCache:
        def get_seq_length(self):
            return 5

    class DummyManager:
        def can_use(self, next_input_ids, **kwargs):
            captured["manager_position_ids"] = kwargs["position_ids"].clone()
            return False

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = torch.nn.Embedding(32, 4)

        def forward(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                last_hidden_state=kwargs["inputs_embeds"],
                past_key_values=kwargs["past_key_values"],
            )

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = DummyTransformer()

    model = object.__new__(hf_molmoact2_modeling.MolmoAct2ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = DummyBackbone()
    model.depth_decode_cuda_graph_manager = DummyManager()
    cache = DummyCache()
    logical_position_ids = torch.tensor([[2, 3]], dtype=torch.long)

    model._run_ar_decode_step(
        torch.tensor([[7, 8]], dtype=torch.long),
        past_key_values=cache,
        attention_bias=torch.zeros(1, 1, 10, 10),
        position_ids=logical_position_ids,
    )

    assert torch.equal(captured["manager_position_ids"], logical_position_ids)
    assert torch.equal(captured["position_ids"], logical_position_ids)
    assert torch.equal(captured["cache_position"], torch.tensor([5, 6]))
    assert captured["attention_mask"].shape == (1, 1, 2, 7)


def test_cuda_graph_rope_selection_uses_logical_not_physical_position():
    cos_cache = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2)
    sin_cache = cos_cache + 100
    rotary = SimpleNamespace(_pos_cos_cache=cos_cache, _pos_sin_cache=sin_cache)
    transformer = SimpleNamespace(rotary_emb=rotary)
    backbone = SimpleNamespace(transformer=transformer)
    manager = hf_molmoact2_modeling.DepthDecodeCudaGraphManager(SimpleNamespace(model=backbone))
    cos = torch.empty(1, 1, 2)
    sin = torch.empty_like(cos)

    manager._select_depth_decode_rope(
        cos,
        sin,
        position_ids=torch.tensor([[2]], dtype=torch.long),
    )

    assert torch.equal(cos, cos_cache[0, :, 2:3, :])
    assert torch.equal(sin, sin_cache[0, :, 2:3, :])


def test_molmoact2_scheduler_auto_scales_to_training_steps():
    from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=0.001)
    config = CosineDecayWithWarmupSchedulerConfig(
        peak_lr=0.01,
        decay_lr=0.001,
        num_warmup_steps=10,
        num_decay_steps=100_000,
    )

    scheduler = config.build(optimizer, num_training_steps=100)
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == pytest.approx([0.0001])


def test_molmoact2_rollout_generator_uses_eval_seed_per_task():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(per_episode_seed=True, eval_seed=1000)
    policy._rollout_action_generator = None
    policy._rollout_task_key = None
    policy._rollout_index_for_task = -1

    policy.reset()
    first = policy._rollout_generator_for_inputs(
        {"task": ["pick", "pick", "pick"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_first = torch.Generator().manual_seed(_combine_rollout_seeds(first_seed=1000, batch_size=3))
    assert torch.allclose(torch.rand(4, generator=first), torch.rand(4, generator=expected_first))

    policy.reset()
    second = policy._rollout_generator_for_inputs(
        {"task": ["pick", "pick", "pick"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_second = torch.Generator().manual_seed(_combine_rollout_seeds(first_seed=1003, batch_size=3))
    assert torch.allclose(torch.rand(4, generator=second), torch.rand(4, generator=expected_second))

    policy.reset()
    new_task = policy._rollout_generator_for_inputs(
        {"task": ["place", "place", "place"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_new_task = torch.Generator().manual_seed(_combine_rollout_seeds(first_seed=1000, batch_size=3))
    assert torch.allclose(torch.rand(4, generator=new_task), torch.rand(4, generator=expected_new_task))


def test_molmoact2_gripper_mask_uses_feature_names(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    ACTION: {"names": {"motors": ["x", "gripper"]}},
                    OBS_STATE: {"names": {"motors": ["joint", "gripper"]}},
                }
            }
        ),
        encoding="utf-8",
    )
    dataset_meta = SimpleNamespace(root=tmp_path)
    stats = {
        ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]},
        OBS_STATE: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]},
    }

    masked_stats = _add_gripper_masks_to_stats(stats, dataset_meta, normalize_gripper=False)

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, False]
    assert masked_stats[OBS_STATE]["mask"] == [True, False]

    features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.QUANTILES,
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[5.0, 0.7]])},
        TransitionKey.ACTION: torch.tensor([[5.0, -0.7]]),
    }
    normalizer = MolmoAct2MaskedNormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=masked_stats,
    )
    normalized = normalizer(transition)

    assert torch.equal(normalized[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[0.0, 0.7]]))
    assert torch.equal(normalized[TransitionKey.ACTION], torch.tensor([[0.0, -0.7]]))

    with pytest.raises(ValueError, match="gripper values are not under \\[-1, 1\\]"):
        normalizer(
            {
                TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[5.0, 7.0]])},
                TransitionKey.ACTION: torch.tensor([[5.0, -0.7]]),
            }
        )

    unnormalizer = MolmoAct2MaskedUnnormalizerProcessorStep(
        features={ACTION: features[ACTION]},
        norm_map=norm_map,
        stats=masked_stats,
    )
    unnormalized = unnormalizer({TransitionKey.ACTION: torch.tensor([[0.0, -0.7]])})

    assert torch.equal(unnormalized[TransitionKey.ACTION], torch.tensor([[5.0, -0.7]]))


def test_molmoact2_gripper_mask_validates_dataset_stats(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps({"features": {ACTION: {"names": ["x", "gripper"]}}}),
        encoding="utf-8",
    )
    stats = {
        ACTION: {
            "min": [-0.5, -2.0],
            "max": [0.5, 0.5],
        }
    }

    with pytest.raises(ValueError, match="gripper values are not under \\[-1, 1\\]"):
        _add_gripper_masks_to_stats(stats, SimpleNamespace(root=tmp_path), normalize_gripper=False)

    masked_stats = _add_gripper_masks_to_stats(stats, SimpleNamespace(root=tmp_path), normalize_gripper=True)
    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, True]


def test_molmoact2_gripper_mask_fails_closed_for_ambiguous_feature_names():
    stats = {ACTION: {"q01": [0.0, 0.0], "q99": [1.0, 1.0]}}

    with pytest.raises(ValueError, match="cannot identify the gripper dimension"):
        _add_gripper_masks_to_stats(
            stats,
            None,
            normalize_gripper=False,
            dataset_feature_names={ACTION: ["actions"]},
        )


def test_molmoact2_explicit_norm_stats_path_and_mask(tmp_path):
    stats_path = tmp_path / "official_norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "metadata_by_tag": {
                    "libero": {
                        "action_stats": {
                            "q01": [-1.0, -1.0],
                            "q99": [1.0, 1.0],
                            "mask": [True, False],
                            "names": ["x", "gripper"],
                        },
                        "state_stats": {
                            "q01": [-2.0, -2.0],
                            "q99": [2.0, 2.0],
                            "mask": [True, False],
                            "names": ["x", "gripper"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    tagged_stats, metadata = _load_hf_norm_stats_for_tag(
        "/tmp/model-weights-are-deliberately-independent",
        revision=None,
        force_download=False,
        norm_tag="libero",
        norm_stats_path=str(stats_path),
    )

    assert tagged_stats[ACTION]["q01"] == [-1.0, -1.0]
    assert tagged_stats[ACTION]["mask"] == [True, False]
    assert tagged_stats[OBS_STATE]["mask"] == [True, False]
    assert metadata["action_stats"]["names"] == ["x", "gripper"]


def test_molmoact2_norm_tag_overrides_training_dataset_stats(monkeypatch):
    calls = []
    tagged_stats = {
        ACTION: {"q01": [-1.0, -1.0], "q99": [1.0, 1.0], "mask": [True, False]},
        OBS_STATE: {"q01": [-2.0, -2.0], "q99": [2.0, 2.0], "mask": [True, False]},
    }

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return tagged_stats, {"action_horizon": 10}

    monkeypatch.setattr(molmoact2_processor, "_load_hf_norm_stats_for_tag", fake_load)
    monkeypatch.setattr(MolmoAct2PackInputsProcessorStep, "__post_init__", lambda self: None)
    cfg = MolmoAct2Config(
        checkpoint_path="/tmp/generic-base-weights",
        norm_tag="libero",
        norm_stats_path="/tmp/official-libero-norm-stats.json",
        chunk_size=10,
        n_action_steps=10,
        image_keys=["observation.images.image"],
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    wrong_dataset_stats = {
        ACTION: {"q01": [10.0, 10.0], "q99": [20.0, 20.0]},
        OBS_STATE: {"q01": [30.0, 30.0], "q99": [40.0, 40.0]},
    }

    preprocessor, _ = make_molmoact2_pre_post_processors(
        cfg,
        dataset_stats=wrong_dataset_stats,
    )
    normalizer = next(
        step for step in preprocessor.steps if isinstance(step, MolmoAct2MaskedNormalizerProcessorStep)
    )

    assert len(calls) == 1
    assert calls[0][1]["norm_stats_path"] == cfg.norm_stats_path
    assert torch.equal(normalizer._tensor_stats[ACTION]["q01"], torch.tensor([-1.0, -1.0]))
    assert torch.equal(normalizer._tensor_stats[OBS_STATE]["q99"], torch.tensor([2.0, 2.0]))
    assert normalizer._tensor_stats[ACTION]["mask"].tolist() == [True, False]


def test_molmoact2_clamp_normalized_respects_masked_gripper_dims():
    step = MolmoAct2ClampNormalizedProcessorStep(
        normalization_masks={
            ACTION: [True, False],
            OBS_STATE: [True, False],
        }
    )
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[-2.0, 0.8]])},
        TransitionKey.ACTION: torch.tensor([[2.0, -0.8]]),
    }

    clamped = step(transition)

    assert torch.equal(clamped[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[-1.0, 0.8]]))
    assert torch.equal(clamped[TransitionKey.ACTION], torch.tensor([[1.0, -0.8]]))

    with pytest.raises(ValueError, match="gripper values are not under \\[-1, 1\\]"):
        step({TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.0, 1.2]])}})


def test_molmoact2_normalize_gripper_true_keeps_all_dims_normalized(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps({"features": {ACTION: {"names": ["x", "gripper"]}}}),
        encoding="utf-8",
    )
    stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        stats,
        SimpleNamespace(root=tmp_path),
        normalize_gripper=True,
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, True]


def test_molmoact2_uses_supplied_stats_with_repo_scoped_names(tmp_path):
    repo_root = tmp_path / "test-org" / "libero"
    (repo_root / "meta").mkdir(parents=True)
    (repo_root / "meta" / "info.json").write_text(
        json.dumps({"features": {ACTION: {"names": ["x", "gripper"]}}}),
        encoding="utf-8",
    )
    base_stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        base_stats,
        SimpleNamespace(root=tmp_path, repo_id="test-org/libero"),
        normalize_gripper=False,
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["q01"] == [0.0, 0.0]
    assert masked_stats[ACTION]["mask"] == [True, False]


def test_molmoact2_uses_config_feature_names_without_dataset_meta():
    base_stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        base_stats,
        None,
        normalize_gripper=False,
        dataset_feature_names={ACTION: ["x", "gripper"]},
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, False]


def test_molmoact2_processor_uses_available_visual_features_over_missing_metadata_keys(monkeypatch):
    monkeypatch.setattr(
        molmoact2_processor,
        "_load_hf_norm_stats_for_tag",
        lambda *args, **kwargs: (
            {},
            {"camera_keys": ["observation.images.image", "observation.images.wrist_image"]},
        ),
    )
    monkeypatch.setattr(MolmoAct2PackInputsProcessorStep, "__post_init__", lambda self: None)
    cfg = MolmoAct2Config(
        checkpoint_path="/tmp/not-a-real-checkpoint",
        norm_tag="libero",
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    preprocessor, _ = make_molmoact2_pre_post_processors(cfg)
    pack_step = next(
        step for step in preprocessor.steps if isinstance(step, MolmoAct2PackInputsProcessorStep)
    )

    assert pack_step.image_keys == ["observation.images.image", "observation.images.image2"]
    assert pack_step.allow_image_key_fallback is True


def test_molmoact2_metadata_image_keys_can_fall_back_to_observation_keys():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.image_keys = ["observation.images.image", "observation.images.wrist_image"]
    step.allow_image_key_fallback = True
    observation = {
        "observation.images.image": torch.zeros(3, 4, 4),
        "observation.images.image2": torch.zeros(3, 4, 4),
    }

    assert step._resolve_image_keys(observation) == ["observation.images.image", "observation.images.image2"]


def test_molmoact2_explicit_image_keys_stay_strict():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.image_keys = ["observation.images.image", "observation.images.wrist_image"]
    step.allow_image_key_fallback = False
    observation = {
        "observation.images.image": torch.zeros(3, 4, 4),
        "observation.images.image2": torch.zeros(3, 4, 4),
    }

    with pytest.raises(ValueError, match="wrist_image"):
        step._resolve_image_keys(observation)


def test_train_mode_vlm_lora_builds_policy_local_peft_config():
    pytest.importorskip("peft")
    policy_cfg = MolmoAct2Config(
        checkpoint_path="/tmp/not-a-real-checkpoint",
        device="cpu",
        train_mode_vlm="lora",
        lora_rank=64,
        push_to_hub=False,
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = policy_cfg

    peft_config = policy._build_inner_lora_config()

    assert peft_config.r == 64
    assert peft_config.target_modules == policy._get_inner_peft_targets()["target_modules"]
    assert peft_config.init_lora_weights is False
    assert not policy_cfg.use_peft


def test_cuda_graph_managers_are_inference_only():
    class DummyManager:
        def __init__(self):
            self.enabled = None

        def set_enabled(self, enabled):
            self.enabled = enabled

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_cuda_graph_manager = DummyManager()

        def _require_action_expert(self):
            return torch.nn.Linear(1, 1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.depth_decode_cuda_graph_manager = DummyManager()

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(train_mode_vlm="fft", enable_inference_cuda_graph=True)
    policy.model = DummyModel()

    policy.train()
    assert policy.model.model.action_cuda_graph_manager.enabled is False
    assert policy.model.depth_decode_cuda_graph_manager.enabled is False

    policy.eval()
    assert policy.model.model.action_cuda_graph_manager.enabled is True
    assert policy.model.depth_decode_cuda_graph_manager.enabled is True

    policy.config.enable_inference_cuda_graph = False
    policy.eval()
    assert policy.model.model.action_cuda_graph_manager.enabled is False
    assert policy.model.depth_decode_cuda_graph_manager.enabled is False


def test_lora_targets_exclude_action_expert():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
    )

    targets = policy._get_default_peft_targets()["target_modules"]

    assert "transformer|vision_backbone|image_pooling_2d|image_projector" in targets
    assert "lm_head" in targets
    assert "action_expert" not in targets
    assert "state_encoder" not in targets
    assert "state_norm" not in targets
    assert "kv_proj" not in targets


def test_lora_initialization_order_matches_native_module_registration():
    names = [
        "base_model.model.model.vision_backbone.image_vit.transformer.resblocks.1.feed_forward.w2",
        "base_model.model.lm_head",
        "base_model.model.model.transformer.blocks.1.self_attn.att_proj",
        "base_model.model.model.image_projector.w3",
        "base_model.model.model.transformer.blocks.0.mlp.ff_proj",
        "base_model.model.model.vision_backbone.image_vit.patch_embedding",
        "base_model.model.model.image_pooling_2d.wk",
        "base_model.model.model.transformer.blocks.0.self_attn.attn_out",
        "base_model.model.model.vision_backbone.image_vit.transformer.resblocks.1.attention.wo",
    ]

    ordered = sorted(names, key=MolmoAct2Policy._official_lora_initialization_sort_key)

    assert ordered == [
        "base_model.model.model.transformer.blocks.0.self_attn.attn_out",
        "base_model.model.model.transformer.blocks.0.mlp.ff_proj",
        "base_model.model.model.transformer.blocks.1.self_attn.att_proj",
        "base_model.model.lm_head",
        "base_model.model.model.image_pooling_2d.wk",
        "base_model.model.model.image_projector.w3",
        "base_model.model.model.vision_backbone.image_vit.patch_embedding",
        "base_model.model.model.vision_backbone.image_vit.transformer.resblocks.1.attention.wo",
        "base_model.model.model.vision_backbone.image_vit.transformer.resblocks.1.feed_forward.w2",
    ]

    with pytest.raises(RuntimeError, match="official initialization order"):
        MolmoAct2Policy._official_lora_initialization_sort_key("model.transformer.unknown")


def test_train_mode_vlm_lora_wraps_loaded_hf_model_locally():
    pytest.importorskip("peft")

    class DummyInnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.att_proj = torch.nn.Linear(2, 2)
            self.transformer.blocks = torch.nn.ModuleList([block])
            # Native MolmoAct2 stores these connector linears inside
            # vision_backbone; the HF conversion moves them to these top-level
            # backbone paths. They must still receive LoRA adapters.
            self.image_pooling_2d = torch.nn.Module()
            self.image_pooling_2d.wq = torch.nn.Linear(2, 2)
            self.image_projector = torch.nn.Module()
            self.image_projector.w1 = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Module()
            self.action_expert.action_embed = torch.nn.Linear(2, 2)

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.model = DummyInnerModel()
            self.lm_head = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.model.transformer.blocks[0].self_attn.att_proj(x)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        checkpoint_path="/tmp/base",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias="none",
        train_mode_vlm="lora",
        enable_inference_cuda_graph=False,
    )
    policy.model = DummyHFModel()

    policy._apply_lora_adapters()

    assert policy._backbone() is policy.model.base_model.model.model
    trainable = [name for name, param in policy.named_parameters() if param.requires_grad]
    assert trainable
    assert any("lora_" in name for name in trainable)
    assert any("lm_head" in name and "lora_" in name for name in trainable)
    assert any("image_pooling_2d" in name and "lora_" in name for name in trainable)
    assert any("image_projector" in name and "lora_" in name for name in trainable)
    assert any("action_expert.action_embed" in name and "lora_" not in name for name in trainable)
    assert policy.model(torch.ones(1, 2)).shape == (1, 2)


def test_lora_vlm_unfreezes_action_expert_base_weights():
    class DummyInnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.wq = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Module()
            self.action_expert.action_embed = torch.nn.Linear(2, 2)

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyInnerModel()

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = DummyHFModel()

    for param in policy.parameters():
        param.requires_grad_(False)
    policy._unfreeze_action_expert_parameters()

    trainable = [name for name, param in policy.named_parameters() if param.requires_grad]
    assert trainable
    assert all("action_expert" in name for name in trainable)


def test_train_mode_vlm_freeze_requires_continuous_action_mode():
    with pytest.raises(ValueError, match="requires action_mode='continuous'"):
        MolmoAct2Config(action_mode="both", train_mode_vlm="freeze")

    cfg = MolmoAct2Config(action_mode="continuous", train_mode_vlm="freeze")
    assert cfg.train_mode_vlm == "freeze"


def test_train_mode_vlm_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported train_mode_vlm"):
        MolmoAct2Config(train_mode_vlm="frozen")


def test_molmoact2_pi05_style_precision_config():
    assert MolmoAct2Config().dtype == "bfloat16"
    assert MolmoAct2Config(dtype="float32").dtype == "float32"

    with pytest.raises(ValueError, match="Unsupported dtype"):
        MolmoAct2Config(dtype="float64")

    with pytest.raises(ValueError, match="Unsupported dtype"):
        MolmoAct2Config(dtype="float16")


def test_molmoact2_pi05_compile_defaults():
    config = MolmoAct2Config()

    assert config.compile_model is False
    assert config.compile_mode == "default"
    for compile_mode in ("default", "max-autotune-no-cudagraphs"):
        assert MolmoAct2Config(compile_mode=compile_mode).compile_mode == compile_mode

    for compile_mode in ("reduce-overhead", "max-autotune", "invalid"):
        with pytest.raises(ValueError, match="Unsupported compile_mode"):
            MolmoAct2Config(compile_mode=compile_mode)


def test_joint_checkpoint_bypasses_only_nested_transformers_wrapper():
    from transformers.modeling_layers import GradientCheckpointingLayer

    class RecordingLayer(GradientCheckpointingLayer):
        def forward(self, value):
            return value + 1

    layer = RecordingLayer().train()
    layer.gradient_checkpointing = True
    checkpoint_calls = []

    def checkpoint(function, *args):
        checkpoint_calls.append(True)
        return function(*args)

    layer._gradient_checkpointing_func = checkpoint
    hook_calls = []
    hook = layer.register_forward_hook(lambda *_: hook_calls.append(True))
    value = torch.tensor(1.0, requires_grad=True)

    direct = layer(value)
    assert direct.item() == 2.0
    assert checkpoint_calls == [True]
    assert hook_calls == [True]

    checkpoint_calls.clear()
    hook_calls.clear()
    joint = _call_module_without_gradient_checkpointing_layer(layer, value)
    assert joint.item() == 2.0
    assert checkpoint_calls == []
    assert hook_calls == [True]
    hook.remove()

    compiled_calls = []
    layer._compiled_call_impl = lambda tensor: compiled_calls.append(True) or tensor + 2
    compiled = _call_module_without_gradient_checkpointing_layer(layer, value)
    assert compiled.item() == 3.0
    assert compiled_calls == [True]


def _make_compile_test_policy(
    *,
    compile_model: bool = True,
    gradient_checkpointing: bool = True,
    train_mode_vlm: str = "fft",
):
    class RecordingModule(torch.nn.Linear):
        def __init__(self, name: str):
            super().__init__(2, 2, bias=False)
            self.name = name
            self.compile_calls = []

        def compile(self, *args, **kwargs):
            self.compile_calls.append((args, kwargs))

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([RecordingModule("text.0"), RecordingModule("text.1")])
            self.input_embeddings = RecordingModule("text.input_embeddings")

    class DummyImageViT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList(
                [RecordingModule("vision.0"), RecordingModule("vision.1")]
            )
            self.patch_embedding = RecordingModule("vision.patch_embedding")

    class DummyVisionBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_vit = DummyImageViT()
            self.image_pooling_2d = RecordingModule("vision.image_pooling_2d")
            self.image_projector = RecordingModule("vision.image_projector")

    class DummyActionExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([RecordingModule("action.0"), RecordingModule("action.1")])
            self.action_embed = RecordingModule("action.action_embed")

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = DummyTransformer()
            self.vision_backbone = DummyVisionBackbone()
            self.action_expert = DummyActionExpert()

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.lm_head = RecordingModule("lm_head")

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        compile_model=compile_model,
        compile_mode="default",
        gradient_checkpointing=gradient_checkpointing,
        train_mode_vlm=train_mode_vlm,
    )
    policy.model = DummyHFModel()
    policy._compile_applied = False
    policy._compiled_module_names = ()
    return policy


def test_molmoact2_compile_applies_precision_safe_action_partition(monkeypatch):
    policy = _make_compile_test_policy()
    backbone = policy._backbone()
    state_dict_keys = tuple(policy.state_dict())
    matmul_precision_calls = []
    lru_cache_calls = []
    ddp_optimizer_disable_calls = []
    monkeypatch.setattr(
        torch,
        "set_float32_matmul_precision",
        lambda precision: matmul_precision_calls.append(precision),
    )
    monkeypatch.setattr(
        molmoact2_modeling,
        "_set_dynamo_lru_cache",
        lambda enabled: lru_cache_calls.append(enabled),
    )
    monkeypatch.setattr(
        molmoact2_modeling,
        "_disable_dynamo_ddp_optimizer",
        lambda: ddp_optimizer_disable_calls.append(True),
    )

    policy._apply_compile()
    policy._apply_compile()

    compile_targets = [*backbone.action_expert.blocks]
    non_targets = [
        *backbone.transformer.blocks,
        *backbone.vision_backbone.image_vit.transformer.resblocks,
        backbone.vision_backbone.image_pooling_2d,
        backbone.vision_backbone.image_projector,
        backbone.transformer.input_embeddings,
        backbone.vision_backbone.image_vit.patch_embedding,
        backbone.action_expert.action_embed,
        policy.model.lm_head,
    ]
    common_kwargs = {
        "backend": "inductor",
        "options": {"emulate_precision_casts": True},
        "fullgraph": False,
    }

    assert matmul_precision_calls == ["high"]
    assert lru_cache_calls == [False]
    assert ddp_optimizer_disable_calls == [True]
    assert tuple(policy.state_dict()) == state_dict_keys
    for module in compile_targets:
        assert module.compile_calls == [((), {**common_kwargs, "dynamic": None})]
    for module in non_targets:
        assert module.compile_calls == []


def test_molmoact2_compile_uses_automatic_shapes_without_gradient_checkpointing(monkeypatch):
    policy = _make_compile_test_policy(gradient_checkpointing=False)
    backbone = policy._backbone()
    lru_cache_calls = []
    ddp_optimizer_disable_calls = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)
    monkeypatch.setattr(
        molmoact2_modeling,
        "_set_dynamo_lru_cache",
        lambda enabled: lru_cache_calls.append(enabled),
    )
    monkeypatch.setattr(
        molmoact2_modeling,
        "_disable_dynamo_ddp_optimizer",
        lambda: ddp_optimizer_disable_calls.append(True),
    )

    policy._apply_compile()

    auto_dynamic_targets = [*backbone.action_expert.blocks]
    for module in auto_dynamic_targets:
        assert module.compile_calls[0][1]["dynamic"] is None
    assert lru_cache_calls == []
    assert ddp_optimizer_disable_calls == []


def test_molmoact2_compile_marks_only_context_sequence_dynamic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch._dynamo,
        "mark_dynamic",
        lambda tensor, dim: calls.append((tensor, dim)),
    )
    key_states = torch.zeros(2, 17, 4, 8)
    value_states = torch.zeros(2, 17, 4, 8)
    attention_mask = torch.zeros(2, 1, 1, 17)

    molmoact2_modeling._mark_action_context_dynamic(key_states, value_states, attention_mask)

    assert calls == [
        (key_states, 1),
        (value_states, 1),
        (attention_mask, 3),
    ]


@pytest.mark.parametrize("train_mode_vlm", ["fft", "lora", "freeze"])
def test_molmoact2_compile_is_limited_to_action_expert_for_every_train_mode(monkeypatch, train_mode_vlm):
    policy = _make_compile_test_policy(train_mode_vlm=train_mode_vlm)
    backbone = policy._backbone()
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)
    monkeypatch.setattr(molmoact2_modeling, "_set_dynamo_lru_cache", lambda enabled: None)
    monkeypatch.setattr(molmoact2_modeling, "_disable_dynamo_ddp_optimizer", lambda: None)

    policy._apply_compile()

    assert policy._compiled_module_names == (
        "action_expert.blocks.0",
        "action_expert.blocks.1",
    )
    for module in backbone.action_expert.blocks:
        assert len(module.compile_calls) == 1
    skipped = [
        *backbone.transformer.blocks,
        *backbone.vision_backbone.image_vit.transformer.resblocks,
        backbone.vision_backbone.image_pooling_2d,
        backbone.vision_backbone.image_projector,
    ]
    assert all(module.compile_calls == [] for module in skipped)


def test_molmoact2_checkpoint_compile_requires_dynamo_cache_order_control(monkeypatch):
    policy = _make_compile_test_policy()
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)

    class MissingEvalFrame:
        pass

    monkeypatch.setattr(torch._C._dynamo, "eval_frame", MissingEvalFrame())

    with pytest.raises(RuntimeError, match="_set_lru_cache"):
        policy._apply_compile()


def test_molmoact2_checkpoint_compile_disables_dynamo_ddp_optimizer(monkeypatch):
    original_optimize_ddp = torch._dynamo.config.optimize_ddp
    try:
        torch._dynamo.config.optimize_ddp = True
        molmoact2_modeling._disable_dynamo_ddp_optimizer()
        assert torch._dynamo.config.optimize_ddp is False
    finally:
        torch._dynamo.config.optimize_ddp = original_optimize_ddp


def test_molmoact2_compile_disabled_is_a_noop(monkeypatch):
    policy = _make_compile_test_policy(compile_model=False)
    matmul_precision_calls = []
    monkeypatch.setattr(
        torch,
        "set_float32_matmul_precision",
        lambda precision: matmul_precision_calls.append(precision),
    )

    policy._apply_compile()

    assert matmul_precision_calls == []
    assert all(module.compile_calls == [] for module in policy.modules() if hasattr(module, "compile_calls"))


def test_molmoact2_train_cli_choice_remains_available(tmp_path):
    from lerobot.configs.train import TrainPipelineConfig

    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=lerobot/libero",
            "--policy.type=molmoact2",
            f"--output_dir={tmp_path / 'output'}",
        ],
    )

    assert isinstance(config.policy, MolmoAct2Config)
    assert config.policy.type == "molmoact2"


def test_model_inputs_keep_continuous_values_float32_before_autocast():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype="bfloat16")

    model_inputs = policy._model_inputs(
        {
            "pixel_values": torch.ones(1, dtype=torch.float64),
            "input_ids": torch.ones(1, dtype=torch.long),
            "ignored_float": torch.ones(1, dtype=torch.float64),
        }
    )

    assert model_inputs["pixel_values"].dtype == torch.float32
    assert model_inputs["input_ids"].dtype == torch.long
    assert "ignored_float" not in model_inputs


def test_explicit_flow_matching_tensors_follow_fp32_action_dtype():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(num_flow_timesteps=2, mask_action_dim_padding=False)

    actions, timesteps, xt, target_velocity = policy._prepare_flow_matching_tensors(
        actions=torch.ones(1, 3, 4, dtype=torch.float64),
        action_dim_is_pad=None,
        timesteps=torch.tensor([[0.25, 0.75]], dtype=torch.float64),
        noise=torch.zeros(1, 2, 3, 4, dtype=torch.float64),
    )

    assert actions.dtype == torch.float32
    assert timesteps.dtype == torch.float32
    assert xt.dtype == torch.float32
    assert target_velocity.dtype == torch.float32


def test_joint_training_external_embeddings_use_active_autocast_dtype():
    class DummyBackbone:
        def merge_visual_inputs(self, **kwargs):
            del kwargs
            return None, None

        def _build_native_attention_bias(self, **kwargs):
            return kwargs["attention_mask"]

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy._backbone = lambda: DummyBackbone()
    model_inputs = {
        "inputs_embeds": torch.ones(1, 3, 4, dtype=torch.float32),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inputs_embeds, _causal_mask, _position_ids, _cache_position = (
            policy._prepare_joint_training_backbone_inputs(model_inputs)
        )

    assert inputs_embeds.dtype == torch.bfloat16


def test_bfloat16_parameter_policy_keeps_action_expert_float32():
    class DummyActionExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_embed = torch.nn.Linear(4, 8)
            self.time_embed = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.SiLU())
            self.final_layer = torch.nn.Linear(8, 4)
            self.block = torch.nn.Linear(8, 8)
            self.modulation = ActionExpertModulation(8, 2)
            self.norm = torch.nn.LayerNorm(8)

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Linear(8, 8)
            self.rotary_emb = object.__new__(MolmoAct2RotaryEmbedding)
            torch.nn.Module.__init__(self.rotary_emb)
            self.rotary_emb.register_buffer("inv_freq", torch.tensor([1.000123, 0.500321]), persistent=True)
            self.rotary_emb.original_inv_freq = self.rotary_emb.inv_freq
            self.rotary_emb.register_buffer("_pos_sin_cache", torch.ones(1), persistent=False)
            self.rotary_emb.register_buffer("_pos_cos_cache", torch.ones(1), persistent=False)

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = DummyTransformer()
            self.vision_backbone = torch.nn.Linear(8, 8)
            self.action_expert = DummyActionExpert()
            self.action_expert_depth_gate = torch.nn.Linear(8, 1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.lm_head = torch.nn.Linear(8, 16)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype="bfloat16")
    policy.model = DummyModel()

    policy._apply_bfloat16_parameter_policy()

    assert policy.model.lm_head.weight.dtype == torch.bfloat16
    assert policy.model.model.transformer.matrix.weight.dtype == torch.bfloat16
    assert policy.model.model.action_expert.block.weight.dtype == torch.float32
    assert policy.model.model.vision_backbone.weight.dtype == torch.bfloat16
    assert policy.model.model.action_expert.action_embed.weight.dtype == torch.float32
    assert policy.model.model.action_expert.time_embed[0].weight.dtype == torch.float32
    assert policy.model.model.action_expert.final_layer.weight.dtype == torch.float32
    assert policy.model.model.action_expert.modulation.linear.weight.dtype == torch.float32
    assert policy.model.model.action_expert.norm.weight.dtype == torch.float32
    assert all(
        parameter.dtype == torch.float32 for parameter in policy.model.model.action_expert.parameters()
    )
    assert policy.model.model.action_expert_depth_gate.weight.dtype == torch.float32
    rotary_emb = policy.model.model.transformer.rotary_emb
    assert rotary_emb.inv_freq.dtype == torch.float32
    assert rotary_emb.original_inv_freq is rotary_emb.inv_freq
    assert rotary_emb._pos_sin_cache.numel() == 0
    assert rotary_emb._pos_cos_cache.numel() == 0

    optimizer = torch.optim.AdamW(policy.model.model.action_expert.block.parameters())
    with policy._autocast_context():
        action_output = policy.model.model.action_expert.block(torch.ones(2, 8, dtype=torch.float32))
    assert action_output.dtype == torch.bfloat16
    action_output.float().sum().backward()
    assert policy.model.model.action_expert.block.weight.grad.dtype == torch.float32
    optimizer.step()
    block_state = optimizer.state[policy.model.model.action_expert.block.weight]
    assert block_state["exp_avg"].dtype == torch.float32
    assert block_state["exp_avg_sq"].dtype == torch.float32

    float32_policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(float32_policy)
    float32_policy.config = SimpleNamespace(dtype="float32")
    float32_policy.model = DummyModel()

    float32_policy._apply_bfloat16_parameter_policy()

    assert all(param.dtype == torch.float32 for param in float32_policy.model.parameters())


def test_action_expert_context_uses_projected_activation_dtype():
    class DummyActionExpert(ActionExpert):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.blocks = torch.nn.ModuleList()
            self.config = SimpleNamespace(causal_attn=False)

        def _prepare_kv_context(self, encoder_kv_states):
            del encoder_kv_states
            shape = (1, 2, 3, 4)
            return [
                (
                    torch.zeros(shape, dtype=torch.bfloat16),
                    torch.zeros(shape, dtype=torch.bfloat16),
                )
            ]

    expert = DummyActionExpert()
    context = expert.prepare_context(
        encoder_kv_states=[
            (
                torch.zeros(1, 2, 3, 4, dtype=torch.float32),
                torch.zeros(1, 2, 3, 4, dtype=torch.float32),
            )
        ],
        encoder_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        action_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        batch_size=1,
        seq_len=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert context.kv_contexts[0][0].dtype == torch.bfloat16
    assert context.valid_action.dtype == torch.bfloat16
    assert context.cross_mask.dtype == torch.bfloat16
    assert context.self_mask.dtype == torch.bfloat16


def test_fp32_rmsnorm_parameters_return_activation_dtype():
    action_norm = ActionExpertRMSNorm(4, elementwise_affine=True).to(torch.float32)
    text_norm = MolmoAct2RMSNorm(4).to(torch.float32)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)

    action_output = action_norm(inputs)
    text_output = text_norm(inputs)

    assert action_norm.weight.dtype == torch.float32
    assert text_norm.weight.dtype == torch.float32
    assert action_output.dtype == torch.bfloat16
    assert text_output.dtype == torch.bfloat16


def test_text_rope_computes_in_float32_and_restores_activation_dtype(monkeypatch):
    seen_dtypes = []
    original_rotate_half = hf_molmoact2_modeling.rotate_half

    def recording_rotate_half(inputs):
        seen_dtypes.append(inputs.dtype)
        return original_rotate_half(inputs)

    monkeypatch.setattr(hf_molmoact2_modeling, "rotate_half", recording_rotate_half)
    query = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    cos = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    sin = torch.randn(1, 3, 4, dtype=torch.bfloat16)

    rotated_query, rotated_key = hf_molmoact2_modeling.apply_rotary_pos_emb(query, key, cos, sin)

    assert seen_dtypes == [torch.float32, torch.float32]
    assert rotated_query.dtype == torch.bfloat16
    assert rotated_key.dtype == torch.bfloat16


def test_text_eager_attention_uses_fp32_qk_scores(monkeypatch):
    matmul_dtypes = []
    original_matmul = torch.matmul

    def recording_matmul(left, right):
        matmul_dtypes.append((left.dtype, right.dtype))
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", recording_matmul)
    module = SimpleNamespace(num_key_value_groups=1, training=False)
    query = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)

    output, attention_weights = hf_molmoact2_modeling.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
    )

    assert matmul_dtypes[0] == (torch.float32, torch.float32)
    assert matmul_dtypes[1] == (torch.bfloat16, torch.bfloat16)
    assert attention_weights.dtype == torch.float32
    assert output.dtype == torch.bfloat16


def test_text_sdpa_keeps_bfloat16_qkv_for_fused_kernels(monkeypatch):
    config = hf_molmoact2_modeling.MolmoAct2TextConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=16,
        additional_vocab_size=None,
        num_hidden_layers=1,
        intermediate_size=16,
        max_position_embeddings=8,
        attn_implementation="sdpa",
    )
    config._attn_implementation = "sdpa"
    attention = hf_molmoact2_modeling.MolmoAct2Attention(config, layer_idx=0).to(dtype=torch.bfloat16)
    seen_dtypes = []
    original_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(query, key, value, **kwargs):
        seen_dtypes.append((query.dtype, key.dtype, value.dtype))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    hidden_states = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    cos = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    sin = torch.randn(1, 3, 4, dtype=torch.bfloat16)

    output, attention_weights = attention(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )

    assert seen_dtypes == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]
    assert output.dtype == torch.bfloat16
    assert attention_weights is None


def test_embedding_boundary_uses_active_autocast_dtype():
    embeddings = torch.ones(2, 4, dtype=torch.float32)

    assert hf_molmoact2_modeling._cast_to_autocast_dtype(embeddings).dtype == torch.float32
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cast_embeddings = hf_molmoact2_modeling._cast_to_autocast_dtype(embeddings)

    assert cast_embeddings.dtype == torch.bfloat16


def test_bfloat16_policy_autocast_bridges_fp32_heads_to_bf16_blocks():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype="bfloat16")
    policy.fp32_head = torch.nn.Linear(4, 4).to(dtype=torch.float32)
    policy.bf16_block = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)

    with policy._autocast_context():
        hidden = policy.fp32_head(torch.ones(2, 4, dtype=torch.float32))
        output = policy.bf16_block(hidden)

    assert hidden.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


def test_float32_policy_disables_surrounding_autocast():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype="float32")
    policy.fp32_layer = torch.nn.Linear(4, 4).to(dtype=torch.float32)
    inputs = torch.ones(2, 4, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outer_output = policy.fp32_layer(inputs)
        with policy._autocast_context():
            policy_output = policy.fp32_layer(inputs)

    assert outer_output.dtype == torch.bfloat16
    assert policy_output.dtype == torch.float32


def test_bfloat16_policy_rejects_device_without_autocast(monkeypatch):
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype="bfloat16")
    policy.layer = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        torch.amp.autocast_mode,
        "is_autocast_available",
        lambda device_type: False,
    )

    with pytest.raises(RuntimeError, match="requires autocast support"):
        policy._autocast_context()


def test_bfloat16_checkpoint_reload_happens_after_dtype_tree_is_applied(monkeypatch):
    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Linear(2, 2, bias=False)
            self.rotary_emb = object.__new__(MolmoAct2RotaryEmbedding)
            torch.nn.Module.__init__(self.rotary_emb)
            self.rotary_emb.register_buffer("inv_freq", torch.ones(2, dtype=torch.float32), persistent=True)
            self.rotary_emb.original_inv_freq = self.rotary_emb.inv_freq
            self.rotary_emb.register_buffer("_pos_sin_cache", torch.empty(0), persistent=False)
            self.rotary_emb.register_buffer("_pos_cos_cache", torch.empty(0), persistent=False)

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = DummyTransformer()
            self.vision_backbone = torch.nn.Linear(2, 2, bias=False)
            self.action_expert = torch.nn.Module()
            self.action_expert.action_embed = torch.nn.Linear(2, 2, bias=False)
            self.action_expert.time_embed = torch.nn.Linear(2, 2, bias=False)
            self.action_expert.final_layer = torch.nn.Linear(2, 2, bias=False)
            self.action_expert.block = torch.nn.Linear(2, 2, bias=False)
            self.action_expert_depth_gate = None

    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                max_action_horizon=30,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = DummyBackbone()

    loaded_model = DummyLoadedModel()
    action_sentinel = torch.tensor([[1.000123, 0.500321], [0.250111, -0.125077]], dtype=torch.float32)
    rope_sentinel = torch.tensor([1.000123, 0.500321], dtype=torch.float32)
    load_dtype = None

    class DummyHFConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

    class DummyMolmoAct2ForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            nonlocal load_dtype
            del args
            load_dtype = kwargs["dtype"]
            return loaded_model

    def fake_strict_load(model, checkpoint_location):
        assert checkpoint_location == "/tmp/bfloat16-checkpoint"
        assert model.model.transformer.matrix.weight.dtype == torch.bfloat16
        assert model.model.vision_backbone.weight.dtype == torch.bfloat16
        assert model.model.action_expert.action_embed.weight.dtype == torch.float32
        assert model.model.action_expert.block.weight.dtype == torch.float32
        assert model.model.transformer.rotary_emb.inv_freq.dtype == torch.float32
        with torch.no_grad():
            model.model.action_expert.action_embed.weight.copy_(action_sentinel)
            model.model.transformer.rotary_emb.inv_freq.copy_(rope_sentinel)

    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: checkpoint_path,
    )
    monkeypatch.setattr(molmoact2_modeling, "HFMolmoAct2Config", DummyHFConfig)
    monkeypatch.setattr(
        molmoact2_modeling,
        "MolmoAct2ForConditionalGeneration",
        DummyMolmoAct2ForConditionalGeneration,
    )
    monkeypatch.setattr(molmoact2_modeling, "_strict_load_safetensors_weights", fake_strict_load)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/bfloat16-checkpoint",
        action_mode="both",
        train_mode_vlm="fft",
        freeze_embedding=False,
        dtype="bfloat16",
    )

    policy._load_hf_model()

    assert load_dtype is torch.bfloat16
    assert torch.equal(policy.model.model.action_expert.action_embed.weight, action_sentinel)
    rotary_emb = policy.model.model.transformer.rotary_emb
    assert torch.equal(rotary_emb.inv_freq, rope_sentinel)
    assert rotary_emb.original_inv_freq is rotary_emb.inv_freq


def test_molmoact2_sequence_length_is_inferred_from_fixed_token_budget():
    assert (
        infer_molmoact2_max_sequence_length(
            num_images=2, state_dim=8, action_dim=7, action_horizon=10, include_discrete_action=True
        )
        == 640
    )
    assert (
        infer_molmoact2_max_sequence_length(
            num_images=2, state_dim=8, action_dim=7, action_horizon=10, include_discrete_action=False
        )
        == 576
    )
    assert (
        infer_molmoact2_max_sequence_length(
            num_images=2, state_dim=8, action_dim=7, action_horizon=30, include_discrete_action=True
        )
        == 768
    )


def test_train_mode_vlm_freeze_freezes_non_action_expert_params():
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2)
            self.vision_backbone = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Linear(2, 2)

        def _require_action_expert(self):
            return self.action_expert

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.lm_head = torch.nn.Linear(2, 2)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(train_mode_vlm="freeze")
    policy.model = DummyModel()

    policy._freeze_vlm_parameters()
    policy.train()

    assert policy.model.model.action_expert.training
    assert not policy.model.training
    assert not policy.model.model.transformer.training
    assert all(param.requires_grad for param in policy.model.model.action_expert.parameters())
    assert not any(param.requires_grad for param in policy.model.model.transformer.parameters())
    assert not any(param.requires_grad for param in policy.model.model.vision_backbone.parameters())
    assert not any(param.requires_grad for param in policy.model.lm_head.parameters())


def test_load_hf_model_accepts_max_action_horizon_schema(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                max_action_horizon=30,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()
            self.embed_tokens = torch.nn.Embedding(4, 4)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    loaded_model = DummyLoadedModel()
    resolved_kwargs = {}

    def fake_resolve_checkpoint_location(checkpoint_path, **kwargs):
        resolved_kwargs.update(kwargs)
        return checkpoint_path

    config_kwargs = {}
    model_kwargs = {}

    class DummyHFConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args
            config_kwargs.update(kwargs)
            return SimpleNamespace()

    class DummyMolmoAct2ForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args
            model_kwargs.update(kwargs)
            return loaded_model

    monkeypatch.setattr(molmoact2_modeling, "_resolve_checkpoint_location", fake_resolve_checkpoint_location)
    monkeypatch.setattr(molmoact2_modeling, "HFMolmoAct2Config", DummyHFConfig)
    monkeypatch.setattr(
        molmoact2_modeling,
        "MolmoAct2ForConditionalGeneration",
        DummyMolmoAct2ForConditionalGeneration,
    )
    monkeypatch.setattr(molmoact2_modeling, "_strict_load_safetensors_weights", lambda *args: None)
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/new-schema-checkpoint",
        checkpoint_revision="main",
        checkpoint_force_download=True,
        chunk_size=10,
        n_action_steps=10,
        action_mode="both",
        dtype="float32",
    )

    policy._load_hf_model()

    assert policy.model is loaded_model
    assert not hasattr(policy.model.config, "action_horizon")
    assert policy.model.config.max_action_horizon == 10
    assert policy._generation_action_horizon() == 10
    assert resolved_kwargs == {"revision": "main", "force_download": True}
    assert "trust_remote_code" not in config_kwargs
    assert "trust_remote_code" not in model_kwargs
    assert model_kwargs["dtype"] is torch.float32


def test_load_hf_model_chunk_size_overrides_larger_than_checkpoint_horizon(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                max_action_horizon=10,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()
            self.embed_tokens = torch.nn.Embedding(4, 4)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    loaded_model = DummyLoadedModel()
    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: checkpoint_path,
    )

    class DummyHFConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

    class DummyMolmoAct2ForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return loaded_model

    monkeypatch.setattr(molmoact2_modeling, "HFMolmoAct2Config", DummyHFConfig)
    monkeypatch.setattr(
        molmoact2_modeling,
        "MolmoAct2ForConditionalGeneration",
        DummyMolmoAct2ForConditionalGeneration,
    )
    monkeypatch.setattr(molmoact2_modeling, "_strict_load_safetensors_weights", lambda *args: None)
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/new-schema-checkpoint",
        chunk_size=30,
        n_action_steps=30,
        action_mode="both",
    )

    policy._load_hf_model()

    assert policy.model.config.max_action_horizon == 30
    assert policy._generation_action_horizon() == 30


def test_load_hf_model_rejects_legacy_action_horizon_schema(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                action_horizon=30,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()

    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: checkpoint_path,
    )

    class DummyHFConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

    class DummyMolmoAct2ForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return DummyLoadedModel()

    monkeypatch.setattr(molmoact2_modeling, "HFMolmoAct2Config", DummyHFConfig)
    monkeypatch.setattr(
        molmoact2_modeling,
        "MolmoAct2ForConditionalGeneration",
        DummyMolmoAct2ForConditionalGeneration,
    )
    monkeypatch.setattr(molmoact2_modeling, "_strict_load_safetensors_weights", lambda *args: None)
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/legacy-schema-checkpoint",
        chunk_size=10,
        n_action_steps=10,
        action_mode="both",
    )

    with pytest.raises(ValueError, match="max_action_horizon"):
        policy._load_hf_model()


def test_rtc_processor_initialization_and_select_action_guard():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(rtc_config=RTCConfig(enabled=True))

    policy.init_rtc_processor()

    assert policy.rtc_processor is not None
    with pytest.raises(AssertionError, match="RTC is not supported for select_action"):
        policy.select_action({})


def test_select_action_uses_single_full_batch_queue():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(rtc_config=None, n_action_steps=2)
    policy._action_queue = deque(maxlen=2)
    calls = 0

    def predict_action_chunk(batch, **kwargs):
        nonlocal calls
        del batch, kwargs
        calls += 1
        return torch.tensor(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )

    policy.predict_action_chunk = predict_action_chunk

    first = policy.select_action({})
    second = policy.select_action({})

    assert calls == 1
    assert torch.equal(first, torch.tensor([[1.0], [3.0]]))
    assert torch.equal(second, torch.tensor([[2.0], [4.0]]))


def test_inference_action_mode_is_explicit_and_has_no_action_mode_alias():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(action_mode="both", inference_action_mode=None)
    policy._checkpoint_action_mode = None

    with pytest.raises(ValueError, match="inference_action_mode.*explicitly"):
        policy._resolve_inference_action_mode(None)
    with pytest.raises(TypeError, match="unexpected keyword argument 'action_mode'"):
        policy.predict_action_chunk({}, action_mode="continuous")


def test_continuous_action_expert_mask_does_not_inherit_both_checkpoint_masking():
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _get_encoder_attention_mask(self, input_ids, attention_mask):
            del input_ids
            self.calls += 1
            return torch.zeros_like(attention_mask, dtype=torch.bool)

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.config = SimpleNamespace(
                eos_token_id=2,
                action_start_token_id=3,
                action_end_token_id=4,
            )

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_mode="continuous")
    policy.model = DummyHFModel()
    input_ids = torch.tensor([[2, 3, 9, 4, 7]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)

    mask = policy._encoder_attention_mask_for_action_expert(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert torch.equal(mask, attention_mask.bool())
    assert policy.model.model.calls == 0


def test_saved_continuous_checkpoint_forwards_outer_mask_to_hf_generation():
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def generate_actions_from_inputs(self, **kwargs):
            self.calls.append(kwargs)
            batch_size = int(kwargs["input_ids"].shape[0])
            return torch.zeros(batch_size, 2, 32)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    backbone = DummyBackbone()
    policy.model = backbone
    policy._backbone = lambda: backbone
    policy.config = MolmoAct2Config(
        action_mode="continuous",
        inference_action_mode="continuous",
        dtype="float32",
        chunk_size=2,
        n_action_steps=2,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy._checkpoint_action_mode = "continuous"
    policy._rollout_action_generator = None
    policy._rollout_task_key = None
    policy._rollout_index_for_task = -1
    policy.rtc_processor = None
    batch = {
        "input_ids": torch.tensor([[2, 3, 9, 4, 7]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long),
    }

    actions = policy.predict_action_chunk(batch, generator=torch.Generator().manual_seed(0))

    assert actions.shape == (1, 2, 7)
    assert torch.equal(
        backbone.calls[0]["encoder_attention_mask"],
        batch["attention_mask"].bool(),
    )

    # An original HF checkpoint has no saved outer LeRobot action mode and
    # must retain the checkpoint's own BOTH-mode inference behavior.
    policy._checkpoint_action_mode = None
    policy.predict_action_chunk(batch, generator=torch.Generator().manual_seed(0))
    assert "encoder_attention_mask" not in backbone.calls[1]


def test_both_action_expert_mask_retains_checkpoint_span_masking():
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _get_encoder_attention_mask(self, input_ids, attention_mask):
            del input_ids
            self.calls += 1
            return attention_mask.to(dtype=torch.bool).clone()

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.config = SimpleNamespace(
                eos_token_id=2,
                action_start_token_id=3,
                action_end_token_id=4,
            )

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_mode="both")
    policy.model = DummyHFModel()
    input_ids = torch.tensor([[2, 3, 9, 4, 7]])
    attention_mask = torch.ones_like(input_ids)

    mask = policy._encoder_attention_mask_for_action_expert(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert torch.equal(mask, torch.tensor([[False, False, False, False, True]]))
    assert policy.model.model.calls == 1


def test_rtc_generation_uses_previous_chunk_prefix():
    class DummyActionExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def prepare_context(self, **kwargs):
            del kwargs
            return SimpleNamespace()

        def get_or_prepare_modulation_cache(self, timesteps, *, cache_key=None):
            del cache_key
            return [SimpleNamespace(conditioning=timestep) for timestep in timesteps]

        def forward_with_context(self, actions, timesteps, *, context, modulation=None):
            del timesteps, context, modulation
            return torch.ones_like(actions) * self.weight

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                flow_matching_num_steps=2,
                max_action_horizon=4,
                max_action_dim=3,
            )
            self.action_expert = DummyActionExpert()
            self.batch_size = 1

        def _require_action_expert(self):
            return self.action_expert

        def forward(self, **kwargs):
            self.batch_size = int(kwargs["input_ids"].shape[0])
            return SimpleNamespace(past_key_values=object())

        def _extract_kv_states(self, past_key_values):
            del past_key_values
            kv = torch.zeros(self.batch_size, 1, 1)
            return [(kv, kv)]

        def _get_encoder_attention_mask(self, input_ids, attention_mask):
            del input_ids
            return attention_mask

        def _depth_gate_from_condition(self, **kwargs):
            del kwargs
            return None, None

        def _apply_depth_gate_to_layer_kv_states(self, encoder_kv_states, depth_mask, depth_gate):
            del depth_mask, depth_gate
            return encoder_kv_states

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        mask_action_dim_padding=True,
        rtc_config=RTCConfig(enabled=True, execution_horizon=2, max_guidance_weight=1.0),
    )
    policy.rtc_processor = None
    policy.model = torch.nn.Module()
    policy.model.model = DummyBackbone()
    policy.init_rtc_processor()
    model_inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    action_dim_is_pad = torch.tensor([[False, False, False]])

    without_prefix = policy._generate_actions_from_inputs_with_rtc(
        model_inputs=model_inputs,
        action_dim_is_pad=action_dim_is_pad,
        num_steps=2,
        generator=torch.Generator().manual_seed(0),
        inference_delay=0,
        prev_chunk_left_over=None,
        execution_horizon=None,
    )
    with_prefix = policy._generate_actions_from_inputs_with_rtc(
        model_inputs=model_inputs,
        action_dim_is_pad=action_dim_is_pad,
        num_steps=2,
        generator=torch.Generator().manual_seed(0),
        inference_delay=0,
        prev_chunk_left_over=torch.zeros(1, 4, 3),
        execution_horizon=None,
    )

    assert without_prefix.shape == (1, 4, 3)
    assert not torch.allclose(without_prefix, with_prefix)


def test_discrete_state_string_matches_molmoact2_bins():
    state = np.asarray([-1.0, 0.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    assert _build_discrete_state_string(state, 256) == (
        "<state_start><state_0><state_128><state_255><state_128><state_255><state_0><state_end>"
    )


def test_question_normalization_matches_release_prompt_style():
    assert _normalize_question_text("Instruction: Pick up the cube, please!") == "pick up the cube, please"
    assert (
        _normalize_question_text("The task is to open drawer. Then close it.") == "open drawer; then close it"
    )


def test_joint_frame_transform_round_trip():
    signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
    original_state = torch.tensor([[10.0, -90.0, -120.0, 30.0, 0.0, -45.0]])

    state_step = MolmoAct2StateFrameTransformStep(joint_signs=signs, joint_offsets=offsets)
    action_step = MolmoAct2ActionFrameTransformStep(joint_signs=signs, joint_offsets=offsets)

    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: original_state.clone()},
    }
    transformed = state_step(transition)
    model_state = transformed[TransitionKey.OBSERVATION][OBS_STATE]

    action_transition = {TransitionKey.ACTION: model_state.clone()}
    recovered = action_step(action_transition)
    recovered_state = recovered[TransitionKey.ACTION]

    assert torch.allclose(recovered_state, original_state)


def test_joint_frame_transform_noop_when_none():
    state_step = MolmoAct2StateFrameTransformStep(joint_signs=None, joint_offsets=None)
    action_step = MolmoAct2ActionFrameTransformStep(joint_signs=None, joint_offsets=None)
    state = torch.tensor([[10.0, -90.0, -120.0]])

    state_transition = {TransitionKey.OBSERVATION: {OBS_STATE: state}}
    assert state_step(state_transition) is state_transition

    action_transition = {TransitionKey.ACTION: state}
    assert action_step(action_transition) is action_transition


def test_action_padding_marks_only_real_dimensions():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32
    action = torch.ones(2, 3, 7)

    padded, horizon_is_pad, dim_is_pad = step._pad_action(action)

    assert padded.shape == (2, 3, 32)
    assert torch.equal(padded[..., :7], action)
    assert torch.count_nonzero(padded[..., 7:]) == 0
    assert not horizon_is_pad.any()
    assert not dim_is_pad[:, :7].any()
    assert dim_is_pad[:, 7:].all()


def test_action_dim_padding_loss_reduces_like_old_trainer():
    loss = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    action_dim_is_pad = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
        ]
    )

    reduced = _apply_action_dim_padding_mask(loss, action_dim_is_pad)

    expected = torch.stack(
        [
            loss[0, :, :, :2].sum(dim=-1) / 2,
            loss[1, :, :, :1].sum(dim=-1) / 1,
        ],
        dim=0,
    )
    assert torch.equal(reduced, expected)


def test_action_chunk_padding_keeps_old_mean_denominator():
    loss = torch.ones(1, 2, 4, 3)
    action_horizon_is_pad = torch.tensor([[False, False, True, True]])

    masked = _apply_action_chunk_padding_mask(loss, action_horizon_is_pad)

    assert masked.mean().item() == 0.5


def test_selected_discrete_loss_matches_full_causal_lm_loss():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=False,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, 1, 2, -100],
            [-100, -100, 3, 4],
        ]
    )

    selected_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state)
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    expected_loss = F.cross_entropy(logits.float().view(-1, 5), shift_labels.view(-1), ignore_index=-100)
    assert torch.allclose(selected_loss, expected_loss)
    assert z_loss is None


def test_discrete_z_loss_matches_old_trainer_formula():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, 1, 2, -100],
            [-100, -100, 3, 4],
        ]
    )

    ce_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    valid = shift_labels != -100
    expected_ce = F.cross_entropy(logits.view(-1, 5), shift_labels.view(-1), ignore_index=-100)
    expected_z = 1e-4 * logits.logsumexp(dim=-1)[valid].pow(2).mean()
    assert torch.allclose(ce_loss, expected_ce)
    assert z_loss is not None
    assert torch.allclose(z_loss, expected_z)


def test_discrete_reduction_none_preserves_mean_loss():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="root_subsegments_root_tokens",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(3, 5, 3)
    labels = torch.tensor(
        [
            [-100, 1, -100, -100, -100],
            [-100, -100, 2, 3, -100],
            [-100, 4, 3, 2, 1],
        ]
    )

    ce_mean, z_mean = policy._discrete_loss_from_backbone_outputs(
        {"labels": labels},
        outputs,
        reduction="mean",
    )
    ce_none, z_none = policy._discrete_loss_from_backbone_outputs(
        {"labels": labels},
        outputs,
        reduction="none",
    )

    assert ce_none.shape == (3,)
    assert z_none is not None
    assert z_none.shape == (3,)
    assert torch.allclose(ce_none.mean(), ce_mean)
    assert torch.allclose(z_none.mean(), z_mean)


def test_forward_reduction_none_returns_per_sample_discrete_loss():
    class DummyBackbone(torch.nn.Module):
        def __init__(self, hidden_states):
            super().__init__()
            self.hidden_states = hidden_states

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(last_hidden_state=self.hidden_states)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_mode="discrete",
        inference_action_mode="discrete",
        dtype="float32",
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    hidden_states = torch.randn(2, 4, 3)
    policy._backbone = lambda: DummyBackbone(hidden_states)
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor(
            [
                [-100, 1, 2, -100],
                [-100, -100, 3, 4],
            ]
        ),
    }

    loss_none, metrics_none = policy.forward(batch, reduction="none")
    loss_mean, metrics_mean = policy.forward(batch, reduction="mean")

    assert loss_none.shape == (2,)
    assert torch.allclose(loss_none.mean(), loss_mean)
    assert metrics_none["loss"] == pytest.approx(metrics_mean["loss"])


def test_discrete_root_token_weighting_matches_old_loss_mask_scaling():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="root_subsegments_root_tokens",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, -100, 1, -100],
            [-100, 2, 3, 4],
        ]
    )

    ce_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    valid = shift_labels != -100
    log_z = logits.logsumexp(dim=-1)
    token_ce = log_z - logits.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    weights = torch.zeros_like(token_ce)
    counts = valid.sum(dim=1).float()
    weights[valid] = (2.0 / torch.sqrt(counts))[:, None].expand_as(weights)[valid]
    expected_ce = (token_ce * weights).sum() / weights.sum()
    expected_z = 1e-4 * (log_z.pow(2) * weights).sum() / weights.sum()
    assert torch.allclose(ce_loss, expected_ce)
    assert z_loss is not None
    assert torch.allclose(z_loss, expected_z)


class _DummyActionTokenizer:
    def decode(self, tokens, *, time_horizon=None, action_dim=None):
        decoded = []
        for token_row in tokens:
            decoded.append(np.full((time_horizon, action_dim), sum(token_row), dtype=np.float32))
        return np.stack(decoded)


def test_discrete_decode_extracts_action_bins_for_each_batch():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(chunk_size=2)
    policy.action_tokenizer = _DummyActionTokenizer()
    policy.model = torch.nn.Module()
    policy.model.config = SimpleNamespace(
        action_start_token_id=10,
        action_end_token_id=11,
        action_token_start_id=100,
        num_action_tokens=4,
        action_horizon=2,
    )

    actions = policy._decode_discrete_action_chunk(
        torch.tensor(
            [
                [10, 100, 101, 11, 2],
                [10, 102, 103, 11, 2],
            ]
        ),
        action_dim=2,
    )

    assert actions.shape == (2, 2, 2)
    assert torch.equal(actions[0], torch.ones(2, 2))
    assert torch.equal(actions[1], torch.full((2, 2), 5.0))


def test_discrete_predict_action_chunk_uses_hf_cached_generation_path():
    class DummyOutput:
        def __init__(self, token_id, batch_size):
            logits = torch.full((batch_size, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            self.logits = logits
            self.past_key_values = object()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(
                action_start_token_id=10,
                action_end_token_id=11,
                action_token_start_id=100,
                num_action_tokens=4,
                action_horizon=2,
            )
            self.tokens = [10, 100, 101, 11, 2]
            self.index = 0

        def forward(self, **kwargs):
            batch_size = int(kwargs["input_ids"].shape[0])
            return DummyOutput(self.tokens[self.index], batch_size)

        def _consume_generation_tokens(self, token_ids, *, past_key_values, attention_mask):
            del past_key_values
            self.index += 1
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(token_ids[:, None])], dim=-1)
            return DummyOutput(self.tokens[self.index], int(token_ids.shape[0])), attention_mask

        def _require_eos_token_id(self):
            return 2

        def _action_token_id_to_bin(self):
            return {100: 0, 101: 1, 102: 2, 103: 3}

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        action_mode="discrete",
        inference_action_mode="discrete",
        dtype="float32",
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        discrete_generation_max_steps=None,
        discrete_action_tokenizer="unused",
        chunk_size=2,
        n_action_steps=1,
        rtc_config=None,
    )
    policy._checkpoint_action_mode = None
    policy.model = DummyModel()
    policy.action_tokenizer = _DummyActionTokenizer()

    actions = policy.predict_action_chunk(
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    )

    assert policy.model.index == 4
    assert actions.shape == (1, 1, 2)
    assert torch.equal(actions, torch.ones(1, 1, 2))


def test_discrete_predict_action_chunk_uses_graph_backed_ar_decode_when_enabled():
    class DummyOutput:
        def __init__(self, token_id, past_key_values):
            logits = torch.full((1, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            self.logits = logits
            self.past_key_values = past_key_values

    class DummyLmHead(torch.nn.Module):
        def forward(self, hidden_states):
            token_id = int(hidden_states[0, 0, 0].item())
            logits = torch.full((1, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            return logits

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.lm_head = DummyLmHead()
            self.config = SimpleNamespace(
                action_start_token_id=10,
                action_end_token_id=11,
                action_token_start_id=100,
                num_action_tokens=4,
                action_horizon=2,
            )
            self.tokens = [10, 100, 101, 11, 2]
            self.index = 0
            self.used_static_cache = False
            self.graph_steps = 0
            self.graph_position_ids = []

        def forward(self, **kwargs):
            self.used_static_cache = kwargs.get("past_key_values") == "static-cache"
            return DummyOutput(self.tokens[self.index], kwargs.get("past_key_values"))

        def _make_ar_decode_static_cache(self, inputs, *, max_steps):
            assert int(inputs["input_ids"].shape[1]) == 3
            assert max_steps == 32
            return "static-cache"

        def _make_depth_decode_attention_bias(self, inputs, past_key_values):
            assert past_key_values == "static-cache"
            return torch.ones(1, 1, 35, 35, dtype=torch.float32)

        def _run_ar_decode_step(
            self,
            token_ids,
            *,
            past_key_values,
            attention_bias,
            position_ids,
        ):
            assert past_key_values == "static-cache"
            assert attention_bias.shape == (1, 1, 35, 35)
            self.graph_position_ids.append(position_ids.detach().clone())
            self.index += 1
            self.graph_steps += 1
            return torch.tensor([[[float(self.tokens[self.index])]]]), past_key_values

        def _require_eos_token_id(self):
            return 2

        def _action_token_id_to_bin(self):
            return {100: 0, 101: 1, 102: 2, 103: 3}

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        action_mode="discrete",
        inference_action_mode="discrete",
        dtype="float32",
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        discrete_generation_max_steps=None,
        discrete_action_tokenizer="unused",
        chunk_size=2,
        n_action_steps=1,
        rtc_config=None,
        enable_inference_cuda_graph=True,
    )
    policy._checkpoint_action_mode = None
    policy.model = DummyModel()
    policy.action_tokenizer = _DummyActionTokenizer()
    torch.nn.Module.train(policy, False)

    actions = policy.predict_action_chunk(
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1]], dtype=torch.long),
        }
    )

    assert policy.model.used_static_cache
    assert policy.model.graph_steps == 4
    assert [position.item() for position in policy.model.graph_position_ids] == [2, 3, 4, 5]
    assert actions.shape == (1, 1, 2)
    assert torch.equal(actions, torch.ones(1, 1, 2))


def test_hf_static_continuation_advances_mask_relative_positions():
    class DummyCache:
        def get_seq_length(self):
            return 4

    class DummyLmHead:
        def __call__(self, hidden_states):
            token_id = int(hidden_states[0, 0, 0].item())
            logits = torch.full((1, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            return logits

    class DummyModel:
        def __init__(self):
            self.tokens = [10, 100, 2]
            self.index = 0
            self.positions = []
            self.lm_head = DummyLmHead()

        def _run_ar_decode_step(
            self,
            token_ids,
            *,
            past_key_values,
            attention_bias,
            position_ids,
        ):
            del token_ids, attention_bias
            self.positions.append(position_ids.clone())
            self.index += 1
            hidden = torch.tensor([[[float(self.tokens[self.index])]]])
            return hidden, past_key_values

    dummy = DummyModel()
    initial_logits = torch.full((1, 1, 128), -1e9)
    initial_logits[:, :, 10] = 1.0

    generated = (
        hf_molmoact2_modeling.MolmoAct2ForConditionalGeneration._continue_discrete_generation_from_output(
            dummy,
            SimpleNamespace(logits=initial_logits),
            past_key_values=DummyCache(),
            attention_mask=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
            end_token_id=2,
            max_steps=4,
            attention_bias=torch.zeros(1, 1, 8, 8),
        )
    )

    assert torch.equal(generated, torch.tensor([[10, 100, 2]]))
    assert [position.item() for position in dummy.positions] == [2, 3]


class _DummyMolmoBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 3)

    def get_input_embeddings(self):
        return self.embed


class _DummyMolmoModel(torch.nn.Module):
    def __init__(self, *, tie_lm_head: bool = False):
        super().__init__()
        self.model = _DummyMolmoBackbone()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.model.embed.weight

    def get_input_embeddings(self):
        return self.model.embed


def test_freeze_embedding_freezes_input_embeddings_only_when_untied():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = _DummyMolmoModel()

    policy._freeze_input_embeddings()

    assert not policy.model.model.embed.weight.requires_grad
    assert policy.model.lm_head.weight.requires_grad


def test_freeze_embedding_rejects_tied_lm_head_without_mutating():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = _DummyMolmoModel(tie_lm_head=True)

    with pytest.raises(RuntimeError, match="would also freeze lm_head"):
        policy._freeze_input_embeddings()

    assert policy.model.model.embed.weight.requires_grad
