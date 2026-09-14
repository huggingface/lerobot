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

import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="lawam requires the `lawam` extra (transformers)")
pytest.importorskip("diffusers", reason="lawam requires the `lawam` extra (diffusers)")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.lawam.configuration_lawam import (
    LaWAMConfig,
    LaWAMCosineWithMinLRSchedulerConfig,
)
from lerobot.policies.lawam.lam_core.core.lam_model import LatentLAMModel, build_latent_action_model
from lerobot.policies.lawam.lam_core.core.utils.modules import build_modal_block_attention_mask
from lerobot.policies.lawam.modeling_lawam import (
    LaWAMPolicy,
    _build_freeze_config,
    _build_native_policy_config,
)
from lerobot.policies.lawam.processor_lawam import (
    LaWAMBinarizeGripperProcessorStep,
    LaWAMClipActionsProcessorStep,
    LaWAMPrepareBatchProcessorStep,
    LaWAMPreSnapGripperProcessorStep,
    LaWAMQwenInputsProcessorStep,
    LaWAMResizeImagesProcessorStep,
)
from lerobot.policies.lawam.vlas.flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
    build_time_grid,
)
from lerobot.policies.lawam.vlas.qwen3vl import (
    freeze_qwen3vl,
    keep_first_n_llm_layers,
    remove_lm_head,
    unfreeze_last_n_llm_layers,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import NormalizerProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config() -> LaWAMConfig:
    return LaWAMConfig(
        device="cpu",
        chunk_size=4,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=2,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        base_vlm="dummy-qwen",
        embodiment_id=25,
        primary_image_features=["observation.images.front"],
        wrist_image_features=["observation.images.wrist"],
    )


class _FakeProcessor:
    def __init__(self, placeholder_token_id: int = 99) -> None:
        self.placeholder_token_id = placeholder_token_id
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.messages = messages
        batch_size = len(messages)
        return {
            "input_ids": torch.full((batch_size, 16), self.placeholder_token_id, dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 16), dtype=torch.long),
            "pixel_values": torch.zeros((batch_size, 3, 256, 256)),
            "image_grid_thw": torch.ones((batch_size, 3), dtype=torch.long),
        }


class _FakeNativeLaWAM(nn.Module):
    def __init__(self, chunk_size: int = 4, action_dim: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.policy_cfg = SimpleNamespace(
            num_action_queries=8,
            flow_action_num_queries=8,
            latent_action_placeholder_token="<ACT_PH>",
            flow_cfg=SimpleNamespace(state_dim=7),
        )
        self.predict_calls = 0
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        loss_flow = batch["actions"].mean() * self.weight
        loss_total = loss_flow + batch["state"].mean() * 0.0
        return {"total_loss": loss_total, "loss_flow": loss_flow}

    def predict_action(self, batch, **kwargs):
        del kwargs
        self.predict_calls += 1
        self.last_batch = batch
        batch_size = int(batch["input_ids"].shape[0])
        actions = torch.arange(
            batch_size * self.chunk_size * self.action_dim,
            dtype=torch.float32,
        ).reshape(batch_size, self.chunk_size, self.action_dim)
        return actions


class _FakeQwen3VL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.encoder = nn.Linear(2, 2)
        self.model.visual.merger = nn.Linear(2, 2)
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 2)
        self.model.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.lm_head = nn.Linear(2, 4, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def make_batch(batch_size: int = 2) -> dict:
    return {
        "observation.images.front": torch.rand(batch_size, 2, 3, 8, 8),
        "observation.images.wrist": torch.rand(batch_size, 2, 3, 8, 8),
        OBS_STATE: torch.rand(batch_size, 7),
        ACTION: torch.rand(batch_size, 4, 7),
        "task": [f"task {idx}" for idx in range(batch_size)],
    }


def make_policy(config: LaWAMConfig | None = None):
    native_model = _FakeNativeLaWAM()
    policy = LaWAMPolicy(config or make_config(), native_model=native_model)
    return policy, native_model


def inject_fake_processor(preprocessor) -> LaWAMQwenInputsProcessorStep:
    step = next(step for step in preprocessor.steps if isinstance(step, LaWAMQwenInputsProcessorStep))
    step._processor = _FakeProcessor()
    step._placeholder_token_id = 99
    return step


def make_prepared_batch(batch_size: int = 2) -> dict:
    return {
        "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        "pixel_values": torch.zeros((batch_size, 3, 8, 8)),
        "primary_image": torch.zeros((batch_size, 3, 8, 8)),
        "state": torch.rand(batch_size, 32),
        "state_mask": torch.ones(batch_size, 32, dtype=torch.bool),
        "embodiment_id": torch.full((batch_size,), 25, dtype=torch.long),
        "action_hz": torch.full((batch_size,), 20.0),
        "actions": torch.rand(batch_size, 4, 32),
        "actions_mask": torch.ones(batch_size, 4, 32, dtype=torch.bool),
    }


def test_factory_registers_lawam() -> None:
    assert get_policy_class("lawam") is LaWAMPolicy
    assert isinstance(make_policy_config("lawam", device="cpu"), LaWAMConfig)


def test_lawam_scheduler_matches_upstream_warmup_relative_cosine() -> None:
    cfg = make_config()
    scheduler_cfg = cfg.get_scheduler_preset()
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=cfg.optimizer_lr)
    scheduler = scheduler_cfg.build(optimizer, num_training_steps=cfg.scheduler_decay_steps)

    assert isinstance(scheduler_cfg, LaWAMCosineWithMinLRSchedulerConfig)
    assert scheduler_cfg.type == "lawam_cosine_with_min_lr"

    min_lr_ratio = cfg.scheduler_decay_lr / cfg.optimizer_lr
    for step in (0, 1, 1_499, 1_500, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000):
        if step < cfg.scheduler_warmup_steps:
            expected_factor = step / cfg.scheduler_warmup_steps
        else:
            progress = (step - cfg.scheduler_warmup_steps) / (
                cfg.scheduler_decay_steps - cfg.scheduler_warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            expected_factor = 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - min_lr_ratio) + min_lr_ratio
        assert scheduler.lr_lambdas[0](step) == pytest.approx(expected_factor)


def test_lawam_declares_real_fsdp_wrap_units() -> None:
    from transformers.models.dinov3_vit import modeling_dinov3_vit
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    from lerobot.policies.lawam.lam_core.core.utils import lam_decoder
    from lerobot.policies.lawam.vlas import cross_attention_dit

    wrap_classes = {
        "Qwen3VLVisionBlock": modeling_qwen3_vl.Qwen3VLVisionBlock,
        "Qwen3VLTextDecoderLayer": modeling_qwen3_vl.Qwen3VLTextDecoderLayer,
        "DINOv3ViTLayer": modeling_dinov3_vit.DINOv3ViTLayer,
        "TransformerEncoderLayer": nn.TransformerEncoderLayer,
        "AdaLNBlock": lam_decoder.AdaLNBlock,
        "BasicTransformerBlock": cross_attention_dit.BasicTransformerBlock,
    }

    assert set(LaWAMPolicy._fsdp_wrap_modules) == set(wrap_classes)
    assert all(isinstance(cls, type) for cls in wrap_classes.values())


def test_make_pre_post_processors_for_lawam() -> None:
    preprocessor, postprocessor = make_pre_post_processors(
        make_config(), dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_lawam_defaults_match_native_state_normalization() -> None:
    assert make_config().normalization_mapping["STATE"] is NormalizationMode.MIN_MAX


def test_unused_state_stats_are_excluded_from_eval_processors() -> None:
    dataset_stats = {
        OBS_STATE: {
            "min": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "max": torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        ACTION: {
            "min": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0]),
            "max": torch.tensor([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 1.0]),
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "mask": torch.tensor([True, True, True, True, True, True, False]),
        },
    }
    cfg = make_config()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_stats, dataset_meta=SimpleNamespace(fps=20)
    )
    inject_fake_processor(preprocessor)
    batch = {
        "observation.images.front": torch.zeros(3, 8, 8),
        "observation.images.wrist": torch.zeros(3, 8, 8),
        OBS_STATE: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 7.0]),
        "task": "task",
    }
    processed_batch = preprocessor(batch)
    processed_action = postprocessor(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
    )

    assert torch.equal(processed_batch["state"], torch.zeros(1, 32))
    assert not processed_batch["state_mask"].any()
    assert torch.allclose(
        processed_action[:, :2],
        torch.tensor([[11.0, 22.0], [12.0, 24.0]]),
    )
    assert processed_action[:, -1].tolist() == [0.5, 1.0]


def test_state_conditioned_flow_uses_standard_lerobot_state() -> None:
    cfg = make_config()
    cfg.flow_use_state = True
    dataset_stats = {
        OBS_STATE: {
            "min": torch.zeros(7),
            "max": torch.full((7,), 2.0),
        }
    }
    preprocessor, _ = make_pre_post_processors(
        cfg, dataset_stats=dataset_stats, dataset_meta=SimpleNamespace(fps=20)
    )
    inject_fake_processor(preprocessor)
    batch = make_batch(batch_size=1)
    batch[OBS_STATE] = torch.ones(1, 7)

    processed_batch = preprocessor(batch)

    assert torch.allclose(processed_batch["state"][:, :7], torch.zeros(1, 7))
    assert processed_batch["state_mask"][:, :7].all()


def test_native_checkpoint_freeze_config_is_loaded(tmp_path) -> None:
    del tmp_path
    cfg = make_config()
    cfg.freeze_embedding = True
    cfg.keep_llm_first_n_layers = 16
    cfg.unfreeze_lam_decoder = True

    freeze_config = _build_freeze_config(cfg)

    assert freeze_config is not None
    assert freeze_config.freeze_embedding is True
    assert freeze_config.keep_llm_first_n_layers == 16
    assert freeze_config.unfreeze_lam_decoder is True


@pytest.mark.parametrize(("dataset_open", "dataset_closed"), [(0.0, 1.0), (-1.0, 1.0)])
def test_lawam_postprocessor_matches_libero_gripper_convention(
    dataset_open: float, dataset_closed: float
) -> None:
    cfg = make_config()
    cfg.clip_normalized_actions = True
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    action_min = torch.zeros(7)
    action_min[cfg.gripper_dim] = dataset_open
    action_max = torch.ones(7)
    action_max[cfg.gripper_dim] = dataset_closed
    dataset_stats = {
        ACTION: {
            "min": action_min,
            "max": action_max,
        }
    }
    _, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_stats, dataset_meta=SimpleNamespace(fps=20)
    )
    action = torch.zeros(2, 7)
    action[:, -1] = torch.tensor([0.0, 1.0])

    processed = postprocessor(action)

    assert processed[:, -1].tolist() == [-1.0, 1.0]


@pytest.mark.parametrize(("dataset_open", "dataset_closed"), [(0.0, 1.0), (-1.0, 1.0)])
def test_lawam_training_preprocessor_preserves_binary_gripper_targets(
    dataset_open: float, dataset_closed: float
) -> None:
    cfg = make_config()
    cfg.pre_snap_gripper_action = True
    action_min = torch.zeros(7)
    action_min[cfg.gripper_dim] = dataset_open
    action_max = torch.ones(7)
    action_max[cfg.gripper_dim] = dataset_closed
    dataset_stats = {
        ACTION: {
            "min": action_min,
            "max": action_max,
        }
    }
    preprocessor, _ = make_pre_post_processors(
        cfg, dataset_stats=dataset_stats, dataset_meta=SimpleNamespace(fps=20)
    )
    inject_fake_processor(preprocessor)
    batch = make_batch()
    batch[ACTION] = torch.zeros(2, 4, 7)
    batch[ACTION][0, :, cfg.gripper_dim] = dataset_open
    batch[ACTION][1, :, cfg.gripper_dim] = dataset_closed

    processed = preprocessor(batch)

    assert processed["actions"][:, :, cfg.gripper_dim].tolist() == [[0.0] * 4, [1.0] * 4]


def test_lawam_gripper_processing_is_opt_in() -> None:
    cfg = make_config()
    _, postprocessor = make_pre_post_processors(cfg, dataset_stats=None, dataset_meta=SimpleNamespace(fps=20))
    action = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])

    assert cfg.clip_normalized_actions is False
    assert cfg.pre_snap_gripper_action is False
    assert cfg.binarize_gripper_action is False
    assert torch.equal(postprocessor(action), action)


def test_lawam_postprocessor_config_round_trip(tmp_path) -> None:
    cfg = make_config()
    cfg.clip_normalized_actions = True
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    cfg.gripper_dim = 3
    cfg.gripper_threshold = 0.25
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    configs = [step.get_config() for step in loaded_postprocessor.steps]
    loaded_qwen_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, LaWAMQwenInputsProcessorStep)
    )
    loaded_resize_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, LaWAMResizeImagesProcessorStep)
    )

    assert {"gripper_dim": 3, "threshold": 0.25} in configs
    assert loaded_qwen_step.model_id == "dummy-qwen"
    assert loaded_resize_step.image_features == [
        "observation.images.front",
        "observation.images.wrist",
    ]


def test_sft_rebuilds_pretraining_processors_for_current_config(tmp_path) -> None:
    pretrain_cfg = make_config()
    pretrain_cfg.chunk_size = 50
    pretrain_cfg.action_horizon = 24
    pretrain_cfg.n_action_steps = 24
    pretrain_preprocessor, pretrain_postprocessor = make_pre_post_processors(
        pretrain_cfg,
        dataset_stats=None,
        dataset_meta=SimpleNamespace(fps=20),
    )
    pretrain_preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    pretrain_postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    sft_cfg = make_config()
    sft_cfg.chunk_size = 50
    sft_cfg.action_horizon = 8
    sft_cfg.n_action_steps = 8
    sft_cfg.clip_normalized_actions = True
    sft_cfg.pre_snap_gripper_action = True
    sft_cfg.binarize_gripper_action = True
    sft_cfg._runtime_dataset_meta = SimpleNamespace(fps=25)
    sft_stats = {
        ACTION: {
            "min": torch.zeros(7),
            "max": torch.ones(7),
        }
    }
    preprocessor, postprocessor = make_pre_post_processors(
        sft_cfg,
        pretrained_path=str(tmp_path),
        dataset_stats=sft_stats,
        preprocessor_overrides={
            "rename_observations_processor": {
                "rename_map": {"observation.images.raw_wrist": "observation.images.wrist"}
            }
        },
    )

    prepare_step = next(
        step for step in preprocessor.steps if isinstance(step, LaWAMPrepareBatchProcessorStep)
    )
    assert prepare_step.action_horizon == 8
    assert prepare_step.chunk_size == 50
    assert prepare_step.action_hz == 25.0
    assert preprocessor.steps[0].get_config() == {
        "rename_map": {"observation.images.raw_wrist": "observation.images.wrist"}
    }
    assert any(isinstance(step, LaWAMResizeImagesProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, LaWAMPreSnapGripperProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, LaWAMClipActionsProcessorStep) for step in postprocessor.steps)
    assert any(isinstance(step, LaWAMBinarizeGripperProcessorStep) for step in postprocessor.steps)

    inject_fake_processor(preprocessor)
    batch = make_batch()
    batch[ACTION] = torch.rand(2, 8, 7)
    processed = preprocessor(batch)
    policy, native_model = make_policy(sft_cfg)
    policy(processed)

    assert native_model.last_batch is not None
    assert native_model.last_batch["actions"].shape == (2, 50, 32)
    assert native_model.last_batch["actions_mask"][:, :8, :7].all()
    assert not native_model.last_batch["actions_mask"][:, 8:].any()
    assert torch.equal(native_model.last_batch["action_hz"], torch.full((2,), 25.0))


def test_resume_loads_serialized_lawam_processors(tmp_path) -> None:
    checkpoint_cfg = make_config()
    checkpoint_stats = {
        ACTION: {
            "min": torch.zeros(7),
            "max": torch.full((7,), 2.0),
        }
    }
    checkpoint_preprocessor, checkpoint_postprocessor = make_pre_post_processors(
        checkpoint_cfg,
        dataset_stats=checkpoint_stats,
        dataset_meta=SimpleNamespace(fps=20),
    )
    checkpoint_preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    checkpoint_postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    resume_cfg = make_config()
    resume_cfg.action_horizon = 2
    resume_cfg._runtime_dataset_meta = SimpleNamespace(fps=25)
    preprocessor, postprocessor = make_pre_post_processors(
        resume_cfg,
        pretrained_path=str(tmp_path),
        dataset_stats=None,
        preprocessor_overrides={
            "rename_observations_processor": {
                "rename_map": {"observation.images.raw_wrist": "observation.images.wrist"}
            }
        },
    )

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    prepare_step = next(
        step for step in preprocessor.steps if isinstance(step, LaWAMPrepareBatchProcessorStep)
    )
    assert torch.equal(torch.as_tensor(normalizer.stats[ACTION]["max"]), torch.full((7,), 2.0))
    assert prepare_step.action_horizon == checkpoint_cfg.action_horizon
    assert prepare_step.action_hz == 20.0
    assert preprocessor.steps[0].get_config() == {
        "rename_map": {"observation.images.raw_wrist": "observation.images.wrist"}
    }
    assert torch.equal(postprocessor(torch.zeros(1, 7)), torch.ones(1, 7))


def test_processor_frequency_is_used_without_dataset_metadata() -> None:
    policy, native_model = make_policy()

    policy(make_prepared_batch())

    assert native_model.last_batch is not None
    assert torch.equal(native_model.last_batch["action_hz"], torch.full((2,), 20.0))


def test_native_config_uses_padded_lawam_action_space() -> None:
    cfg = make_config()
    policy_cfg = _build_native_policy_config(cfg)

    assert cfg.action_feature.shape == (7,)
    assert policy_cfg.flow_cfg.action_dim == 32
    assert policy_cfg.flow_cfg.state_dim == 32
    assert policy_cfg.action_horizon == 4
    assert policy_cfg.effective_action_horizon == 4


def test_lam_constructs_without_checkpoint_or_pretrained_dino() -> None:
    model_config = {
        "dim": 32,
        "num_heads": 4,
        "ffn_expansion_factor": 2,
        "enc_layers": 1,
        "code_dim": 8,
        "max_state_dim": 7,
        "num_frames": 2,
        "num_queries": 1,
        "vq_kwargs": {"layer_norm": True},
        "dec_layers": 1,
        "dropout": 0.0,
        "norm_latents": True,
        "norm_latents_type": "ln",
        "enc_modal_mask": True,
        "latent_layer_to_use": -2,
        "num_embodiments": 32,
        "image_hw": (32, 32),
        "patch_size": 16,
        "decoder_last_ln": True,
        "dinov3_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_register_tokens": 4,
        },
    }
    loaded_model = build_latent_action_model(model_config)

    assert isinstance(loaded_model, LatentLAMModel)
    assert all(not parameter.requires_grad for parameter in loaded_model.parameters())


def test_action_horizon_is_independent_of_control_frequency() -> None:
    cfg = make_config()
    cfg.chunk_size = 50
    cfg.action_horizon = 8

    assert cfg.action_delta_indices == list(range(8))
    policy_cfg = _build_native_policy_config(cfg)
    assert policy_cfg.action_horizon == 50
    assert policy_cfg.effective_action_horizon == 8


def test_sft_cli_overrides_pretraining_action_horizon(tmp_path) -> None:
    pretrain_cfg = make_config()
    pretrain_cfg.chunk_size = 50
    pretrain_cfg.action_horizon = 24
    pretrain_cfg.n_action_steps = 24
    pretrain_cfg.save_pretrained(tmp_path)

    sft_cfg = LaWAMConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--action_horizon=8",
            "--n_action_steps=8",
            "--clip_normalized_actions=true",
            "--pre_snap_gripper_action=true",
            "--binarize_gripper_action=true",
        ],
    )

    assert sft_cfg.chunk_size == 50
    assert sft_cfg.action_horizon == 8
    assert sft_cfg.n_action_steps == 8
    assert sft_cfg.clip_normalized_actions is True
    assert sft_cfg.pre_snap_gripper_action is True
    assert sft_cfg.binarize_gripper_action is True


def test_processor_action_hz_is_derived_from_dataset_metadata() -> None:
    preprocessor, _ = make_pre_post_processors(
        make_config(), dataset_stats=None, dataset_meta=SimpleNamespace(fps=25)
    )
    prepare_step = next(
        step for step in preprocessor.steps if isinstance(step, LaWAMPrepareBatchProcessorStep)
    )

    assert prepare_step.action_hz == 25.0
    assert not hasattr(make_config(), "action_hz")


def test_natural_time_grid_uses_dataset_frequency() -> None:
    grid = build_time_grid(hz=torch.tensor([20.0, 25.0]), seq_len=3)

    assert torch.allclose(grid, torch.tensor([[0.0, 0.05, 0.1], [0.0, 0.04, 0.08]]))


def test_dataset_sampling_uses_the_configured_action_horizon() -> None:
    cfg = make_config()
    cfg.chunk_size = 50
    cfg.action_horizon = 10
    dataset_meta = SimpleNamespace(
        fps=25,
        features={ACTION: {}, "observation.images.front": {}},
    )

    delta_timestamps = resolve_delta_timestamps(cfg, dataset_meta)

    assert delta_timestamps is not None
    assert len(delta_timestamps[ACTION]) == 10
    assert delta_timestamps[ACTION][-1] == pytest.approx(9 / 25)
    assert delta_timestamps["observation.images.front"] == pytest.approx([0.0, 9 / 25])


def test_flow_uses_padded_horizon_and_returns_only_effective_actions(monkeypatch) -> None:
    flow = ConditionalFlowMatchingHead(
        ConditionalFlowMatchingConfig(
            action_dim=4,
            hidden_dim=8,
            num_layers=2,
            attention_heads=2,
            num_inference_steps=1,
            vlm_dim=8,
            vision_dim=8,
            num_vision_tokens=2,
            use_state=False,
            num_embodiments=32,
            interleave_self_attention=True,
            use_alternate_vldit=False,
        )
    )
    flow.action_horizon = 5
    flow.effective_action_horizon = 2
    sampled_shapes = []

    def sample_noise(shape, device, dtype):
        sampled_shapes.append(shape)
        return torch.zeros(shape, device=device, dtype=dtype)

    monkeypatch.setattr(flow, "sample_noise", sample_noise)
    actions = flow.sample_actions_cfg(
        h_t=torch.zeros(2, 2, 8),
        h_t1_star=torch.zeros(2, 2, 8),
        h_vlm=torch.zeros(2, 3, 8),
        state=torch.zeros(2, 8),
        state_mask=torch.zeros(2, 8, dtype=torch.bool),
        action_hz=torch.full((2,), 20.0),
        embodiment_id=torch.full((2,), 25, dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert sampled_shapes == [(2, 5, 4)]
    assert actions.shape == (2, 2, 4)
    assert torch.isfinite(actions).all()


def test_preprocessor_builds_complete_resized_training_batch() -> None:
    preprocessor, _ = make_pre_post_processors(
        make_config(), dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    prepare_step = inject_fake_processor(preprocessor)
    raw_batch = {
        "observation.images.front": torch.rand(1, 2, 3, 480, 640),
        "observation.images.wrist": torch.rand(1, 2, 3, 480, 640),
        OBS_STATE: torch.rand(1, 7),
        ACTION: torch.rand(1, 4, 7),
        "task": ["pick"],
    }

    batch = preprocessor(raw_batch)

    assert batch["primary_video"].shape == (1, 2, 3, 256, 256)
    assert batch["primary_image"].shape == (1, 3, 256, 256)
    assert batch["actions"].shape == (1, 4, 32)
    assert batch["actions_mask"][:, :, :7].all()
    assert batch["state"].shape == (1, 32)
    assert prepare_step._processor.messages is not None


@pytest.mark.parametrize("shape", [(2, 3, 20, 40), (2, 4, 3, 20, 40)])
def test_lawam_resize_processor_supports_current_and_temporal_images(shape: tuple[int, ...]) -> None:
    image_key = "observation.images.front"
    images = torch.rand(shape)
    step = LaWAMResizeImagesProcessorStep(image_features=[image_key], image_hw=(8, 12))

    processed = step({TransitionKey.OBSERVATION: {image_key: images}})

    output = processed[TransitionKey.OBSERVATION][image_key]
    assert output.shape == (*shape[:-2], 8, 12)
    assert output.dtype == images.dtype
    assert output.device == images.device
    assert torch.isfinite(output).all()


def test_action_steps_cannot_exceed_action_horizon() -> None:
    with pytest.raises(ValueError, match="n_action_steps.*action_horizon"):
        LaWAMConfig(chunk_size=50, action_horizon=8, n_action_steps=9)


def test_lam_modal_mask_allows_query_and_same_modality_attention() -> None:
    mask = build_modal_block_attention_mask(
        num_frames=2,
        grid_height=1,
        grid_width=1,
        add_tokens=1,
        num_queries=1,
    )

    assert mask.shape == (5, 5)
    assert mask[4].all()
    assert mask[0, 2]
    assert mask[1, 3]
    assert not mask[0, 1]


def test_qwen3vl_freeze_and_layer_selection() -> None:
    model = _FakeQwen3VL()

    freeze_qwen3vl(
        model,
        freeze_vision_backbone=True,
        freeze_llm_backbone=True,
        freeze_embedding=False,
        unfreeze_vision_merger=True,
    )
    keep_first_n_llm_layers(model, 2)
    unfreeze_last_n_llm_layers(model, 1)
    remove_lm_head(model)

    assert not model.model.visual.encoder.weight.requires_grad
    assert model.model.visual.merger.weight.requires_grad
    assert model.get_input_embeddings().weight.requires_grad
    assert len(model.model.language_model.layers) == 2
    assert not model.model.language_model.layers[0].weight.requires_grad
    assert model.model.language_model.layers[1].weight.requires_grad
    assert not hasattr(model, "lm_head")


def test_training_forward_consumes_processor_prepared_batch() -> None:
    policy, _ = make_policy()
    loss, logs = policy.forward(make_prepared_batch())

    assert loss.ndim == 0
    assert "loss" in logs


def test_saved_policy_uses_safetensors_without_torch_load(tmp_path, monkeypatch) -> None:
    cfg = make_config()
    policy, _ = make_policy(cfg)

    policy.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "config.json").read_text())

    assert saved_config["base_vlm"] == "dummy-qwen"

    def fail_torch_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("LaWAM native checkpoints must not call torch.load")

    monkeypatch.setattr(torch, "load", fail_torch_load)

    loaded_policy = LaWAMPolicy.from_pretrained(
        tmp_path,
        native_model=_FakeNativeLaWAM(),
    )
    assert torch.equal(loaded_policy.model.weight, policy.model.weight)


def test_pretrained_load_stages_lawam_checkpoint_on_cpu(tmp_path, monkeypatch) -> None:
    cfg = make_config()
    cfg.device = "cuda"
    events = {}

    def fake_load(cls, model, model_file, map_location, strict):
        del cls, model_file, strict
        events["load_device"] = map_location
        events["parameter_device_during_load"] = next(model.parameters()).device.type
        return model

    def fake_to(self, device):
        events["final_device"] = str(device)
        if str(device).startswith("cpu"):
            return nn.Module.to(self, device)
        return self

    monkeypatch.setattr(PreTrainedPolicy, "_load_as_safetensor", classmethod(fake_load))
    monkeypatch.setattr(LaWAMPolicy, "to", fake_to)

    policy = LaWAMPolicy.from_pretrained(
        tmp_path,
        config=cfg,
        native_model=_FakeNativeLaWAM(),
    )

    assert events == {
        "load_device": "cpu",
        "parameter_device_during_load": "cpu",
        "final_device": "cuda",
    }
    assert cfg.device == "cuda"
    assert policy.config.device == "cuda"


def test_base_vlm_rejects_local_paths() -> None:
    with pytest.raises(ValueError, match="portable Hugging Face model ID"):
        LaWAMConfig(base_vlm="/local/qwen3-vl")


@pytest.mark.parametrize("primary_features", [None, ["observation.images.front", "observation.images.wrist"]])
def test_primary_image_feature_override(primary_features: list[str] | None) -> None:
    cfg = make_config()
    cfg.primary_image_features = primary_features
    if primary_features is not None:
        cfg.wrist_image_features = []
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=None, dataset_meta=SimpleNamespace(fps=20))
    prepare_step = inject_fake_processor(preprocessor)
    preprocessor(make_batch(batch_size=1))

    expected_views = 1 if primary_features is None else 2
    assert len(prepare_step._processor.messages[0][0]["content"]) >= expected_views
    assert len(prepare_step.primary_image_features) == expected_views


def test_multiple_cameras_require_explicit_roles() -> None:
    cfg = make_config()
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    cfg.primary_image_features = None
    cfg.wrist_image_features = None

    with pytest.raises(ValueError, match="explicit `primary_image_features`"):
        make_policy(cfg)


def test_preprocessor_keeps_inference_inputs_as_tensors() -> None:
    preprocessor, _ = make_pre_post_processors(
        make_config(), dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    prepare_step = inject_fake_processor(preprocessor)
    batch = make_batch(batch_size=1)
    batch.pop(ACTION)

    prepared = preprocessor(batch)

    assert prepared["primary_image"].shape == (1, 3, 256, 256)
    assert "actions" not in prepared
    image_items = [
        item for item in prepare_step._processor.messages[0][0]["content"] if item["type"] == "image"
    ]
    assert image_items
    assert all(torch.is_tensor(item["image"]) for item in image_items)


def test_select_action_uses_action_queue_before_refill() -> None:
    policy, native_model = make_policy()
    batch = make_prepared_batch(batch_size=1)

    first = policy.select_action(batch)
    second = policy.select_action(batch)

    assert native_model.predict_calls == 1
    assert first.shape == (1, 7)
    assert second.shape == (1, 7)
    assert not torch.equal(first, second)
