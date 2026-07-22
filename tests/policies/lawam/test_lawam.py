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
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="lawam requires the `lawam` extra (transformers)")
pytest.importorskip("diffusers", reason="lawam requires the `lawam` extra (diffusers)")
pytest.importorskip("qwen_vl_utils", reason="lawam requires the `lawam` extra (qwen-vl-utils)")
pytest.importorskip("timm", reason="lawam requires the `lawam` extra (timm)")
pytest.importorskip("yaml", reason="lawam requires the `lawam` extra (PyYAML)")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig
from lerobot.policies.lawam.latent_world.train_collator import valid_action_horizon_steps
from lerobot.policies.lawam.modeling_lawam import (
    LaWAMPolicy,
    _build_native_policy_config,
    _load_lawam_checkpoint_freeze_config,
    _normalize_lawam_checkpoint_state_dict,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config() -> LaWAMConfig:
    return LaWAMConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=2,
        num_video_frames=2,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        lam_ckpt_path="lam.pt",
        lam_yaml_path="lam.yaml",
        lawam_checkpoint_path="dummy.pt",
        base_vlm="dummy-qwen",
        action_hz=20.0,
        embodiment_id=25,
    )


class _FakeCollator:
    def __init__(self) -> None:
        self.samples = None

    def __call__(self, samples):
        self.samples = samples
        return {
            "actions": torch.stack([sample["action"] for sample in samples]),
            "state": torch.stack([sample["state"][-1] for sample in samples]),
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

    def forward(self, batch):
        loss_flow = batch["actions"].mean() * self.weight
        loss_total = loss_flow + batch["state"].mean() * 0.0
        return {"total_loss": loss_total, "loss_flow": loss_flow}

    def predict_action(self, examples, **kwargs):
        del kwargs
        self.predict_calls += 1
        batch_size = len(examples)
        actions = torch.arange(
            batch_size * self.chunk_size * self.action_dim,
            dtype=torch.float32,
        ).reshape(batch_size, self.chunk_size, self.action_dim)
        return {"normalized_actions": actions}


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
    collator = _FakeCollator()
    policy = LaWAMPolicy(config or make_config(), native_model=native_model, native_collator=collator)
    return policy, native_model, collator


def test_factory_registers_lawam() -> None:
    assert get_policy_class("lawam") is LaWAMPolicy
    assert isinstance(make_policy_config("lawam", device="cpu"), LaWAMConfig)


def test_make_pre_post_processors_for_lawam() -> None:
    preprocessor, postprocessor = make_pre_post_processors(make_config(), dataset_stats=None)
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_lawam_defaults_match_native_state_normalization() -> None:
    assert make_config().normalization_mapping["STATE"] is NormalizationMode.MIN_MAX


def test_native_checkpoint_stats_are_used_for_eval_processors(tmp_path) -> None:
    run_dir = tmp_path / "lawam_run"
    final_model_dir = run_dir / "final_model"
    final_model_dir.mkdir(parents=True)
    checkpoint_path = final_model_dir / "pytorch_model.pt"
    checkpoint_path.touch()
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "franka": {
                    "state": {
                        "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "max": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    },
                    "action": {
                        "min": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0],
                        "max": [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 1.0],
                        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        "mask": [True, True, True, True, True, True, False],
                    },
                }
            }
        )
    )
    cfg = make_config()
    cfg.lawam_checkpoint_path = str(checkpoint_path)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
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

    assert torch.allclose(processed_batch[OBS_STATE], torch.zeros(1, 7))
    assert torch.allclose(
        processed_action[:, :2],
        torch.tensor([[11.0, 22.0], [12.0, 24.0]]),
    )
    assert processed_action[:, -1].tolist() == [-1.0, 1.0]


def test_native_checkpoint_freeze_config_is_loaded(tmp_path) -> None:
    run_dir = tmp_path / "lawam_run"
    final_model_dir = run_dir / "final_model"
    final_model_dir.mkdir(parents=True)
    checkpoint_path = final_model_dir / "pytorch_model.pt"
    checkpoint_path.touch()
    (run_dir / "config.yaml").write_text(
        """
trainer:
  freeze:
    freeze_embedding: true
    keep_llm_first_n_layers: 16
    unfreeze_lam_decoder: true
"""
    )
    cfg = make_config()
    cfg.lawam_checkpoint_path = str(checkpoint_path)

    freeze_config = _load_lawam_checkpoint_freeze_config(cfg)

    assert freeze_config is not None
    assert freeze_config.freeze_embedding is True
    assert freeze_config.keep_llm_first_n_layers == 16
    assert freeze_config.unfreeze_lam_decoder is True


def test_lawam_postprocessor_matches_libero_gripper_convention() -> None:
    _, postprocessor = make_pre_post_processors(make_config(), dataset_stats=None)
    action = torch.zeros(2, 7)
    action[0, -1] = 0.0
    action[1, -1] = 1.0

    processed = postprocessor(action)

    assert processed[:, -1].tolist() == [-1.0, 1.0]


def test_native_config_uses_padded_lawam_action_space() -> None:
    cfg = make_config()
    policy_cfg = _build_native_policy_config(cfg)

    assert cfg.action_feature.shape == (7,)
    assert policy_cfg.flow_cfg.action_dim == 32
    assert policy_cfg.flow_cfg.state_dim == 32


def test_checkpoint_normalization_populates_shared_vlm_adapter_alias() -> None:
    state = {"policy_backend.vlm.model.visual.weight": torch.tensor([1.0])}
    model_state = {
        "policy_backend.vlm.model.visual.weight": torch.tensor([0.0]),
        "policy_vlm_adapter.model.model.visual.weight": torch.tensor([0.0]),
    }

    normalized = _normalize_lawam_checkpoint_state_dict(state, model_state)

    assert set(normalized) == set(model_state)


def test_train_collator_masks_only_flow_horizon_steps() -> None:
    assert valid_action_horizon_steps(window_size=50, horizon_sec=1.2, action_hz=20.0) == 24
    assert valid_action_horizon_steps(window_size=8, horizon_sec=0.4, action_hz=20.0) == 8


def test_training_forward_converts_batch_to_lawam_samples() -> None:
    policy, _, collator = make_policy()
    loss, logs = policy.forward(make_batch())

    assert loss.ndim == 0
    assert "loss" in logs
    assert len(collator.samples) == 2
    first = collator.samples[0]
    assert first["primary_videos"].shape == (1, 2, 3, 8, 8)
    assert first["wrist_images"].shape == (1, 3, 8, 8)
    assert first["action"].shape == (4, 7)
    assert first["state"].shape == (1, 7)
    assert first["lang"] == "task 0"
    assert first["embodiment_id"] == 25
    assert first["action_hz"] == 20.0


@pytest.mark.parametrize("primary_features", [None, ["observation.images.front", "observation.images.wrist"]])
def test_primary_image_feature_override(primary_features: list[str] | None) -> None:
    cfg = make_config()
    cfg.primary_image_features = primary_features
    if primary_features is not None:
        cfg.wrist_image_features = []
    policy, _, collator = make_policy(cfg)

    policy.forward(make_batch(batch_size=1))

    expected_views = 1 if primary_features is None else 2
    assert collator.samples[0]["primary_videos"].shape[0] == expected_views


def test_libero_image2_defaults_to_wrist_view() -> None:
    cfg = make_config()
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    batch = {
        "observation.images.image": torch.rand(1, 2, 3, 8, 8),
        "observation.images.image2": torch.rand(1, 2, 3, 8, 8),
        OBS_STATE: torch.rand(1, 7),
        ACTION: torch.rand(1, 4, 7),
        "task": ["task 0"],
    }
    policy, _, collator = make_policy(cfg)

    policy.forward(batch)

    assert collator.samples[0]["primary_videos"].shape[0] == 1
    assert collator.samples[0]["wrist_images"].shape[0] == 1


def test_select_action_uses_action_queue_before_refill() -> None:
    policy, native_model, _ = make_policy()
    batch = make_batch(batch_size=1)

    first = policy.select_action(batch)
    second = policy.select_action(batch)

    assert native_model.predict_calls == 1
    assert first.shape == (1, 7)
    assert second.shape == (1, 7)
    assert not torch.equal(first, second)
