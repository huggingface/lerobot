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

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig
from lerobot.policies.lawam.lam_core.core.lam_model import LatentLAMModel, load_latent_action_model
from lerobot.policies.lawam.lam_core.core.utils.modules import build_modal_block_attention_mask
from lerobot.policies.lawam.latent_world.processor_utils import LatentWorldProcessorSpec
from lerobot.policies.lawam.latent_world.train_collator import (
    LatentWorldTrainCollator,
    valid_action_horizon_steps,
)
from lerobot.policies.lawam.modeling_lawam import (
    LaWAMPolicy,
    _build_freeze_config,
    _build_native_policy_config,
    _normalize_lawam_checkpoint_state_dict,
)
from lerobot.policies.lawam.vlas.qwen3vl import (
    freeze_qwen3vl,
    keep_first_n_llm_layers,
    remove_lm_head,
    unfreeze_last_n_llm_layers,
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
    cfg.lawam_dataset_stats_path = str(run_dir / "dataset_statistics.json")

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
    assert processed_action[:, -1].tolist() == [0.5, 1.0]


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


def test_lawam_postprocessor_matches_libero_gripper_convention() -> None:
    cfg = make_config()
    cfg.clip_normalized_actions = True
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    _, postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
    action = torch.zeros(2, 7)
    action[0, -1] = 0.0
    action[1, -1] = 1.0

    processed = postprocessor(action)

    assert processed[:, -1].tolist() == [-1.0, 1.0]


def test_lawam_gripper_processing_is_opt_in() -> None:
    cfg = make_config()
    cfg.clip_normalized_actions = False
    cfg.pre_snap_gripper_action = False
    cfg.binarize_gripper_action = False
    _, postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
    action = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])

    assert torch.equal(postprocessor(action), action)


def test_lawam_postprocessor_config_round_trip(tmp_path) -> None:
    cfg = make_config()
    cfg.clip_normalized_actions = True
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    cfg.gripper_dim = 3
    cfg.gripper_threshold = 0.25
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    _, loaded_postprocessor = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    configs = [step.get_config() for step in loaded_postprocessor.steps]

    assert {"gripper_dim": 3, "threshold": 0.25} in configs


def test_native_config_uses_padded_lawam_action_space() -> None:
    cfg = make_config()
    policy_cfg = _build_native_policy_config(cfg)

    assert cfg.action_feature.shape == (7,)
    assert policy_cfg.flow_cfg.action_dim == 32
    assert policy_cfg.flow_cfg.state_dim == 32
    assert policy_cfg.action_horizon == 4


def test_lam_constructs_without_pretrained_dino_and_strictly_loads_legacy_checkpoint(tmp_path) -> None:
    model_config = {
        "dim": 32,
        "num_heads": 4,
        "ffn_expansion_factor": 2,
        "enc_layers": 1,
        "codebook_size": 8,
        "code_dim": 8,
        "max_state_dim": 7,
        "num_frames": 2,
        "num_queries": 1,
        "vq_kwargs": {"layer_norm": True},
        "dec_layers": 1,
        "dropout": 0.0,
        "vq_type": "vae",
        "norm_latents": True,
        "norm_latents_type": "ln",
        "enc_add_state": False,
        "enc_modal_mask": True,
        "latent_layer_to_use": -2,
        "multi_input": False,
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
    source_model = LatentLAMModel(**model_config)
    legacy_state = {}
    for key, value in source_model.state_dict().items():
        legacy_key = key.replace(
            "vision_encoder.model.model.layer.",
            "vision_encoder.model.layer.",
            1,
        )
        legacy_state[f"lam.{legacy_key}"] = value
    checkpoint_path = tmp_path / "lam.ckpt"
    torch.save({"state_dict": legacy_state}, checkpoint_path)

    loaded_model = load_latent_action_model(model_config, checkpoint_path=str(checkpoint_path))

    assert set(loaded_model.state_dict()) == set(source_model.state_dict())
    assert all(not parameter.requires_grad for parameter in loaded_model.parameters())


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


def test_action_hz_is_derived_from_dataset_metadata() -> None:
    cfg = make_config()
    cfg.action_hz = None
    cfg.n_action_steps = None

    policy = LaWAMPolicy(
        cfg,
        dataset_meta=SimpleNamespace(fps=25),
        native_model=_FakeNativeLaWAM(),
        native_collator=_FakeCollator(),
    )

    assert policy.config.action_hz == 25.0
    assert policy.config.n_action_steps == 4


def test_dataset_sampling_uses_the_derived_action_horizon() -> None:
    cfg = make_config()
    cfg.action_hz = None
    cfg.chunk_size = 50
    cfg.n_action_steps = None
    dataset_meta = SimpleNamespace(
        fps=25,
        features={ACTION: {}, "observation.images.front": {}},
    )

    delta_timestamps = resolve_delta_timestamps(cfg, dataset_meta)

    assert cfg.action_hz == 25.0
    assert cfg.n_action_steps == 10
    assert delta_timestamps is not None
    assert len(delta_timestamps[ACTION]) == 10
    assert delta_timestamps[ACTION][-1] == pytest.approx(9 / 25)


def test_train_collator_resizes_all_model_inputs() -> None:
    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {
                "input_ids": torch.full((1, 4), 99, dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 256, 256)),
            }

    policy_cfg = SimpleNamespace(
        action_horizon=4,
        flow_cfg=SimpleNamespace(action_dim=7, state_dim=7, horizon_sec=0.4),
        lam_config={"image_hw": (256, 256)},
    )
    collator = LatentWorldTrainCollator(
        policy_cfg=policy_cfg,
        processor_spec=LatentWorldProcessorSpec(model_id="unused", placeholder_token="<ACT_PH>"),
        act_queries=2,
        flow_queries=2,
        enable_primary_video_aug=False,
        enable_primary_random_resized_crop=False,
    )
    collator._processor = FakeProcessor()
    collator._placeholder_token_id = 99
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = {
        "primary_videos": torch.rand(1, 2, 3, 480, 640, device=device),
        "wrist_images": torch.rand(1, 3, 480, 640, device=device),
        "lang": "pick",
        "state": torch.rand(1, 7, device=device),
        "action": torch.rand(4, 7, device=device),
        "embodiment_id": 25,
        "action_hz": 20.0,
    }

    batch = collator([sample])

    assert batch["primary_video"].shape == (1, 2, 3, 256, 256)
    assert batch["actions"].device.type == device.type


def test_action_steps_cannot_exceed_flow_horizon() -> None:
    cfg = make_config()
    cfg.chunk_size = 50
    cfg.n_action_steps = 9

    with pytest.raises(ValueError, match="cannot exceed the flow horizon"):
        make_policy(cfg)


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


def test_saved_policy_config_drops_local_initialization_paths(tmp_path) -> None:
    cfg = make_config()
    cfg.base_vlm_path = "/private/qwen3-vl"
    cfg.lawam_dataset_stats_path = "/private/dataset_statistics.json"
    cfg.hf_cache_dir = "/private/huggingface-cache"
    policy, _, _ = make_policy(cfg)

    policy.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "config.json").read_text())

    assert saved_config["base_vlm"] == "dummy-qwen"
    assert saved_config["base_vlm_path"] is None
    assert saved_config["lam_ckpt_path"] is None
    assert saved_config["lawam_checkpoint_path"] is None
    assert saved_config["lawam_dataset_stats_path"] is None
    assert saved_config["hf_cache_dir"] is None

    loaded_policy = LaWAMPolicy.from_pretrained(
        tmp_path,
        native_model=_FakeNativeLaWAM(),
        native_collator=_FakeCollator(),
    )
    assert torch.equal(loaded_policy.model.weight, policy.model.weight)


def test_base_vlm_path_overrides_portable_model_id_at_runtime() -> None:
    cfg = make_config()

    assert cfg.base_vlm_source == "dummy-qwen"

    cfg.base_vlm_path = "/local/qwen3-vl"

    assert cfg.base_vlm_source == "/local/qwen3-vl"


def test_base_vlm_rejects_local_paths() -> None:
    with pytest.raises(ValueError, match="Use `base_vlm_path` for a local Qwen directory"):
        LaWAMConfig(base_vlm="/local/qwen3-vl")


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


def test_inference_examples_convert_device_images_to_numpy() -> None:
    policy, _, _ = make_policy()
    batch = make_batch(batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    examples = policy._prepare_infer_examples(batch)

    assert examples[0]["primary_image"][0].shape == (8, 8, 3)
    assert examples[0]["wrist_image"][0].shape == (8, 8, 3)
    assert examples[0]["state"].shape == (1, 7)


def test_select_action_uses_action_queue_before_refill() -> None:
    policy, native_model, _ = make_policy()
    batch = make_batch(batch_size=1)

    first = policy.select_action(batch)
    second = policy.select_action(batch)

    assert native_model.predict_calls == 1
    assert first.shape == (1, 7)
    assert second.shape == (1, 7)
    assert not torch.equal(first, second)
