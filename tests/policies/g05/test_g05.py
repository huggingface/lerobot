#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.g05.configuration_g05 import G05_CAMERA_PROFILES, G05_EMBODIMENT_MAPPINGS, G05Config
from lerobot.policies.g05.modeling_g05 import (
    G05_RUNTIME_PREDICT_COT,
    G05GatedDeltaNet,
    G05NativeBackend,
    G05Policy,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME


class TinyG05Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(20, 20)
        self.last_samples = None
        self.last_runtime_predict_cot = None

    def predict_action(self, batch):
        self.last_samples = batch["samples"]
        self.last_runtime_predict_cot = batch[G05_RUNTIME_PREDICT_COT]
        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state.unsqueeze(1)
        step = self.proj(state[:, -1])
        return {
            ACTION: step.unsqueeze(1).expand(-1, 4, -1),
            "ar_action": (step + 1).unsqueeze(1).expand(-1, 4, -1),
            "cot_text": ["Subtask: move carefully"] * step.shape[0],
        }

    def forward(self, batch):
        prediction = self.proj(batch[OBS_STATE][:, -1])
        target = batch[ACTION][:, 0]
        loss = torch.nn.functional.mse_loss(prediction, target)
        return loss, {"fm_loss": loss.detach()}


class GroupedTinyG05Backend(TinyG05Backend):
    def __init__(self):
        super().__init__()
        self.action_scale = nn.Parameter(torch.ones(()))
        self.vision_scale = nn.Parameter(torch.ones(()))
        self.optim_kwargs = None

    def get_optim_param_groups(
        self,
        lr,
        weight_decay,
        apply_decay_on_norm_and_bias=False,
        backbone_lr_multiplier=1.0,
        vision_lr_multiplier=1.0,
    ):
        self.optim_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            "apply_decay_on_norm_and_bias": apply_decay_on_norm_and_bias,
            "backbone_lr_multiplier": backbone_lr_multiplier,
            "vision_lr_multiplier": vision_lr_multiplier,
        }
        return [
            {
                "params": [self.proj.weight, self.proj.bias],
                "lr": lr * backbone_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "backbone_decay",
            },
            {
                "params": [self.action_scale],
                "lr": lr,
                "weight_decay": 0.0,
                "name": "action_no_decay",
            },
            {
                "params": [self.vision_scale],
                "lr": lr * backbone_lr_multiplier * vision_lr_multiplier,
                "weight_decay": 0.0,
                "name": "vision_no_decay",
            },
        ]


class TinyLanguageTrainingBackend(G05NativeBackend):
    def __init__(self, *, ce_weight: float, z_loss_scale: float = 0.0):
        nn.Module.__init__(self)
        self.model_config = {
            "ar": {"ce_weight": ce_weight, "ce_z_loss_scale": z_loss_scale},
            "continuous_action": False,
            "discrete_action": True,
            "predict_cot": True,
        }
        self.head = nn.Linear(2, 3, bias=False)
        self.head.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, -0.5],
                    [-0.25, 0.75],
                    [0.5, 0.25],
                ]
            )
        )
        self.model = SimpleNamespace(vlm=SimpleNamespace(logits=self.head))
        self.hidden = nn.Parameter(
            torch.tensor(
                [
                    [0.5, -0.5],
                    [1.0, 0.25],
                    [-0.25, 0.75],
                ]
            )
        )
        self.processor = SimpleNamespace(
            encode_train=lambda samples, device, action_codec: SimpleNamespace(
                labels=torch.tensor([[-100, 0, 2]], device=device),
                token_types=torch.zeros(1, 3, device=device),
                split_index=3,
            )
        )
        self.action_tokenizer = None

    def _proprio(self, samples, device):
        return torch.zeros(len(samples), 1, 2, device=device)

    def _prefill(self, sequence, pixel_values, proprio):
        return (
            self.hidden.unsqueeze(0),
            object(),
            torch.zeros(3, 1, 3, dtype=torch.long),
        )


def _features():
    return {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }


def _config(**kwargs):
    normalization_mode = kwargs.pop("normalization_mode", "identity")
    return G05Config(
        checkpoint_profile="custom",
        normalization_mode=normalization_mode,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=4,
        n_action_steps=kwargs.pop("n_action_steps", 4),
        device="cpu",
        **kwargs,
    )


def _policy_batch(task: str = "  Pick café cup\nverbatim  "):
    return {
        OBS_STATE: torch.zeros(1, 1, 20),
        ACTION: torch.zeros(1, 4, 20),
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        "observation.images.wrist_image": torch.zeros(1, 3, 8, 8),
        "task": [task],
        "proprio_dim_is_pad": torch.zeros(20, dtype=torch.bool),
    }


def test_factory_wiring_is_lazy():
    assert make_policy_config("g05", checkpoint_profile="custom").type == "g05"
    assert get_policy_class("g05") is G05Policy


def test_system2_fm_only_builder_uses_exact_cot_template_without_action_tokens():
    config = _config(
        action_head="flow",
        runtime_system="system2",
        predict_cot=True,
        discrete_action=False,
        continuous_action=True,
        return_continuous_action=True,
        processor_metadata={
            "samples_builder": {
                "_target_": ("g05.data_processor.processor.samples_builder.SubtaskCoTBuilderFMOnly")
            }
        },
    )

    assert "<prompt_text_!>\n<EOC><atomic_task_text>|Action: <EOV><eos>" in config.prompt_template
    assert "<action_action" not in config.prompt_template


def test_so101_runtime_pads_optional_left_wrist():
    config = G05Config(
        checkpoint_profile="g05-so101",
        embodiment="so100",
        action_head="flow",
        runtime_system="system2",
        predict_cot=True,
        discrete_action=True,
        continuous_action=True,
        return_continuous_action=True,
        policy_action_dim=20,
        policy_state_dim=20,
        raw_action_dim=6,
        raw_state_dim=6,
        chunk_size=32,
        n_action_steps=16,
        normalization_mode="identity",
        camera_order=(
            "observation.images.exterior",
            "observation.images.wrist_left",
            "observation.images.wrist_right",
        ),
        camera_sizes={
            "observation.images.exterior": (8, 8),
            "observation.images.wrist_left": (8, 8),
            "observation.images.wrist_right": (8, 8),
        },
        optional_camera_keys=("observation.images.wrist_left",),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.exterior": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            "observation.images.wrist_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
    )
    preprocessor, _ = make_pre_post_processors(config)

    processed = preprocessor(
        {
            OBS_STATE: torch.zeros(6),
            "observation.images.exterior": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "observation.images.wrist_right": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "task": "pick up the cube",
        }
    )

    assert processed["observation.images.wrist_left"].shape == (1, 3, 8, 8)
    assert torch.all(processed["observation.images.wrist_left"] == -1)
    assert not processed["action_dim_is_pad"][0, 10:16].any()


def test_libero_runtime_executes_ten_step_window_and_binarizes_gripper():
    config = G05Config(
        checkpoint_profile="custom",
        embodiment="libero",
        action_head="flow",
        discrete_action=False,
        continuous_action=True,
        return_continuous_action=True,
        chunk_size=32,
        n_action_steps=10,
        normalization_mode="identity",
        libero_gripper_binarize=True,
    )
    _, postprocessor = make_pre_post_processors(config)
    policy_action = torch.zeros(5, 20)
    policy_action[:, 19] = torch.tensor([0.0, 0.5, 1.0, -0.2, 1.2])

    env_action = postprocessor(policy_action)

    torch.testing.assert_close(env_action[:, -1], torch.tensor([1.0, 1.0, -1.0, 1.0, -1.0]))


def test_select_action_discards_tail_beyond_execution_window():
    config = _config(n_action_steps=2)
    policy = G05Policy(config, backend=TinyG05Backend())
    calls = 0

    def predict_action_chunk(batch, **kwargs):
        nonlocal calls
        calls += 1
        return torch.full((1, 4, 20), float(calls))

    policy.predict_action_chunk = predict_action_chunk
    batch = _policy_batch()

    assert policy.select_action(batch)[0, 0].item() == 1
    assert policy.select_action(batch)[0, 0].item() == 1
    assert policy.select_action(batch)[0, 0].item() == 2
    assert calls == 2


def test_libero_projection_mask_and_inverse_roundtrip():
    config = _config()
    preprocessor, postprocessor = make_pre_post_processors(config)
    raw_action = torch.arange(7, dtype=torch.float32).repeat(4, 1)
    batch = {
        OBS_STATE: torch.arange(7, dtype=torch.float32),
        ACTION: raw_action,
        "observation.images.image": torch.zeros(3, 8, 8),
        "observation.images.wrist_image": torch.zeros(3, 8, 8),
        "task": "test",
    }

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape == (1, 20)
    assert processed[ACTION].shape == (4, 20)
    assert processed["action_dim_is_pad"].shape == (1, 20)
    assert processed["action_dim_is_pad"].sum() == 13
    assert torch.equal(processed["action_op_mask"], ~processed["action_dim_is_pad"])
    assert processed["action_parts_meta"] == {
        "left_control": 9,
        "left_gripper": 1,
        "right_control": 9,
        "right_gripper": 1,
    }
    assert torch.equal(processed[ACTION][:, [10, 11, 12, 13, 14, 15, 19]], raw_action)
    restored = postprocessor(processed[ACTION])
    assert torch.equal(restored, raw_action)


def test_inference_without_ground_truth_action_still_emits_action_dimension_mask():
    config = _config()
    preprocessor, _ = make_pre_post_processors(config)

    processed = preprocessor(
        {
            OBS_STATE: torch.arange(7, dtype=torch.float32),
            "observation.images.image": torch.zeros(3, 8, 8),
            "observation.images.wrist_image": torch.zeros(3, 8, 8),
            "task": "inference",
        }
    )

    assert processed["action_dim_is_pad"].shape == (1, 20)
    assert processed["action_dim_is_pad"].sum() == 13


def test_lerobot_libero_two_finger_state_matches_author_first_qpos_contract():
    config = _config()
    preprocessor, _ = make_pre_post_processors(config)
    env_state = torch.arange(8, dtype=torch.float32)

    processed = preprocessor(
        {
            OBS_STATE: env_state,
            "observation.images.image": torch.zeros(3, 8, 8),
            "observation.images.wrist_image": torch.zeros(3, 8, 8),
            "task": "libero env",
        }
    )

    checkpoint_slots = G05_EMBODIMENT_MAPPINGS["libero"]["state"]
    assert torch.equal(processed[OBS_STATE][0, list(checkpoint_slots)], env_state[:7])


def test_quantile_mode_refuses_minmax_substitution():
    config = _config(normalization_mode="q01_q99")
    stats = {
        OBS_STATE: {"min": torch.zeros(7), "max": torch.ones(7)},
        ACTION: {"min": torch.zeros(7), "max": torch.ones(7)},
    }
    with pytest.raises(ValueError, match="real q01/q99"):
        make_pre_post_processors(config, dataset_stats=stats)


def test_checkpoint_normalization_clips_to_author_finite_range():
    config = _config(normalization_mode="q01_q99", normalization_clip=(-5.0, 5.0))
    stats = {
        OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7)},
        ACTION: {"q01": torch.zeros(4, 7), "q99": torch.ones(4, 7)},
    }
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)

    processed = preprocessor(
        {
            OBS_STATE: torch.full((7,), -100.0),
            ACTION: torch.full((4, 7), 100.0),
            "observation.images.image": torch.zeros(3, 8, 8),
            "observation.images.wrist_image": torch.zeros(3, 8, 8),
            "task": "clip",
        }
    )

    assert processed[OBS_STATE].min() == -5
    assert processed[ACTION].max() == 5


def test_stepwise_quantiles_constant_dimension_are_finite_and_serializable(tmp_path: Path):
    config = _config(
        normalization_mode="q01_q99",
        use_stepwise_action_norm=True,
        n_action_steps=2,
    )
    q01_action = torch.zeros(4, 7)
    q99_action = torch.ones(4, 7)
    q99_action[:, 2] = 0
    stats = {
        OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7)},
        ACTION: {"q01": q01_action, "q99": q99_action},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    processed = preprocessor(
        {
            OBS_STATE: torch.zeros(7),
            ACTION: torch.zeros(4, 7),
            "observation.images.image": torch.zeros(3, 8, 8),
            "observation.images.wrist_image": torch.zeros(3, 8, 8),
            "task": "constant",
        }
    )
    assert torch.isfinite(processed[ACTION]).all()
    torch.testing.assert_close(postprocessor(processed[ACTION]), torch.zeros(4, 7))

    step_q01 = torch.arange(4, dtype=torch.float32).view(4, 1).expand(4, 7)
    step_stats = {
        OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7)},
        ACTION: {"q01": step_q01, "q99": step_q01 + 2},
    }
    _, stepwise_postprocessor = make_pre_post_processors(config, dataset_stats=step_stats)
    normalized_action = torch.zeros(1, config.policy_action_dim)
    torch.testing.assert_close(stepwise_postprocessor(normalized_action), torch.ones(1, 7))
    torch.testing.assert_close(stepwise_postprocessor(normalized_action), torch.full((1, 7), 2.0))
    torch.testing.assert_close(stepwise_postprocessor(normalized_action), torch.ones(1, 7))
    stepwise_postprocessor.reset()
    torch.testing.assert_close(stepwise_postprocessor(normalized_action), torch.ones(1, 7))

    preprocessor.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    assert [step.__class__.__name__ for step in loaded.steps] == [
        step.__class__.__name__ for step in preprocessor.steps
    ]


def test_exact_raw_task_reaches_author_command_and_head_selection():
    backend = TinyG05Backend()
    policy = G05Policy(_config(), backend=backend)
    raw_task = "  把 red cup 放到左边\nexactly as written  "

    action, metadata = policy.predict_action_chunk_with_runtime(_policy_batch(), task=raw_task)

    assert backend.last_samples[0]["command"] == raw_task
    assert backend.last_runtime_predict_cot is False
    assert action.shape == (1, 4, 20)
    assert "cot_text" not in metadata


def test_same_predict_cot_checkpoint_switches_prompt_and_backend_runtime_path():
    backend = TinyG05Backend()
    policy = G05Policy(_config(predict_cot=True, runtime_system="system2"), backend=backend)

    _, system1_metadata = policy.predict_action_chunk_with_runtime(
        _policy_batch(),
        task="pick",
        system_mode="system1",
    )
    system1_sample = backend.last_samples[0]
    assert backend.last_runtime_predict_cot is False
    assert "prompt" not in system1_sample
    assert "<atomic_task_text>" not in system1_sample["template"]
    assert "cot_text" not in system1_metadata

    _, system2_metadata = policy.predict_action_chunk_with_runtime(
        _policy_batch(),
        task="pick",
        system_mode="system2",
    )
    system2_sample = backend.last_samples[0]
    assert backend.last_runtime_predict_cot is True
    assert system2_sample["prompt"] == "predict subtask"
    assert "<atomic_task_text>" in system2_sample["template"]
    assert system2_metadata["cot_text"] == ["Subtask: move carefully"]


def test_system1_config_disables_cot_on_predict_cot_checkpoint_without_override():
    backend = TinyG05Backend()
    policy = G05Policy(_config(predict_cot=True, runtime_system="system1"), backend=backend)

    _, metadata = policy.predict_action_chunk_with_runtime(_policy_batch(), task="pick")

    assert backend.last_runtime_predict_cot is False
    assert "<atomic_task_text>" not in backend.last_samples[0]["template"]
    assert "cot_text" not in metadata


def test_native_backend_uses_per_call_cot_gate_instead_of_checkpoint_default():
    class TinyNativeBackend(G05NativeBackend):
        def __init__(self):
            nn.Module.__init__(self)
            self.model_config = {
                "predict_cot": True,
                "continuous_action": True,
                "discrete_action": False,
                "ar": {"max_new_tokens": 4},
            }
            self.processor = SimpleNamespace(
                encode_inference=lambda samples, device: SimpleNamespace(
                    token_types=torch.zeros(len(samples), 1)
                ),
                eov_token_id=2,
                decode=lambda ids: "Subtask: pick",
            )
            self.generated = 0

        def _prefill(self, sequence, pixel_values, proprio):
            batch_size = len(proprio)
            return (
                torch.zeros(batch_size, 1, 4),
                object(),
                torch.zeros(3, batch_size, 1, dtype=torch.long),
            )

        def _generate_text(self, last_hidden, *, token_types, positions, cache, **kwargs):
            self.generated += 1
            generated = torch.tensor([[1, 2]] * last_hidden.shape[0])
            return generated, cache, last_hidden, token_types, positions

        def _infer_flow(self, *, token_types, **kwargs):
            return torch.zeros(token_types.shape[0], 4, 20)

    backend = TinyNativeBackend()
    batch = {
        "samples": [{"proprio": torch.zeros(1, 20)}],
        "pixel_values": {"camera": torch.zeros(1, 1, 3, 8, 8)},
    }

    system1 = backend.predict_action({**batch, G05_RUNTIME_PREDICT_COT: False})
    assert backend.generated == 0
    assert "cot_text" not in system1

    system2 = backend.predict_action({**batch, G05_RUNTIME_PREDICT_COT: True})
    assert backend.generated == 1
    assert system2["cot_text"] == ["Subtask: pick"]


def test_native_training_applies_ar_loss_config_and_reaches_language_head():
    ce_weight = 0.25
    z_loss_scale = 0.2
    backend = TinyLanguageTrainingBackend(ce_weight=ce_weight, z_loss_scale=z_loss_scale)
    batch = {
        "samples": [{}],
        "pixel_values": {"camera": torch.zeros(1, 1, 3, 2, 2)},
    }

    loss, metrics = backend(batch)

    logits = backend.head(backend.hidden[:2])
    labels = torch.tensor([0, 2])
    expected = (
        ce_weight
        * (
            torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            + z_loss_scale * torch.logsumexp(logits, dim=-1).square()
        ).mean()
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["ce_loss"], expected)
    loss.backward()
    language_head_grad = backend.head.weight.grad
    assert language_head_grad is not None
    assert torch.isfinite(language_head_grad).all()
    assert language_head_grad.abs().sum() > 0

    disabled = TinyLanguageTrainingBackend(ce_weight=0.0)
    disabled_loss, disabled_metrics = disabled(batch)
    assert disabled_loss.requires_grad
    torch.testing.assert_close(disabled_loss, torch.zeros_like(disabled_loss))
    torch.testing.assert_close(disabled_metrics["ce_loss"], torch.zeros_like(disabled_loss))


def test_author_action_payload_fills_required_tokenizer_metadata():
    policy = G05Policy(_config(), backend=TinyG05Backend())

    prepared = policy._prepare_author_batch(_policy_batch())

    assert set(prepared["samples"][0]["action"]) == {
        "value",
        "action_dim_is_pad",
        "action_op_mask",
        "parts_meta",
    }


def test_system2_training_target_is_forwarded_without_replacing_operator_task():
    config = _config(predict_cot=True, runtime_system="system2")
    policy = G05Policy(config, backend=TinyG05Backend())
    batch = _policy_batch("  operator task\n")
    batch["atomic_task"] = ["grasp the cup"]

    prepared = policy._prepare_author_batch(batch)

    assert prepared["samples"][0]["command"] == "  operator task\n"
    assert prepared["samples"][0]["atomic_task"] == "Subtask: grasp the cup"


def test_system2_recipe_subtask_target_selects_author_template():
    policy = G05Policy(_config(predict_cot=True, runtime_system="system2"), backend=TinyG05Backend())
    batch = _policy_batch("operator task")
    batch["messages"] = [
        [
            {"role": "user", "content": "operator task"},
            {"role": "assistant", "content": "Subtask: grasp the cup"},
        ]
    ]
    batch["target_message_indices"] = [[1]]

    sample = policy._prepare_author_batch(batch)["samples"][0]

    assert sample["command"] == "operator task"
    assert sample["prompt"] == "predict subtask"
    assert sample["atomic_task"] == "Subtask: grasp the cup"
    assert "<EOC><atomic_task_text>|Action: <EOV><action_action>|<eos>" in sample["template"]


def test_system2_recipe_bbox_and_subtask_use_checkpoint_field_order():
    policy = G05Policy(_config(predict_cot=True, runtime_system="system2"), backend=TinyG05Backend())
    batch = _policy_batch("operator task")
    batch["messages"] = [
        [
            {"role": "user", "content": "operator task"},
            {
                "role": "assistant",
                "content": (
                    'BBoxJSON: {"detections": [{"label": "cup", "bbox_format": "xyxy", '
                    '"bbox": [20, 10, 100, 50]}]}'
                ),
            },
            {"role": "assistant", "content": "Subtask: grasp the cup"},
        ]
    ]
    batch["target_message_indices"] = [[1, 2]]
    batch["g05_bbox_image_size"] = (100, 200)

    sample = policy._prepare_author_batch(batch)["samples"][0]

    assert sample["prompt"] == "predict bbox, subtask and action"
    assert sample["bbox"] == "BBox: cup <loc0102><loc0102><loc0512><loc0512>"
    assert sample["atomic_task"] == "Subtask: grasp the cup"
    assert "<EOC><bbox_text>|<atomic_task_text>|Action:" in sample["template"]


def test_system2_recipe_no_cot_branch_uses_action_only_training_template():
    policy = G05Policy(_config(predict_cot=True, runtime_system="system2"), backend=TinyG05Backend())
    batch = _policy_batch("operator task")
    batch["messages"] = [[{"role": "user", "content": "operator task"}]]
    batch["target_message_indices"] = [[]]

    sample = policy._prepare_author_batch(batch)["samples"][0]

    assert "prompt" not in sample
    assert "atomic_task" not in sample
    assert "<chat_assistant_prefix>Action: <EOV><EOC><action_action>|<eos>" in sample["template"]


def test_recipe_preprocessor_resolves_lerobot_subtask_and_bbox_annotations():
    pytest.importorskip("datasets", reason="recipe rendering requires lerobot[dataset]")
    config = _config(
        predict_cot=True,
        runtime_system="system2",
        recipe_path="recipes/g05_bbox_subtask.yaml",
    )
    preprocessor, _ = make_pre_post_processors(config)
    policy = G05Policy(config, backend=TinyG05Backend())
    raw = {
        OBS_STATE: torch.zeros(7),
        ACTION: torch.zeros(4, 7),
        "observation.images.image": torch.zeros(3, 100, 200, dtype=torch.uint8),
        "observation.images.wrist_image": torch.zeros(3, 100, 200, dtype=torch.uint8),
        "task": "operator task",
        "timestamp": torch.tensor(0.0),
        "language_persistent": [
            {
                "role": "assistant",
                "content": "grasp the cup",
                "style": "subtask",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            }
        ],
        "language_events": [
            {
                "role": "assistant",
                "content": (
                    '{"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [20, 10, 100, 50]}]}'
                ),
                "style": "vqa",
                "camera": "observation.images.exterior",
                "tool_calls": None,
            }
        ],
    }

    processed = next(
        candidate
        for sample_index in range(100)
        if (candidate := preprocessor({**raw, "index": torch.tensor(sample_index)}))["target_message_indices"]
        == [[1, 2]]
    )
    sample = policy._prepare_author_batch(processed)["samples"][0]

    assert "language_persistent" not in processed
    assert "language_events" not in processed
    assert sample["bbox"] == "BBox: cup <loc0102><loc0102><loc0512><loc0512>"
    assert sample["atomic_task"] == "Subtask: grasp the cup"
    assert "<EOC><bbox_text>|<atomic_task_text>|Action:" in sample["template"]


def test_author_inference_payload_synthesizes_required_dummy_action():
    policy = G05Policy(_config(), backend=TinyG05Backend())
    batch = _policy_batch()
    del batch[ACTION]

    prepared = policy._prepare_author_batch(batch)

    assert prepared["samples"][0]["action"]["value"].shape == (4, 20)


def test_policy_to_moves_non_module_action_tokenizer_sidecar():
    class TrackingTokenizer:
        device = None

        def to(self, device):
            self.device = device

    backend = TinyG05Backend()
    backend.action_tokenizer = TrackingTokenizer()
    policy = G05Policy(_config(), backend=backend).to("cpu")

    assert backend.action_tokenizer.device == next(policy.parameters()).device


def test_author_inference_precision_preserves_declared_fp32_parameters():
    class MixedPrecisionBackend(TinyG05Backend):
        def __init__(self):
            super().__init__()
            self.bulk_weight = nn.Parameter(torch.ones(2))
            self.precision_weight = nn.Parameter(torch.ones(2))

        def apply_fp32_params(self):
            self.precision_weight.data = self.precision_weight.data.float()

    backend = MixedPrecisionBackend()
    policy = G05Policy(_config(), backend=backend)

    policy._apply_author_inference_precision()

    assert backend.bulk_weight.dtype is torch.bfloat16
    assert backend.precision_weight.dtype is torch.float32


def test_system2_precision_keeps_tied_lm_head_compatible_with_fp32_final_norm():
    class CoTPrecisionBackend(TinyG05Backend):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.vlm = nn.Module()
            self.model.vlm.input_proj = nn.Embedding(8, 4)
            self.model.vlm.output_proj = nn.Linear(4, 8, bias=False)
            self.model.vlm.output_proj.weight = self.model.vlm.input_proj.weight

        def apply_fp32_params(self):
            pass

    backend = CoTPrecisionBackend()
    policy = G05Policy(_config(predict_cot=True, runtime_system="system2"), backend=backend)

    policy._apply_author_inference_precision()

    assert backend.model.vlm.output_proj.weight.dtype is torch.float32
    assert backend.model.vlm.input_proj.weight is backend.model.vlm.output_proj.weight


def test_batch_two_preserves_each_raw_task_and_every_camera_slot():
    backend = TinyG05Backend()
    policy = G05Policy(_config(), backend=backend)
    batch = _policy_batch()
    batch[OBS_STATE] = batch[OBS_STATE].expand(2, -1, -1)
    batch[ACTION] = batch[ACTION].expand(2, -1, -1)
    batch["observation.images.image"] = batch["observation.images.image"].expand(2, -1, -1, -1)
    batch["observation.images.wrist_image"] = batch["observation.images.wrist_image"].expand(2, -1, -1, -1)
    batch["proprio_dim_is_pad"] = torch.zeros(2, 20, dtype=torch.bool)
    batch["task"] = [" first\n", "第二个 task"]

    action = policy.predict_action_chunk(batch)

    assert action.shape == (2, 4, 20)
    assert [sample["command"] for sample in backend.last_samples] == batch["task"]
    assert all(sample["image0"] == (224, 224) for sample in backend.last_samples)
    assert all(sample["image1"] == (224, 224) for sample in backend.last_samples)


def test_forward_backward_update_and_save_reload(tmp_path: Path):
    policy = G05Policy(_config(), backend=TinyG05Backend())
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    loss, metrics = policy(_policy_batch("train"))
    loss.backward()
    grad_norm = torch.stack(
        [parameter.grad.norm() for parameter in policy.parameters() if parameter.grad is not None]
    ).sum()
    assert torch.isfinite(loss)
    assert grad_norm > 0 and torch.isfinite(grad_norm)
    optimizer.step()
    assert metrics is not None and metrics["fm_loss"] >= 0

    policy.save_pretrained(tmp_path)
    reloaded = G05Policy.from_pretrained(
        tmp_path, backend=TinyG05Backend(), local_files_only=True, strict=True
    )
    expected = policy.predict_action_chunk(_policy_batch("save"))
    actual = reloaded.predict_action_chunk(_policy_batch("save"))
    torch.testing.assert_close(actual, expected)


def test_gated_delta_cached_suffix_matches_tokenwise_decode():
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    torch.manual_seed(1337)
    layer = G05GatedDeltaNet(config, 0).eval()
    prefix = torch.randn(2, 7, config.hidden_size)
    suffix = torch.randn(2, 3, config.hidden_size)
    bulk_cache = DynamicCache(config=config)
    tokenwise_cache = DynamicCache(config=config)

    with torch.inference_mode():
        layer(prefix, bulk_cache)
        layer(prefix, tokenwise_cache)
        bulk = layer(suffix, bulk_cache)
        tokenwise = torch.cat(
            [layer(suffix[:, index : index + 1], tokenwise_cache) for index in range(suffix.shape[1])],
            dim=1,
        )

    torch.testing.assert_close(bulk, tokenwise, atol=2e-7, rtol=2e-6)


def test_from_pretrained_constructs_on_meta_and_assigns_directly(tmp_path: Path, monkeypatch):
    reference = G05Policy(_config(), backend=TinyG05Backend())
    reference.save_pretrained(tmp_path)
    constructed_on_meta = False

    def make_tiny_backend(config):
        nonlocal constructed_on_meta
        backend = TinyG05Backend()
        constructed_on_meta = next(backend.parameters()).is_meta
        return backend

    monkeypatch.setattr("lerobot.policies.g05.modeling_g05._native_backend", make_tiny_backend)
    loaded = G05Policy.from_pretrained(tmp_path, local_files_only=True, strict=True)

    assert constructed_on_meta
    assert not next(loaded.parameters()).is_meta
    assert next(loaded.parameters()).device.type == "cpu"
    torch.testing.assert_close(loaded.backend.proj.weight, reference.backend.proj.weight)


def test_meta_loader_materializes_transformers_rotary_buffers():
    config = Qwen3_5TextConfig(
        hidden_size=32,
        num_attention_heads=2,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    backend = G05NativeBackend.__new__(G05NativeBackend)
    nn.Module.__init__(backend)
    with torch.device("meta"):
        backend.text_rotary = Qwen3_5TextRotaryEmbedding(config)
        backend.vision_rotary = Qwen3_5VisionRotaryEmbedding(dim=8)

    assert all(buffer.is_meta for buffer in backend.buffers())
    backend.materialize_runtime_buffers("cpu")
    assert all(not buffer.is_meta and buffer.device.type == "cpu" for buffer in backend.buffers())


def test_training_forward_uses_policy_autocast_context(monkeypatch):
    policy = G05Policy(_config(), backend=TinyG05Backend())
    autocast_calls = []

    class AutocastContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def track_autocast(**kwargs):
        autocast_calls.append(kwargs)
        return AutocastContext()

    monkeypatch.setattr(torch, "autocast", track_autocast)

    policy(_policy_batch("train"))

    assert autocast_calls == [{"device_type": "cpu", "dtype": torch.bfloat16, "enabled": False}]


def test_save_pretrained_copies_required_gated_sidecars_portably(tmp_path: Path):
    source = tmp_path / "checkpoint"
    processor = source / "hf_processor"
    processor.mkdir(parents=True)
    (processor / "tokenizer.json").write_text("{}")
    tokenizer = source / "action_tokenizer.pt"
    torch.save({"codec": "ActionCodec"}, tokenizer)
    for name in ("LICENSE-G0.5", "NOTICE"):
        (source / name).write_text("{}")
    config = _config(
        author_model_config={
            "hf_processor_path": str(processor),
            "AT_CONFIG": {"ckpt_dir": str(tokenizer)},
        }
    )
    output = tmp_path / "saved"

    G05Policy(config, backend=TinyG05Backend()).save_pretrained(output)

    assert (output / "hf_processor" / "tokenizer.json").is_file()
    assert (output / "action_tokenizer.pt").is_file()
    assert (output / "LICENSE-G0.5").is_file()
    loaded_config = PreTrainedConfig.from_pretrained(output)
    assert isinstance(loaded_config, G05Config)
    assert loaded_config.author_model_config["hf_processor_path"] == "hf_processor"
    assert loaded_config.author_model_config["AT_CONFIG"]["ckpt_dir"] == "action_tokenizer.pt"


def test_tiny_fixed_batch_overfit_reduces_loss():
    policy = G05Policy(_config(), backend=TinyG05Backend())
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=5e-2)
    batch = _policy_batch("overfit")
    initial = policy(batch)[0].item()
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()
    final = policy(batch)[0].item()
    assert final < initial * 0.25


def test_training_preset_uses_author_optimizer_parameter_groups():
    config = _config(
        optimizer_lr=2e-4,
        optimizer_weight_decay=0.03,
        optimizer_backbone_lr_multiplier=0.5,
        optimizer_vision_lr_multiplier=0.2,
        optimizer_apply_decay_on_norm_and_bias=True,
    )
    backend = GroupedTinyG05Backend()
    policy = G05Policy(config, backend=backend)

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())

    assert backend.optim_kwargs == {
        "lr": 2e-4,
        "weight_decay": 0.03,
        "apply_decay_on_norm_and_bias": True,
        "backbone_lr_multiplier": 0.5,
        "vision_lr_multiplier": 0.2,
    }
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone_decay",
        "action_no_decay",
        "vision_no_decay",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 2e-4, 2e-5])


@pytest.mark.skipif(
    not os.environ.get("LEROBOT_G05_CHECKPOINT"),
    reason="requires an accepted gated OpenGalaxea/G05 checkpoint and author CUDA environment",
)
def test_gated_checkpoint_loads_strictly():
    checkpoint = Path(os.environ["LEROBOT_G05_CHECKPOINT"])
    policy = G05Policy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    assert policy.config.source_checkpoint_revision


def test_project_stats_passes_dataset_count_through():
    config = _config(normalization_mode="q01_q99")
    stats = {
        OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7), "count": torch.tensor([100])},
        ACTION: {"q01": torch.zeros(7), "q99": torch.ones(7), "count": torch.tensor([100])},
    }

    make_pre_post_processors(config, dataset_stats=stats)


def test_named_embodiment_rebuilds_stale_camera_sizes():
    config = G05Config(
        checkpoint_profile="custom",
        embodiment="robotwin",
        raw_state_dim=14,
        raw_action_dim=14,
        camera_order=G05_CAMERA_PROFILES["robotwin"],
        camera_sizes={
            **dict.fromkeys(G05_CAMERA_PROFILES["robotwin"], (256, 256)),
            "observation.images.stale_camera": (256, 256),
        },
        device="cpu",
    )

    assert set(config.camera_sizes) == set(G05_CAMERA_PROFILES["robotwin"])
