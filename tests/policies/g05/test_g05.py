# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.g05.configuration_g05 import G05_EMBODIMENT_MAPPINGS, G05Config
from lerobot.policies.g05.modeling_g05 import G05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME


class TinyG05Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(20, 20)
        self.last_samples = None

    def predict_action(self, batch):
        self.last_samples = batch["samples"]
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


def test_libero_and_atomic4_are_distinct_validated_mappings():
    with pytest.raises(ValueError, match="27D"):
        G05Config(
            checkpoint_profile="custom",
            embodiment="atomic_4",
            raw_state_dim=16,
            raw_action_dim=12,
            camera_order=(
                "observation.images.robot0_agentview_left",
                "observation.images.robot0_eye_in_hand",
                "observation.images.robot0_agentview_right",
            ),
        )

    cfg = G05Config(
        checkpoint_profile="custom",
        embodiment="atomic_4",
        raw_state_dim=16,
        raw_action_dim=12,
        policy_state_dim=27,
        policy_action_dim=27,
        camera_order=(
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ),
    )
    assert cfg.embodiment == "atomic_4"


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


def test_atomic4_projection_has_mobile_base_control_mode_and_exact_inverse():
    config = G05Config(
        checkpoint_profile="custom",
        embodiment="atomic_4",
        raw_state_dim=16,
        raw_action_dim=12,
        policy_state_dim=27,
        policy_action_dim=27,
        normalization_mode="identity",
        camera_order=(
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.robot0_agentview_left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 8, 8)
            ),
            "observation.images.robot0_eye_in_hand": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            "observation.images.robot0_agentview_right": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 8, 8)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
        device="cpu",
    )
    preprocessor, postprocessor = make_pre_post_processors(config)
    raw_action = torch.arange(12, dtype=torch.float32).repeat(3, 1)
    batch = {
        OBS_STATE: torch.arange(16, dtype=torch.float32),
        ACTION: raw_action,
        **{camera: torch.zeros(3, 8, 8) for camera in config.camera_order},
        "task": "atomic",
    }

    processed = preprocessor(batch)
    indices = G05_EMBODIMENT_MAPPINGS["atomic_4"]["action"]
    assert torch.equal(processed[ACTION][..., list(indices)], raw_action)
    assert torch.equal(postprocessor(processed[ACTION]), raw_action)
    # Last five raw dimensions are base motion[4] and control mode.
    assert indices[-5:] == (20, 21, 22, 23, 24)


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
    assert action.shape == (1, 4, 20)
    assert metadata["cot_text"] == ["Subtask: move carefully"]


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
