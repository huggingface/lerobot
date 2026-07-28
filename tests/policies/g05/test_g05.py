# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.g05.configuration_g05 import G05_EMBODIMENT_MAPPINGS, G05Config
from lerobot.policies.g05.convert_g05_checkpoint import (
    _camera_sizes,
    _profile_config,
    convert_checkpoint,
    convert_dataset_stats,
    convert_state_dict,
    save_converted_state_dict,
)
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


def _base_r1lite_hydra():
    state_action = [
        {"key": "left_arm", "shape": 6},
        {"key": "left_gripper", "shape": 1},
        {"key": "right_arm", "shape": 6},
        {"key": "right_gripper", "shape": 1},
    ]
    images = [
        {
            "key": "head_rgb",
            "camera_type": "exterior",
            "lerobot_key": "observation.images.head_rgb",
            "shape": [3, 224, 224],
        },
        {
            "key": "left_wrist_rgb",
            "camera_type": "wrist_left",
            "lerobot_key": "observation.images.left_wrist_rgb",
            "shape": [3, 224, 224],
        },
        {
            "key": "right_wrist_rgb",
            "camera_type": "wrist_right",
            "lerobot_key": "observation.images.right_wrist_rgb",
            "shape": [3, 224, 224],
        },
    ]
    return {
        "model": {
            "model_arch": {
                "action_dim": 27,
                "proprio_dim": 27,
                "num_input_images": 18,
                "horizon_steps": 32,
                "predict_cot": True,
                "discrete_action": True,
                "continuous_action": True,
            },
            "processor": {
                "num_obs_steps": 6,
                "use_stepwise_action_norm": True,
                "camera_size_config": {
                    "exterior": [256, 256],
                    "wrist_left": [256, 256],
                    "wrist_right": [256, 256],
                },
                "samples_builder": {
                    "_target_": "g05.data_processor.processor.samples_builder.MixedSamplesBuilder"
                },
            },
        },
        "data": {
            "action_size": 32,
            "processors": {
                "galaxea_r1lite": {
                    "shape_meta": {
                        "state": state_action,
                        "action": state_action,
                        "images": images,
                    },
                    "norm_default_mode": "z-score-tail",
                    "norm_exception_mode": {
                        "state": {"left_gripper": "q01/q99", "right_gripper": "q01/q99"},
                        "action": {"left_gripper": "q01/q99", "right_gripper": "q01/q99"},
                    },
                    "action_filter": {
                        "_target_": (
                            "g05.data_processor.processor.galaxea_action_processor.R1LiteJointActionFilter"
                        ),
                        "joint_threshold": 0.002,
                        "gripper_threshold": 0.01,
                    },
                }
            },
        },
    }


def _base_r1lite_stats():
    result = {"state": {}, "action": {}}
    for category in result:
        for key, width in (
            ("left_arm", 6),
            ("left_gripper", 1),
            ("right_arm", 6),
            ("right_gripper", 1),
        ):
            if category == "action":
                shape = (32, width)
                prefix = "stepwise"
            else:
                shape = (width,)
                prefix = "global"
            result[category][key] = {
                f"{prefix}_mean": torch.zeros(shape).tolist(),
                f"{prefix}_std": torch.ones(shape).tolist(),
                f"{prefix}_q01": torch.full(shape, -1.0).tolist(),
                f"{prefix}_q99": torch.full(shape, 1.0).tolist(),
            }
    return {"galaxea_r1lite": result}


def test_base_system2_requires_named_embodiment_and_roundtrips_mixed_tail_stats(tmp_path):
    hydra = _base_r1lite_hydra()
    with pytest.raises(ValueError, match="concrete --embodiment"):
        _profile_config("g05-base", hydra)

    config = _profile_config("g05-base", hydra, embodiment="galaxea_r1lite")
    actioncodec_config = _profile_config(
        "g05-base", hydra, embodiment="galaxea_r1lite", action_head="actioncodec"
    )
    config.camera_sizes = _camera_sizes(config.processor_metadata, config.camera_order)
    config.camera_sizes = dict.fromkeys(config.camera_order, (8, 8))
    stats = convert_dataset_stats(_base_r1lite_stats(), config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    raw_state = torch.linspace(-2, 2, 14).repeat(1, 6, 1)
    raw_action = raw_state[:, -1] + torch.linspace(-0.2, 0.2, 14).repeat(1, 32, 1)

    raw_batch = {
        OBS_STATE: raw_state,
        ACTION: raw_action,
        **{camera: torch.zeros(1, 6, 3, 8, 8, dtype=torch.uint8) for camera in config.camera_order},
        "task": ["native system 2"],
    }
    processed = preprocessor(raw_batch)
    restored = postprocessor(processed[ACTION])

    assert config.runtime_system == "system2"
    assert config.predict_cot and config.discrete_action and config.continuous_action
    assert config.action_head == "flow" and actioncodec_config.action_head == "actioncodec"
    assert actioncodec_config.runtime_system == "system2"
    assert not actioncodec_config.return_continuous_action
    assert config.policy_action_dim == 27
    assert config.num_input_images == 18
    assert "<prompt_text_!>" in config.prompt_template
    # G0.5-base's 32-step head includes n_obs_steps - 1 alignment steps.
    assert processed["action_dim_is_pad"].shape == (1, 27)
    assert not processed["action_op_mask"].any()
    assert restored.shape == (1, 27, 14)
    torch.testing.assert_close(restored, raw_action[:, 5:], atol=2e-5, rtol=2e-5)

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)
    reloaded = loaded_preprocessor(raw_batch)
    reloaded_restored = loaded_postprocessor(reloaded[ACTION])
    torch.testing.assert_close(reloaded[ACTION], processed[ACTION])
    torch.testing.assert_close(reloaded_restored, restored)


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
    config = _config(normalization_mode="q01_q99")
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
    optimizer = torch.optim.AdamW(policy.get_optim_params()["params"], lr=1e-3)
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


def test_save_pretrained_copies_required_gated_sidecars_portably(tmp_path: Path):
    source = tmp_path / "converted"
    processor = source / "hf_processor"
    processor.mkdir(parents=True)
    (processor / "tokenizer.json").write_text("{}")
    tokenizer = source / "action_tokenizer.pt"
    torch.save({"codec": "ActionCodec"}, tokenizer)
    for name in ("LICENSE-G0.5", "NOTICE", "conversion_report.json"):
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
    optimizer = torch.optim.AdamW(policy.get_optim_params()["params"], lr=5e-2)
    batch = _policy_batch("overfit")
    initial = policy(batch)[0].item()
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()
    final = policy(batch)[0].item()
    assert final < initial * 0.25


def test_conversion_reports_mapping_duplicates_shapes_and_required_prefixes():
    source = {
        "model.embed_tokens.weight": torch.zeros(2, 3),
        "model.vision_tower.block.weight": torch.ones(2, 2),
        "model.action_expert.block.weight": torch.ones(2, 2),
    }
    converted, report = convert_state_dict(source)
    assert "backend.model.vlm.input_proj.weight" in converted
    assert report.missing == []

    expected = {
        "backend.model.vlm.input_proj.weight": torch.zeros(3, 3),
        "backend.model.vision_tower.block.weight": torch.zeros(2, 2),
        "backend.model.action_expert.block.weight": torch.ones(2, 2),
    }
    _, strict_report = convert_state_dict(source, expected)
    assert strict_report.shape_mismatched["backend.model.vlm.input_proj.weight"]["source"] == [2, 3]


def test_conversion_records_and_deduplicates_tied_weight_aliases(tmp_path: Path):
    tied = torch.zeros(2, 3)
    aliases = save_converted_state_dict(
        {
            "backend.model.vlm.input_proj.weight": tied,
            "backend.model.vlm.output_proj.weight": tied,
        },
        tmp_path / "model.safetensors",
    )

    assert aliases == {"backend.model.vlm.output_proj.weight": "backend.model.vlm.input_proj.weight"}


def test_libero_conversion_packages_model_processors_and_provenance(tmp_path: Path):
    source = tmp_path / "author"
    output = tmp_path / "lerobot"
    (source / ".hydra").mkdir(parents=True)
    (source / "hf_processor").mkdir()
    (source / "hf_processor" / "tokenizer.json").write_text("{}")
    (source / ".hydra" / "config.yaml").write_text(
        """
model:
  model_arch:
    num_input_images: 2
    horizon_steps: 32
    predict_cot: false
    discrete_action: false
    continuous_action: true
  processor:
    use_stepwise_action_norm: true
    norm_default_mode: q01/q99
    camera_size_config:
      exterior: [256, 256]
      wrist_right: [256, 256]
data:
  action_size: 32
  processors:
    libero:
      shape_meta:
        state:
          - {key: right_ee_pose, shape: 6}
          - {key: right_gripper, shape: 1}
        action:
          - {key: right_ee_pose, shape: 6}
          - {key: right_gripper, shape: 1}
        images:
          - {key: image, camera_type: exterior, lerobot_key: observation.images.image, shape: [3, 224, 224]}
          - {key: wrist_image, camera_type: wrist_right, lerobot_key: observation.images.wrist_image, shape: [3, 224, 224]}
tokenizer:
  vq_config: {block_wise_autoregressive: false}
"""
    )
    action_stats = {
        "right_ee_pose": {
            "stepwise_q01": torch.zeros(32, 6).tolist(),
            "stepwise_q99": torch.ones(32, 6).tolist(),
        },
        "right_gripper": {
            "stepwise_q01": torch.zeros(32, 1).tolist(),
            "stepwise_q99": torch.ones(32, 1).tolist(),
        },
    }
    state_stats = {
        "right_ee_pose": {"global_q01": [0.0] * 6, "global_q99": [1.0] * 6},
        "right_gripper": {"global_q01": [0.0], "global_q99": [1.0]},
    }
    (source / "dataset_stats.json").write_text(
        json.dumps({"libero": {"state": state_stats, "action": action_stats}})
    )
    torch.save(
        {
            "model.vlm.block.weight": torch.zeros(2, 2),
            "model.vision_tower.block.weight": torch.zeros(2, 2),
            "model.action_expert.block.weight": torch.zeros(2, 2),
        },
        source / "model.pt",
    )
    torch.save({"tokenizer_meta": {"codec": "ActionCodec"}}, source / "action_tokenizer.pt")
    license_file = source / "LICENSE-G0.5"
    license_file.write_text("test fixture license")

    report = convert_checkpoint(source, output, "g05-libero", license_file=license_file)

    assert len(report.mapped) == 3
    assert (output / "model.safetensors").is_file()
    assert (output / "policy_preprocessor.json").is_file()
    assert (output / "policy_postprocessor.json").is_file()
    assert (output / "conversion_report.json").is_file()
    config = PreTrainedConfig.from_pretrained(output)
    assert isinstance(config, G05Config)
    assert config.source_checkpoint_revision
    assert config.prompt_template.startswith("<chat_user_prefix><image0_image_!><image1_image_!>")
    assert config.camera_sizes == {
        "observation.images.image": (256, 256),
        "observation.images.wrist_image": (256, 256),
    }


@pytest.mark.skipif(
    not os.environ.get("LEROBOT_G05_CHECKPOINT"),
    reason="requires an accepted gated OpenGalaxea/G05 checkpoint and author CUDA environment",
)
def test_gated_checkpoint_loads_strictly():
    checkpoint = Path(os.environ["LEROBOT_G05_CHECKPOINT"])
    policy = G05Policy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    assert policy.config.source_checkpoint_revision
