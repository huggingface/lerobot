#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

import fnmatch
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

pytest.importorskip("transformers")

from lerobot.common.train_utils import generate_model_card, publish_trained_model
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.dm05.constants import ACTION_REFERENCE_OFFSET
from lerobot.policies.dm05.conversion_dm05 import DM05LerobotBatchConverter
from lerobot.policies.dm05.modeling_dm05 import DM05Policy, prepare_compiled_suffix_inputs
from lerobot.policies.dm05.modeling_dm05_core import (
    DM05CoreModelConfig,
    DM05ForCausalLM,
    masked_flow_matching_loss,
)
from lerobot.policies.dm05.prepare_stats_dm05 import (
    _training_episodes,
    compute_dm05_stats,
    prepare_dm05_stats,
)
from lerobot.policies.dm05.processor_dm05 import make_dm05_pre_post_processors
from lerobot.policies.dm05.tokenization_dm05 import DM05Tokenization
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


class _FakeTokenizer:
    def tokenize_robot_batch(self, samples):
        batch_size = len(samples)
        return {
            "input_ids": torch.ones(batch_size, 1, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
            "token_type_ids": torch.zeros(batch_size, 1, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size, 3, 2, 2),
        }


class _FakeProcessor:
    pass


def _dm05_config(**kwargs) -> DM05Config:
    config = DM05Config(
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        max_state_dim=32,
        max_action_dim=32,
        **kwargs,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    return config


def _convert(
    config: DM05Config,
    batch: dict,
    include_actions: bool = True,
) -> dict:
    converter = DM05LerobotBatchConverter(
        config=config,
        tokenization_cls=lambda **_: _FakeTokenizer(),
        processor=_FakeProcessor(),
    )
    return converter.convert_lerobot_batch(batch, include_actions=include_actions)


def _tiny_core_config() -> DM05CoreModelConfig:
    text = {
        "model_type": "gemma3_text",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention", "full_attention"],
        "sliding_window": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }
    vision = {
        "model_type": "siglip_vision_model",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_channels": 3,
        "image_size": 16,
        "patch_size": 8,
    }
    action = {**text, "vocab_size": 16}
    return DM05CoreModelConfig(
        vlm_config={
            "model_type": "gemma3",
            "text_config": text,
            "vision_config": vision,
            "mm_tokens_per_image": 4,
            "image_token_index": 127,
            "boi_token_index": 125,
            "eoi_token_index": 126,
        },
        action_config=action,
        action_dim=4,
        chunk_size=2,
        bf16=False,
        gradient_checkpointing=False,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
        llm_attn_implementation="eager",
        vision_attn_implementation="eager",
        action_attn_implementation="eager",
    )


def _save_tiny_real_processor(path: Path):
    """Create a network-free Gemma3Processor with DM05's real multimodal code path."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Gemma3ImageProcessor, Gemma3Processor, PreTrainedTokenizerFast

    tokens = [f"<unused_{index}>" for index in range(128)]
    replacements = {
        0: "<pad>",
        1: "<eos>",
        2: "<bos>",
        3: "<unk>",
        4: "<start_of_turn>",
        5: "<end_of_turn>",
        6: "user",
        7: "model",
        8: "Task:",
        9: "State:",
        10: "Head",
        11: "image:",
        12: "pick",
        13: "0",
        14: "255",
        125: "<start_of_image>",
        126: "<end_of_image>",
        127: "<image_soft_token>",
    }
    for index, token in replacements.items():
        tokens[index] = token
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        image_token="<image_soft_token>",
        boi_token="<start_of_image>",
        eoi_token="<end_of_image>",
        additional_special_tokens=[
            "<start_of_turn>",
            "<end_of_turn>",
            "<start_of_image>",
            "<end_of_image>",
            "<image_soft_token>",
        ],
    )
    chat_template = """{% for message in messages %}<start_of_turn>{{ message['role'] }}
{% for item in message['content'] %}{% if item['type'] == 'image' %}<start_of_image>{% else %}{{ item['text'] }}{% endif %}{% endfor %}<end_of_turn>
{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""
    processor = Gemma3Processor(
        Gemma3ImageProcessor(size={"height": 16, "width": 16}, do_pan_and_scan=False),
        tokenizer,
        chat_template=chat_template,
        image_seq_length=4,
    )
    processor.save_pretrained(path)
    return processor


def test_dm05_config_defaults_and_validation(monkeypatch, tmp_path):
    config = make_policy_config(policy_type="dm05", chunk_size=10, n_action_steps=10)
    config.validate_features()

    assert isinstance(config, DM05Config)
    assert get_policy_class("dm05") is DM05Policy
    assert config.pretrained_name_or_path == "Dexmal/DM05"
    assert config.license == "gemma"
    assert (
        config.use_relative_actions,
        config.add_state,
    ) == (False, True)
    assert config.input_features[OBS_STATE].shape == (14,)
    assert config.output_features[ACTION].shape == (14,)
    assert config.get_optimizer_preset().type == "adamw"
    monkeypatch.setattr("lerobot.common.train_utils.ModelCard.validate", lambda _self: None)
    card = generate_model_card(config)
    assert card.data.base_model == "Dexmal/DM05"
    assert card.data.license == "gemma"
    for invalid_steps in (0, -1):
        with pytest.raises(ValueError, match="diffusion_steps must be positive"):
            DM05Config(diffusion_steps=invalid_steps)

    relative_config = _dm05_config(use_relative_actions=True)
    relative_config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    with pytest.raises(ValueError, match="equal state/action dimensions"):
        relative_config.validate_features()
    relative_config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(3,))
    relative_config.normalization_mapping["ACTION"] = NormalizationMode.MEAN_STD
    with pytest.raises(ValueError, match="do not support MEAN_STD"):
        relative_config.validate_features()

    core_mismatch_config = _dm05_config()
    core_mismatch_config.core_config = {"action_dim": 31}
    with pytest.raises(ValueError, match="must match DM05 core action_dim"):
        core_mismatch_config.validate_features()

    config.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert not {"norm_stats_sample_size", "norm_stats_sample_seed"} & saved_config.keys()


def test_dm05_explicit_stats_preparation_uses_training_chunks(tmp_path):
    states = np.asarray([[10 * i, 100 + i, 1000 + i] for i in range(6)], dtype=np.float32)
    actions = np.asarray([[10 * i + 1, 200 + i, 2000 + i] for i in range(6)], dtype=np.float32)

    class NumericDataset:
        root = tmp_path
        meta = SimpleNamespace(
            stats=None,
            features={
                OBS_STATE: {"dtype": "float32", "shape": (3,), "names": ["s0", "s1", "s2"]},
                ACTION: {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": {"gripper": 1, "tool": 2, "joint": 0},
                },
            },
        )

        def __len__(self):
            return len(states)

        def select_columns(self, _):
            return self

        def __getitem__(self, key):
            if key == "episode_index":
                return [0, 0, 0, 1, 1, 1]
            raise KeyError(key)

        def select(self, indices):
            return {OBS_STATE: states[indices].tolist(), ACTION: actions[indices].tolist()}

    dataset = NumericDataset()
    meta = dataset.meta
    config = _dm05_config()
    config.set_dataset_feature_metadata(meta.features)
    assert config.action_feature_names == ["joint", "gripper", "tool"]

    meta.repo_id = "local/numeric"
    meta.root = tmp_path
    config._runtime_dataset_meta = meta
    with pytest.raises(ValueError, match=r"prepare_stats_dm05.*--root=.*--chunk-size=2"):
        make_dm05_pre_post_processors(config, None)

    meta.stats = compute_dm05_stats(config, dataset, sample_size=6)
    absolute_action_stats = meta.stats[ACTION]
    flat_indices = np.asarray([0, 1, 1, 2, 3, 4, 4, 5])
    np.testing.assert_array_equal(meta.stats[OBS_STATE]["count"], [4])
    np.testing.assert_array_equal(meta.stats[ACTION]["count"], [8])
    np.testing.assert_allclose(
        meta.stats[ACTION]["q01"],
        np.quantile(actions[flat_indices], 0.01, axis=0),
        rtol=0,
        atol=1e-5,
    )

    meta.stats = None
    config.use_relative_actions = True
    meta.stats = compute_dm05_stats(config, dataset, sample_size=6)
    owner_frames = np.asarray([0, 0, 1, 1, 3, 3, 4, 4])
    expected_relative = actions[flat_indices].copy()
    expected_relative[:, [0, 2]] -= states[owner_frames][:, [0, 2]]
    np.testing.assert_allclose(
        meta.stats[ACTION]["q99"],
        np.quantile(expected_relative, 0.99, axis=0),
        rtol=0,
        atol=1e-5,
    )
    excluded_constant_stats = {
        OBS_STATE: {"q01": [0.0] * 3, "q99": [1.0] * 3},
        ACTION: {"q01": [0.0] * 3, "q99": [1.0, 0.0, 1.0]},
    }
    make_dm05_pre_post_processors(config, excluded_constant_stats)
    invalid_relative_stats = {
        **excluded_constant_stats,
        ACTION: {"q01": [0.0] * 3, "q99": [0.0, 0.0, 1.0]},
    }
    with pytest.raises(ValueError, match="non-degenerate.*invalid indices: \\[0\\]"):
        make_dm05_pre_post_processors(config, invalid_relative_stats)
    meta.stats = {ACTION: absolute_action_stats}
    with pytest.raises(ValueError, match="--force"):
        compute_dm05_stats(config, dataset, sample_size=6)

    meta.stats = None
    config.use_relative_actions = False
    stats_path, changed = prepare_dm05_stats(config, dataset)
    assert changed is True
    assert stats_path == tmp_path / "meta/stats.json"
    written = json.loads(stats_path.read_text())
    assert set(written[OBS_STATE]) >= {"q01", "q10", "q90", "q99"}
    assert set(written[ACTION]) >= {"q01", "q10", "q90", "q99"}

    selected_dataset = SimpleNamespace(
        episodes=[1, 2, 4, 5],
        meta=SimpleNamespace(
            total_episodes=6,
            episodes={"tasks": [["a"], ["a"], ["a"], ["b"], ["b"], ["b"]]},
        ),
    )
    assert _training_episodes(selected_dataset, eval_split=0) == [1, 2, 4, 5]
    assert _training_episodes(selected_dataset, eval_split=0.5) == [1, 4]


def test_dm05_tokenization_builds_opendm_style_user_content_without_random_branches():
    tokenization = DM05Tokenization(
        processor=_FakeProcessor(),
        n_bins=256,
        add_state=False,
    )
    images = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    meta = {
        "robot_type": "franka",
        "control_mode": "relative",
        "dataset_meta": {"image_keys": ["images_1", "observation.images.left_wrist"]},
    }

    user_content = tokenization._build_user_content(
        prompt="Pick up the mug",
        images=images,
        state=np.array([-1.0, 1.0], dtype=np.float32),
        meta_data=meta,
        speed_text="0.5",
    )

    assert user_content[0]["text"] == (
        "Robot: franka\nControl mode: relative\nOverall speed: 0.5\nTask: Pick up the mug.\nHead image: "
    )
    assert user_content[1]["type"] == "image"
    assert user_content[2]["text"] == "Left wrist image: "
    assert user_content[3]["type"] == "image"
    assert all("State:" not in item.get("text", "") for item in user_content)

    state_tokenization = DM05Tokenization(processor=_FakeProcessor(), n_bins=256, add_state=True)
    state_content = state_tokenization._build_user_content(
        prompt="Pick up the mug",
        images=[images[0]],
        state=np.array([-1.0, 1.0], dtype=np.float32),
        meta_data={"dataset_meta": {"image_keys": ["images_1"]}},
        speed_text=None,
    )

    assert state_content[0]["text"].startswith("Overall speed: 0.5\nTask: Pick up the mug.\n")
    assert state_content[-1]["text"] == "States: 0 255"


def test_dm05_processors_roundtrip(tmp_path):
    config = _dm05_config(use_relative_actions=False)
    config.action_feature_names = ["joint_0", "joint_1", "gripper"]
    stats = {
        OBS_STATE: {"q01": [-10.0] * 3, "q99": [10.0] * 3},
        ACTION: {"q01": [-10.0] * 3, "q99": [10.0] * 3},
    }
    preprocessor, postprocessor = make_dm05_pre_post_processors(config=config, dataset_stats=stats)

    preprocessor.save_pretrained(tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(tmp_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)

    assert {path.name for path in tmp_path.glob("*normalizer_processor.safetensors")} == {
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
    }
    assert not list(tmp_path.glob("*dm05*normalizer*.safetensors"))

    default_processed = loaded_preprocessor(
        {
            OBS_STATE: torch.tensor([1.0, 2.0, 3.0]),
            ACTION: torch.tensor([[4.0, 5.0, 6.0]]),
            "observation.images.front": torch.zeros(3, 2, 2),
        }
    )

    assert default_processed["task"] == "Execute the robot action."
    assert ACTION_REFERENCE_OFFSET in default_processed
    relative_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )
    assert not relative_step.enabled and absolute_step.relative_step is relative_step
    torch.testing.assert_close(default_processed[ACTION], torch.tensor([[0.4, 0.5, 0.6]]))
    torch.testing.assert_close(
        loaded_postprocessor(default_processed[ACTION]),
        torch.tensor([[4.0, 5.0, 6.0]]),
        atol=2e-6,
        rtol=0,
    )
    numpy_action = loaded_postprocessor(default_processed[ACTION].to(torch.bfloat16))
    assert numpy_action.dtype == torch.float32

    replacement_stats = {
        OBS_STATE: {"q01": [-2.0] * 3, "q99": [2.0] * 3},
        ACTION: {"q01": [-4.0] * 3, "q99": [4.0] * 3},
    }
    overridden_preprocessor, overridden_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=tmp_path,
        preprocessor_overrides={
            "normalizer_processor": {
                "features": {**config.input_features, **config.output_features},
                "stats": replacement_stats,
            },
            "rename_observations_processor": {
                "rename_map": {"observation.images.camera": "observation.images.front"}
            },
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": replacement_stats}},
    )
    overridden_normalizer = next(
        step for step in overridden_preprocessor.steps if isinstance(step, NormalizerProcessorStep)
    )
    overridden_unnormalizer = next(
        step for step in overridden_postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
    )
    assert set(overridden_normalizer.normalize_observation_keys) == {OBS_STATE}
    torch.testing.assert_close(overridden_normalizer._tensor_stats[ACTION]["q99"], torch.full((3,), 4.0))
    torch.testing.assert_close(overridden_unnormalizer._tensor_stats[ACTION]["q99"], torch.full((3,), 4.0))
    renamed = overridden_preprocessor(
        {
            OBS_STATE: torch.zeros(3),
            ACTION: torch.zeros(1, 3),
            "observation.images.camera": Image.new("RGB", (2, 2)),
        }
    )
    assert "observation.images.front" in renamed
    assert isinstance(renamed["observation.images.front"][0], Image.Image)

    def prepare_policy_batch(policy_config, processed):
        policy = object.__new__(DM05Policy)
        torch.nn.Module.__init__(policy)
        policy.config = policy_config
        return policy._prepare_policy_batch(processed, include_actions=True)

    outlier = {
        OBS_STATE: torch.zeros(3),
        ACTION: torch.tensor([[20.0, 0.0, 0.0]]),
        "observation.images.front": torch.zeros(3, 2, 2),
    }
    clipped = prepare_policy_batch(config, loaded_preprocessor(outlier.copy()))
    unclipped_config = _dm05_config(use_relative_actions=False, norm_clip=False)
    unclipped_preprocessor, _ = make_dm05_pre_post_processors(unclipped_config, stats)
    unclipped = prepare_policy_batch(unclipped_config, unclipped_preprocessor(outlier.copy()))
    assert clipped[ACTION][0, 0] == 1
    assert unclipped[ACTION][0, 0] > 1

    # Absolute actions do not require state/action dimensions to match, and PIL
    # observations bypass identity visual normalization.
    asymmetric_config = _dm05_config()
    asymmetric_config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(2,))
    asymmetric_stats = {
        OBS_STATE: {"q01": [-1.0] * 2, "q99": [1.0] * 2},
        ACTION: {"q01": [-1.0] * 3, "q99": [1.0] * 3},
    }
    asymmetric_preprocessor, _ = make_dm05_pre_post_processors(asymmetric_config, asymmetric_stats)
    asymmetric = asymmetric_preprocessor(
        {
            OBS_STATE: torch.zeros(2),
            ACTION: torch.zeros(3),
            "observation.images.front": Image.new("RGB", (2, 2)),
        }
    )
    assert ACTION_REFERENCE_OFFSET not in asymmetric
    assert isinstance(asymmetric["observation.images.front"][0], Image.Image)


def test_dm05_relative_actions_use_generation_state_for_training_and_inference():
    config = _dm05_config(use_relative_actions=True)
    config.action_feature_names = ["joint_0", "joint_1", "gripper"]
    stats = {
        OBS_STATE: {"q01": [-100.0] * 3, "q99": [100.0] * 3},
        ACTION: {"q01": [-10.0] * 3, "q99": [10.0] * 3},
    }
    preprocessor, postprocessor = make_dm05_pre_post_processors(config, stats)

    class InferenceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(chunk_size=2, action_dim=32)
            normalized_relative = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.actions = torch.nn.functional.pad(normalized_relative, (0, 29))
            self.calls = 0

        def inference_action(self, **_kwargs):
            self.calls += 1
            return self.actions

    policy = object.__new__(DM05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = InferenceModel()
    policy._compile_suffix_active = False
    policy._prepare_model_inputs = lambda _batch, include_actions: {
        "input_ids": torch.ones(1, 1, dtype=torch.long)
    }
    policy.reset()

    generation_batch = preprocessor({OBS_STATE: torch.tensor([10.0, 20.0, 30.0])})
    direct_chunk = policy.predict_action_chunk(generation_batch)
    torch.testing.assert_close(
        postprocessor(direct_chunk),
        torch.tensor([[[11.0, 22.0, 3.0], [14.0, 25.0, 6.0]]]),
        atol=2e-5,
        rtol=0,
    )

    policy.reset()
    generation_batch = preprocessor({OBS_STATE: torch.tensor([10.0, 20.0, 30.0])})
    first = policy.select_action(generation_batch)
    first = postprocessor(first)
    current_batch = preprocessor({OBS_STATE: torch.tensor([100.0, 200.0, 300.0])})
    second = policy.select_action(current_batch)
    second = postprocessor(second)

    torch.testing.assert_close(first, torch.tensor([[11.0, 22.0, 3.0]]), atol=2e-5, rtol=0)
    torch.testing.assert_close(second, torch.tensor([[14.0, 25.0, 6.0]]), atol=2e-5, rtol=0)
    assert policy.model.calls == 2

    with pytest.raises(ValueError, match="checkpoint preprocessor"):
        policy.select_action({})

    processed = preprocessor(
        {
            OBS_STATE: torch.tensor([[10.0, 20.0, 30.0]]),
            ACTION: torch.tensor([[[11.0, 22.0, 3.0], [14.0, 25.0, 6.0]]]),
            "observation.images.front": torch.zeros(1, 3, 2, 2),
        }
    )

    policy = object.__new__(DM05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    prepared = policy._prepare_policy_batch(processed, include_actions=True)

    # Both timesteps use the same generation-time state; the gripper stays absolute.
    torch.testing.assert_close(
        prepared[ACTION],
        torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
        atol=2e-6,
        rtol=0,
    )


def test_dm05_action_dim_mask_and_fixed_noise_cover_training_and_inference():
    model = DM05ForCausalLM(_tiny_core_config()).eval()
    captured = []
    model.model.action_in_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    common = {
        "input_ids": torch.tensor([[2, 5]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
        "action_dim_mask": torch.tensor([[True, True, False, False]]),
    }
    model(
        **common,
        actions=torch.full((1, 2, 4), 7.0),
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
        has_actions=torch.ones(1, dtype=torch.bool),
    )
    assert torch.count_nonzero(captured[-1][..., 2:]) == 0

    with pytest.raises(ValueError, match="must be provided together"):
        model(**common, prefill_actions=torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="must be provided together"):
        model.inference_action(**common, action_prefill_len=torch.tensor([1]))
    with pytest.raises(ValueError, match="only supported with use_compiled_suffix"):
        model.inference_action(**common, action_prefix_mask=torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="must be provided together"):
        prepare_compiled_suffix_inputs(
            SimpleNamespace(compile_suffix_pad_length=None),
            model,
            None,
            {"input_ids": common["input_ids"], "prefill_actions": torch.zeros(1, 2, 4)},
            dtype=torch.float32,
        )

    first_actions = torch.tensor([[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]])
    leaked_actions = first_actions.clone()
    leaked_actions[:, 1] = 1_000_000
    padded = torch.tensor([[False, True]])
    torch.manual_seed(123)
    first_loss = model(
        **common,
        actions=first_actions,
        action_is_pad=padded,
        has_actions=torch.ones(1, dtype=torch.bool),
    ).loss
    first_flow_input = captured[-1]
    torch.manual_seed(123)
    leaked_loss = model(
        **common,
        actions=leaked_actions,
        action_is_pad=padded,
        has_actions=torch.ones(1, dtype=torch.bool),
    ).loss
    leaked_flow_input = captured[-1]
    assert torch.count_nonzero(first_flow_input[:, 1]) == 0
    assert torch.count_nonzero(leaked_flow_input[:, 1]) == 0
    torch.testing.assert_close(first_loss, leaked_loss)

    noise = torch.full((1, 2, 4), 7.0)
    first = model.inference_action(**common, diffusion_steps=1, initial_noise=noise)
    second = model.inference_action(**common, diffusion_steps=1, initial_noise=noise)
    assert torch.count_nonzero(captured[-1][..., 2:]) == 0
    assert torch.count_nonzero(first[..., 2:]) == 0
    torch.testing.assert_close(first, second)

    class InferenceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(chunk_size=2, action_dim=32)
            self.kwargs = None

        def inference_action(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["initial_noise"]

    policy = object.__new__(DM05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = _dm05_config()
    policy.model = InferenceModel()
    policy._compile_suffix_active = False
    policy._prepare_model_inputs = lambda _batch, include_actions: {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "action_dim_mask": torch.cat(
            [torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 29, dtype=torch.bool)], dim=1
        ),
    }
    policy.config.device = "cuda:7"
    policy.to("cpu")
    assert policy.config.device == "cpu"

    policy_noise = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    actions = policy.predict_action_chunk({}, noise=policy_noise)

    assert policy.model.kwargs["initial_noise"].shape == (1, 2, 32)
    torch.testing.assert_close(actions, policy_noise)
    assert torch.count_nonzero(policy.model.kwargs["initial_noise"][..., 3:]) == 0
    with pytest.raises(ValueError, match="noise action dimension"):
        policy.predict_action_chunk({}, noise=torch.zeros(1, 2, 1))
    with pytest.raises(ValueError, match="diffusion_steps must be positive"):
        policy.predict_action_chunk({}, noise=policy_noise, diffusion_steps=0)
    with pytest.raises(ValueError, match="diffusion_steps must be positive"):
        model.inference_action(**common, diffusion_steps=0, initial_noise=noise)


def test_dm05_gradient_checkpointing_includes_vision_tower():
    model = DM05ForCausalLM(_tiny_core_config())
    model.enable_gradient_checkpointing(
        vlm_gradient_checkpointing=True,
        ae_gradient_checkpointing=True,
        ae_layers=2,
    )
    assert model.model.vlm.model.language_model.gradient_checkpointing
    assert model.model.vlm.model.vision_tower.vision_model.encoder.gradient_checkpointing
    assert model.model.action_expert.gradient_checkpointing
    assert model.model.action_expert.gradient_checkpointing_layers == 2


def test_dm05_tiny_real_processor_core_save_and_offline_hub_reload(monkeypatch, tmp_path):
    processor_path = tmp_path / "processor"
    original_processor = _save_tiny_real_processor(processor_path)
    core_config = _tiny_core_config()
    config = _dm05_config(
        core_config=core_config.to_dict(),
        pretrained_name_or_path=".",
        processor_name_or_path=".",
        dtype="float32",
        use_relative_actions=True,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
    )
    config.action_feature_names = ["joint_0", "joint_1", "joint_2", "gripper"]
    config.max_state_dim = 4
    config.max_action_dim = 4
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    policy = DM05Policy(config, checkpoint_source=processor_path)
    checkpoint = tmp_path / "checkpoint" / "pretrained_model"
    (checkpoint.parent / "training_state").mkdir(parents=True)

    # The gathered weights deliberately differ from the live model so this cannot pass by accident.
    gathered_state_dict = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    sentinel_key = next(key for key, value in gathered_state_dict.items() if value.is_floating_point())
    gathered_state_dict[sentinel_key] = gathered_state_dict[sentinel_key] + 1
    with monkeypatch.context() as save_monkeypatch:
        save_monkeypatch.setattr(
            "lerobot.distributed.checkpoint.full_model_state_dict", lambda _policy: gathered_state_dict
        )

        non_main_checkpoint = tmp_path / "non_main"
        save_monkeypatch.setattr("lerobot.distributed.utils.is_main_process", lambda: False)
        policy.save_pretrained(non_main_checkpoint)
        assert not any(non_main_checkpoint.iterdir())

        save_monkeypatch.setattr("lerobot.distributed.utils.is_main_process", lambda: True)
        policy.save_pretrained(checkpoint)

    stats = {
        OBS_STATE: {"q01": [-1.0] * 4, "q99": [1.0] * 4},
        ACTION: {"q01": [-1.0] * 4, "q99": [1.0] * 4},
    }
    preprocessor, postprocessor = make_dm05_pre_post_processors(policy.config, stats)
    preprocessor.save_pretrained(
        checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        checkpoint,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )

    saved_state_dict = load_file(checkpoint / "model.safetensors")
    expected_state_dict = {
        key: value
        for key, value in gathered_state_dict.items()
        if key != "model.model.vlm.model.language_model.embed_tokens.weight"
    }
    saved_config = json.loads((checkpoint / "config.json").read_text())
    torch.testing.assert_close(saved_state_dict, expected_state_dict)
    assert saved_config["pretrained_name_or_path"] == "."
    assert saved_config["processor_name_or_path"] == "."
    assert saved_config["core_config"]["model_type"] == "dexbotic_dm05"
    assert saved_config["core_config"]["_name_or_path"] == "."
    assert PreTrainedConfig.from_pretrained(checkpoint).type == "dm05"

    sample = {
        "prompt": "pick",
        "images": [Image.new("RGB", (16, 16), color="white")],
        "state": np.zeros(4, dtype=np.float32),
        "meta_data": {},
    }
    before = DM05Tokenization(original_processor, add_state=False).tokenize_robot_batch([sample])
    after = DM05Tokenization(policy.processor, add_state=False).tokenize_robot_batch([sample])
    torch.testing.assert_close(after, before)
    assert "labels" not in before

    cache_dir = tmp_path / "hub_cache"
    snapshot = cache_dir / "models--org--tiny-dm05" / "snapshots" / ("a" * 40)

    class FakeHubApi:
        def create_repo(self, repo_id, **kwargs):
            assert repo_id == "org/tiny-dm05"
            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, *, folder_path, allow_patterns, **kwargs):
            snapshot.mkdir(parents=True, exist_ok=True)
            for source_file in Path(folder_path).iterdir():
                if any(fnmatch.fnmatch(source_file.name, pattern) for pattern in allow_patterns):
                    shutil.copy2(source_file, snapshot / source_file.name)
            refs = snapshot.parents[1] / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text("a" * 40)
            return SimpleNamespace(repo_url=SimpleNamespace(url="https://huggingface.co/org/tiny-dm05"))

    class FakeCard:
        def save(self, path):
            Path(path).write_text("# Tiny DM05\n")

    class FakeTrainConfig:
        dataset = SimpleNamespace(repo_id="org/tiny-dataset")

        def save_pretrained(self, path):
            Path(path, "train_config.json").write_text("{}\n")

    def fake_model_push_to_hub(repo_id, **_kwargs):
        assert repo_id == "org/tiny-dm05"
        snapshot.mkdir(parents=True)
        policy.save_pretrained(snapshot)
        refs = snapshot.parents[1] / "refs"
        refs.mkdir()
        (refs / "main").write_text("a" * 40)

    policy.config.repo_id = "org/tiny-dm05"
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", FakeHubApi)
    monkeypatch.setattr("lerobot.common.train_utils.generate_model_card", lambda *args, **kwargs: FakeCard())
    monkeypatch.setattr(policy, "push_to_hub", fake_model_push_to_hub)
    publish_trained_model(FakeTrainConfig(), policy, None, None, None)
    assert (snapshot / "chat_template.jinja").exists()
    assert "chat_template" in json.loads((snapshot / "tokenizer_config.json").read_text())
    assert "chat_template" not in json.loads((snapshot / "processor_config.json").read_text())
    hub_reloaded = DM05Policy.from_pretrained(
        "org/tiny-dm05",
        cache_dir=cache_dir,
        local_files_only=True,
        strict=True,
    )
    torch.testing.assert_close(hub_reloaded.state_dict(), policy.state_dict())
    hub_tokens = DM05Tokenization(hub_reloaded.processor, add_state=False).tokenize_robot_batch([sample])
    torch.testing.assert_close(hub_tokens, before)

    monkeypatch.chdir(checkpoint.parent)
    reloaded = DM05Policy.from_pretrained(checkpoint.name, strict=True)
    assert not hasattr(reloaded, "_action_codec")
    torch.testing.assert_close(reloaded.state_dict(), gathered_state_dict)

    resumed_preprocessor, _ = make_pre_post_processors(reloaded.config, pretrained_path=checkpoint.name)
    resumed_normalizer = next(
        step for step in resumed_preprocessor.steps if isinstance(step, NormalizerProcessorStep)
    )
    torch.testing.assert_close(resumed_normalizer._tensor_stats[ACTION]["q99"], torch.ones(4))
    relative_step = next(
        step for step in resumed_preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    assert relative_step.enabled

    absolute_config = PreTrainedConfig.from_pretrained(checkpoint.name)
    absolute_config.use_relative_actions = False
    with pytest.raises(ValueError, match="cannot be loaded"):
        DM05Policy.from_pretrained(checkpoint.name, config=absolute_config)

    config_path = checkpoint / "config.json"
    relative_payload = json.loads(config_path.read_text())
    relative_payload["use_relative_actions"] = False
    config_path.write_text(json.dumps(relative_payload))
    relative_config = PreTrainedConfig.from_pretrained(checkpoint.name)
    relative_config.use_relative_actions = True
    with pytest.raises(ValueError, match="requires complete relative-action dataset statistics"):
        DM05Policy.from_pretrained(
            checkpoint.name,
            config=relative_config,
            dataset_meta=SimpleNamespace(repo_id="org/dataset", root=tmp_path, stats=None),
            dataset_stats=None,
        )
    relative_payload["use_relative_actions"] = True
    config_path.write_text(json.dumps(relative_payload))

    inference_batch = preprocessor(
        {
            OBS_STATE: torch.zeros(4),
            "observation.images.front": torch.ones(3, 16, 16),
            "task": "pick",
        }
    )
    predicted_actions = reloaded.predict_action_chunk(
        inference_batch,
        noise=torch.zeros(1, 2, 4),
        diffusion_steps=1,
    )
    actions = postprocessor(predicted_actions)
    assert actions.shape == (1, 2, 4)
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all()

    if importlib.util.find_spec("grpc") is not None:
        # Exercise the supported async-server boundary with the same real tiny
        # checkpoint and processors. No socket is needed for this policy contract.
        import pickle  # nosec
        import time

        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
        from lerobot.async_inference.policy_server import PolicyServer

        server = PolicyServer(PolicyServerConfig(host="localhost", port=9999))
        lerobot_features = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": [4],
                "names": ["joint_0", "joint_1", "joint_2", "joint_3"],
            },
            "observation.images.front": {
                "dtype": "image",
                "shape": [16, 16, 3],
                "names": ["height", "width", "channel"],
            },
        }
        instructions = RemotePolicyConfig(
            policy_type="dm05",
            pretrained_name_or_path=str(checkpoint),
            lerobot_features=lerobot_features,
            actions_per_chunk=2,
            device="cpu",
        )
        server.SendPolicyInstructions(
            SimpleNamespace(data=pickle.dumps(instructions)),  # nosec
            SimpleNamespace(peer=lambda: "test-client"),
        )
        assert server.policy.config.device == "cpu"
        timed_actions = server._predict_action_chunk(
            TimedObservation(
                observation={
                    "joint_0": 0.0,
                    "joint_1": 0.0,
                    "joint_2": 0.0,
                    "joint_3": 0.0,
                    "front": np.full((16, 16, 3), 255, dtype=np.uint8),
                    "task": "pick",
                },
                timestamp=time.time(),
                timestep=3,
            )
        )
        assert [item.get_timestep() for item in timed_actions] == [3, 4]
        assert all(torch.isfinite(item.get_action()).all() for item in timed_actions)

    policy_batch = preprocessor(
        {
            OBS_STATE: torch.zeros(1, 4),
            ACTION: torch.zeros(1, 2, 4),
            "action_is_pad": torch.tensor([[False, True]]),
            "observation.images.front": torch.ones(1, 3, 16, 16),
            "task": ["pick"],
        }
    )
    loss, metrics = reloaded(policy_batch)
    assert loss.isfinite()
    assert metrics == {}
    loss.backward()
    assert reloaded.model.model.action_out_proj.weight.grad is not None
    assert reloaded.model.model.vlm.model.language_model.layers[-1].self_attn.k_proj.weight.grad is not None


def test_dm05_padding_mask_excludes_fabricated_targets_and_preserves_sample_weighting():
    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]]),
        ACTION: torch.tensor([[[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]]),
        "action_is_pad": torch.tensor([[False, True]]),
        "observation.images.front": torch.zeros(1, 3, 2, 2),
    }
    converted = _convert(_dm05_config(), batch)

    assert torch.equal(converted["action_is_pad"], torch.tensor([[False, True]]))

    prediction = torch.zeros(1, 2, 3)
    target = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    loss = masked_flow_matching_loss(
        prediction,
        target,
        action_dim_mask=torch.tensor([[True, True, False]]),
        action_is_pad=converted["action_is_pad"],
        has_actions=torch.tensor([True]),
    )
    torch.testing.assert_close(loss, torch.tensor(2.5))
    all_pad_prediction = prediction.clone().requires_grad_()
    all_pad_loss = masked_flow_matching_loss(
        all_pad_prediction,
        target,
        action_is_pad=torch.ones(1, 2, dtype=torch.bool),
        has_actions=torch.tensor([True]),
    )
    assert all_pad_loss.isfinite()
    all_pad_loss.backward()
    assert torch.count_nonzero(all_pad_prediction.grad) == 0

    sample_balanced_loss = masked_flow_matching_loss(
        torch.zeros(2, 2, 1),
        torch.tensor([[[1.0], [1.0]], [[3.0], [100.0]]]),
        action_is_pad=torch.tensor([[False, False], [False, True]]),
        has_actions=torch.tensor([True, True]),
    )
    # Preserve DM05's original sample-balanced reduction: (1^2 + 3^2) / 2.
    torch.testing.assert_close(sample_balanced_loss, torch.tensor(5.0))
