#!/usr/bin/env python

# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for OpenEAI processor pipeline (pre/post processing)."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.openeai.configuration_openeai import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    OpenEAIVLAConfig,
)
from lerobot.policies.openeai.processor_openeai import (  # noqa: E402
    make_openeai_pre_post_processors,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# == Config-dependent defaults ==


def test_default_image_size():
    assert DEFAULT_IMAGE_SIZE == 224


def test_openeai_config_defaults():
    config = OpenEAIVLAConfig()
    assert config.n_obs_steps == 1
    assert config.chunk_size == 50
    assert config.n_action_steps == 50
    assert config.empty_cameras == 0
    assert config.image_resolution == (224, 224)


def test_openeai_config_n_action_steps_validation():
    """Test that n_action_steps > chunk_size raises at construction."""
    with pytest.raises(ValueError, match="n_action_steps.*cannot be greater"):
        OpenEAIVLAConfig(n_action_steps=60, chunk_size=50)


# == Helpers ==


CAM_KEY = f"{OBS_IMAGES}.cam"


def _make_tiny_config():
    """Create a tiny config for processor testing."""
    config = OpenEAIVLAConfig(
        hidden_dim=64,
        n_layers=2,
        num_heads=4,
        ff_ratio=2.0,
        chunk_size=10,
        n_action_steps=10,
        denoise_steps=5,
        qwen_dim=2560,
        qwen_path="Qwen/Qwen3-VL-4B-Instruct",
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        CAM_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return config


def _make_dummy_dataset_stats(config):
    """Create dummy dataset stats matching config features."""
    state_dim = config.input_features[OBS_STATE].shape[0]
    action_dim = config.output_features[ACTION].shape[0]
    img_shape = config.input_features[CAM_KEY].shape

    return {
        OBS_STATE: {
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
        },
        ACTION: {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
        },
        CAM_KEY: {
            "mean": torch.zeros(img_shape),
            "std": torch.ones(img_shape),
        },
    }


# == make_openeai_pre_post_processors Tests ==


def test_pre_post_processor_pipeline_creation():
    """Test pre/post processor pipelines can be created."""
    config = _make_tiny_config()
    config.device = "cpu"
    dataset_stats = _make_dummy_dataset_stats(config)

    pre, post = make_openeai_pre_post_processors(config, dataset_stats)

    assert pre is not None
    assert post is not None
    # Pre-processor: Rename -> Batch -> Qwen3VL -> Device -> Normalizer
    assert len(pre.steps) == 5
    # Post-processor: Unnormalizer -> Device
    assert len(post.steps) == 2


def test_preprocessor_steps_order():
    """Test pre-processor step order: Rename -> Batch -> Qwen3VL -> Device -> Normalizer."""
    config = _make_tiny_config()
    config.device = "cpu"
    dataset_stats = _make_dummy_dataset_stats(config)

    pre, _ = make_openeai_pre_post_processors(config, dataset_stats)

    step_types = [type(s).__name__ for s in pre.steps]
    assert step_types[0] == "RenameObservationsProcessorStep"
    assert step_types[1] == "AddBatchDimensionProcessorStep"
    assert step_types[2] == "Qwen3VLProcessorStep"
    assert step_types[3] == "DeviceProcessorStep"
    assert step_types[4] == "NormalizerProcessorStep"


def test_postprocessor_steps_order():
    """Test post-processor step order: Unnormalizer -> Device."""
    config = _make_tiny_config()
    config.device = "cpu"
    dataset_stats = _make_dummy_dataset_stats(config)

    _, post = make_openeai_pre_post_processors(config, dataset_stats)

    step_types = [type(s).__name__ for s in post.steps]
    assert step_types[0] == "UnnormalizerProcessorStep"
    assert step_types[1] == "DeviceProcessorStep"


def test_processor_with_empty_cameras():
    """Test processor pipeline handles empty cameras."""
    config = _make_tiny_config()
    config.empty_cameras = 2
    config.device = "cpu"
    # empty_cameras is only added during validate_features()
    config.validate_features()
    dataset_stats = _make_dummy_dataset_stats(config)
    # Add stats for the auto-added empty cameras
    img_shape = (3, *config.image_resolution)
    for i in range(config.empty_cameras):
        dataset_stats[f"{OBS_IMAGES}.empty_camera_{i}"] = {
            "mean": torch.zeros(img_shape),
            "std": torch.ones(img_shape),
        }

    pre, post = make_openeai_pre_post_processors(config, dataset_stats)

    # Original features (state + 1 cam) + 2 empty_camera_*
    assert len(config.input_features) == 4
    assert pre is not None
    assert post is not None


def test_normalization_mapping_is_correct():
    """Test that normalization mapping matches OpenEAI requirements."""
    config = OpenEAIVLAConfig()
    mapping = config.normalization_mapping
    # VISUAL should use identity (no normalization)
    assert mapping["VISUAL"] == NormalizationMode.IDENTITY
    # STATE and ACTION should use mean-std normalization
    assert mapping["STATE"] == NormalizationMode.MEAN_STD
    assert mapping["ACTION"] == NormalizationMode.MEAN_STD


# == Qwen3VLProcessorStep Tests ==


def test_qwen3vl_processor_step_class_constants():
    """Test Qwen3VLProcessorStep has VLA_TEMPLATE / IMAGE_TOKEN as ClassVars.

    These are class-level constants and should be accessible without instantiation.
    """
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep

    assert hasattr(Qwen3VLProcessorStep, "VLA_TEMPLATE")
    assert isinstance(Qwen3VLProcessorStep.VLA_TEMPLATE, str)
    assert "<|im_start|>" in Qwen3VLProcessorStep.VLA_TEMPLATE
    assert "{}" in Qwen3VLProcessorStep.VLA_TEMPLATE  # task placeholder

    assert hasattr(Qwen3VLProcessorStep, "IMAGE_TOKEN")
    assert isinstance(Qwen3VLProcessorStep.IMAGE_TOKEN, str)
    assert "<|vision_start|>" in Qwen3VLProcessorStep.IMAGE_TOKEN
    assert "<|image_pad|>" in Qwen3VLProcessorStep.IMAGE_TOKEN
    assert "<|vision_end|>" in Qwen3VLProcessorStep.IMAGE_TOKEN


def test_qwen3vl_processor_step_default_dataclass_fields():
    """Test Qwen3VLProcessorStep dataclass field defaults (without instantiation).

    Verified via dataclass.fields() to avoid loading the actual Qwen3 processor.
    """
    from dataclasses import fields

    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep

    field_defaults = {f.name: f.default for f in fields(Qwen3VLProcessorStep) if f.init}
    assert field_defaults["processor_name"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert field_defaults["max_length"] == 128
    assert field_defaults["padding"] == "longest"
    assert field_defaults["padding_side"] == "left"
    assert field_defaults["truncation"] is True


def test_qwen3vl_processor_step_transform_features():
    """Test transform_features declares OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK.

    Uses patched __post_init__ to skip loading the actual Qwen3VLProcessor.
    """
    from lerobot.configs import PipelineFeatureType
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep

    with patch.object(Qwen3VLProcessorStep, "__post_init__", lambda self: None):
        step = Qwen3VLProcessorStep(max_length=128)

    features: dict = {
        PipelineFeatureType.OBSERVATION: {},
        PipelineFeatureType.ACTION: {},
    }
    out = step.transform_features(features)

    obs_features = out[PipelineFeatureType.OBSERVATION]
    assert OBS_LANGUAGE_TOKENS in obs_features
    assert OBS_LANGUAGE_ATTENTION_MASK in obs_features
    assert obs_features[OBS_LANGUAGE_TOKENS].type == FeatureType.LANGUAGE
    assert obs_features[OBS_LANGUAGE_TOKENS].shape == (128,)
    assert obs_features[OBS_LANGUAGE_ATTENTION_MASK].type == FeatureType.LANGUAGE
    assert obs_features[OBS_LANGUAGE_ATTENTION_MASK].shape == (128,)


def test_qwen3vl_processor_step_get_task_prompt_priority():
    """Test _get_task prefers 'prompt' over 'task' in complementary data."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep
    from lerobot.types import TransitionKey

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "prompt": "pick up the cup",
            "task": "task_42",
        },
    }
    result = Qwen3VLProcessorStep._get_task(transition)
    assert result == ["pick up the cup"]


def test_qwen3vl_processor_step_get_task_fallback_to_task():
    """Test _get_task falls back to 'task' when 'prompt' is absent."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep
    from lerobot.types import TransitionKey

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {"task": "task_42"},
    }
    result = Qwen3VLProcessorStep._get_task(transition)
    assert result == ["task_42"]


def test_qwen3vl_processor_step_get_task_none_when_missing():
    """Test _get_task returns None when no task/prompt info is provided."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep
    from lerobot.types import TransitionKey

    assert Qwen3VLProcessorStep._get_task({}) is None
    assert Qwen3VLProcessorStep._get_task({TransitionKey.COMPLEMENTARY_DATA: None}) is None
    assert Qwen3VLProcessorStep._get_task({TransitionKey.COMPLEMENTARY_DATA: {}}) is None


def test_qwen3vl_processor_step_get_task_list_input():
    """Test _get_task handles list input (multi-batch task)."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep
    from lerobot.types import TransitionKey

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "prompt": ["task_a", "task_b"],
        },
    }
    result = Qwen3VLProcessorStep._get_task(transition)
    assert result == ["task_a", "task_b"]


def test_qwen3vl_processor_step_detect_device_from_observation():
    """Test _detect_device finds device from observation tensors."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep
    from lerobot.types import TransitionKey

    cpu = torch.device("cpu")
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(8, device=cpu),
        },
    }
    assert Qwen3VLProcessorStep._detect_device(transition) == cpu


def test_qwen3vl_processor_step_detect_device_returns_none():
    """Test _detect_device returns None when no tensors are present."""
    from lerobot.policies.openeai.processor_openeai import Qwen3VLProcessorStep

    assert Qwen3VLProcessorStep._detect_device({}) is None
