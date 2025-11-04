#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Tests for the TokenizerProcessorStep class.
"""

import tempfile
from unittest.mock import patch

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import DataProcessorPipeline, TokenizerProcessorStep, TransitionKey
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_LANGUAGE, OBS_STATE
from tests.utils import require_package


class MockTokenizer:
    """Mock tokenizer for testing that mimics transformers tokenizer interface."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
        padding_side: str = "right",
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Mock tokenization that returns deterministic tokens based on text."""
        texts = [text] if isinstance(text, str) else text

        batch_size = len(texts)

        # Create mock input_ids and attention_mask
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

        for i, txt in enumerate(texts):
            # Simple mock: use hash of text to generate deterministic tokens
            text_hash = hash(txt) % self.vocab_size
            seq_len = min(len(txt.split()), max_length)

            # Fill input_ids with simple pattern based on text
            for j in range(seq_len):
                input_ids[i, j] = (text_hash + j) % self.vocab_size

            # Set attention mask for non-padded positions
            attention_mask[i, :seq_len] = 1

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Return single sequence for single input to match transformers behavior
        if len(texts) == 1:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    return MockTokenizer(vocab_size=100)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_basic_tokenization(mock_auto_tokenizer):
    """Test basic string tokenization functionality."""
    # Mock AutoTokenizer.from_pretrained to return our mock tokenizer
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=10)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "pick up the red cube"},
    )

    result = processor(transition)

    # Check that original task is preserved
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "pick up the red cube"

    # Check that tokens were added to observation
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation

    # Check token structure
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert tokens.shape == (10,)
    assert attention_mask.shape == (10,)


@require_package("transformers")
def test_basic_tokenization_with_tokenizer_object():
    """Test basic string tokenization functionality using tokenizer object directly."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "pick up the red cube"},
    )

    result = processor(transition)

    # Check that original task is preserved
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "pick up the red cube"

    # Check that tokens were added to observation
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation

    # Check token structure
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert tokens.shape == (10,)
    assert attention_mask.shape == (10,)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_list_of_strings_tokenization(mock_auto_tokenizer):
    """Test tokenization of a list of strings."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=8)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": ["pick up cube", "place on table"]},
    )

    result = processor(transition)

    # Check that original task is preserved
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick up cube", "place on table"]

    # Check that tokens were added to observation
    observation = result[TransitionKey.OBSERVATION]
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.shape == (2, 8)  # batch_size=2, seq_len=8
    assert attention_mask.shape == (2, 8)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_custom_keys(mock_auto_tokenizer):
    """Test using custom task_key."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", task_key="instruction", max_length=5)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"instruction": "move forward"},
    )

    result = processor(transition)

    # Check that tokens are stored in observation regardless of task_key
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation

    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    assert tokens.shape == (5,)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_none_complementary_data(mock_auto_tokenizer):
    """Test handling of None complementary_data."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    transition = create_transition(observation={}, complementary_data=None)

    # create_transition converts None complementary_data to empty dict, so task key is missing
    with pytest.raises(KeyError, match="task"):
        processor(transition)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_missing_task_key(mock_auto_tokenizer):
    """Test handling when task key is missing."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    transition = create_transition(observation={}, complementary_data={"other_field": "some value"})

    with pytest.raises(KeyError, match="task"):
        processor(transition)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_none_task_value(mock_auto_tokenizer):
    """Test handling when task value is None."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    transition = create_transition(observation={}, complementary_data={"task": None})

    with pytest.raises(ValueError, match="Task extracted from Complementary data is None"):
        processor(transition)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_unsupported_task_type(mock_auto_tokenizer):
    """Test handling of unsupported task types."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    # Test with integer task - get_task returns None, observation raises ValueError
    transition = create_transition(observation={}, complementary_data={"task": 123})

    with pytest.raises(ValueError, match="Task cannot be None"):
        processor(transition)

    # Test with mixed list - get_task returns None, observation raises ValueError
    transition = create_transition(observation={}, complementary_data={"task": ["text", 123, "more text"]})

    with pytest.raises(ValueError, match="Task cannot be None"):
        processor(transition)


@require_package("transformers")
def test_no_tokenizer_error():
    """Test that ValueError is raised when neither tokenizer nor tokenizer_name is provided."""
    with pytest.raises(ValueError, match="Either 'tokenizer' or 'tokenizer_name' must be provided"):
        TokenizerProcessorStep()


@require_package("transformers")
def test_invalid_tokenizer_name_error():
    """Test that error is raised when invalid tokenizer_name is provided."""
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer") as mock_auto_tokenizer:
        # Mock import error
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Model not found"):
            TokenizerProcessorStep(tokenizer_name="invalid-tokenizer")


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_get_config_with_tokenizer_name(mock_auto_tokenizer):
    """Test configuration serialization when using tokenizer_name."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(
        tokenizer_name="test-tokenizer",
        max_length=256,
        task_key="instruction",
        padding="longest",
        truncation=False,
    )

    config = processor.get_config()

    expected = {
        "tokenizer_name": "test-tokenizer",
        "max_length": 256,
        "task_key": "instruction",
        "padding_side": "right",
        "padding": "longest",
        "truncation": False,
    }

    assert config == expected


@require_package("transformers")
def test_get_config_with_tokenizer_object():
    """Test configuration serialization when using tokenizer object."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    processor = TokenizerProcessorStep(
        tokenizer=mock_tokenizer,
        max_length=256,
        task_key="instruction",
        padding="longest",
        truncation=False,
    )

    config = processor.get_config()

    # tokenizer_name should not be in config when tokenizer object is used
    expected = {
        "max_length": 256,
        "task_key": "instruction",
        "padding_side": "right",
        "padding": "longest",
        "truncation": False,
    }

    assert config == expected
    assert "tokenizer_name" not in config


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_state_dict_methods(mock_auto_tokenizer):
    """Test state_dict and load_state_dict methods."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    # Should return empty dict
    state = processor.state_dict()
    assert state == {}

    # load_state_dict should not raise error
    processor.load_state_dict({})


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_reset_method(mock_auto_tokenizer):
    """Test reset method."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    # Should not raise error
    processor.reset()


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_integration_with_robot_processor(mock_auto_tokenizer):
    """Test integration with RobotProcessor."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    tokenizer_processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=6)
    robot_processor = DataProcessorPipeline(
        [tokenizer_processor], to_transition=identity_transition, to_output=identity_transition
    )

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "test task"},
    )

    result = robot_processor(transition)

    # Check that observation exists and tokenization was applied
    assert TransitionKey.OBSERVATION in result
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.shape == (6,)
    assert attention_mask.shape == (6,)

    # Check that other data is preserved
    assert torch.equal(
        result[TransitionKey.OBSERVATION]["state"], transition[TransitionKey.OBSERVATION]["state"]
    )
    assert torch.equal(result[TransitionKey.ACTION], transition[TransitionKey.ACTION])


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_save_and_load_pretrained_with_tokenizer_name(mock_auto_tokenizer):
    """Test saving and loading processor with tokenizer_name."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    original_processor = TokenizerProcessorStep(
        tokenizer_name="test-tokenizer", max_length=32, task_key="instruction"
    )

    robot_processor = DataProcessorPipeline(
        [original_processor], to_transition=identity_transition, to_output=identity_transition
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save processor
        robot_processor.save_pretrained(temp_dir)

        # Load processor - tokenizer will be recreated from saved config
        loaded_processor = DataProcessorPipeline.from_pretrained(
            temp_dir,
            config_filename="dataprocessorpipeline.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        # Test that loaded processor works
        transition = create_transition(
            observation={"state": torch.tensor([1.0, 2.0])},
            action=torch.tensor([0.1, 0.2]),
            complementary_data={"instruction": "test instruction"},
        )

        result = loaded_processor(transition)
        assert TransitionKey.OBSERVATION in result
        assert f"{OBS_LANGUAGE}.tokens" in result[TransitionKey.OBSERVATION]
        assert f"{OBS_LANGUAGE}.attention_mask" in result[TransitionKey.OBSERVATION]


@require_package("transformers")
def test_save_and_load_pretrained_with_tokenizer_object():
    """Test saving and loading processor with tokenizer object using overrides."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    original_processor = TokenizerProcessorStep(
        tokenizer=mock_tokenizer, max_length=32, task_key="instruction"
    )

    robot_processor = DataProcessorPipeline(
        [original_processor], to_transition=identity_transition, to_output=identity_transition
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save processor
        robot_processor.save_pretrained(temp_dir)

        # Load processor with tokenizer override (since tokenizer object wasn't saved)
        loaded_processor = DataProcessorPipeline.from_pretrained(
            temp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides={"tokenizer_processor": {"tokenizer": mock_tokenizer}},
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        # Test that loaded processor works
        transition = create_transition(
            observation={"state": torch.tensor([1.0, 2.0])},
            action=torch.tensor([0.1, 0.2]),
            complementary_data={"instruction": "test instruction"},
        )

        result = loaded_processor(transition)
        assert TransitionKey.OBSERVATION in result
        assert f"{OBS_LANGUAGE}.tokens" in result[TransitionKey.OBSERVATION]
        assert f"{OBS_LANGUAGE}.attention_mask" in result[TransitionKey.OBSERVATION]


@require_package("transformers")
def test_registry_functionality():
    """Test that the processor is properly registered."""
    from lerobot.processor import ProcessorStepRegistry

    # Check that the processor is registered
    assert "tokenizer_processor" in ProcessorStepRegistry.list()

    # Check that we can retrieve it
    retrieved_class = ProcessorStepRegistry.get("tokenizer_processor")
    assert retrieved_class is TokenizerProcessorStep


@require_package("transformers")
def test_features_basic():
    """Test basic feature contract functionality."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=128)

    input_features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
    }

    output_features = processor.transform_features(input_features)

    # Check that original features are preserved
    assert OBS_STATE in output_features[PipelineFeatureType.OBSERVATION]
    assert ACTION in output_features[PipelineFeatureType.ACTION]

    # Check that tokenized features are added
    assert f"{OBS_LANGUAGE}.tokens" in output_features[PipelineFeatureType.OBSERVATION]
    assert f"{OBS_LANGUAGE}.attention_mask" in output_features[PipelineFeatureType.OBSERVATION]

    # Check feature properties
    tokens_feature = output_features[PipelineFeatureType.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask_feature = output_features[PipelineFeatureType.OBSERVATION][
        f"{OBS_LANGUAGE}.attention_mask"
    ]

    assert tokens_feature.type == FeatureType.LANGUAGE
    assert tokens_feature.shape == (128,)
    assert attention_mask_feature.type == FeatureType.LANGUAGE
    assert attention_mask_feature.shape == (128,)


@require_package("transformers")
def test_features_with_custom_max_length():
    """Test feature contract with custom max_length."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=64)

    input_features = {PipelineFeatureType.OBSERVATION: {}}
    output_features = processor.transform_features(input_features)

    # Check that features use correct max_length
    assert f"{OBS_LANGUAGE}.tokens" in output_features[PipelineFeatureType.OBSERVATION]
    assert f"{OBS_LANGUAGE}.attention_mask" in output_features[PipelineFeatureType.OBSERVATION]

    tokens_feature = output_features[PipelineFeatureType.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask_feature = output_features[PipelineFeatureType.OBSERVATION][
        f"{OBS_LANGUAGE}.attention_mask"
    ]

    assert tokens_feature.shape == (64,)
    assert attention_mask_feature.shape == (64,)


@require_package("transformers")
def test_features_existing_features():
    """Test feature contract when tokenized features already exist."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=256)

    input_features = {
        PipelineFeatureType.OBSERVATION: {
            f"{OBS_LANGUAGE}.tokens": PolicyFeature(type=FeatureType.LANGUAGE, shape=(100,)),
            f"{OBS_LANGUAGE}.attention_mask": PolicyFeature(type=FeatureType.LANGUAGE, shape=(100,)),
        }
    }

    output_features = processor.transform_features(input_features)

    # Should not overwrite existing features
    assert output_features[PipelineFeatureType.OBSERVATION][f"{OBS_LANGUAGE}.tokens"].shape == (
        100,
    )  # Original shape preserved
    assert output_features[PipelineFeatureType.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"].shape == (100,)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_tokenization_parameters(mock_auto_tokenizer):
    """Test that tokenization parameters are correctly passed to tokenizer."""

    # Create a custom mock that tracks calls
    class TrackingMockTokenizer:
        def __init__(self):
            self.last_call_args = None
            self.last_call_kwargs = None

        def __call__(self, *args, **kwargs):
            self.last_call_args = args
            self.last_call_kwargs = kwargs
            # Return minimal valid output
            return {
                "input_ids": torch.zeros(16, dtype=torch.long),
                "attention_mask": torch.ones(16, dtype=torch.long),
            }

    tracking_tokenizer = TrackingMockTokenizer()
    mock_auto_tokenizer.from_pretrained.return_value = tracking_tokenizer

    processor = TokenizerProcessorStep(
        tokenizer_name="test-tokenizer",
        max_length=16,
        padding="longest",
        truncation=False,
        padding_side="left",
    )

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "test task"},
    )

    processor(transition)

    # Check that parameters were passed correctly (task is converted to list)
    assert tracking_tokenizer.last_call_args == (["test task"],)
    assert tracking_tokenizer.last_call_kwargs["max_length"] == 16
    assert tracking_tokenizer.last_call_kwargs["padding"] == "longest"
    assert tracking_tokenizer.last_call_kwargs["padding_side"] == "left"
    assert tracking_tokenizer.last_call_kwargs["truncation"] is False
    assert tracking_tokenizer.last_call_kwargs["return_tensors"] == "pt"


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_preserves_other_complementary_data(mock_auto_tokenizer):
    """Test that other complementary data fields are preserved."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer")

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={
            "task": "test task",
            "episode_id": 123,
            "timestamp": 456.789,
            "other_field": {"nested": "data"},
        },
    )

    result = processor(transition)
    comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check that all original fields are preserved
    assert comp_data["task"] == "test task"
    assert comp_data["episode_id"] == 123
    assert comp_data["timestamp"] == 456.789
    assert comp_data["other_field"] == {"nested": "data"}

    # Check that tokens were added to observation
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_deterministic_tokenization(mock_auto_tokenizer):
    """Test that tokenization is deterministic for the same input."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=10)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "consistent test"},
    )

    result1 = processor(transition)
    result2 = processor(transition)

    tokens1 = result1[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask1 = result1[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]
    tokens2 = result2[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask2 = result2[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    # Results should be identical
    assert torch.equal(tokens1, tokens2)
    assert torch.equal(attention_mask1, attention_mask2)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_empty_string_task(mock_auto_tokenizer):
    """Test handling of empty string task."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=8)

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": ""},
    )

    result = processor(transition)

    # Should still tokenize (mock tokenizer handles empty strings)
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    assert tokens.shape == (8,)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_very_long_task(mock_auto_tokenizer):
    """Test handling of very long task strings."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=5, truncation=True)

    long_task = " ".join(["word"] * 100)  # Very long task
    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": long_task},
    )

    result = processor(transition)

    # Should be truncated to max_length
    observation = result[TransitionKey.OBSERVATION]
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.shape == (5,)
    assert attention_mask.shape == (5,)


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_custom_padding_side(mock_auto_tokenizer):
    """Test using custom padding_side parameter."""

    # Create a mock tokenizer that tracks padding_side calls
    class PaddingSideTrackingTokenizer:
        def __init__(self):
            self.padding_side_calls = []

        def __call__(
            self,
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
            **kwargs,
        ):
            self.padding_side_calls.append(padding_side)
            # Return minimal valid output
            return {
                "input_ids": torch.zeros(max_length, dtype=torch.long),
                "attention_mask": torch.ones(max_length, dtype=torch.long),
            }

    tracking_tokenizer = PaddingSideTrackingTokenizer()
    mock_auto_tokenizer.from_pretrained.return_value = tracking_tokenizer

    # Test left padding
    processor_left = TokenizerProcessorStep(
        tokenizer_name="test-tokenizer", max_length=10, padding_side="left"
    )

    transition = create_transition(
        observation={"state": torch.tensor([1.0, 2.0])},
        action=torch.tensor([0.1, 0.2]),
        complementary_data={"task": "test task"},
    )
    processor_left(transition)

    assert tracking_tokenizer.padding_side_calls[-1] == "left"

    # Test right padding (default)
    processor_right = TokenizerProcessorStep(
        tokenizer_name="test-tokenizer", max_length=10, padding_side="right"
    )

    processor_right(transition)

    assert tracking_tokenizer.padding_side_calls[-1] == "right"


@require_package("transformers")
def test_device_detection_cpu():
    """Test that tokenized tensors stay on CPU when other tensors are on CPU."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Create transition with CPU tensors
    observation = {OBS_STATE: torch.randn(10)}  # CPU tensor
    action = torch.randn(5)  # CPU tensor
    transition = create_transition(
        observation=observation, action=action, complementary_data={"task": "test task"}
    )

    result = processor(transition)

    # Check that tokenized tensors are on CPU
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device.type == "cpu"
    assert attention_mask.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@require_package("transformers")
def test_device_detection_cuda():
    """Test that tokenized tensors are moved to CUDA when other tensors are on CUDA."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Create transition with CUDA tensors
    observation = {OBS_STATE: torch.randn(10).cuda()}  # CUDA tensor
    action = torch.randn(5).cuda()  # CUDA tensor
    transition = create_transition(
        observation=observation, action=action, complementary_data={"task": "test task"}
    )

    result = processor(transition)

    # Check that tokenized tensors are on CUDA
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device.type == "cuda"
    assert attention_mask.device.type == "cuda"
    assert tokens.device.index == 0  # Should be on same device as input


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@require_package("transformers")
def test_device_detection_multi_gpu():
    """Test that tokenized tensors match device in multi-GPU setup."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Test with tensors on cuda:1
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(10).to(device)}
    action = torch.randn(5).to(device)
    transition = create_transition(
        observation=observation, action=action, complementary_data={"task": "multi gpu test"}
    )

    result = processor(transition)

    # Check that tokenized tensors are on cuda:1
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device == device
    assert attention_mask.device == device


@require_package("transformers")
def test_device_detection_no_tensors():
    """Test that tokenized tensors stay on CPU when no other tensors exist."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Create transition with no tensors
    transition = create_transition(
        observation={"metadata": {"key": "value"}},  # No tensors
        complementary_data={"task": "no tensor test"},
    )

    result = processor(transition)

    # Check that tokenized tensors are on CPU (default)
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device.type == "cpu"
    assert attention_mask.device.type == "cpu"


@require_package("transformers")
def test_device_detection_mixed_devices():
    """Test device detection when tensors are on different devices (uses first found)."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    if torch.cuda.is_available():
        # Create transition with mixed devices
        observation = {
            "observation.cpu": torch.randn(10),  # CPU
            "observation.cuda": torch.randn(10).cuda(),  # CUDA
        }
        transition = create_transition(
            observation=observation, complementary_data={"task": "mixed device test"}
        )

        result = processor(transition)

        # The device detection should use the first tensor found
        # (iteration order depends on dict, but result should be consistent)
        tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
        attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

        # Both should be on the same device
        assert tokens.device == attention_mask.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@require_package("transformers")
def test_device_detection_from_action():
    """Test that device is detected from action tensor when no observation tensors exist."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Create transition with action on CUDA but no observation tensors
    observation = {"metadata": {"key": "value"}}  # No tensors in observation
    action = torch.randn(5).cuda()
    transition = create_transition(
        observation=observation, action=action, complementary_data={"task": "action device test"}
    )

    result = processor(transition)

    # Check that tokenized tensors match action's device
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device.type == "cuda"
    assert attention_mask.device.type == "cuda"


@require_package("transformers")
def test_device_detection_preserves_dtype():
    """Test that device detection doesn't affect dtype of tokenized tensors."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Create transition with float tensor (to test dtype isn't affected)
    observation = {OBS_STATE: torch.randn(10, dtype=torch.float16)}
    transition = create_transition(observation=observation, complementary_data={"task": "dtype test"})

    result = processor(transition)

    # Check that tokenized tensors have correct dtypes (not affected by input dtype)
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.dtype == torch.long  # Should remain long
    assert attention_mask.dtype == torch.bool  # Should be bool (converted in processor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_integration_with_device_processor(mock_auto_tokenizer):
    """Test that TokenizerProcessorStep works correctly with DeviceProcessorStep in pipeline."""
    from lerobot.processor import DeviceProcessorStep

    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    # Create pipeline with TokenizerProcessorStep then DeviceProcessorStep
    tokenizer_processor = TokenizerProcessorStep(tokenizer_name="test-tokenizer", max_length=6)
    device_processor = DeviceProcessorStep(device="cuda:0")
    robot_processor = DataProcessorPipeline(
        [tokenizer_processor, device_processor],
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    # Start with CPU tensors
    transition = create_transition(
        observation={OBS_STATE: torch.randn(10)},  # CPU
        action=torch.randn(5),  # CPU
        complementary_data={"task": "pipeline test"},
    )

    result = robot_processor(transition)

    # All tensors should end up on CUDA (moved by DeviceProcessorStep)
    assert result[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert result[TransitionKey.ACTION].device.type == "cuda"

    # Tokenized tensors should also be on CUDA
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.device.type == "cuda"
    assert attention_mask.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@require_package("transformers")
def test_simulated_accelerate_scenario():
    """Test scenario simulating Accelerate with data already on GPU."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessorStep(tokenizer=mock_tokenizer, max_length=10)

    # Simulate Accelerate scenario: batch already on GPU
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(1, 10).to(device),  # Batched, on GPU
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),  # Batched, on GPU
    }
    action = torch.randn(1, 5).to(device)  # Batched, on GPU

    transition = create_transition(
        observation=observation,
        action=action,
        complementary_data={"task": ["accelerate test"]},  # List for batched task
    )

    result = processor(transition)

    # Tokenized tensors should match GPU placement
    tokens = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask = result[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens.device == device
    assert attention_mask.device == device
    # MockTokenizer squeezes single-item batches, so shape is (max_length,) not (1, max_length)
    assert tokens.shape == (10,)  # MockTokenizer behavior for single string in list
    assert attention_mask.shape == (10,)
