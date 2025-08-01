"""
Tests for the TokenizerProcessor class.
"""

import tempfile
from unittest.mock import patch

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import OBS_LANGUAGE
from lerobot.processor.pipeline import RobotProcessor, TransitionKey
from lerobot.processor.tokenizer_processor import TokenizerProcessor


def create_transition(
    observation=None, action=None, reward=None, done=None, truncated=None, info=None, complementary_data=None
):
    """Helper function to create test transitions."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }


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
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Mock tokenization that returns deterministic tokens based on text."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

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


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_basic_tokenization(mock_auto_tokenizer):
    """Test basic string tokenization functionality."""
    # Mock AutoTokenizer.from_pretrained to return our mock tokenizer
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=10)

    transition = create_transition(complementary_data={"task": "pick up the red cube"})

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


def test_basic_tokenization_with_tokenizer_object():
    """Test basic string tokenization functionality using tokenizer object directly."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    processor = TokenizerProcessor(tokenizer=mock_tokenizer, max_length=10)

    transition = create_transition(complementary_data={"task": "pick up the red cube"})

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


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_list_of_strings_tokenization(mock_auto_tokenizer):
    """Test tokenization of a list of strings."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=8)

    transition = create_transition(complementary_data={"task": ["pick up cube", "place on table"]})

    result = processor(transition)

    # Check that original task is preserved
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick up cube", "place on table"]

    # Check that tokens were added to observation
    observation = result[TransitionKey.OBSERVATION]
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.shape == (2, 8)  # batch_size=2, seq_len=8
    assert attention_mask.shape == (2, 8)


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_custom_keys(mock_auto_tokenizer):
    """Test using custom task_key."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", task_key="instruction", max_length=5)

    transition = create_transition(complementary_data={"instruction": "move forward"})

    result = processor(transition)

    # Check that tokens are stored in observation regardless of task_key
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    assert f"{OBS_LANGUAGE}.attention_mask" in observation

    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    assert tokens.shape == (5,)


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_none_complementary_data(mock_auto_tokenizer):
    """Test handling of None complementary_data."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    transition = create_transition(complementary_data=None)

    result = processor(transition)
    assert result == transition  # Should return unchanged


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_missing_task_key(mock_auto_tokenizer):
    """Test handling when task key is missing."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    transition = create_transition(complementary_data={"other_field": "some value"})

    result = processor(transition)
    assert result == transition  # Should return unchanged


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_none_task_value(mock_auto_tokenizer):
    """Test handling when task value is None."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    transition = create_transition(complementary_data={"task": None})

    result = processor(transition)
    assert result == transition  # Should return unchanged


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_unsupported_task_type(mock_auto_tokenizer):
    """Test handling of unsupported task types."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    # Test with integer task
    transition = create_transition(complementary_data={"task": 123})

    result = processor(transition)
    assert result == transition  # Should return unchanged

    # Test with mixed list
    transition = create_transition(complementary_data={"task": ["text", 123, "more text"]})

    result = processor(transition)
    assert result == transition  # Should return unchanged


def test_no_tokenizer_error():
    """Test that ValueError is raised when neither tokenizer nor tokenizer_name is provided."""
    with pytest.raises(ValueError, match="Either 'tokenizer' or 'tokenizer_name' must be provided"):
        TokenizerProcessor()


def test_invalid_tokenizer_name_error():
    """Test that error is raised when invalid tokenizer_name is provided."""
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer") as mock_auto_tokenizer:
        # Mock import error
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Model not found"):
            TokenizerProcessor(tokenizer_name="invalid-tokenizer")


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_get_config_with_tokenizer_name(mock_auto_tokenizer):
    """Test configuration serialization when using tokenizer_name."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(
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
        "padding": "longest",
        "truncation": False,
    }

    assert config == expected


def test_get_config_with_tokenizer_object():
    """Test configuration serialization when using tokenizer object."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    processor = TokenizerProcessor(
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
        "padding": "longest",
        "truncation": False,
    }

    assert config == expected
    assert "tokenizer_name" not in config


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_state_dict_methods(mock_auto_tokenizer):
    """Test state_dict and load_state_dict methods."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    # Should return empty dict
    state = processor.state_dict()
    assert state == {}

    # load_state_dict should not raise error
    processor.load_state_dict({})


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_reset_method(mock_auto_tokenizer):
    """Test reset method."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    # Should not raise error
    processor.reset()


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_integration_with_robot_processor(mock_auto_tokenizer):
    """Test integration with RobotProcessor."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    tokenizer_processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=6)
    robot_processor = RobotProcessor([tokenizer_processor])

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


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_save_and_load_pretrained_with_tokenizer_name(mock_auto_tokenizer):
    """Test saving and loading processor with tokenizer_name."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    original_processor = TokenizerProcessor(
        tokenizer_name="test-tokenizer", max_length=32, task_key="instruction"
    )

    robot_processor = RobotProcessor([original_processor])

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save processor
        robot_processor.save_pretrained(temp_dir)

        # Load processor - tokenizer will be recreated from saved config
        loaded_processor = RobotProcessor.from_pretrained(temp_dir)

        # Test that loaded processor works
        transition = create_transition(complementary_data={"instruction": "test instruction"})

        result = loaded_processor(transition)
        assert TransitionKey.OBSERVATION in result
        assert f"{OBS_LANGUAGE}.tokens" in result[TransitionKey.OBSERVATION]
        assert f"{OBS_LANGUAGE}.attention_mask" in result[TransitionKey.OBSERVATION]


def test_save_and_load_pretrained_with_tokenizer_object():
    """Test saving and loading processor with tokenizer object using overrides."""
    mock_tokenizer = MockTokenizer(vocab_size=100)

    original_processor = TokenizerProcessor(tokenizer=mock_tokenizer, max_length=32, task_key="instruction")

    robot_processor = RobotProcessor([original_processor])

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save processor
        robot_processor.save_pretrained(temp_dir)

        # Load processor with tokenizer override (since tokenizer object wasn't saved)
        loaded_processor = RobotProcessor.from_pretrained(
            temp_dir, overrides={"tokenizer_processor": {"tokenizer": mock_tokenizer}}
        )

        # Test that loaded processor works
        transition = create_transition(complementary_data={"instruction": "test instruction"})

        result = loaded_processor(transition)
        assert TransitionKey.OBSERVATION in result
        assert f"{OBS_LANGUAGE}.tokens" in result[TransitionKey.OBSERVATION]
        assert f"{OBS_LANGUAGE}.attention_mask" in result[TransitionKey.OBSERVATION]


def test_registry_functionality():
    """Test that the processor is properly registered."""
    from lerobot.processor.pipeline import ProcessorStepRegistry

    # Check that the processor is registered
    assert "tokenizer_processor" in ProcessorStepRegistry.list()

    # Check that we can retrieve it
    retrieved_class = ProcessorStepRegistry.get("tokenizer_processor")
    assert retrieved_class is TokenizerProcessor


def test_feature_contract_basic():
    """Test basic feature contract functionality."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessor(tokenizer=mock_tokenizer, max_length=128)

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    }

    output_features = processor.feature_contract(input_features)

    # Check that original features are preserved
    assert "observation.state" in output_features
    assert "action" in output_features

    # Check that tokenized features are added
    assert f"{OBS_LANGUAGE}.tokens" in output_features
    assert f"{OBS_LANGUAGE}.attention_mask" in output_features

    # Check feature properties
    tokens_feature = output_features[f"{OBS_LANGUAGE}.tokens"]
    attention_mask_feature = output_features[f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens_feature.type == FeatureType.LANGUAGE
    assert tokens_feature.shape == (128,)
    assert attention_mask_feature.type == FeatureType.LANGUAGE
    assert attention_mask_feature.shape == (128,)


def test_feature_contract_with_custom_max_length():
    """Test feature contract with custom max_length."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessor(tokenizer=mock_tokenizer, max_length=64)

    input_features = {}
    output_features = processor.feature_contract(input_features)

    # Check that features use correct max_length
    assert f"{OBS_LANGUAGE}.tokens" in output_features
    assert f"{OBS_LANGUAGE}.attention_mask" in output_features

    tokens_feature = output_features[f"{OBS_LANGUAGE}.tokens"]
    attention_mask_feature = output_features[f"{OBS_LANGUAGE}.attention_mask"]

    assert tokens_feature.shape == (64,)
    assert attention_mask_feature.shape == (64,)


def test_feature_contract_existing_features():
    """Test feature contract when tokenized features already exist."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    processor = TokenizerProcessor(tokenizer=mock_tokenizer, max_length=256)

    input_features = {
        f"{OBS_LANGUAGE}.tokens": PolicyFeature(type=FeatureType.LANGUAGE, shape=(100,)),
        f"{OBS_LANGUAGE}.attention_mask": PolicyFeature(type=FeatureType.LANGUAGE, shape=(100,)),
    }

    output_features = processor.feature_contract(input_features)

    # Should not overwrite existing features
    assert output_features[f"{OBS_LANGUAGE}.tokens"].shape == (100,)  # Original shape preserved
    assert output_features[f"{OBS_LANGUAGE}.attention_mask"].shape == (100,)


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

    processor = TokenizerProcessor(
        tokenizer_name="test-tokenizer", max_length=16, padding="longest", truncation=False
    )

    transition = create_transition(complementary_data={"task": "test task"})

    processor(transition)

    # Check that parameters were passed correctly (task is converted to list)
    assert tracking_tokenizer.last_call_args == (["test task"],)
    assert tracking_tokenizer.last_call_kwargs["max_length"] == 16
    assert tracking_tokenizer.last_call_kwargs["padding"] == "longest"
    assert tracking_tokenizer.last_call_kwargs["truncation"] is False
    assert tracking_tokenizer.last_call_kwargs["return_tensors"] == "pt"


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_preserves_other_complementary_data(mock_auto_tokenizer):
    """Test that other complementary data fields are preserved."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer")

    transition = create_transition(
        complementary_data={
            "task": "test task",
            "episode_id": 123,
            "timestamp": 456.789,
            "other_field": {"nested": "data"},
        }
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


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_deterministic_tokenization(mock_auto_tokenizer):
    """Test that tokenization is deterministic for the same input."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=10)

    transition = create_transition(complementary_data={"task": "consistent test"})

    result1 = processor(transition)
    result2 = processor(transition)

    tokens1 = result1[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask1 = result1[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]
    tokens2 = result2[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.tokens"]
    attention_mask2 = result2[TransitionKey.OBSERVATION][f"{OBS_LANGUAGE}.attention_mask"]

    # Results should be identical
    assert torch.equal(tokens1, tokens2)
    assert torch.equal(attention_mask1, attention_mask2)


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_empty_string_task(mock_auto_tokenizer):
    """Test handling of empty string task."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=8)

    transition = create_transition(complementary_data={"task": ""})

    result = processor(transition)

    # Should still tokenize (mock tokenizer handles empty strings)
    observation = result[TransitionKey.OBSERVATION]
    assert f"{OBS_LANGUAGE}.tokens" in observation
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    assert tokens.shape == (8,)


@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_very_long_task(mock_auto_tokenizer):
    """Test handling of very long task strings."""
    mock_tokenizer = MockTokenizer(vocab_size=100)
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    processor = TokenizerProcessor(tokenizer_name="test-tokenizer", max_length=5, truncation=True)

    long_task = " ".join(["word"] * 100)  # Very long task
    transition = create_transition(complementary_data={"task": long_task})

    result = processor(transition)

    # Should be truncated to max_length
    observation = result[TransitionKey.OBSERVATION]
    tokens = observation[f"{OBS_LANGUAGE}.tokens"]
    attention_mask = observation[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.shape == (5,)
    assert attention_mask.shape == (5,)
