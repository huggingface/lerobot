"""
Tokenizer processor for handling text tokenization in robot transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.processor.pipeline import (
    EnvTransition,
    ObservationProcessor,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessor(ObservationProcessor):
    """Tokenizes text tasks in complementary data using a huggingface tokenizer.

    This processor handles tokenization of task strings found in the complementary_data
    using a specified pretrained tokenizer from Hugging Face. It adds tokenized versions
    to the observation data for model processing while preserving the original task string.

    The processor supports both single strings and lists of strings as task inputs.

    Args:
        tokenizer_name: Name of the pretrained tokenizer to load from Hugging Face Hub
            (e.g., "bert-base-uncased", "microsoft/DialoGPT-medium"). This will be used
            with AutoTokenizer.from_pretrained(). If tokenizer is provided, this is ignored.
        tokenizer: A tokenizer object (e.g., from transformers library) that implements
            the __call__ method. If provided, tokenizer_name is ignored. This parameter
            is not serialized and must be provided via overrides when loading.
        max_length: Maximum sequence length for tokenization. Defaults to 512.
        task_key: Key in complementary_data containing the task text. Defaults to "task".
        padding: Padding strategy for tokenization. Defaults to "max_length".
        truncation: Whether to truncate sequences longer than max_length. Defaults to True.

    Examples:
        Using tokenizer name (auto-loaded):
        ```python
        processor = TokenizerProcessor(tokenizer_name="bert-base-uncased", max_length=128)
        ```

        Using custom tokenizer object:
        ```python
        from transformers import AutoTokenizer

        custom_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        processor = TokenizerProcessor(tokenizer=custom_tokenizer, max_length=128)
        ```
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # Otherwise transformers is not available in the core dependencies
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # Internal tokenizer instance (not serialized)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the tokenizer from the provided tokenizer or tokenizer name."""
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessor."
            )

        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            self._tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """Extract and normalize task from complementary data.

        Args:
            transition: Input transition containing complementary_data.

        Returns:
            List of task strings if task is present, None otherwise.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return None

        if self.task_key not in complementary_data:
            return None

        task = complementary_data[self.task_key]
        if task is None:
            return None

        # Convert to list of strings
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def observation(self, observation):
        """Process the transition by tokenizing the task text.

        Args:
            transition: Input transition containing complementary_data with task text.

        Returns:
            Modified transition with tokenized task added to observation.

        Raises:
            ValueError: If tokenizer initialization failed.
        """
        task = self.get_task(self.transition)
        if task is None:
            return observation

        # Tokenize the task (creates CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect device from existing tensors in the transition
        target_device = self._detect_device(self.transition)

        # Move tokenized tensors to match the device of other data
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Get or create observation dict
        new_observation = dict(observation)

        # Add tokenized data to observation
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """Detect device from existing tensors in the transition.

        This allows the tokenized tensors to match the device of other data,
        which is especially important for multi-GPU training with Accelerate.

        Args:
            transition: The transition to search for existing tensors.

        Returns:
            The device of the first tensor found, or None if no tensors exist.
        """
        # Check observation tensors first (most likely to exist)
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # Check action tensor
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # No tensors found, keep on CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """Tokenize text using the configured tokenizer.

        Args:
            text: Text string or list of strings to tokenize.

        Returns:
            Dictionary containing tokenized output with keys like 'input_ids', 'attention_mask'.
        """
        return self._tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization.

        Note: Only tokenizer_name is saved, not the tokenizer object itself.
        When loading, provide the tokenizer via overrides if needed.
        """
        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

        # Only include tokenizer_name if it was used (not when tokenizer object was provided)
        # TODO(steven): Consider saving the name of the _tokenizer if it was loaded
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Add tokenized task features to the feature contract.

        Args:
            features: Input feature dictionary.

        Returns:
            Updated feature dictionary with tokenized task features added.
        """
        # Add features for tokenized output if they don't exist
        # Standard tokenizer output includes tokens and attention_mask

        if OBS_LANGUAGE_TOKENS not in features:
            features[OBS_LANGUAGE_TOKENS] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.max_length,))

        if OBS_LANGUAGE_ATTENTION_MASK not in features:
            features[OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features
