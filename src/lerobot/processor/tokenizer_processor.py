"""
Tokenizer processor for handling text tokenization in robot transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_LANGUAGE
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessor:
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
    tokenizer: AutoTokenizer | None = None
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # Internal tokenizer instance (not serialized)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the tokenizer from the provided tokenizer or tokenizer name."""
        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            self._tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process the transition by tokenizing the task text.

        Args:
            transition: Input transition containing complementary_data with task text.

        Returns:
            Modified transition with tokenized task added to observation.

        Raises:
            ValueError: If tokenizer initialization failed.
        """
        task = self.get_task(transition)
        if task is None:
            return transition

        # Tokenize the task
        tokenized_prompt = self._tokenize_text(task)

        # Get or create observation dict
        if TransitionKey.OBSERVATION not in transition or transition[TransitionKey.OBSERVATION] is None:
            transition[TransitionKey.OBSERVATION] = {}
        observation = transition[TransitionKey.OBSERVATION]

        # Add tokenized data to observation
        observation[f"{OBS_LANGUAGE}.tokens"] = tokenized_prompt["input_ids"]
        observation[f"{OBS_LANGUAGE}.attention_mask"] = tokenized_prompt["attention_mask"].to(
            dtype=torch.bool
        )

        return transition

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
        if self.tokenizer_name is not None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary (empty for this processor)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary (no-op for this processor)."""
        pass

    def reset(self) -> None:
        """Reset processor state (no-op for this processor)."""
        pass

    def features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
