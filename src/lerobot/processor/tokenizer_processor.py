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
This script defines a processor for tokenizing natural language instructions from an environment transition.

It uses a tokenizer from the Hugging Face `transformers` library to convert task descriptions (text) into
token IDs and attention masks, which are then added to the observation dictionary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

from .core import EnvTransition, RobotObservation, TransitionKey
from .pipeline import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    Processor step to tokenize a natural language task description.

    This step extracts a task string from the `complementary_data` of an `EnvTransition`,
    tokenizes it using a Hugging Face `transformers` tokenizer, and adds the resulting
    token IDs and attention mask to the `observation` dictionary.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained tokenizer from the Hugging Face Hub (e.g., "bert-base-uncased").
        tokenizer: A pre-initialized tokenizer object. If provided, `tokenizer_name` is ignored.
        max_length: The maximum length to pad or truncate sequences to.
        task_key: The key in `complementary_data` where the task string is stored.
        padding_side: The side to pad on ('left' or 'right').
        padding: The padding strategy ('max_length', 'longest', etc.).
        truncation: Whether to truncate sequences longer than `max_length`.
        input_tokenizer: The internal tokenizer instance, loaded during initialization.
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # Use `Any` for compatibility without a hard dependency
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # Internal tokenizer instance (not part of the config)
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings for the tokenizer
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task

        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Tokenizes the task description and adds it to the observation dictionary.

        This method retrieves the task, tokenizes it, moves the resulting tensors to the
        same device as other data in the transition, and updates the observation.

        Args:
            observation: The original observation dictionary.

        Returns:
            The updated observation dictionary including token IDs and an attention mask.
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Tokenize the task (this will create CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect the device from existing tensors in the transition to ensure consistency
        target_device = self._detect_device(self.transition)

        # Move new tokenized tensors to the detected device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Create a new observation dict to avoid modifying the original in place
        new_observation = dict(observation)

        # Add tokenized data to the observation
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        Detects the torch.device from existing tensors in the transition.

        It checks tensors in the observation dictionary first, then the action tensor.

        Args:
            transition: The environment transition.

        Returns:
            The detected `torch.device`, or None if no tensors are found.
        """
        # Check observation tensors first (most likely place to find tensors)
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # Fallback to checking the action tensor
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # No tensors found, default will be CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        A wrapper around the tokenizer call.

        Args:
            text: A string or list of strings to tokenize.

        Returns:
            A dictionary containing tokenized 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Note: The tokenizer object itself is not serialized. If the processor was initialized
        with a tokenizer name, that name will be included in the config.

        Returns:
            A dictionary with the processor's configuration parameters.
        """
        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

        # Only save tokenizer_name if it was used to create the tokenizer
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for the language tokens and attention mask.

        This updates the policy features dictionary to include the new data added to the
        observation, ensuring downstream components are aware of their shape and type.

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        # Add a feature for the token IDs if it doesn't already exist
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # Add a feature for the attention mask if it doesn't already exist
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features


@dataclass
@ProcessorStepRegistry.register(name="action_tokenizer_processor")
class ActionTokenizerProcessorStep(ActionProcessorStep):
    """
    Processor step to tokenize action data using a fast action tokenizer.

    This step takes action tensors from an `EnvTransition`, tokenizes them using
    a Hugging Face `transformers` AutoProcessor (such as the Physical Intelligence "fast" tokenizer),
    and returns the tokenized action.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained processor from the Hugging Face Hub (e.g., "physical-intelligence/fast").
        tokenizer: A pre-initialized processor/tokenizer object. If provided, `tokenizer_name` is ignored.
        trust_remote_code: Whether to trust remote code when loading the tokenizer (required for some tokenizers).
        action_tokenizer: The internal tokenizer/processor instance, loaded during initialization.
        paligemma_tokenizer_name: The name of a pretrained PaliGemma tokenizer from the Hugging Face Hub (e.g., "google/paligemma-3b-pt-224").
    """

    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_skip_tokens: int = 128
    paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224"
    # Internal tokenizer instance (not part of the config)
    action_tokenizer: Any = field(default=None, init=False, repr=False)
    _paligemma_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the action tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use ActionTokenizerProcessorStep."
            )

        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object

        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=self.trust_remote_code,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies action tokenization to the transition.

        This overrides the base class to handle both tokens and mask.

        Args:
            transition: The input transition with action data.

        Returns:
            The processed transition with tokenized actions and mask in complementary data.
        """
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            # During inference, no action is available, skip tokenization
            return new_transition

        # Tokenize and get both tokens and mask
        tokens, mask = self._tokenize_action(action)

        # Store mask in complementary data
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary_data is None:
            complementary_data = {}
        complementary_data[ACTION_TOKEN_MASK] = mask
        complementary_data[ACTION_TOKENS] = tokens
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts action tokens to PaliGemma tokens.
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes the action tensor and creates a mask.

        Args:
            action: The input action tensor to tokenize. Shape: (B, H, action_dim) or (H, action_dim,)

        Returns:
            A tuple of (tokens, mask) where:
            - tokens: Tensor of token IDs with shape (B, max_action_tokens)
            - mask: Boolean mask with shape (B, max_action_tokens), True for real tokens, False for padding
        """
        if action is None:
            raise ValueError("Action cannot be None")

        # Get the device and dtype of the input action
        device = action.device if isinstance(action, torch.Tensor) else None

        # Handle single sample (add batch dimension)
        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]

        # Tokenize the action batch
        # The fast tokenizer expects action data and returns token IDs
        tokens_list = []
        masks_list = []

        for i in range(batch_size):
            # Tokenize single action (move to CPU first as tokenizer uses scipy which requires numpy)
            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)

            # Convert to numpy array if it's a list
            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:
                # Move tokens back to the same device as input action
                tokens = tokens.to(device=action.device)

            # Flatten to 1D if needed
            if tokens.dim() > 1:
                tokens = tokens.flatten()

            bos_id = self._paligemma_tokenizer.bos_token_id
            # add bos
            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=action.device),
                    torch.tensor(
                        self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                        device=action.device,
                    ),
                    self._act_tokens_to_paligemma_tokens(tokens),
                    torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device),
                ]
            )

            # Truncate or pad to max_action_tokens
            if len(tokens) > self.max_action_tokens:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
                    "Consider increasing the `max_action_tokens` in your model config if this happens frequently."
                )
                tokens = tokens[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(
                            self.max_action_tokens - len(tokens), dtype=torch.bool, device=action.device
                        ),
                    ]
                )
                # Pad tokens with zeros
                tokens = torch.nn.functional.pad(tokens, (0, self.max_action_tokens - len(tokens)), value=0)

            tokens_list.append(tokens)
            masks_list.append(mask)

        # Stack into batched tensors
        tokens_batch = torch.stack(tokens_list, dim=0)  # (B, max_action_tokens)
        masks_batch = torch.stack(masks_list, dim=0)  # (B, max_action_tokens)

        # Remove batch dimension if input was single sample
        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)

        # Move to the same device as the input
        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)

        return tokens_batch, masks_batch

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """
        This method is not used since we override __call__.
        Required by ActionProcessorStep ABC.
        """
        tokens, _ = self._tokenize_action(action)
        return tokens

    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Note: The tokenizer object itself is not serialized. If the processor was initialized
        with a tokenizer name, that name will be included in the config.

        Returns:
            A dictionary with the processor's configuration parameters.
        """
        config = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
        }

        # Only save tokenizer_name if it was used to create the tokenizer
        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates feature definitions to reflect tokenized actions.

        This updates the policy features dictionary to indicate that the action
        has been tokenized into a sequence of token IDs with shape (max_action_tokens,).

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        return features
