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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from .core import EnvTransition, TransitionKey
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry, ProcessorStep

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
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

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
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
@ProcessorStepRegistry.register(name="pi0fast_tokenizer_processor")
class PI0FASTTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to tokenize state, language, and actions for PI0FAST models.

    This step handles the complete tokenization pipeline for PI0FAST:
    1. Discretizes state observations
    2. Formats task descriptions with state
    3. Tokenizes actions using the FAST tokenizer
    4. Combines everything into the proper format with masks

    Example usage:
        ```python
        from transformers import AutoTokenizer, AutoProcessor
        from lerobot.processor.tokenizer_processor import PI0FASTTokenizerProcessorStep

        # Initialize tokenizers
        paligemma_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        paligemma_processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
        fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

        # Create processor step
        processor = PI0FASTTokenizerProcessorStep(
            paligemma_tokenizer=paligemma_tokenizer,
            fast_tokenizer=fast_tokenizer,
            paligemma_processor=paligemma_processor,
            max_action_dim=7,
            fast_skip_tokens=2,
            max_input_seq_len=180,
            task_key="task",
            state_key="observation.state"
        )

        # Apply to a transition
        tokenized_transition = processor(transition)

        # Access tokenized data from observation
        input_ids = tokenized_transition["observation"]["pi0fast_input_ids"]
        attention_mask = tokenized_transition["observation"]["pi0fast_attention_mask"]
        loss_mask = tokenized_transition["observation"]["pi0fast_loss_mask"]
        token_type_ids = tokenized_transition["observation"]["pi0fast_token_type_ids"]
        ```

    Attributes:
        paligemma_tokenizer: The PaliGemma tokenizer for text
        fast_tokenizer: The FAST tokenizer for actions
        paligemma_processor: The PaliGemma processor
        max_action_dim: Maximum dimension for actions (default: 7)
        fast_skip_tokens: Number of tokens to skip in FAST tokenizer mapping (default: 2)
        max_input_seq_len: Maximum input sequence length (default: 180)
        padding_side: The side to pad on ('left' or 'right', default: 'right')
        task_key: The key in complementary_data where the task string is stored (default: 'task')
        state_key: The key in observation where the state is stored (default: 'observation.state')
    """

    paligemma_tokenizer: Any = None
    fast_tokenizer: Any = None
    paligemma_processor: Any = None
    max_action_dim: int = 7
    fast_skip_tokens: int = 2
    max_input_seq_len: int = 180
    padding_side: str = "right"
    task_key: str = "task"
    state_key: str = OBS_STATE

    def __post_init__(self):
        """Initialize the tokenizers."""
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use PI0FASTTokenizerProcessorStep."
            )

        if self.paligemma_tokenizer is None or self.fast_tokenizer is None or self.paligemma_processor is None:
            raise ValueError(
                "paligemma_tokenizer, fast_tokenizer, and paligemma_processor must all be provided. "
                "These should be initialized tokenizer/processor objects."
            )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1] range per batch element."""
        mins = actions.amin(dim=(1, 2), keepdim=True)
        maxs = actions.amax(dim=(1, 2), keepdim=True)
        return 2 * (actions - mins) / (maxs - mins + 1e-8) - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert FAST tokens to PaliGemma vocabulary space."""
        vocab_size = getattr(self.paligemma_tokenizer, "vocab_size", 257152)
        return vocab_size - 1 - self.fast_skip_tokens - tokens

    def fast_tokenizer_wrapper(self, actions_norm):
        """Wrapper for FAST tokenizer that ensures batch processing and returns PyTorch tensors."""
        batch_tokens = self.fast_tokenizer(actions_norm)
        fast_out = self.paligemma_processor.tokenizer.pad({"input_ids": batch_tokens}, return_tensors="pt")
        return fast_out

    def create_token_type_ids(self, padded_mask: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
        """Create token type IDs to distinguish prefix from action tokens."""
        token_type_ids = torch.zeros_like(padded_mask, dtype=torch.bool)
        cumsum_mask = (padded_mask != 0).cumsum(dim=1)
        suffix_mask = cumsum_mask > prefix_len
        return suffix_mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Process the transition and add tokenized inputs.

        Args:
            transition: The environment transition to process

        Returns:
            The transition with added tokenized data
        """
        self.transition = transition
        
        # Extract components from transition
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        
        if observation is None:
            raise ValueError("Observation is None in transition")
        
        # Get state and language
        state = observation.get(self.state_key)
        if state is None:
            raise ValueError(f"State key '{self.state_key}' not found in observation")
        
        # Get task description
        if complementary_data is None:
            raise ValueError("Complementary data is None, cannot extract task")
        
        task_data = complementary_data.get(self.task_key)
        if task_data is None:
            raise ValueError(f"Task key '{self.task_key}' not found in complementary data")
        
        # Standardize task to list of strings
        if isinstance(task_data, str):
            lang_text = [task_data]
        elif isinstance(task_data, list) and all(isinstance(t, str) for t in task_data):
            lang_text = task_data
        else:
            raise ValueError(f"Task must be string or list of strings, got {type(task_data)}")
        
        # Create tokenized inputs
        tokenized_data = self.create_input_tokens(state, lang_text, action)
        
        # Add tokenized data to observation
        new_observation = dict(observation)
        new_observation["pi0fast_input_ids"] = tokenized_data["input_ids"]
        new_observation["pi0fast_attention_mask"] = tokenized_data["attention_mask"]
        new_observation["pi0fast_padded_mask"] = tokenized_data["padded_mask"]
        new_observation["pi0fast_loss_mask"] = tokenized_data["loss_mask"]
        new_observation["pi0fast_token_type_ids"] = tokenized_data["token_type_ids"]
        
        # Create new transition with updated observation
        new_transition = dict(transition)
        new_transition[TransitionKey.OBSERVATION] = new_observation
        
        return new_transition

    def create_input_tokens(self, state, lang_text, actions=None):
        """
        Create tokenized input from state, language, and actions.

        This method follows the same logic as the original PI0FAST create_input_tokens method.

        Args:
            state: State tensor [batch_size, state_dim]
            lang_text: List of task description strings
            actions: Optional action tensor [batch_size, horizon, action_dim]

        Returns:
            Dictionary containing input_ids, attention_mask, padded_mask, loss_mask, and token_type_ids
        """
        bsize = state.shape[0]
        device = state.device
        
        # Discretize state
        bins = torch.linspace(-1, 1, 256 + 1, device=device)[:-1]
        discretized = torch.bucketize(state, bins) - 1
        discretized = discretized[:, :32]

        # Create prefix texts with task and state
        prefix_texts = []
        for txt, disc in zip(lang_text, discretized, strict=False):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(str(val.item()) for val in disc)
            prefix_texts.append(f"Task: {cleaned}, State: {state_str};\n")

        # Tokenize prefix
        prefix_out = self.paligemma_tokenizer(
            prefix_texts, add_special_tokens=True, return_tensors="pt", padding="longest", truncation=False
        )
        prefix_ids = prefix_out["input_ids"].to(device)
        prefix_mask = prefix_out["attention_mask"].to(device)
        prefix_lens = prefix_mask.sum(dim=1)[:, None].cpu()

        # Get pad token ID
        pad_token_id = (
            self.paligemma_tokenizer.pad_token_id
            if hasattr(self.paligemma_tokenizer, "pad_token_id")
            else self.paligemma_tokenizer.eos_token_id
        )

        if actions is not None:
            # pad actions
            actions_pad = F.pad(
                actions, (0, max(0, self.max_action_dim - actions.shape[2])), value=0
            )[:, :, : self.max_action_dim]
            
            # Tokenize actions with FAST tokenizer
            fast_out = self.fast_tokenizer_wrapper(actions_pad.cpu())
            act_ids = fast_out["input_ids"]
            act_mask = fast_out["attention_mask"].to(device)

            # Convert FAST tokens to PaliGemma token space
            act_ids = self._act_tokens_to_paligemma_tokens(act_ids).to(device)
            
            # Replace padding tokens
            vocab_size = getattr(self.paligemma_tokenizer, "vocab_size", 257152)
            act_ids = torch.where(
                act_ids == vocab_size - 1 - self.fast_skip_tokens,
                pad_token_id,
                act_ids,
            )

            # Add BOS ("Action: ") and EOS tokens
            eos_token = torch.tensor(
                [self.paligemma_tokenizer.eos_token_id], dtype=torch.long, device=device
            ).expand(bsize, -1)
            eos_mask = torch.tensor([1], dtype=torch.long, device=device).expand(bsize, -1)
            
            bos = self.paligemma_tokenizer("Action: ", add_special_tokens=False, return_tensors="pt")
            bos_token = bos["input_ids"].expand(act_ids.shape[0], -1).to(device)
            bos_mask = bos["attention_mask"].expand(act_ids.shape[0], -1).to(device)
            
            act_ids = torch.cat([bos_token, act_ids, eos_token], dim=1)
            act_mask = torch.cat([bos_mask, act_mask, eos_mask], dim=1)
            act_mask = act_mask.to(device)
        else:
            # No actions provided
            act_ids = torch.empty(bsize, 0, dtype=torch.long, device=device)
            act_mask = torch.empty(bsize, 0, dtype=torch.long, device=device)

        # Concatenate prefix and action tokens
        final_ids = torch.cat([prefix_ids, act_ids], dim=1)
        final_mask = torch.cat([prefix_mask, act_mask], dim=1)
        
        batch_inputs = {"input_ids": final_ids.tolist(), "attention_mask": final_mask.tolist()}

        # Pad to max length
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding="longest", max_length=self.max_input_seq_len, return_tensors="pt"
        )
        padded_mask = padded_output["attention_mask"]

        # Create attention mask (excludes prefix)
        att_mask = (padded_mask != 0).cumsum(dim=1) > prefix_lens

        # Create token type IDs
        token_type_ids = self.create_token_type_ids(padded_mask=padded_mask, prefix_len=prefix_lens)

        # Return all masks
        return {
            "input_ids": padded_output["input_ids"],
            "attention_mask": att_mask,
            "padded_mask": padded_mask,
            "loss_mask": att_mask & padded_mask,  # loss is computed not on prefix, and not on padding
            "token_type_ids": token_type_ids,
        }

    def get_config(self) -> dict[str, Any]:
        """Returns the serializable configuration of the processor."""
        return {
            "max_action_dim": self.max_action_dim,
            "fast_skip_tokens": self.fast_skip_tokens,
            "max_input_seq_len": self.max_input_seq_len,
            "padding_side": self.padding_side,
            "task_key": self.task_key,
            "state_key": self.state_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for the tokenized PI0FAST inputs.

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        # Add features for tokenized inputs
        if "pi0fast_input_ids" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["pi0fast_input_ids"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_input_seq_len,)
            )

        if "pi0fast_attention_mask" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["pi0fast_attention_mask"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_input_seq_len,)
            )

        if "pi0fast_padded_mask" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["pi0fast_padded_mask"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_input_seq_len,)
            )

        if "pi0fast_loss_mask" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["pi0fast_loss_mask"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_input_seq_len,)
            )

        if "pi0fast_token_type_ids" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["pi0fast_token_type_ids"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_input_seq_len,)
            )

        return features
