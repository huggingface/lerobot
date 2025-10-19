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
This module defines a generic, sequential data processing pipeline framework, primarily designed for
transforming robotics data (observations, actions, rewards, etc.).

The core components are:
- ProcessorStep: An abstract base class for a single data transformation operation.
- ProcessorStepRegistry: A mechanism to register and retrieve ProcessorStep classes by name.
- DataProcessorPipeline: A class that chains multiple ProcessorStep instances together to form a complete
  data processing workflow. It integrates with the Hugging Face Hub for easy sharing and versioning of
  pipelines, including their configuration and state.
- Specialized abstract ProcessorStep subclasses (e.g., ObservationProcessorStep, ActionProcessorStep)
  to simplify the creation of steps that target specific parts of a data transition.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypedDict, TypeVar, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.hub import HubMixin

from .converters import batch_to_transition, create_transition, transition_to_batch
from .core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey

# Generic type variables for pipeline input and output.
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:
    """A registry for ProcessorStep classes to allow instantiation from a string name.

    This class provides a way to map string identifiers to `ProcessorStep` classes,
    which is useful for deserializing pipelines from configuration files without

    hardcoding class imports.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """A class decorator to register a ProcessorStep.

        Args:
            name: The name to register the class under. If None, the class's `__name__` is used.

        Returns:
            A decorator function that registers the class and returns it.

        Raises:
            ValueError: If a step with the same name is already registered.
        """

        def decorator(step_class: type) -> type:
            """The actual decorator that performs the registration."""
            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )

            cls._registry[registration_name] = step_class
            # Store the registration name on the class for easy lookup during serialization.
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieves a processor step class from the registry by its name.

        Args:
            name: The name of the step to retrieve.

        Returns:
            The processor step class corresponding to the given name.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Processor step '{name}' not found in registry. "
                f"Available steps: {available}. "
                f"Make sure the step is registered using @ProcessorStepRegistry.register()"
            )
        return cls._registry[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """Removes a processor step from the registry.

        Args:
            name: The name of the step to unregister.
        """
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:
        """Returns a list of all registered processor step names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears all processor steps from the registry."""
        cls._registry.clear()


class ProcessorStep(ABC):
    """Abstract base class for a single step in a data processing pipeline.

    Each step must implement the `__call__` method to perform its transformation
    on a data transition and the `transform_features` method to describe how it
    alters the shape or type of data features.

    Subclasses can optionally be stateful by implementing `state_dict` and `load_state_dict`.
    """

    _current_transition: EnvTransition | None = None

    @property
    def transition(self) -> EnvTransition:
        """Provides access to the most recent transition being processed.

        This is useful for steps that need to access other parts of the transition
        data beyond their primary target (e.g., an action processing step that
        needs to look at the observation).

        Raises:
            ValueError: If accessed before the step has been called with a transition.
        """
        if self._current_transition is None:
            raise ValueError("Transition is not set. Make sure to call the step with a transition first.")
        return self._current_transition

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Processes an environment transition.

        This method should contain the core logic of the processing step.

        Args:
            transition: The input data transition to be processed.

        Returns:
            The processed transition.
        """
        return transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A JSON-serializable dictionary of configuration parameters.
        """
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns the state of the step (e.g., learned parameters, running means).

        Returns:
            A dictionary mapping state names to tensors.
        """
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Loads the step's state from a state dictionary.

        Args:
            state: A dictionary of state tensors.
        """
        return None

    def reset(self) -> None:
        """Resets the internal state of the processor step, if any."""
        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Defines how this step modifies the description of pipeline features.

        This method is used to track changes in data shapes, dtypes, or modalities
        as data flows through the pipeline, without needing to process actual data.

        Args:
            features: A dictionary describing the input features for observations, actions, etc.

        Returns:
            A dictionary describing the output features after this step's transformation.
        """
        return features


class ProcessorKwargs(TypedDict, total=False):
    """A TypedDict for optional keyword arguments used in pipeline construction."""

    to_transition: Callable[[dict[str, Any]], EnvTransition] | None
    to_output: Callable[[EnvTransition], Any] | None
    name: str | None
    before_step_hooks: list[Callable[[int, EnvTransition], None]] | None
    after_step_hooks: list[Callable[[int, EnvTransition], None]] | None


class ProcessorMigrationError(Exception):
    """Raised when a model needs migration to the processor format"""

    def __init__(self, model_path: str | Path, migration_command: str, original_error: str):
        self.model_path = model_path
        self.migration_command = migration_command
        self.original_error = original_error
        super().__init__(
            f"Model '{model_path}' requires migration to processor format. "
            f"Run: {migration_command}\n\nOriginal error: {original_error}"
        )


@dataclass
class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):
    """A sequential pipeline for processing data, integrated with the Hugging Face Hub.

    This class chains together multiple `ProcessorStep` instances to form a complete
    data processing workflow. It's generic, allowing for custom input and output types,
    which are handled by the `to_transition` and `to_output` converters.

    Attributes:
        steps: A sequence of `ProcessorStep` objects that make up the pipeline.
        name: A descriptive name for the pipeline.
        to_transition: A function to convert raw input data into the standardized `EnvTransition` format.
        to_output: A function to convert the final `EnvTransition` into the desired output format.
        before_step_hooks: A list of functions to be called before each step is executed.
        after_step_hooks: A list of functions to be called after each step is executed.
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "DataProcessorPipeline"

    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: TInput) -> TOutput:
        """Processes input data through the full pipeline.

        Args:
            data: The input data to process.

        Returns:
            The processed data in the specified output format.
        """
        transition = self.to_transition(data)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

    def _forward(self, transition: EnvTransition) -> EnvTransition:
        """Executes all processing steps and hooks in sequence.

        Args:
            transition: The initial `EnvTransition` object.

        Returns:
            The final `EnvTransition` after all steps have been applied.
        """
        for idx, processor_step in enumerate(self.steps):
            # Execute pre-hooks
            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)

            # Execute post-hooks
            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    def step_through(self, data: TInput) -> Iterable[EnvTransition]:
        """Processes data step-by-step, yielding the transition at each stage.

        This is a generator method useful for debugging and inspecting the intermediate
        state of the data as it passes through the pipeline.

        Args:
            data: The input data.

        Yields:
            The `EnvTransition` object, starting with the initial state and then after
            each processing step.
        """
        transition = self.to_transition(data)

        # Yield the initial state before any processing.
        yield transition

        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """Internal method to comply with `HubMixin`'s saving mechanism.

        This method does the actual saving work and is called by HubMixin.save_pretrained.
        """
        config_filename = kwargs.pop("config_filename", None)

        # Sanitize the pipeline name to create a valid filename prefix.
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        config: dict[str, Any] = {
            "name": self.name,
            "steps": [],
        }

        # Iterate through each step to build its configuration entry.
        for step_index, processor_step in enumerate(self.steps):
            registry_name = getattr(processor_step.__class__, "_registry_name", None)

            step_entry: dict[str, Any] = {}
            # Prefer registry name for portability, otherwise fall back to full class path.
            if registry_name:
                step_entry["registry_name"] = registry_name
            else:
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )

            # Save step configuration if `get_config` is implemented.
            if hasattr(processor_step, "get_config"):
                step_entry["config"] = processor_step.get_config()

            # Save step state if `state_dict` is implemented and returns a non-empty dict.
            if hasattr(processor_step, "state_dict"):
                state = processor_step.state_dict()
                if state:
                    # Clone tensors to avoid modifying the original state.
                    cloned_state = {key: tensor.clone() for key, tensor in state.items()}

                    # Create a unique filename for the state file.
                    if registry_name:
                        state_filename = f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"
                    else:
                        state_filename = f"{sanitized_name}_step_{step_index}.safetensors"

                    save_file(cloned_state, os.path.join(str(save_directory), state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

        # Write the main configuration JSON file.
        with open(os.path.join(str(save_directory), config_filename), "w") as file_pointer:
            json.dump(config, file_pointer, indent=2)

    def save_pretrained(
        self,
        save_directory: str | Path | None = None,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        config_filename: str | None = None,
        **push_to_hub_kwargs,
    ):
        """Saves the pipeline's configuration and state to a directory.

        This method creates a JSON configuration file that defines the pipeline's structure
        (name and steps). For each stateful step, it also saves a `.safetensors` file
        containing its state dictionary.

        Args:
            save_directory: The directory where the pipeline will be saved. If None, saves to
                HF_LEROBOT_HOME/processors/{sanitized_pipeline_name}.
            repo_id: ID of your repository on the Hub. Used only if `push_to_hub=True`.
            push_to_hub: Whether or not to push your object to the Hugging Face Hub after saving it.
            card_kwargs: Additional arguments passed to the card template to customize the card.
            config_filename: The name of the JSON configuration file. If None, a name is
                generated from the pipeline's `name` attribute.
            **push_to_hub_kwargs: Additional key word arguments passed along to the push_to_hub method.
        """
        if save_directory is None:
            # Use default directory in HF_LEROBOT_HOME
            from lerobot.utils.constants import HF_LEROBOT_HOME

            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
            save_directory = HF_LEROBOT_HOME / "processors" / sanitized_name

        # For direct saves (not through hub), handle config_filename
        if not push_to_hub and config_filename is not None:
            # Call _save_pretrained directly with config_filename
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self._save_pretrained(save_directory, config_filename=config_filename)
            return None

        # Pass config_filename through kwargs for _save_pretrained when using hub
        if config_filename is not None:
            push_to_hub_kwargs["config_filename"] = config_filename

        # Call parent's save_pretrained which will call our _save_pretrained
        return super().save_pretrained(
            save_directory=save_directory,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            card_kwargs=card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_filename: str,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
        **kwargs,
    ) -> DataProcessorPipeline[TInput, TOutput]:
        """Loads a pipeline from a local directory, single file, or Hugging Face Hub repository.

        This method implements a simplified loading pipeline with intelligent migration detection:

        **Simplified Loading Strategy**:
        1. **Config Loading** (_load_config):
           - **Directory**: Load specified config_filename from directory
           - **Single file**: Load file directly (config_filename ignored)
           - **Hub repository**: Download specified config_filename from Hub

        2. **Config Validation** (_validate_loaded_config):
           - Format validation: Ensure config is valid processor format
           - Migration detection: Guide users to migrate old LeRobot models
           - Clear errors: Provide actionable error messages

        3. **Step Construction** (_build_steps_with_overrides):
           - Class resolution: Registry lookup or dynamic imports
           - Override merging: User parameters override saved config
           - State loading: Load .safetensors files for stateful steps

        4. **Override Validation** (_validate_overrides_used):
           - Ensure all user overrides were applied (catch typos)
           - Provide helpful error messages with available keys

        **Migration Detection**:
        - **Smart detection**: Analyzes JSON files to detect old LeRobot models
        - **Precise targeting**: Avoids false positives on other HuggingFace models
        - **Clear guidance**: Provides exact migration command to run
        - **Error mode**: Always raises ProcessorMigrationError for clear user action

        **Loading Examples**:
        ```python
        # Directory loading
        pipeline = DataProcessorPipeline.from_pretrained("/models/my_model", config_filename="processor.json")

        # Single file loading
        pipeline = DataProcessorPipeline.from_pretrained(
            "/models/my_model/processor.json", config_filename="processor.json"
        )

        # Hub loading
        pipeline = DataProcessorPipeline.from_pretrained("user/repo", config_filename="processor.json")

        # Multiple configs (preprocessor/postprocessor)
        preprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_preprocessor.json"
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_postprocessor.json"
        )
        ```

        **Override System**:
        - **Key matching**: Use registry names or class names as override keys
        - **Config merging**: User overrides take precedence over saved config
        - **Validation**: Ensure all override keys match actual steps (catch typos)
        - **Example**: overrides={"NormalizeStep": {"device": "cuda"}}

        Args:
            pretrained_model_name_or_path: The identifier of the repository on the Hugging Face Hub,
                a path to a local directory, or a path to a single config file.
            config_filename: The name of the pipeline's JSON configuration file. Always required
                to prevent ambiguity when multiple configs exist (e.g., preprocessor vs postprocessor).
            force_download: Whether to force (re)downloading the files.
            resume_download: Whether to resume a previously interrupted download.
            proxies: A dictionary of proxy servers to use.
            token: The token to use as HTTP bearer authorization for private Hub repositories.
            cache_dir: The path to a specific cache folder to store downloaded files.
            local_files_only: If True, avoid downloading files from the Hub.
            revision: The specific model version to use (e.g., a branch name, tag name, or commit id).
            overrides: A dictionary to override the configuration of specific steps. Keys should
                match the step's class name or registry name.
            to_transition: A custom function to convert input data to `EnvTransition`.
            to_output: A custom function to convert the final `EnvTransition` to the output format.
            **kwargs: Additional arguments (not used).

        Returns:
            An instance of `DataProcessorPipeline` loaded with the specified configuration and state.

        Raises:
            FileNotFoundError: If the config file cannot be found.
            ValueError: If configuration is ambiguous or instantiation fails.
            ImportError: If a step's class cannot be imported.
            KeyError: If an override key doesn't match any step in the pipeline.
            ProcessorMigrationError: If the model requires migration to processor format.
        """
        model_id = str(pretrained_model_name_or_path)
        hub_download_kwargs = {
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        # 1. Load configuration using simplified 3-way logic
        loaded_config, base_path = cls._load_config(model_id, config_filename, hub_download_kwargs)

        # 2. Validate configuration and handle migration
        cls._validate_loaded_config(model_id, loaded_config, config_filename)

        # 3. Build steps with overrides
        steps, validated_overrides = cls._build_steps_with_overrides(
            loaded_config, overrides or {}, model_id, base_path, hub_download_kwargs
        )

        # 4. Validate that all overrides were used
        cls._validate_overrides_used(validated_overrides, loaded_config)

        # 5. Construct and return the final pipeline instance
        return cls(
            steps=steps,
            name=loaded_config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or cast(Callable[[TInput], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )

    @classmethod
    def _load_config(
        cls,
        model_id: str,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        """Load configuration from local file or Hugging Face Hub.

        This method implements a super-simplified 3-way loading strategy:

        1. **Local directory**: Load config_filename from directory
           - Example: model_id="/models/my_model", config_filename="processor.json"
           - Loads: "/models/my_model/processor.json"

        2. **Single file**: Load file directly (ignore config_filename)
           - Example: model_id="/models/my_model/processor.json"
           - Loads: "/models/my_model/processor.json" (config_filename ignored)

        3. **Hub repository**: Download config_filename from Hub
           - Example: model_id="user/repo", config_filename="processor.json"
           - Downloads and loads: config_filename from Hub repo

        **Benefits of Explicit config_filename**:
        - No auto-detection complexity or edge cases
        - No risk of loading wrong config (preprocessor vs postprocessor)
        - Consistent behavior across local and Hub usage
        - Clear, predictable errors

        Args:
            model_id: The model identifier (Hub repo ID, local directory, or file path)
            config_filename: The explicit config filename to load (always required)
            hub_download_kwargs: Parameters for hf_hub_download (tokens, cache, etc.)

        Returns:
            Tuple of (loaded_config, base_path)
            - loaded_config: Parsed JSON config dict (always loaded, never None)
            - base_path: Directory containing config file (for state file resolution)

        Raises:
            FileNotFoundError: If config file cannot be found locally or on Hub
        """
        model_path = Path(model_id)

        if model_path.is_dir():
            # Directory: load specified config from directory
            config_path = model_path / config_filename
            if not config_path.exists():
                # Check for migration before giving clear error
                if cls._should_suggest_migration(model_path):
                    cls._suggest_processor_migration(model_id, f"Config file '{config_filename}' not found")
                raise FileNotFoundError(
                    f"Config file '{config_filename}' not found in directory '{model_id}'"
                )

            with open(config_path) as f:
                return json.load(f), model_path

        elif model_path.is_file():
            # File: load file directly (config_filename is ignored for single files)
            with open(model_path) as f:
                return json.load(f), model_path.parent

        else:
            # Hub: download specified config
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=config_filename,
                    repo_type="model",
                    **hub_download_kwargs,
                )

                with open(config_path) as f:
                    return json.load(f), Path(config_path).parent

            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find '{config_filename}' on the HuggingFace Hub at '{model_id}'"
                ) from e

    @classmethod
    def _validate_loaded_config(
        cls, model_id: str, loaded_config: dict[str, Any], config_filename: str
    ) -> None:
        """Validate that a config was loaded and is a valid processor config.

        This method validates processor config format with intelligent migration detection:

        **Config Format Validation**:
        - Use _is_processor_config() to validate structure
          - Must have "steps" field with list of step configurations
          - Each step needs "class" or "registry_name"
        - If validation fails AND local directory: Check for migration need
        - If migration needed: Raise ProcessorMigrationError with command
        - If no migration: Raise ValueError with helpful error message

        **Migration Detection Logic**:
        - Only triggered for local directories (not Hub repos)
        - Analyzes all JSON files in directory to detect old LeRobot models
        - Provides exact migration command with model path

        Args:
            model_id: The model identifier (used for migration detection)
            loaded_config: The loaded config dictionary (guaranteed non-None)
            config_filename: The config filename that was loaded (for error messages)

        Raises:
            ValueError: If config format is invalid
            ProcessorMigrationError: If model needs migration to processor format
        """
        # Validate that this is actually a processor config
        if not cls._is_processor_config(loaded_config):
            if Path(model_id).is_dir() and cls._should_suggest_migration(Path(model_id)):
                cls._suggest_processor_migration(
                    model_id,
                    f"Config file '{config_filename}' is not a valid processor configuration",
                )
            raise ValueError(
                f"Config file '{config_filename}' is not a valid processor configuration. "
                f"Expected a config with 'steps' field, but got: {list(loaded_config.keys())}"
            )

    @classmethod
    def _build_steps_with_overrides(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[list[ProcessorStep], set[str]]:
        """Build all processor steps with overrides and state loading.

        This method orchestrates the complete step construction pipeline:

        **For each step in loaded_config["steps"]**:

        1. **Class Resolution** (via _resolve_step_class):
           - **If "registry_name" exists**: Look up in ProcessorStepRegistry
             Example: {"registry_name": "normalize_step"} -> Get registered class
           - **Else use "class" field**: Dynamic import from full module path
             Example: {"class": "lerobot.processor.normalize.NormalizeStep"}
           - **Result**: (step_class, step_key) where step_key is used for overrides

        2. **Step Instantiation** (via _instantiate_step):
           - **Merge configs**: saved_config + user_overrides
           - **Override priority**: User overrides take precedence over saved config
           - **Example**: saved={"mean": 0.0}, override={"mean": 1.0} -> final={"mean": 1.0}
           - **Result**: Instantiated ProcessorStep object

        3. **State Loading** (via _load_step_state):
           - **If step has "state_file"**: Load tensor state from .safetensors
           - **Local first**: Check base_path/state_file.safetensors
           - **Hub fallback**: Download state file if not found locally
           - **Optional**: Only load if step has load_state_dict method

        4. **Override Tracking**:
           - **Track used overrides**: Remove step_key from remaining set
           - **Purpose**: Validate all user overrides were applied (detect typos)

        **Error Handling**:
        - Class resolution errors -> ImportError with helpful message
        - Instantiation errors -> ValueError with config details
        - State loading errors -> Propagated from load_state_dict

        Args:
            loaded_config: The loaded processor configuration (must have "steps" field)
            overrides: User-provided parameter overrides (keyed by class/registry name)
            model_id: The model identifier (needed for Hub state file downloads)
            base_path: Local directory path for finding state files
            hub_download_kwargs: Parameters for hf_hub_download (tokens, cache, etc.)

        Returns:
            Tuple of (instantiated_steps_list, unused_override_keys)
            - instantiated_steps_list: List of ready-to-use ProcessorStep instances
            - unused_override_keys: Override keys that didn't match any step (for validation)

        Raises:
            ImportError: If a step class cannot be imported or found in registry
            ValueError: If a step cannot be instantiated with its configuration
        """
        steps: list[ProcessorStep] = []
        override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:
            # 1. Get step class and key
            step_class, step_key = cls._resolve_step_class(step_entry)

            # 2. Instantiate step with overrides
            step_instance = cls._instantiate_step(step_entry, step_class, step_key, overrides)

            # 3. Load step state if available
            cls._load_step_state(step_instance, step_entry, model_id, base_path, hub_download_kwargs)

            # 4. Track used overrides
            if step_key in override_keys:
                override_keys.discard(step_key)

            steps.append(step_instance)

        return steps, override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry: dict[str, Any]) -> tuple[type[ProcessorStep], str]:
        """Resolve step class from registry or import path.

        This method implements a two-tier resolution strategy:

        **Tier 1: Registry-based resolution** (preferred):
        - **If "registry_name" in step_entry**: Look up in ProcessorStepRegistry
          - **Advantage**: Faster, no imports needed, guaranteed compatibility
          - **Example**: {"registry_name": "normalize_step"} -> Get pre-registered class
          - **Error**: KeyError if registry_name not found -> Convert to ImportError

        **Tier 2: Dynamic import fallback**:
        - **Else use "class" field**: Full module.ClassName import path
          - **Process**: Split "module.path.ClassName" into module + class parts
          - **Import**: Use importlib.import_module() + getattr()
          - **Example**: "lerobot.processor.normalize.NormalizeStep"
            a. Import module: "lerobot.processor.normalize"
            b. Get class: getattr(module, "NormalizeStep")
          - **step_key**: Use class_name ("NormalizeStep") for overrides

        **Override Key Strategy**:
        - Registry steps: Use registry_name ("normalize_step")
        - Import steps: Use class_name ("NormalizeStep")
        - This allows users to override with: {"normalize_step": {...}} or {"NormalizeStep": {...}}

        **Error Handling**:
        - Registry KeyError -> ImportError with registry context
        - Import/Attribute errors -> ImportError with helpful suggestions
        - All errors include troubleshooting guidance

        Args:
            step_entry: The step configuration dictionary (must have "registry_name" or "class")

        Returns:
            Tuple of (step_class, step_key)
            - step_class: The resolved ProcessorStep class (ready for instantiation)
            - step_key: The key used for user overrides (registry_name or class_name)

        Raises:
            ImportError: If step class cannot be loaded from registry or import path
        """
        if "registry_name" in step_entry:
            try:
                step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                return step_class, step_entry["registry_name"]
            except KeyError as e:
                raise ImportError(f"Failed to load processor step from registry. {str(e)}") from e
        else:
            # Fallback to dynamic import using the full class path
            full_class_path = step_entry["class"]
            module_path, class_name = full_class_path.rsplit(".", 1)

            try:
                module = importlib.import_module(module_path)
                step_class = getattr(module, class_name)
                return step_class, class_name
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load processor step '{full_class_path}'. "
                    f"Make sure the module '{module_path}' is installed and contains class '{class_name}'. "
                    f"Consider registering the step using @ProcessorStepRegistry.register() for better portability. "
                    f"Error: {str(e)}"
                ) from e

    @classmethod
    def _instantiate_step(
        cls,
        step_entry: dict[str, Any],
        step_class: type[ProcessorStep],
        step_key: str,
        overrides: dict[str, Any],
    ) -> ProcessorStep:
        """Instantiate a single processor step with config overrides.

        This method handles the configuration merging and instantiation logic:

        **Configuration Merging Strategy**:
        1. **Extract saved config**: Get step_entry.get("config", {}) from saved pipeline
           - Example: {"config": {"mean": 0.0, "std": 1.0}}
        2. **Extract user overrides**: Get overrides.get(step_key, {}) for this step
           - Example: overrides = {"NormalizeStep": {"mean": 2.0, "device": "cuda"}}
        3. **Merge with priority**: {**saved_cfg, **step_overrides}
           - **Override priority**: User values override saved values
           - **Result**: {"mean": 2.0, "std": 1.0, "device": "cuda"}

        **Instantiation Process**:
        - **Call constructor**: step_class(**merged_cfg)
        - **Example**: NormalizeStep(mean=2.0, std=1.0, device="cuda")

        **Error Handling**:
        - **Any exception during instantiation**: Convert to ValueError
        - **Include context**: step name, attempted config, original error
        - **Purpose**: Help users debug configuration issues
        - **Common causes**:
          a. Invalid parameter types (str instead of float)
          b. Missing required parameters
          c. Incompatible parameter combinations

        Args:
            step_entry: The step configuration from saved config (contains "config" dict)
            step_class: The step class to instantiate (already resolved)
            step_key: The key used for overrides ("registry_name" or class name)
            overrides: User-provided parameter overrides (keyed by step_key)

        Returns:
            The instantiated processor step (ready for use)

        Raises:
            ValueError: If step cannot be instantiated, with detailed error context
        """
        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            raise ValueError(
                f"Failed to instantiate processor step '{step_name}' with config: {step_entry.get('config', {})}. "
                f"Error: {str(e)}"
            ) from e

    @classmethod
    def _load_step_state(
        cls,
        step_instance: ProcessorStep,
        step_entry: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> None:
        """Load state dictionary for a processor step if available.

        This method implements conditional state loading with local/Hub fallback:

        **Precondition Checks** (early return if not met):
        1. **"state_file" in step_entry**: Step config specifies a state file
           - **If missing**: Step has no saved state (e.g., stateless transforms)
        2. **hasattr(step_instance, "load_state_dict")**: Step supports state loading
           - **If missing**: Step doesn't implement state loading (rare)

        **State File Resolution Strategy**:
        1. **Local file priority**: Check base_path/state_filename exists
           - **Advantage**: Faster, no network calls
           - **Example**: "/models/my_model/normalize_step_0.safetensors"
           - **Use case**: Loading from local saved model directory

        2. **Hub download fallback**: Download state file from repository
           - **When triggered**: Local file not found or base_path is None
           - **Process**: Use hf_hub_download with same parameters as config
           - **Example**: Download "normalize_step_0.safetensors" from "user/repo"
           - **Result**: Downloaded to local cache, path returned

        **State Loading Process**:
        - **Load tensors**: Use safetensors.torch.load_file()
        - **Apply to step**: Call step_instance.load_state_dict(tensor_dict)
        - **In-place modification**: Updates step's internal tensor state

        **Common state file examples**:
        - "normalize_step_0.safetensors" - normalization statistics
        - "custom_step_1.safetensors" - learned parameters
        - "tokenizer_step_2.safetensors" - vocabulary embeddings

        Args:
            step_instance: The step instance to load state into (must have load_state_dict)
            step_entry: The step configuration dictionary (may contain "state_file")
            model_id: The model identifier (used for Hub downloads if needed)
            base_path: Local directory path for finding state files (None for Hub-only)
            hub_download_kwargs: Parameters for hf_hub_download (tokens, cache, etc.)

        Note:
            This method modifies step_instance in-place and returns None.
            If state loading fails, exceptions from load_state_dict propagate.
        """
        if "state_file" not in step_entry or not hasattr(step_instance, "load_state_dict"):
            return

        state_filename = step_entry["state_file"]

        # Try local file first
        if base_path and (base_path / state_filename).exists():
            state_path = str(base_path / state_filename)
        else:
            # Download from Hub
            state_path = hf_hub_download(
                repo_id=model_id,
                filename=state_filename,
                repo_type="model",
                **hub_download_kwargs,
            )

        step_instance.load_state_dict(load_file(state_path))

    @classmethod
    def _validate_overrides_used(
        cls, remaining_override_keys: set[str], loaded_config: dict[str, Any]
    ) -> None:
        """Validate that all provided overrides were used.

        This method ensures user overrides are valid to catch typos and configuration errors:

        **Validation Logic**:
        1. **If remaining_override_keys is empty**: All overrides were used -> Success
           - **Early return**: No validation needed
           - **Normal case**: User provided correct override keys

        2. **If remaining_override_keys has entries**: Some overrides unused -> Error
           - **Root cause**: User provided keys that don't match any step
           - **Common issues**:
             a. Typos in step names ("NormalizStep" vs "NormalizeStep")
             b. Using wrong key type (class name vs registry name)
             c. Step doesn't exist in saved pipeline

        **Helpful Error Generation**:
        - **Extract available keys**: Build list of valid override keys from config
          a. **Registry steps**: Use "registry_name" directly
          b. **Import steps**: Extract class name from "class" field
          - Example: "lerobot.processor.normalize.NormalizeStep" -> "NormalizeStep"
        - **Error message includes**:
          a. Invalid keys provided by user
          b. List of valid keys they can use
          c. Guidance about registry vs class names

        **Override Key Resolution Rules**:
        - Steps with "registry_name": Use registry_name for overrides
        - Steps with "class": Use final class name for overrides
        - Users must match these exact keys in their overrides dict

        Args:
            remaining_override_keys: Override keys that weren't matched to any step
            loaded_config: The loaded processor configuration (contains "steps" list)

        Raises:
            KeyError: If any override keys were not used, with helpful error message
        """
        if not remaining_override_keys:
            return

        available_keys = [
            step.get("registry_name") or step["class"].rsplit(".", 1)[1] for step in loaded_config["steps"]
        ]

        raise KeyError(
            f"Override keys {list(remaining_override_keys)} do not match any step in the saved configuration. "
            f"Available step keys: {available_keys}. "
            f"Make sure override keys match exact step class names or registry names."
        )

    @classmethod
    def _should_suggest_migration(cls, model_path: Path) -> bool:
        """Check if directory has JSON files but no processor configs.

        This method implements smart migration detection to avoid false positives:

        **Decision Logic**:
        1. **No JSON files found**: Return False
           - **Reason**: Empty directory or only non-config files
           - **Example**: Directory with only .safetensors, .md files
           - **Action**: No migration needed

        2. **JSON files exist**: Analyze each file
           - **Goal**: Determine if ANY file is a valid processor config
           - **Process**:
             a. Try to parse each .json file
             b. Skip files with JSON parse errors (malformed)
             c. Check if parsed config passes _is_processor_config()
           - **If ANY valid processor found**: Return False (no migration)
           - **If NO valid processors found**: Return True (migration needed)

        **Examples**:
        - **No migration**: ["processor.json", "config.json"] where processor.json is valid
        - **Migration needed**: ["config.json", "train.json"] where both are model configs
        - **No migration**: [] (empty directory)
        - **Migration needed**: ["old_model_config.json"] with old LeRobot format

        **Why this works**:
        - **Precise detection**: Only suggests migration for actual old LeRobot models
        - **Avoids false positives**: Won't trigger on other HuggingFace model types
        - **Graceful handling**: Ignores malformed JSON files

        Args:
            model_path: Path to local directory to analyze

        Returns:
            True if directory has JSON configs but none are processor configs (migration needed)
            False if no JSON files or at least one valid processor config exists
        """
        json_files = list(model_path.glob("*.json"))
        if len(json_files) == 0:
            return False

        # Check if any JSON file is a processor config
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config = json.load(f)

                if cls._is_processor_config(config):
                    return False  # Found at least one processor config, no migration needed

            except (json.JSONDecodeError, OSError):
                # Skip files that can't be parsed as JSON
                continue

        # Have JSON files but no processor configs - suggest migration
        return True

    @classmethod
    def _is_processor_config(cls, config: dict) -> bool:
        """Check if config follows DataProcessorPipeline format.

        This method validates the processor configuration structure:

        **Required Structure Validation**:
        1. **"steps" field existence**: Must have top-level "steps" key
           - **If missing**: Not a processor config (e.g., model config, train config)
           - **Example invalid**: {"type": "act", "hidden_dim": 256}

        2. **"steps" field type**: Must be a list, not other types
           - **If not list**: Invalid format
           - **Example invalid**: {"steps": "some_string"} or {"steps": {"key": "value"}}

        3. **Empty steps validation**: Empty list is valid
           - **If len(steps) == 0**: Return True immediately
           - **Use case**: Empty processor pipeline (no-op)
           - **Example valid**: {"name": "EmptyProcessor", "steps": []}

        **Individual Step Validation** (for non-empty steps):
        For each step in the steps list:
        1. **Step type**: Must be a dictionary
           - **If not dict**: Invalid step format
           - **Example invalid**: ["string_step", 123, true]

        2. **Step identifier**: Must have either "class" OR "registry_name"
           - **"registry_name"**: Registered step (preferred)
             Example: {"registry_name": "normalize_step", "config": {...}}
           - **"class"**: Full import path
             Example: {"class": "lerobot.processor.normalize.NormalizeStep"}
           - **If neither**: Invalid step (can't resolve class)
           - **If both**: Also valid (registry_name takes precedence)

        **Valid Processor Config Examples**:
        - {"steps": []} - Empty processor
        - {"steps": [{"registry_name": "normalize"}]} - Registry step
        - {"steps": [{"class": "my.module.Step"}]} - Import step
        - {"name": "MyProcessor", "steps": [...]} - With name

        **Invalid Config Examples**:
        - {"type": "act"} - Missing "steps"
        - {"steps": "normalize"} - Steps not a list
        - {"steps": [{}]} - Step missing class/registry_name
        - {"steps": ["string"]} - Step not a dict

        Args:
            config: The configuration dictionary to validate

        Returns:
            True if config follows valid DataProcessorPipeline format, False otherwise
        """
        # Must have a "steps" field with a list of step configurations
        if not isinstance(config.get("steps"), list):
            return False

        steps = config["steps"]
        if len(steps) == 0:
            return True  # Empty processor is valid

        # Each step must be a dict with either "class" or "registry_name"
        for step in steps:
            if not isinstance(step, dict):
                return False
            if not ("class" in step or "registry_name" in step):
                return False

        return True

    @classmethod
    def _suggest_processor_migration(cls, model_path: str | Path, original_error: str) -> None:
        """Raise migration error when we detect JSON files but no processor configs.

        This method is called when migration detection determines that a model
        directory contains configuration files but none are valid processor configs.
        This typically indicates an old LeRobot model that needs migration.

        **When this is called**:
        - User tries to load DataProcessorPipeline from local directory
        - Directory contains JSON configuration files
        - None of the JSON files follow processor config format
        - _should_suggest_migration() returned True

        **Migration Command Generation**:
        - Constructs exact command user needs to run
        - Uses the migration script: migrate_policy_normalization.py
        - Includes the model path automatically
        - Example: "python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path /models/old_model"

        **Error Structure**:
        - **Always raises**: ProcessorMigrationError (never returns)
        - **Includes**: model_path, migration_command, original_error
        - **Purpose**: Force user attention to migration need
        - **User experience**: Clear actionable error with exact command to run

        **Migration Process**:
        The suggested command will:
        1. Extract normalization stats from old model
        2. Create new processor configs (preprocessor + postprocessor)
        3. Remove normalization layers from model
        4. Save migrated model with processor pipeline

        Args:
            model_path: Path to the model directory needing migration
            original_error: The error that triggered migration detection (for context)

        Raises:
            ProcessorMigrationError: Always raised (this method never returns normally)
        """
        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {model_path}"
        )

        raise ProcessorMigrationError(model_path, migration_command, original_error)

    def __len__(self) -> int:
        """Returns the number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | DataProcessorPipeline[TInput, TOutput]:
        """Retrieves a step or a sub-pipeline by index or slice.

        Args:
            idx: An integer index or a slice object.

        Returns:
            A `ProcessorStep` if `idx` is an integer, or a new `DataProcessorPipeline`
            containing the sliced steps.
        """
        if isinstance(idx, slice):
            # Return a new pipeline instance with the sliced steps.
            return DataProcessorPipeline(
                steps=self.steps[idx],
                name=self.name,
                to_transition=self.to_transition,
                to_output=self.to_output,
                before_step_hooks=self.before_step_hooks.copy(),
                after_step_hooks=self.after_step_hooks.copy(),
            )
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Registers a function to be called before each step.

        Args:
            fn: A callable that accepts the step index and the current transition.
        """
        self.before_step_hooks.append(fn)

    def unregister_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Unregisters a 'before_step' hook.

        Args:
            fn: The exact function object that was previously registered.

        Raises:
            ValueError: If the hook is not found in the list.
        """
        try:
            self.before_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in before_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Registers a function to be called after each step.

        Args:
            fn: A callable that accepts the step index and the current transition.
        """
        self.after_step_hooks.append(fn)

    def unregister_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Unregisters an 'after_step' hook.

        Args:
            fn: The exact function object that was previously registered.

        Raises:
            ValueError: If the hook is not found in the list.
        """
        try:
            self.after_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in after_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def reset(self):
        """Resets the state of all stateful steps in the pipeline."""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:
        """Provides a concise string representation of the pipeline."""
        step_names = [step.__class__.__name__ for step in self.steps]

        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:
            # For long pipelines, show the first, second, and last steps.
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"

        parts = [f"name='{self.name}'", steps_repr]

        return f"DataProcessorPipeline({', '.join(parts)})"

    def __post_init__(self):
        """Validates that all provided steps are instances of `ProcessorStep`."""
        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"Step {i} ({type(step).__name__}) must inherit from ProcessorStep")

    def transform_features(
        self, initial_features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Applies feature transformations from all steps sequentially.

        This method propagates a feature description dictionary through each step's
        `transform_features` method, allowing the pipeline to statically determine
        the output feature specification without processing any real data.

        Args:
            initial_features: A dictionary describing the initial features.

        Returns:
            The final feature description after all transformations.
        """
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = deepcopy(initial_features)

        for _, step in enumerate(self.steps):
            out = step.transform_features(features)
            features = out
        return features

    # Convenience methods for processing individual parts of a transition.
    def process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Processes only the observation part of a transition through the pipeline.

        Args:
            observation: The observation dictionary.

        Returns:
            The processed observation dictionary.
        """
        transition: EnvTransition = create_transition(observation=observation)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.OBSERVATION]

    def process_action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """Processes only the action part of a transition through the pipeline.

        Args:
            action: The action data.

        Returns:
            The processed action.
        """
        transition: EnvTransition = create_transition(action=action)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.ACTION]

    def process_reward(self, reward: float | torch.Tensor) -> float | torch.Tensor:
        """Processes only the reward part of a transition through the pipeline.

        Args:
            reward: The reward value.

        Returns:
            The processed reward.
        """
        transition: EnvTransition = create_transition(reward=reward)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.REWARD]

    def process_done(self, done: bool | torch.Tensor) -> bool | torch.Tensor:
        """Processes only the done flag of a transition through the pipeline.

        Args:
            done: The done flag.

        Returns:
            The processed done flag.
        """
        transition: EnvTransition = create_transition(done=done)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.DONE]

    def process_truncated(self, truncated: bool | torch.Tensor) -> bool | torch.Tensor:
        """Processes only the truncated flag of a transition through the pipeline.

        Args:
            truncated: The truncated flag.

        Returns:
            The processed truncated flag.
        """
        transition: EnvTransition = create_transition(truncated=truncated)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.TRUNCATED]

    def process_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Processes only the info dictionary of a transition through the pipeline.

        Args:
            info: The info dictionary.

        Returns:
            The processed info dictionary.
        """
        transition: EnvTransition = create_transition(info=info)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.INFO]

    def process_complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """Processes only the complementary data part of a transition through the pipeline.

        Args:
            complementary_data: The complementary data dictionary.

        Returns:
            The processed complementary data dictionary.
        """
        transition: EnvTransition = create_transition(complementary_data=complementary_data)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.COMPLEMENTARY_DATA]


# Type aliases for semantic clarity.
RobotProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]
PolicyProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]


class ObservationProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the observation in a transition."""

    @abstractmethod
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Processes an observation dictionary. Subclasses must implement this method.

        Args:
            observation: The input observation dictionary from the transition.

        Returns:
            The processed observation dictionary.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `observation` method to the transition's observation."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ObservationProcessorStep requires an observation in the transition.")

        processed_observation = self.observation(observation.copy())
        new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition


class ActionProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the action in a transition."""

    @abstractmethod
    def action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """Processes an action. Subclasses must implement this method.

        Args:
            action: The input action from the transition.

        Returns:
            The processed action.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `action` method to the transition's action."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("ActionProcessorStep requires an action in the transition.")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RobotActionProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` for processing a `RobotAction` (a dictionary)."""

    @abstractmethod
    def action(self, action: RobotAction) -> RobotAction:
        """Processes a `RobotAction`. Subclasses must implement this method.

        Args:
            action: The input `RobotAction` dictionary.

        Returns:
            The processed `RobotAction`.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `action` method to the transition's action, ensuring it's a `RobotAction`."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")

        processed_action = self.action(action.copy())
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class PolicyActionProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` for processing a `PolicyAction` (a tensor or dict of tensors)."""

    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:
        """Processes a `PolicyAction`. Subclasses must implement this method.

        Args:
            action: The input `PolicyAction`.

        Returns:
            The processed `PolicyAction`.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `action` method to the transition's action, ensuring it's a `PolicyAction`."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RewardProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the reward in a transition."""

    @abstractmethod
    def reward(self, reward) -> float | torch.Tensor:
        """Processes a reward. Subclasses must implement this method.

        Args:
            reward: The input reward from the transition.

        Returns:
            The processed reward.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `reward` method to the transition's reward."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        reward = new_transition.get(TransitionKey.REWARD)
        if reward is None:
            raise ValueError("RewardProcessorStep requires a reward in the transition.")

        processed_reward = self.reward(reward)
        new_transition[TransitionKey.REWARD] = processed_reward
        return new_transition


class DoneProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the 'done' flag in a transition."""

    @abstractmethod
    def done(self, done) -> bool | torch.Tensor:
        """Processes a 'done' flag. Subclasses must implement this method.

        Args:
            done: The input 'done' flag from the transition.

        Returns:
            The processed 'done' flag.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `done` method to the transition's 'done' flag."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        done = new_transition.get(TransitionKey.DONE)
        if done is None:
            raise ValueError("DoneProcessorStep requires a done flag in the transition.")

        processed_done = self.done(done)
        new_transition[TransitionKey.DONE] = processed_done
        return new_transition


class TruncatedProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the 'truncated' flag in a transition."""

    @abstractmethod
    def truncated(self, truncated) -> bool | torch.Tensor:
        """Processes a 'truncated' flag. Subclasses must implement this method.

        Args:
            truncated: The input 'truncated' flag from the transition.

        Returns:
            The processed 'truncated' flag.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `truncated` method to the transition's 'truncated' flag."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        truncated = new_transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            raise ValueError("TruncatedProcessorStep requires a truncated flag in the transition.")

        processed_truncated = self.truncated(truncated)
        new_transition[TransitionKey.TRUNCATED] = processed_truncated
        return new_transition


class InfoProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that specifically targets the 'info' dictionary in a transition."""

    @abstractmethod
    def info(self, info) -> dict[str, Any]:
        """Processes an 'info' dictionary. Subclasses must implement this method.

        Args:
            info: The input 'info' dictionary from the transition.

        Returns:
            The processed 'info' dictionary.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `info` method to the transition's 'info' dictionary."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        info = new_transition.get(TransitionKey.INFO)
        if info is None or not isinstance(info, dict):
            raise ValueError("InfoProcessorStep requires an info dictionary in the transition.")

        processed_info = self.info(info.copy())
        new_transition[TransitionKey.INFO] = processed_info
        return new_transition


class ComplementaryDataProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` that targets the 'complementary_data' in a transition."""

    @abstractmethod
    def complementary_data(self, complementary_data) -> dict[str, Any]:
        """Processes a 'complementary_data' dictionary. Subclasses must implement this method.

        Args:
            complementary_data: The input 'complementary_data' from the transition.

        Returns:
            The processed 'complementary_data' dictionary.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `complementary_data` method to the transition's data."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = self.complementary_data(complementary_data.copy())
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition


class IdentityProcessorStep(ProcessorStep):
    """A no-op processor step that returns the input transition and features unchanged.

    This can be useful as a placeholder or for debugging purposes.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Returns the transition without modification."""
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Returns the features without modification."""
        return features
