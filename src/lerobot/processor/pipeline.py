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
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

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
    def register(cls, name: str = None):
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


@dataclass
class DataProcessorPipeline(ModelHubMixin, Generic[TInput, TOutput]):
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
        """Internal method to comply with `ModelHubMixin`'s saving mechanism."""
        config_filename = kwargs.pop("config_filename", None)
        self.save_pretrained(save_directory, config_filename=config_filename)

    def save_pretrained(self, save_directory: str | Path, config_filename: str | None = None, **kwargs):
        """Saves the pipeline's configuration and state to a directory.

        This method creates a JSON configuration file that defines the pipeline's structure
        (name and steps). For each stateful step, it also saves a `.safetensors` file
        containing its state dictionary.

        Args:
            save_directory: The directory where the pipeline will be saved.
            config_filename: The name of the JSON configuration file. If None, a name is
                generated from the pipeline's `name` attribute.
            **kwargs: Additional arguments (not used, but present for compatibility).
        """
        os.makedirs(str(save_directory), exist_ok=True)

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        config_filename: str | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
        **kwargs,
    ) -> DataProcessorPipeline[TInput, TOutput]:
        """Loads a pipeline from a local directory or a Hugging Face Hub repository.

        This method reconstructs a `DataProcessorPipeline` by:
        1. Loading the main JSON configuration file.
        2. Iterating through the steps defined in the config.
        3. Dynamically importing or looking up each step's class.
        4. Instantiating each step with its saved configuration, potentially with overrides.
        5. Loading the step's state from its `.safetensors` file, if it exists.

        Args:
            pretrained_model_name_or_path: The identifier of the repository on the Hugging Face Hub
                or a path to a local directory.
            force_download: Whether to force (re)downloading the files.
            resume_download: Whether to resume a previously interrupted download.
            proxies: A dictionary of proxy servers to use.
            token: The token to use as HTTP bearer authorization for private Hub repositories.
            cache_dir: The path to a specific cache folder to store downloaded files.
            local_files_only: If True, avoid downloading files from the Hub.
            revision: The specific model version to use (e.g., a branch name, tag name, or commit id).
            config_filename: The name of the pipeline's JSON configuration file. Required when
                loading from the Hub. If loading from a local directory, it's inferred if there's
                only one `.json` file.
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
        """
        source = str(pretrained_model_name_or_path)

        # Heuristic to distinguish a local path from a Hub repository ID.
        is_local_path = (
            Path(source).is_dir()
            or Path(source).is_absolute()
            or source.startswith("./")
            or source.startswith("../")
            # A simple heuristic: repo IDs usually don't have more than one slash.
            or source.count("/") > 1
            or "\\" in source
        )

        # Load configuration from a local directory.
        if is_local_path:
            base_path = Path(source)

            # If config filename is not provided, try to find a unique .json file.
            if config_filename is None:
                json_files = list(base_path.glob("*.json"))
                if len(json_files) == 0:
                    raise FileNotFoundError(f"No .json configuration files found in {source}")
                elif len(json_files) > 1:
                    raise ValueError(
                        f"Multiple .json files found in {source}: {[f.name for f in json_files]}. "
                        f"Please specify which one to load using the config_filename parameter."
                    )
                config_filename = json_files[0].name

            with open(base_path / config_filename) as file_pointer:
                loaded_config: dict[str, Any] = json.load(file_pointer)
        # Load configuration from the Hugging Face Hub.
        else:
            if config_filename is None:
                raise ValueError(
                    f"For Hugging Face Hub repositories ({source}), you must specify the config_filename parameter. "
                    f"Example: DataProcessorPipeline.from_pretrained('{source}', config_filename='processor.json')"
                )
            # Download the configuration file from the Hub.
            config_path = hf_hub_download(
                source,
                config_filename,
                repo_type="model",
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )

            with open(config_path) as file_pointer:
                loaded_config = json.load(file_pointer)

            # The base path for other files (like state tensors) is the directory of the config file.
            base_path = Path(config_path).parent

        if overrides is None:
            overrides = {}

        override_keys = set(overrides.keys())

        steps: list[ProcessorStep] = []
        for step_entry in loaded_config["steps"]:
            # Determine the step class, prioritizing the registry.
            if "registry_name" in step_entry:
                try:
                    step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                    step_key = step_entry["registry_name"]
                except KeyError as e:
                    raise ImportError(f"Failed to load processor step from registry. {str(e)}") from e
            else:
                # Fallback to dynamic import using the full class path.
                full_class_path = step_entry["class"]
                module_path, class_name = full_class_path.rsplit(".", 1)

                try:
                    module = importlib.import_module(module_path)
                    step_class = getattr(module, class_name)
                    step_key = class_name
                except (ImportError, AttributeError) as e:
                    raise ImportError(
                        f"Failed to load processor step '{full_class_path}'. "
                        f"Make sure the module '{module_path}' is installed and contains class '{class_name}'. "
                        f"Consider registering the step using @ProcessorStepRegistry.register() for better portability. "
                        f"Error: {str(e)}"
                    ) from e

            # Instantiate the step, merging saved config with user-provided overrides.
            try:
                saved_cfg = step_entry.get("config", {})
                step_overrides = overrides.get(step_key, {})
                merged_cfg = {**saved_cfg, **step_overrides}
                step_instance: ProcessorStep = step_class(**merged_cfg)

                if step_key in override_keys:
                    override_keys.discard(step_key)

            except Exception as e:
                step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
                raise ValueError(
                    f"Failed to instantiate processor step '{step_name}' with config: {step_entry.get('config', {})}. "
                    f"Error: {str(e)}"
                ) from e

            # Load the step's state if a state file is specified.
            if "state_file" in step_entry and hasattr(step_instance, "load_state_dict"):
                if is_local_path:
                    state_path = str(base_path / step_entry["state_file"])
                else:
                    # Download the state file from the Hub.
                    state_path = hf_hub_download(
                        source,
                        step_entry["state_file"],
                        repo_type="model",
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                        revision=revision,
                    )

                step_instance.load_state_dict(load_file(state_path))

            steps.append(step_instance)

        # Check for any unused override keys, which likely indicates a typo by the user.
        if override_keys:
            available_keys = [
                step.get("registry_name") or step["class"].rsplit(".", 1)[1]
                for step in loaded_config["steps"]
            ]

            raise KeyError(
                f"Override keys {list(override_keys)} do not match any step in the saved configuration. "
                f"Available step keys: {available_keys}. "
                f"Make sure override keys match exact step class names or registry names."
            )

        # Construct and return the final pipeline instance.
        return cls(
            steps=steps,
            name=loaded_config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or batch_to_transition,
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )

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
    def observation(self, observation) -> dict[str, Any]:
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
        if observation is None:
            raise ValueError("ObservationProcessorStep requires an observation in the transition.")

        processed_observation = self.observation(observation)
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
        if not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")

        processed_action = self.action(action=action)
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
        if info is None:
            raise ValueError("InfoProcessorStep requires an info dictionary in the transition.")

        processed_info = self.info(info)
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
        if complementary_data is None:
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = self.complementary_data(complementary_data)
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
