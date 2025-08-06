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
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypedDict

import torch
from huggingface_hub import ModelHubMixin, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file, save_file

from lerobot.configs.types import PolicyFeature


class TransitionKey(str, Enum):
    """Keys for accessing EnvTransition dictionary components."""

    # TODO(Steven): Use consts
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: dict[str, Any] | None,
        TransitionKey.ACTION.value: Any | torch.Tensor | None,
        TransitionKey.REWARD.value: float | torch.Tensor | None,
        TransitionKey.DONE.value: bool | torch.Tensor | None,
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)


class ProcessorStepRegistry:
    """Registry for processor steps that enables saving/loading by name instead of module path."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str = None):
        """Decorator to register a processor step class.

        Args:
            name: Optional registration name. If not provided, uses class name.

        Example:
            @ProcessorStepRegistry.register("adaptive_normalizer")
            class AdaptiveObservationNormalizer:
                ...
        """

        def decorator(step_class: type) -> type:
            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )

            cls._registry[registration_name] = step_class
            # Store the registration name on the class for later reference
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered processor step class by name.

        Args:
            name: The registration name of the step.

        Returns:
            The registered step class.

        Raises:
            KeyError: If the step is not registered.
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
        """Remove a step from the registry."""
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:
        """List all registered step names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations."""
        cls._registry.clear()


class ProcessorStep(Protocol):
    """Structural typing interface for a single processor step.

    A step is any callable accepting a full `EnvTransition` dict and
    returning a (possibly modified) dict of the same structure. Implementers
    are encouraged—but not required—to expose the optional helper methods
    listed below. When present, these hooks let `RobotProcessor`
    automatically serialise the step's configuration and learnable state using
    a safe-to-share JSON + SafeTensors format.


    **Required**:
        - ``__call__(transition: EnvTransition) -> EnvTransition``
        - ``feature_contract(features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]``

    Optional helper protocol:
    * ``get_config() -> dict[str, Any]`` – User-defined JSON-serializable
      configuration and state. YOU decide what to save here. This is where all
      non-tensor state goes (e.g., name, counter, threshold, window_size).
      The config dict will be passed to your class constructor when loading.
    * ``state_dict() -> dict[str, torch.Tensor]`` – PyTorch tensor state ONLY.
      This is exclusively for torch.Tensor objects (e.g., learned weights,
      running statistics as tensors). Never put simple Python types here.
    * ``load_state_dict(state)`` – Inverse of ``state_dict``. Receives a dict
      containing torch tensors only.
    * ``reset()`` – Clear internal buffers at episode boundaries.

    Example separation:
    - get_config(): {"name": "my_step", "learning_rate": 0.01, "window_size": 10}
    - state_dict(): {"weights": torch.tensor(...), "running_mean": torch.tensor(...)}
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition: ...

    def get_config(self) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None: ...

    def reset(self) -> None: ...

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]: ...


def _default_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:  # noqa: D401
    """Convert a *batch* dict coming from Learobot replay/dataset code into an
    ``EnvTransition`` dictionary.

    The function maps well known keys to the EnvTransition structure. Missing keys are
    filled with sane defaults (``None`` or ``0.0``/``False``).

    Keys recognised (case-sensitive):

    * "observation.*" (keys starting with "observation." are grouped into observation dict)
    * "action"
    * "next.reward"
    * "next.done"
    * "next.truncated"
    * "info"

    Additional keys are ignored so that existing dataloaders can carry extra
    metadata without breaking the processor.
    """

    # Extract observation keys
    observation_keys = {k: v for k, v in batch.items() if k.startswith("observation.")}
    observation = observation_keys if observation_keys else None

    # Extract padding and task keys for complementary data
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    complementary_data = {**pad_keys, **task_key} if pad_keys or task_key else {}

    transition: EnvTransition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: batch.get("action"),
        TransitionKey.REWARD: batch.get("next.reward", 0.0),
        TransitionKey.DONE: batch.get("next.done", False),
        TransitionKey.TRUNCATED: batch.get("next.truncated", False),
        TransitionKey.INFO: batch.get("info", {}),
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    return transition


def _default_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:  # noqa: D401
    """Inverse of :pyfunc:`_default_batch_to_transition`. Returns a dict with
    the canonical field names used throughout *LeRobot*.
    """

    batch = {
        "action": transition.get(TransitionKey.ACTION),
        "next.reward": transition.get(TransitionKey.REWARD, 0.0),
        "next.done": transition.get(TransitionKey.DONE, False),
        "next.truncated": transition.get(TransitionKey.TRUNCATED, False),
        "info": transition.get(TransitionKey.INFO, {}),
    }

    # Add padding and task data from complementary_data
    complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
    if complementary_data:
        pad_data = {k: v for k, v in complementary_data.items() if "_is_pad" in k}
        batch.update(pad_data)

        if "task" in complementary_data:
            batch["task"] = complementary_data["task"]

    # Handle observation - flatten dict to observation.* keys if it's a dict
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


@dataclass
class RobotProcessor(ModelHubMixin):
    """
    Composable, debuggable post-processing processor for robot transitions.

    The class orchestrates an ordered collection of small, functional transforms—steps—executed
    left-to-right on each incoming `EnvTransition`. It can process both `EnvTransition` dicts
    and batch dictionaries, automatically converting between formats as needed.

    Args:
        steps: Ordered list of processing steps executed on every call. Defaults to empty list.
        name: Human-readable identifier that is persisted inside the JSON config.
            Defaults to "RobotProcessor".
        to_transition: Function to convert batch dict to EnvTransition dict.
            Defaults to _default_batch_to_transition.
        to_output: Function to convert EnvTransition dict to the desired output format.
            Usually it is a batch dict or EnvTransition dict.
            Defaults to _default_transition_to_batch.
        before_step_hooks: List of hooks called before each step. Each hook receives the step
            index and transition, and can optionally return a modified transition.
        after_step_hooks: List of hooks called after each step. Each hook receives the step
            index and transition, and can optionally return a modified transition.

    Hook Semantics:
        - Hooks are executed sequentially in the order they were registered. There is no way to
          reorder hooks after registration without creating a new pipeline.
        - Hooks are for observation/monitoring only and DO NOT modify transitions. They are called
          with the step index and current transition for logging, debugging, or monitoring purposes.
        - All hooks for a given type (before/after) are executed for every step, or none at all if
          an error occurs. There is no partial execution of hooks.
        - Hooks should generally be stateless to maintain predictable behavior. If you need stateful
          processing, consider implementing a proper ProcessorStep instead.
        - To remove hooks, use the unregister methods. To remove steps, you must create a new pipeline.
        - Hooks ALWAYS receive transitions in EnvTransition format, regardless of the input format
          passed to __call__. This ensures consistent hook behavior whether processing batch dicts
          or EnvTransition objects.
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "RobotProcessor"

    to_transition: Callable[[dict[str, Any]], EnvTransition] = field(
        default_factory=lambda: _default_batch_to_transition, repr=False
    )
    to_output: Callable[[EnvTransition], dict[str, Any] | EnvTransition] = field(
        default_factory=lambda: _default_transition_to_batch, repr=False
    )

    # Processor-level hooks for observation/monitoring
    # Hooks do not modify transitions - they are called for logging, debugging, or monitoring purposes
    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: EnvTransition | dict[str, Any]):
        """Process data through all steps.

        The method accepts either the classic EnvTransition dict or a batch dictionary
        (like the ones returned by ReplayBuffer or LeRobotDataset). If a dict is supplied
        it is first converted to the internal dict format using to_transition; after all
        steps are executed the dict is transformed back into a batch dict with to_batch and the
        result is returned – thereby preserving the caller's original data type.

        Args:
            data: Either an EnvTransition dict or a batch dictionary to process.

        Returns:
            The processed data in the same format as the input (EnvTransition or batch dict).

        Raises:
            ValueError: If the transition is not a valid EnvTransition format.
        """
        # Check if we need to convert back to batch format at the end
        _, called_with_batch = self._prepare_transition(data)

        # Use step_through to get the iterator
        step_iterator = self.step_through(data)

        # Get initial state (before any steps)
        current_transition = next(step_iterator)

        # Process each step with hooks
        for idx, next_transition in enumerate(step_iterator):
            # Apply before hooks with current state (before step execution)
            for hook in self.before_step_hooks:
                hook(idx, current_transition)

            # Move to next state (after step execution)
            current_transition = next_transition

            # Apply after hooks with updated state
            for hook in self.after_step_hooks:
                hook(idx, current_transition)

        # Convert back to original format if needed
        return self.to_output(current_transition) if called_with_batch else current_transition

    def _prepare_transition(self, data: EnvTransition | dict[str, Any]) -> tuple[EnvTransition, bool]:
        """Prepare and validate transition data for processing.

        Args:
            data: Either an EnvTransition dict or a batch dictionary to process.

        Returns:
            A tuple of (prepared_transition, called_with_batch_flag)

        Raises:
            ValueError: If the transition is not a valid EnvTransition format.
        """
        # Check if data is already an EnvTransition or needs conversion
        if isinstance(data, dict) and not all(isinstance(k, TransitionKey) for k in data.keys()):
            # It's a batch dict, convert it
            called_with_batch = True
            transition = self.to_transition(data)
        else:
            # It's already an EnvTransition
            called_with_batch = False
            transition = data

        # Basic validation
        if not isinstance(transition, dict):
            raise ValueError(f"EnvTransition must be a dictionary. Got {type(transition).__name__}")

        return transition, called_with_batch

    def step_through(self, data: EnvTransition | dict[str, Any]) -> Iterable[EnvTransition]:
        """Yield the intermediate results after each processor step.

        This is a low-level method that does NOT apply hooks. It simply executes each step
        and yields the intermediate results. This allows users to debug the pipeline or
        apply custom logic between steps if needed.

        Note: This method always yields EnvTransition objects regardless of input format.
        If you need the results in the original input format, you'll need to convert them
        using `to_output()`.

        Args:
            data: Either an EnvTransition dict or a batch dictionary to process.

        Yields:
            The intermediate EnvTransition results after each step.
        """
        transition, _ = self._prepare_transition(data)

        # Yield initial state
        yield transition

        # Process each step WITHOUT hooks (low-level method)
        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """Internal save method for ModelHubMixin compatibility."""
        # Extract config_filename from kwargs if provided
        config_filename = kwargs.pop("config_filename", None)
        self.save_pretrained(save_directory, config_filename=config_filename)

    def save_pretrained(self, save_directory: str | Path, config_filename: str | None = None, **kwargs):
        """Serialize the processor definition and parameters to *save_directory*.

        Args:
            save_directory: Directory where the processor will be saved.
            config_filename: Optional custom config filename. If not provided, defaults to
                "{self.name}.json" where self.name is sanitized for filesystem compatibility.
        """
        os.makedirs(str(save_directory), exist_ok=True)

        # Sanitize processor name for use in filenames
        import re

        # The huggingface hub does not allow special characters in the repo name, so we sanitize the name
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

        # Use sanitized name for config if not provided
        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        config: dict[str, Any] = {
            "name": self.name,
            "steps": [],
        }

        for step_index, processor_step in enumerate(self.steps):
            # Check if step was registered
            registry_name = getattr(processor_step.__class__, "_registry_name", None)

            step_entry: dict[str, Any] = {}
            if registry_name:
                # Use registry name for registered steps
                step_entry["registry_name"] = registry_name
            else:
                # Fall back to full module path for unregistered steps
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )

            if hasattr(processor_step, "get_config"):
                step_entry["config"] = processor_step.get_config()

            if hasattr(processor_step, "state_dict"):
                state = processor_step.state_dict()
                if state:
                    # Clone tensors to avoid shared memory issues
                    # This ensures each tensor has its own memory allocation
                    # The reason is to avoid the following error:
                    # RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk
                    # and potential differences when loading them again
                    # ------------------------------------------------------------------------------
                    # Since the state_dict of processor will be light, we can just clone the tensors
                    # and save them to the disk.
                    cloned_state = {}
                    for key, tensor in state.items():
                        cloned_state[key] = tensor.clone()

                    # Include pipeline name and step index to ensure unique filenames
                    # This prevents conflicts when multiple processors are saved in the same directory
                    if registry_name:
                        state_filename = f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"
                    else:
                        state_filename = f"{sanitized_name}_step_{step_index}.safetensors"

                    save_file(cloned_state, os.path.join(str(save_directory), state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

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
        **kwargs,
    ) -> RobotProcessor:
        """Load a serialized processor from source (local path or Hugging Face Hub identifier).

        Args:
            pretrained_model_name_or_path: Local path to a saved processor directory or Hugging Face Hub identifier
                (e.g., "username/processor-name").
            config_filename: Optional specific config filename to load. If not provided, will:
                - For local paths: look for any .json file in the directory (error if multiple found)
                - For HF Hub: try common names ("processor.json", "preprocessor.json", "postprocessor.json")
            overrides: Optional dictionary mapping step names to configuration overrides.
                Keys must match exact step class names (for unregistered steps) or registry names
                (for registered steps). Values are dictionaries containing parameter overrides
                that will be merged with the saved configuration. This is useful for providing
                non-serializable objects like environment instances.

        Returns:
            A RobotProcessor instance loaded from the saved configuration.

        Raises:
            ImportError: If a processor step class cannot be loaded or imported.
            ValueError: If a step cannot be instantiated with the provided configuration.
            KeyError: If an override key doesn't match any step in the saved configuration.

        Examples:
            Basic loading:
            ```python
            processor = RobotProcessor.from_pretrained("path/to/processor")
            ```

            Loading specific config file:
            ```python
            processor = RobotProcessor.from_pretrained(
                "username/multi-processor-repo", config_filename="preprocessor.json"
            )
            ```

            Loading with overrides for non-serializable objects:
            ```python
            import gym

            env = gym.make("CartPole-v1")
            processor = RobotProcessor.from_pretrained(
                "username/cartpole-processor", overrides={"ActionRepeatStep": {"env": env}}
            )
            ```

            Multiple overrides:
            ```python
            processor = RobotProcessor.from_pretrained(
                "path/to/processor",
                overrides={
                    "CustomStep": {"param1": "new_value"},
                    "device_processor": {"device": "cuda:1"},  # For registered steps
                },
            )
            ```
        """
        # Use the local variable name 'source' for clarity
        source = str(pretrained_model_name_or_path)

        if Path(source).is_dir():
            # Local path - use it directly
            base_path = Path(source)

            if config_filename is None:
                # Look for any .json file in the directory
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
        else:
            # Hugging Face Hub - download all required files
            if config_filename is None:
                # Try common config names
                common_names = [
                    "processor.json",
                    "preprocessor.json",
                    "postprocessor.json",
                    "robotprocessor.json",
                ]
                config_path = None
                for name in common_names:
                    try:
                        config_path = hf_hub_download(
                            source,
                            name,
                            repo_type="model",
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            token=token,
                            cache_dir=cache_dir,
                            local_files_only=local_files_only,
                            revision=revision,
                        )
                        config_filename = name
                        break
                    except (FileNotFoundError, OSError, HfHubHTTPError):
                        # FileNotFoundError: local file issues
                        # OSError: network/system errors
                        # HfHubHTTPError: file not found on Hub (404) or other HTTP errors
                        continue

                if config_path is None:
                    raise FileNotFoundError(
                        f"No processor configuration file found in {source}. "
                        f"Tried: {common_names}. Please specify the config_filename parameter."
                    )
            else:
                # Download specific config file
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

            # Store downloaded files in the same directory as the config
            base_path = Path(config_path).parent

        # Handle None overrides
        if overrides is None:
            overrides = {}

        # Validate that all override keys will be matched
        override_keys = set(overrides.keys())

        steps: list[ProcessorStep] = []
        for step_entry in loaded_config["steps"]:
            # Check if step uses registry name or module path
            if "registry_name" in step_entry:
                # Load from registry
                try:
                    step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                    step_key = step_entry["registry_name"]
                except KeyError as e:
                    raise ImportError(f"Failed to load processor step from registry. {str(e)}") from e
            else:
                # Fall back to module path loading for backward compatibility
                full_class_path = step_entry["class"]
                module_path, class_name = full_class_path.rsplit(".", 1)

                # Import the module containing the step class
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

            # Instantiate the step with its config
            try:
                saved_cfg = step_entry.get("config", {})
                step_overrides = overrides.get(step_key, {})
                merged_cfg = {**saved_cfg, **step_overrides}
                step_instance: ProcessorStep = step_class(**merged_cfg)

                # Track which override keys were used
                if step_key in override_keys:
                    override_keys.discard(step_key)

            except Exception as e:
                step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
                raise ValueError(
                    f"Failed to instantiate processor step '{step_name}' with config: {step_entry.get('config', {})}. "
                    f"Error: {str(e)}"
                ) from e

            # Load state if available
            if "state_file" in step_entry and hasattr(step_instance, "load_state_dict"):
                if Path(source).is_dir():
                    # Local path - read directly
                    state_path = str(base_path / step_entry["state_file"])
                else:
                    # Hugging Face Hub - download the state file
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

        # Check for unused override keys
        if override_keys:
            available_keys = []
            for step_entry in loaded_config["steps"]:
                if "registry_name" in step_entry:
                    available_keys.append(step_entry["registry_name"])
                else:
                    full_class_path = step_entry["class"]
                    class_name = full_class_path.rsplit(".", 1)[1]
                    available_keys.append(class_name)

            raise KeyError(
                f"Override keys {list(override_keys)} do not match any step in the saved configuration. "
                f"Available step keys: {available_keys}. "
                f"Make sure override keys match exact step class names or registry names."
            )

        return cls(steps, loaded_config.get("name", "RobotProcessor"))

    def __len__(self) -> int:
        """Return the number of steps in the processor."""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | RobotProcessor:
        """Indexing helper exposing underlying steps.
        * ``int`` – returns the idx-th ProcessorStep.
        * ``slice`` – returns a new RobotProcessor with the sliced steps.
        """
        if isinstance(idx, slice):
            return RobotProcessor(self.steps[idx], self.name)
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Attach fn to be executed before every processor step."""
        self.before_step_hooks.append(fn)

    def unregister_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Remove a previously registered before_step hook.

        Args:
            fn: The exact function reference that was registered. Must be the same object.

        Raises:
            ValueError: If the hook is not found in the registered hooks.
        """
        try:
            self.before_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in before_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Attach fn to be executed after every processor step."""
        self.after_step_hooks.append(fn)

    def unregister_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """Remove a previously registered after_step hook.

        Args:
            fn: The exact function reference that was registered. Must be the same object.

        Raises:
            ValueError: If the hook is not found in the registered hooks.
        """
        try:
            self.after_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in after_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def reset(self):
        """Clear state in every step that implements ``reset()`` and fire registered hooks."""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        """Return a readable string representation of the processor."""
        step_names = [step.__class__.__name__ for step in self.steps]

        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:
            # Show first 2 and last 1 with ellipsis for long lists
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"

        parts = [f"name='{self.name}'", steps_repr]

        return f"RobotProcessor({', '.join(parts)})"

    def __post_init__(self):
        for i, step in enumerate(self.steps):
            if not callable(step):
                raise TypeError(
                    f"Step {i} ({type(step).__name__}) must define __call__(transition) -> EnvTransition"
                )

            fc = getattr(step, "feature_contract", None)
            if not callable(fc):
                raise TypeError(
                    f"Step {i} ({type(step).__name__}) must define feature_contract(features) -> dict[str, Any]"
                )

    def feature_contract(self, initial_features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """
        Apply ALL steps in order. Each step must implement
        feature_contract(features) and return a dict (full or incremental schema).
        """
        features: dict[str, PolicyFeature] = deepcopy(initial_features)

        for _, step in enumerate(self.steps):
            out = step.feature_contract(features)
            if not isinstance(out, dict):
                raise TypeError(f"{step.__class__.__name__}.feature_contract must return dict[str, Any]")
            features = out
        return features


class ObservationProcessor:
    """Base class for processors that modify only the observation component of a transition.

    Subclasses should override the `observation` method to implement custom observation processing.
    This class handles the boilerplate of extracting and reinserting the processed observation
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class MyObservationScaler(ObservationProcessor):
            def __init__(self, scale_factor):
                self.scale_factor = scale_factor

            def observation(self, observation):
                return observation * self.scale_factor
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific observation processing logic.
    """

    def observation(self, observation):
        """Process the observation component.

        Args:
            observation: The observation to process

        Returns:
            The processed observation
        """
        return observation

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        processed_observation = self.observation(observation)
        # Create a new transition dict with the processed observation
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class ActionProcessor:
    """Base class for processors that modify only the action component of a transition.

    Subclasses should override the `action` method to implement custom action processing.
    This class handles the boilerplate of extracting and reinserting the processed action
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class ActionClipping(ActionProcessor):
            def __init__(self, min_val, max_val):
                self.min_val = min_val
                self.max_val = max_val

            def action(self, action):
                return np.clip(action, self.min_val, self.max_val)
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific action processing logic.
    """

    def action(self, action):
        """Process the action component.

        Args:
            action: The action to process

        Returns:
            The processed action
        """
        return action

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        processed_action = self.action(action)
        # Create a new transition dict with the processed action
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class RewardProcessor:
    """Base class for processors that modify only the reward component of a transition.

    Subclasses should override the `reward` method to implement custom reward processing.
    This class handles the boilerplate of extracting and reinserting the processed reward
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class RewardScaler(RewardProcessor):
            def __init__(self, scale_factor):
                self.scale_factor = scale_factor

            def reward(self, reward):
                return reward * self.scale_factor
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific reward processing logic.
    """

    def reward(self, reward):
        """Process the reward component.

        Args:
            reward: The reward to process

        Returns:
            The processed reward
        """
        return reward

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        reward = transition.get(TransitionKey.REWARD)
        if reward is None:
            return transition

        processed_reward = self.reward(reward)
        # Create a new transition dict with the processed reward
        new_transition = transition.copy()
        new_transition[TransitionKey.REWARD] = processed_reward
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class DoneProcessor:
    """Base class for processors that modify only the done flag of a transition.

    Subclasses should override the `done` method to implement custom done flag processing.
    This class handles the boilerplate of extracting and reinserting the processed done flag
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class TimeoutDone(DoneProcessor):
            def __init__(self, max_steps):
                self.steps = 0
                self.max_steps = max_steps

            def done(self, done):
                self.steps += 1
                return done or self.steps >= self.max_steps

            def reset(self):
                self.steps = 0
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific done flag processing logic.
    """

    def done(self, done):
        """Process the done flag.

        Args:
            done: The done flag to process

        Returns:
            The processed done flag
        """
        return done

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        done = transition.get(TransitionKey.DONE)
        if done is None:
            return transition

        processed_done = self.done(done)
        # Create a new transition dict with the processed done flag
        new_transition = transition.copy()
        new_transition[TransitionKey.DONE] = processed_done
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class TruncatedProcessor:
    """Base class for processors that modify only the truncated flag of a transition.

    Subclasses should override the `truncated` method to implement custom truncated flag processing.
    This class handles the boilerplate of extracting and reinserting the processed truncated flag
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class EarlyTruncation(TruncatedProcessor):
            def __init__(self, threshold):
                self.threshold = threshold

            def truncated(self, truncated):
                # Additional truncation condition
                return truncated or some_condition > self.threshold
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific truncated flag processing logic.
    """

    def truncated(self, truncated):
        """Process the truncated flag.

        Args:
            truncated: The truncated flag to process

        Returns:
            The processed truncated flag
        """
        return truncated

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        truncated = transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            return transition

        processed_truncated = self.truncated(truncated)
        # Create a new transition dict with the processed truncated flag
        new_transition = transition.copy()
        new_transition[TransitionKey.TRUNCATED] = processed_truncated
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class InfoProcessor:
    """Base class for processors that modify only the info dictionary of a transition.

    Subclasses should override the `info` method to implement custom info processing.
    This class handles the boilerplate of extracting and reinserting the processed info
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class InfoAugmenter(InfoProcessor):
            def __init__(self):
                self.step_count = 0

            def info(self, info):
                info = info.copy()  # Create a copy to avoid modifying the original
                info["steps"] = self.step_count
                self.step_count += 1
                return info

            def reset(self):
                self.step_count = 0
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition dict
    manipulation, focusing only on the specific info dictionary processing logic.
    """

    def info(self, info):
        """Process the info dictionary.

        Args:
            info: The info dictionary to process

        Returns:
            The processed info dictionary
        """
        return info

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO)
        if info is None:
            return transition

        processed_info = self.info(info)
        # Create a new transition dict with the processed info
        new_transition = transition.copy()
        new_transition[TransitionKey.INFO] = processed_info
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class ComplementaryDataProcessor:
    """Base class for processors that modify only the complementary data of a transition.

    Subclasses should override the `complementary_data` method to implement custom complementary data processing.
    This class handles the boilerplate of extracting and reinserting the processed complementary data
    into the transition dict, eliminating the need to implement the `__call__` method in subclasses.
    """

    def complementary_data(self, complementary_data):
        """Process the complementary data.

        Args:
            complementary_data: The complementary data to process

        Returns:
            The processed complementary data
        """
        return complementary_data

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return transition

        processed_complementary_data = self.complementary_data(complementary_data)
        # Create a new transition dict with the processed complementary data
        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


class IdentityProcessor:
    """Identity processor that does nothing."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
