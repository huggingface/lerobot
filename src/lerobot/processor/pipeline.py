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
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Protocol, Sequence, Tuple

import torch
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file


class TransitionIndex(IntEnum):
    """Explicit indices for EnvTransition tuple components."""

    OBSERVATION = 0
    ACTION = 1
    REWARD = 2
    DONE = 3
    TRUNCATED = 4
    INFO = 5
    COMPLEMENTARY_DATA = 6


# (observation, action, reward, done, truncated, info, complementary_data)
EnvTransition = Tuple[
    dict[str, Any] | None,  # observation
    Any | None,  # action
    float | None,  # reward
    bool | None,  # done
    bool | None,  # truncated
    Dict[str, Any] | None,  # info
    Dict[str, Any] | None,  # complementary_data
]


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

    A step is any callable accepting a full `EnvTransition` tuple and
    returning a (possibly modified) tuple of the same structure. Implementers
    are encouraged—but not required—to expose the optional helper methods
    listed below. When present, these hooks let `RobotProcessor`
    automatically serialise the step's configuration and learnable state using
    a safe-to-share JSON + SafeTensors format.

    Optional helper protocol:
    * ``get_config() -> Dict[str, Any]`` – User-defined JSON-serializable
      configuration and state. YOU decide what to save here. This is where all
      non-tensor state goes (e.g., name, counter, threshold, window_size).
      The config dict will be passed to your class constructor when loading.
    * ``state_dict() -> Dict[str, torch.Tensor]`` – PyTorch tensor state ONLY.
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


def _default_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:  # noqa: D401
    """Convert a *batch* dict coming from Learobot replay/dataset code into an
    ``EnvTransition`` tuple.

    The function is intentionally **strictly positional** – it maps well known
    keys to the fixed slot order used inside the pipeline.  Missing keys are
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

    # Handle observation and observation.* keys
    observation_keys = {k: v for k, v in batch.items() if k.startswith("observation.")}

    observation = None
    if observation_keys:
        observation = {}
        # Add observation.* keys to the observation dict, removing the "observation." prefix
        for key, value in observation_keys.items():
            observation[key] = value

    return (
        observation,
        batch.get("action"),
        batch.get("next.reward", 0.0),
        batch.get("next.done", False),
        batch.get("next.truncated", False),
        batch.get("info", {}),
        {},
    )


def _default_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:  # noqa: D401
    """Inverse of :pyfunc:`_default_batch_to_transition`. Returns a dict with
    the canonical field names used throughout *LeRobot*.
    """

    (
        observation,
        action,
        reward,
        done,
        truncated,
        info,
        _,
    ) = transition

    batch = {
        "action": action,
        "next.reward": reward,
        "next.done": done,
        "next.truncated": truncated,
        "info": info,
    }

    # Handle observation - flatten dict to observation.* keys if it's a dict
    if isinstance(observation, dict):
        # Check if this looks like a dict that was created from observation.* keys
        for key, value in observation.items():
            batch[key] = value
    return batch


@dataclass
class RobotProcessor(ModelHubMixin):
    """
    Composable, debuggable post-processing processor for robot transitions.

    The class orchestrates an ordered collection of small, functional transforms—steps—executed
    left-to-right on each incoming `EnvTransition`. It can process both `EnvTransition` tuples
    and batch dictionaries, automatically converting between formats as needed.

    Args:
        steps: Ordered list of processing steps executed on every call. Defaults to empty list.
        name: Human-readable identifier that is persisted inside the JSON config.
            Defaults to "RobotProcessor".
        seed: Global seed forwarded to steps that choose to consume it. Defaults to None.
        to_transition: Function to convert batch dict to EnvTransition tuple.
            Defaults to _default_batch_to_transition.
        to_output: Function to convert EnvTransition tuple to the desired output format.
            Usually it is a batch dict or EnvTransition tuple.
            Defaults to _default_transition_to_batch.
        before_step_hooks: List of hooks called before each step. Each hook receives the step
            index and transition, and can optionally return a modified transition.
        after_step_hooks: List of hooks called after each step. Each hook receives the step
            index and transition, and can optionally return a modified transition.
        reset_hooks: List of hooks called during processor reset.
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "RobotProcessor"
    seed: int | None = None

    to_transition: Callable[[dict[str, Any]], EnvTransition] = field(
        default_factory=lambda: _default_batch_to_transition, repr=False
    )
    to_output: Callable[[EnvTransition], dict[str, Any] | EnvTransition] = field(
        default_factory=lambda: _default_transition_to_batch, repr=False
    )

    # Processor-level hooks
    # A hook can optionally return a modified transition.  If it returns
    # ``None`` the current value is left untouched.
    before_step_hooks: list[Callable[[int, EnvTransition], EnvTransition | None]] = field(
        default_factory=list, repr=False
    )
    after_step_hooks: list[Callable[[int, EnvTransition], EnvTransition | None]] = field(
        default_factory=list, repr=False
    )
    reset_hooks: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: EnvTransition | dict[str, Any]):
        """Process data through all steps.

        The method accepts either the classic EnvTransition tuple or a batch dictionary
        (like the ones returned by ReplayBuffer or LeRobotDataset). If a dict is supplied
        it is first converted to the internal tuple format using to_transition; after all
        steps are executed the tuple is transformed back into a dict with to_batch and the
        result is returned – thereby preserving the caller's original data type.

        Args:
            data: Either an EnvTransition tuple or a batch dictionary to process.

        Returns:
            The processed data in the same format as the input (tuple or dict).

        Raises:
            ValueError: If the transition is not a valid 7-tuple format.
        """

        called_with_batch = isinstance(data, dict)

        transition = self.to_transition(data) if called_with_batch else data

        # Basic validation with helpful error message for tuple input
        if not isinstance(transition, tuple) or len(transition) != 7:
            raise ValueError(
                "EnvTransition must be a 7-tuple of (observation, action, reward, done, "
                "truncated, info, complementary_data). "
                f"Got {type(transition).__name__} with length {len(transition) if hasattr(transition, '__len__') else 'unknown'}."
            )

        for idx, processor_step in enumerate(self.steps):
            for hook in self.before_step_hooks:
                updated = hook(idx, transition)
                if updated is not None:
                    transition = updated

            transition = processor_step(transition)

            for hook in self.after_step_hooks:
                updated = hook(idx, transition)
                if updated is not None:
                    transition = updated

        return self.to_output(transition) if called_with_batch else transition

    def step_through(self, transition: EnvTransition) -> Iterable[EnvTransition]:
        """Yield the intermediate Transition instances after each processor step."""
        yield transition
        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    _CFG_NAME = "processor.json"

    def _save_pretrained(self, destination_path: str, **kwargs):
        """Internal save method for ModelHubMixin compatibility."""
        self.save_pretrained(destination_path)

    def _generate_model_card(self, destination_path: str) -> None:
        """Generate README.md from the RobotProcessor model card template."""
        # Read the template
        template_path = Path(__file__).parent.parent / "templates" / "robotprocessor_modelcard_template.md"

        if not template_path.exists():
            # Fallback: if template doesn't exist, skip model card generation
            return

        with open(template_path) as f:
            model_card_content = f.read()

        # Write the README.md
        readme_path = os.path.join(destination_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card_content)

    def save_pretrained(self, destination_path: str, **kwargs):
        """Serialize the processor definition and parameters to *destination_path*."""
        os.makedirs(destination_path, exist_ok=True)

        config: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "steps": [],
        }

        for step_index, processor_step in enumerate(self.steps):
            # Check if step was registered
            registry_name = getattr(processor_step.__class__, "_registry_name", None)

            if registry_name:
                # Use registry name for registered steps
                step_entry: dict[str, Any] = {
                    "registry_name": registry_name,
                }
            else:
                # Fall back to full module path for unregistered steps
                step_entry: dict[str, Any] = {
                    "class": f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}",
                }

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

                    state_filename = f"step_{step_index}.safetensors"
                    save_file(cloned_state, os.path.join(destination_path, state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

        with open(os.path.join(destination_path, self._CFG_NAME), "w") as file_pointer:
            json.dump(config, file_pointer, indent=2)

        # Generate README.md from template
        self._generate_model_card(destination_path)

    def to(self, device: str | torch.device):
        """Move all tensor states inside each step to device and return self.

        Uses a generic mechanism: fetch each step's state dict, move every tensor
        to the target device, and reload it. Only works for steps that implement
        both state_dict() and load_state_dict() methods.
        """
        device = torch.device(device)

        for step in self.steps:
            if hasattr(step, "state_dict") and hasattr(step, "load_state_dict"):
                state = step.state_dict()
                if state:  # Only process if there's actual state
                    moved_state = {k: v.to(device) for k, v in state.items()}
                    step.load_state_dict(moved_state)

        return self

    @classmethod
    def from_pretrained(cls, source: str) -> RobotProcessor:
        """Load a serialized processor from source (local path or Hugging Face Hub identifier)."""
        if Path(source).is_dir():
            # Local path - use it directly
            base_path = Path(source)
            with open(base_path / cls._CFG_NAME) as file_pointer:
                config: dict[str, Any] = json.load(file_pointer)
        else:
            # Hugging Face Hub - download all required files
            # First download the config file
            config_path = hf_hub_download(source, cls._CFG_NAME, repo_type="model")
            with open(config_path) as file_pointer:
                config: dict[str, Any] = json.load(file_pointer)

            # Store downloaded files in the same directory as the config
            base_path = Path(config_path).parent

        steps: list[ProcessorStep] = []
        for step_entry in config["steps"]:
            # Check if step uses registry name or module path
            if "registry_name" in step_entry:
                # Load from registry
                try:
                    step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
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
                except (ImportError, AttributeError) as e:
                    raise ImportError(
                        f"Failed to load processor step '{full_class_path}'. "
                        f"Make sure the module '{module_path}' is installed and contains class '{class_name}'. "
                        f"Consider registering the step using @ProcessorStepRegistry.register() for better portability. "
                        f"Error: {str(e)}"
                    ) from e

            # Instantiate the step with its config
            try:
                step_instance: ProcessorStep = step_class(**step_entry.get("config", {}))
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
                    state_path = hf_hub_download(source, step_entry["state_file"], repo_type="model")

                step_instance.load_state_dict(load_file(state_path))

            steps.append(step_instance)

        return cls(steps, config.get("name", "RobotProcessor"), config.get("seed"))

    def __len__(self) -> int:
        """Return the number of steps in the processor."""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | RobotProcessor:
        """Indexing helper exposing underlying steps.
        * ``int`` – returns the idx-th ProcessorStep.
        * ``slice`` – returns a new RobotProcessor with the sliced steps.
        """
        if isinstance(idx, slice):
            return RobotProcessor(self.steps[idx], self.name, self.seed)
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], EnvTransition | None]):
        """Attach fn to be executed before every processor step."""
        self.before_step_hooks.append(fn)

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], EnvTransition | None]):
        """Attach fn to be executed after every processor step."""
        self.after_step_hooks.append(fn)

    def register_reset_hook(self, fn: Callable[[], None]):
        """Attach fn to be executed when reset is called."""
        self.reset_hooks.append(fn)

    def reset(self):
        """Clear state in every step that implements ``reset()`` and fire registered hooks."""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()  # type: ignore[attr-defined]
        for fn in self.reset_hooks:
            fn()

    def profile_steps(self, transition: EnvTransition, num_runs: int = 100) -> dict[str, float]:
        """Profile the execution time of each step for performance optimization."""
        import time

        profile_results = {}

        for idx, processor_step in enumerate(self.steps):
            step_name = f"step_{idx}_{processor_step.__class__.__name__}"

            # Warm up
            for _ in range(5):
                _ = processor_step(transition)

            # Time the step
            start_time = time.perf_counter()
            for _ in range(num_runs):
                transition = processor_step(transition)
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
            profile_results[step_name] = avg_time

        return profile_results


class ObservationProcessor:
    """Base class for processors that modify only the observation component of a transition.

    Subclasses should override the `observation` method to implement custom observation processing.
    This class handles the boilerplate of extracting and reinserting the processed observation
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class MyObservationScaler(ObservationProcessor):
            def __init__(self, scale_factor):
                self.scale_factor = scale_factor

            def observation(self, observation):
                return observation * self.scale_factor
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        observation = transition[TransitionIndex.OBSERVATION]
        observation = self.observation(observation)
        transition = (observation, *transition[TransitionIndex.ACTION :])
        return transition


class ActionProcessor:
    """Base class for processors that modify only the action component of a transition.

    Subclasses should override the `action` method to implement custom action processing.
    This class handles the boilerplate of extracting and reinserting the processed action
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class ActionClipping(ActionProcessor):
            def __init__(self, min_val, max_val):
                self.min_val = min_val
                self.max_val = max_val

            def action(self, action):
                return np.clip(action, self.min_val, self.max_val)
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        action = transition[TransitionIndex.ACTION]
        action = self.action(action)
        transition = (transition[TransitionIndex.OBSERVATION], action, *transition[TransitionIndex.REWARD :])
        return transition


class RewardProcessor:
    """Base class for processors that modify only the reward component of a transition.

    Subclasses should override the `reward` method to implement custom reward processing.
    This class handles the boilerplate of extracting and reinserting the processed reward
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class RewardScaler(RewardProcessor):
            def __init__(self, scale_factor):
                self.scale_factor = scale_factor

            def reward(self, reward):
                return reward * self.scale_factor
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        reward = transition[TransitionIndex.REWARD]
        reward = self.reward(reward)
        transition = (
            transition[TransitionIndex.OBSERVATION],
            transition[TransitionIndex.ACTION],
            reward,
            *transition[TransitionIndex.DONE :],
        )
        return transition


class DoneProcessor:
    """Base class for processors that modify only the done flag of a transition.

    Subclasses should override the `done` method to implement custom done flag processing.
    This class handles the boilerplate of extracting and reinserting the processed done flag
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

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

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        done = transition[TransitionIndex.DONE]
        done = self.done(done)
        transition = (
            transition[TransitionIndex.OBSERVATION],
            transition[TransitionIndex.ACTION],
            transition[TransitionIndex.REWARD],
            done,
            *transition[TransitionIndex.TRUNCATED :],
        )
        return transition


class TruncatedProcessor:
    """Base class for processors that modify only the truncated flag of a transition.

    Subclasses should override the `truncated` method to implement custom truncated flag processing.
    This class handles the boilerplate of extracting and reinserting the processed truncated flag
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class EarlyTruncation(TruncatedProcessor):
            def __init__(self, threshold):
                self.threshold = threshold

            def truncated(self, truncated):
                # Additional truncation condition
                return truncated or some_condition > self.threshold
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        truncated = transition[TransitionIndex.TRUNCATED]
        truncated = self.truncated(truncated)
        transition = (
            transition[TransitionIndex.OBSERVATION],
            transition[TransitionIndex.ACTION],
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            truncated,
            *transition[TransitionIndex.INFO :],
        )
        return transition


class InfoProcessor:
    """Base class for processors that modify only the info dictionary of a transition.

    Subclasses should override the `info` method to implement custom info processing.
    This class handles the boilerplate of extracting and reinserting the processed info
    into the transition tuple, eliminating the need to implement the `__call__` method in subclasses.

    Example:
        ```python
        class InfoAugmenter(InfoProcessor):
            def __init__(self):
                self.step_count = 0

            def info(self, info):
                info = info.copy()  # Create a copy to avoid modifying the original
                info['steps'] = self.step_count
                self.step_count += 1
                return info

            def reset(self):
                self.step_count = 0
        ```

    By inheriting from this class, you avoid writing repetitive code to handle transition tuple
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
        info = transition[TransitionIndex.INFO]
        info = self.info(info)
        transition = (
            transition[TransitionIndex.OBSERVATION],
            transition[TransitionIndex.ACTION],
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            info,
            *transition[TransitionIndex.COMPLEMENTARY_DATA :],
        )
        return transition


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
