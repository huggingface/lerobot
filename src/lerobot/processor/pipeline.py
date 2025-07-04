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
    Any | None,  # observation
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


@dataclass
class RobotProcessor(ModelHubMixin):
    """
    Composable, debuggable post-processing processor for robot transitions.
    The class orchestrates an ordered collection of small, functional
    transforms—steps—executed left-to-right on each incoming
    `EnvTransition`.
    Parameters:
    steps : Sequence[ProcessorStep], optional
        Ordered list executed on every call
        name : str, default="RobotProcessor"
        Human-readable identifier that is persisted inside the JSON config.
    seed : int | None, optional
        Global seed forwarded to steps that choose to consume it.
    Examples:
    Basic usage::
        env = gym.make("CartPole-v1")
        proc = RobotProcessor([
            ObservationNormalizer(),
            IntrinsicVelocity(),
            VelocityBonus(0.02),
        ])
        obs, info = env.reset(seed=0)
        tr = (obs, None, 0.0, False, False, info, {})
        obs, *_ = proc(tr)  # agent sees a normalised observation
    Inspecting intermediate results::
        for idx, step_tr in enumerate(proc.step_through(tr)):
            print(idx, step_tr)
    Serialization to the Hugging Face Hub::
        proc.save_pretrained("chkpt")
        proc.push_to_hub("my-org/cartpole_proc")
        loaded = RobotProcessor.from_pretrained("my-org/cartpole_proc")
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "RobotProcessor"
    seed: int | None = None

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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Run *transition* through every step, firing hooks on the way."""

        # Basic validation with helpful error message
        if not isinstance(transition, tuple) or len(transition) != 7:
            raise ValueError(
                f"EnvTransition must be a 7-tuple of (observation, action, reward, done, truncated, info, complementary_data), "
                f"got {type(transition).__name__} with length {len(transition) if hasattr(transition, '__len__') else 'unknown'}"
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

        return transition

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
    def observation(self, observation):
        return observation

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionIndex.OBSERVATION]
        observation = self.observation(observation)
        transition = (observation, *transition[TransitionIndex.ACTION :])
        return transition


class ActionProcessor:
    def action(self, action):
        return action

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition[TransitionIndex.ACTION]
        action = self.action(action)
        transition = (transition[TransitionIndex.OBSERVATION], action, *transition[TransitionIndex.REWARD :])
        return transition


class RewardProcessor:
    def reward(self, reward):
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
    def done(self, done):
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
    def truncated(self, truncated):
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
    def info(self, info):
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
