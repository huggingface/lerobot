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
import os, json
from typing import Any, Dict, Sequence, Iterable, Protocol, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import IntEnum
import numpy as np
import torch
from huggingface_hub import  hf_hub_download, ModelHubMixin
from safetensors.torch import save_file, load_file


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
    Any| None,                # observation
    Any| None,                # action
    float| None,              # reward
    bool| None,               # done
    bool| None,               # truncated
    Dict[str, Any]| None,     # info
    Dict[str, Any]| None,     # complementary_data
]



class PipelineStep(Protocol):
    """Structural typing interface for a single pipeline step.
    
    A step is any callable accepting a full `EnvTransition` tuple and
    returning a (possibly modified) tuple of the same structure. Implementers
    are encouraged—but not required—to expose the optional helper methods
    listed below. When present, these hooks let `RobotPipeline`
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

    def get_config(self) -> Dict[str, Any]: ...

    def state_dict(self) -> Dict[str, torch.Tensor]: ...

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None: ...

    def reset(self) -> None: ...


@dataclass
class RobotPipeline(ModelHubMixin):
    """
    Composable, debuggable post-processing pipeline for RL transitions.
    The class orchestrates an ordered collection of small, functional
    transforms—steps—executed left-to-right on each incoming
    `EnvTransition`.
    Parameters:
    steps : Sequence[PipelineStep], optional
        Ordered list executed on every call
        name : str, default="RobotPipeline"
        Human-readable identifier that is persisted inside the JSON config.
    seed : int | None, optional
        Global seed forwarded to steps that choose to consume it.
    Examples:
    Basic usage::
        env = gym.make("CartPole-v1")
        pipe = RobotPipeline([
            ObservationNormalizer(),
            IntrinsicVelocity(),
            VelocityBonus(0.02),
        ])
        obs, info = env.reset(seed=0)
        tr = (obs, None, 0.0, False, False, info, {})
        obs, *_ = pipe(tr)  # agent sees a normalised observation
    Inspecting intermediate results::
        for idx, step_tr in enumerate(pipe.step_through(tr)):
            print(idx, step_tr)
    Serialization to the Hugging Face Hub::
        pipe.save_pretrained("chkpt")
        pipe.push_to_hub("my-org/cartpole_pipe")
        loaded = RobotPipeline.from_pretrained("my-org/cartpole_pipe")
    """
    steps: Sequence[PipelineStep] = field(default_factory=list)
    name: str = "RobotPipeline"
    seed: Optional[int] = None

    # Pipeline-level hooks
    # A hook can optionally return a modified transition.  If it returns
    # ``None`` the current value is left untouched.
    before_step_hooks: list[Callable[[int, EnvTransition], Optional[EnvTransition]]] = field(
        default_factory=list, repr=False
    )
    after_step_hooks: list[Callable[[int, EnvTransition], Optional[EnvTransition]]] = field(
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

        for idx, pipeline_step in enumerate(self.steps):
            for hook in self.before_step_hooks:
                updated = hook(idx, transition)
                if updated is not None:
                    transition = updated

            transition = pipeline_step(transition)

            for hook in self.after_step_hooks:
                updated = hook(idx, transition)
                if updated is not None:
                    transition = updated

        return transition

    def step_through(self, transition: EnvTransition) -> Iterable[EnvTransition]:
        """Yield the intermediate Transition instances after each pipeline step."""
        yield transition
        for pipeline_step in self.steps:
            transition = pipeline_step(transition)
            yield transition

    _CFG_NAME = "pipeline.json"
    
    def _save_pretrained(self, destination_path: str, **kwargs):
        """Internal save method for ModelHubMixin compatibility."""
        self.save_pretrained(destination_path)
    
    def save_pretrained(self, destination_path: str, **kwargs):
        """Serialize the pipeline definition and parameters to *destination_path*."""
        os.makedirs(destination_path, exist_ok=True)

        config: Dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "steps": [],
        }

        for step_index, pipeline_step in enumerate(self.steps):
            step_entry: Dict[str, Any] = {
                "class": f"{pipeline_step.__class__.__module__}.{pipeline_step.__class__.__name__}",
            }

            if hasattr(pipeline_step, "get_config"):
                step_entry["config"] = pipeline_step.get_config()

            if hasattr(pipeline_step, "state_dict"):
                state = pipeline_step.state_dict()
                if state:
                    state_filename = f"step_{step_index}.safetensors"
                    save_file(state, os.path.join(destination_path, state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

        with open(os.path.join(destination_path, self._CFG_NAME), "w") as file_pointer:
            json.dump(config, file_pointer, indent=2)

    @classmethod
    def from_pretrained(cls, source: str) -> "RobotPipeline":
        """Load a serialized pipeline from *source* (local path or Hugging Face Hub identifier)."""
        if Path(source).is_dir():
            # Local path - use it directly
            base_path = Path(source)
            with open(base_path / cls._CFG_NAME) as file_pointer:
                config: Dict[str, Any] = json.load(file_pointer)
        else:
            # Hugging Face Hub - download all required files
            # First download the config file
            config_path = hf_hub_download(source, cls._CFG_NAME, repo_type="model")
            with open(config_path) as file_pointer:
                config: Dict[str, Any] = json.load(file_pointer)
            
            # Store downloaded files in the same directory as the config
            base_path = Path(config_path).parent

        steps: list[PipelineStep] = []
        for step_entry in config["steps"]:
            module_path, class_name = step_entry["class"].rsplit(".", 1)
            step_class = getattr(__import__(module_path, fromlist=[class_name]), class_name)
            step_instance: PipelineStep = step_class(**step_entry.get("config", {}))

            if "state_file" in step_entry and hasattr(step_instance, "load_state_dict"):
                if Path(source).is_dir():
                    # Local path - read directly
                    state_path = str(base_path / step_entry["state_file"])
                else:
                    # Hugging Face Hub - download the state file
                    state_path = hf_hub_download(source, step_entry["state_file"], repo_type="model")
                
                step_instance.load_state_dict(load_file(state_path))

            steps.append(step_instance)

        return cls(steps, config.get("name", "RobotPipeline"), config.get("seed"))

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> PipelineStep | RobotPipeline:
        """Indexing helper exposing underlying steps.
        * ``int`` – returns the idx-th PipelineStep.
        * ``slice`` – returns a new RobotPipeline with the sliced steps.
        """
        if isinstance(idx, slice):
            return RobotPipeline(self.steps[idx], self.name, self.seed)
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], Optional[EnvTransition]]):
        """Attach fn to be executed before every pipeline step."""
        self.before_step_hooks.append(fn)

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], Optional[EnvTransition]]):
        """Attach fn to be executed after every pipeline step."""
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

    def profile_steps(self, transition: EnvTransition, num_runs: int = 100) -> Dict[str, float]:
        """Profile the execution time of each step for performance optimization."""
        import time
        
        profile_results = {}
        
        for idx, pipeline_step in enumerate(self.steps):
            step_name = f"step_{idx}_{pipeline_step.__class__.__name__}"
            
            # Warm up
            for _ in range(5):
                _ = pipeline_step(transition)
            
            # Time the step
            start_time = time.perf_counter()
            for _ in range(num_runs):
                transition = pipeline_step(transition)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
            profile_results[step_name] = avg_time
            
        return profile_results