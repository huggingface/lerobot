# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Describe a simulator environment without deployment or transport concepts."""

import importlib.metadata
from dataclasses import asdict

import numpy as np

from lerobot.sims.backend import Backend
from lerobot.sims.types import EnvDescriptor, FeatureSpec


def describe_backend(backend: Backend) -> EnvDescriptor:
    """Return canonical spaces, conventions, and tasks exposed by a backend."""
    observation_space, action_space = backend.single_spaces()
    features = []
    for name, space in observation_space.spaces.items():
        names = backend.config.state_names if name == "observation.state" else ()
        if not names and len(space.shape) == 1:
            names = tuple(f"state_{index}" for index in range(space.shape[0]))
        features.append(
            FeatureSpec(
                name,
                space.shape,
                space.dtype.name,
                "rgb" if space.dtype == np.uint8 else "tensor",
                names,
                backend.config.semantics,
            )
        )
    dim = action_space.shape[0]
    action_names = backend.config.action_names or tuple(f"command_{index}" for index in range(dim))
    return EnvDescriptor(
        backend.config.type,
        backend.config.semantics,
        tuple(features),
        FeatureSpec(
            "action",
            (dim,),
            "float32",
            names=action_names,
            semantics=backend.config.semantics,
        ),
        backend.config.control,
        backend.config.fps,
        backend.tasks,
        backend.max_episode_steps,
        {
            "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
        },
        backend.config.gripper_indices,
        backend.config.command_retention_indices,
        backend.config.robot_type,
        asdict(backend.config),
    )
