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

"""Check a checkpoint's declared features against a simulator before stepping."""

from typing import Any

from .contracts import EnvDescriptor

# Placeholder cameras come from the checkpoint's own `empty_cameras` setting;
# their values and shapes belong to policy preparation, not the simulator.
_EMPTY_CAMERA_PREFIX = "observation.images.empty_camera_"


def validate_checkpoint(descriptor: EnvDescriptor, config: Any, rename_map: dict[str, str]) -> None:
    """Reject a checkpoint whose inputs or action do not fit the simulator.

    Simulator keys are renamed with ``rename_map``, as for any robot. RGB spatial
    dimensions are left to policy preprocessing, which owns resizing.
    """
    available = {rename_map.get(f.name, f.name): f for f in descriptor.features}
    for name, expected in config.input_features.items():
        if name.startswith(_EMPTY_CAMERA_PREFIX) and name.removeprefix(_EMPTY_CAMERA_PREFIX).isdigit():
            continue
        if name not in available:
            raise ValueError(f"Checkpoint feature missing from the simulator (check --rename_map): {name}")
        actual = available[name]
        shape: tuple[int, ...]
        if actual.kind == "rgb":
            compatible = len(expected.shape) == 3 and expected.shape[0] == actual.shape[2]
            shape = (actual.shape[2], actual.shape[0], actual.shape[1])
        else:
            compatible = tuple(expected.shape) == actual.shape
            shape = actual.shape
        if not compatible:
            raise ValueError(
                f"Checkpoint feature shape mismatch for {name}: expected {expected.shape}, got {shape}"
            )
    if "action" not in config.output_features:
        raise ValueError("Checkpoint must declare the canonical action feature")
    if tuple(config.output_features["action"].shape) != descriptor.action_feature.shape:
        raise ValueError("Checkpoint action dimension differs from the simulator")
