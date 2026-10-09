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

"""Declarative mappings and explicit simulator semantics for evaluation admission."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from lerobot.transport.wire.features import feature_mismatch

from .contracts import EnvDescriptor


@dataclass
class EvalProfile:
    """Declare an explicit checkpoint mapping and controller convention."""

    semantics: str
    feature_mapping: dict[str, str] = field(default_factory=dict)
    state_names: list[str] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)
    action_name_mapping: dict[str, str] = field(default_factory=dict)
    empty_cameras: int = 0
    control: str = "delta"

    @classmethod
    def load(cls, path: str):
        """Read a declarative evaluation profile from YAML."""
        return cls(**yaml.safe_load(Path(path).read_text()))

    def validate_descriptor(self, descriptor: EnvDescriptor):
        """Reject schema mappings and conventions incompatible with the simulator."""
        if self.semantics != descriptor.semantics or self.control != descriptor.control:
            raise ValueError("Evaluation profile semantics/controller do not match the simulator")
        names = {f.name for f in descriptor.features}
        if set(self.feature_mapping) - names:
            raise ValueError("Profile maps observation keys absent from the simulator")
        targets = [self.feature_mapping.get(name, name) for name in names]
        if len(set(targets)) != len(targets):
            raise ValueError("Evaluation feature mapping must be one-to-one")
        state = next((f for f in descriptor.features if f.name == "observation.state"), None)
        if self.state_names and (
            state is None or feature_mismatch(state, replace(state, names=tuple(self.state_names)))
        ):
            raise ValueError("Simulator state component order differs from the profile")
        if self.action_names and feature_mismatch(
            descriptor.action_feature, replace(descriptor.action_feature, names=tuple(self.action_names))
        ):
            raise ValueError("Simulator action component order differs from the profile")
        self.policy_action_names(descriptor)

    def policy_action_names(self, descriptor: EnvDescriptor) -> tuple[str, ...]:
        """Alias component labels while preserving the native controller order."""
        names = descriptor.action_feature.names
        if set(self.action_name_mapping) - set(names):
            raise ValueError("Profile maps action components absent from the simulator")
        mapped = tuple(self.action_name_mapping.get(name, name) for name in names)
        if len(set(mapped)) != len(mapped) or any(not name for name in mapped):
            raise ValueError("Action component mapping must be one-to-one with nonempty labels")
        return mapped

    def validate_policy(self, descriptor: EnvDescriptor, config):
        """Check simulator features against the authoritative checkpoint declaration."""
        self.validate_descriptor(descriptor)
        if getattr(config, "empty_cameras", 0) != self.empty_cameras:
            raise ValueError("Checkpoint empty-camera configuration differs from the profile")
        available = {self.feature_mapping.get(f.name, f.name): f for f in descriptor.features}
        empty = {f"observation.images.empty_camera_{i}" for i in range(self.empty_cameras)}
        for name, expected in config.input_features.items():
            if name in empty:
                if tuple(expected.shape) != (3, 480, 640):
                    raise ValueError("Checkpoint empty-camera shape is invalid")
                continue
            if name not in available:
                raise ValueError(f"Checkpoint feature missing from simulator/profile: {name}")
            actual = available[name]
            shape = (
                (actual.shape[2], actual.shape[0], actual.shape[1]) if actual.kind == "rgb" else actual.shape
            )
            if tuple(expected.shape) != shape:
                raise ValueError(
                    f"Checkpoint feature shape mismatch for {name}: expected {expected.shape}, got {shape}"
                )
        if "action" not in config.output_features:
            raise ValueError("Checkpoint must declare the canonical action feature")
        if tuple(config.output_features["action"].shape) != descriptor.action_feature.shape:
            raise ValueError("Checkpoint action dimension differs from the simulator")
        checkpoint_names = getattr(config, "action_feature_names", None)
        if checkpoint_names and feature_mismatch(
            replace(descriptor.action_feature, names=tuple(checkpoint_names)),
            replace(descriptor.action_feature, names=self.policy_action_names(descriptor)),
        ):
            raise ValueError("Checkpoint action component order differs from the simulator")
