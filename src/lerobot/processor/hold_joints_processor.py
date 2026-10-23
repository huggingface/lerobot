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

"""Hold selected uncommanded joints at their first measured positions."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("hold_uncommanded_joints")
@dataclass
class HoldUncommandedJointsProcessorStep(RobotActionProcessorStep):
    """Fill missing joint commands from a latched observation, without replacing explicit actions.

    Joint names are configured in robot action space. Targets stay fixed until ``reset()``;
    reset on a new control session, not between recording and scene-reset phases. After an
    explicit command, the next missing command latches a fresh measured position.
    """

    joint_names: tuple[str, ...] = ()
    _targets: RobotAction = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(set(self.joint_names)) != len(self.joint_names) or any(
            not name.endswith(".pos") for name in self.joint_names
        ):
            raise ValueError("Held joints must have distinct position-action names ending in '.pos'.")

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        pending = [name for name in self.joint_names if name not in action and name not in self._targets]
        missing = [name for name in pending if name not in observation]
        if missing:
            raise ValueError(f"Cannot hold joints without measured positions: {missing}")
        for name in self.joint_names:
            if name in action:
                self._targets.pop(name, None)
            else:
                if name not in self._targets:
                    self._targets[name] = observation[name]
                action[name] = self._targets[name]
        return action

    def reset(self) -> None:
        self._targets.clear()

    def get_config(self) -> dict[str, Any]:
        return {"joint_names": list(self.joint_names)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return {
            **features,
            PipelineFeatureType.ACTION: {
                **{name: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for name in self.joint_names},
                **features.get(PipelineFeatureType.ACTION, {}),
            },
        }
