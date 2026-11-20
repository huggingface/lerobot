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

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("hold_uncommanded_joints")
@dataclass
class HoldUncommandedJointsProcessorStep(RobotActionProcessorStep):
    """Fill missing joint commands with the position measured the first time each joint was missing.

    Latching (rather than echoing each new observation) keeps a sagging arm from drifting.
    Targets persist across recording/reset phases; ``reset()`` re-latches, e.g. after policy control.
    """

    joint_names: tuple[str, ...] = ()
    _targets: RobotAction = field(default_factory=dict, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        for name in self.joint_names:
            if name not in action:
                action[name] = self._targets.setdefault(name, observation[name])
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
                **features.get(PipelineFeatureType.ACTION, {}),
                **{name: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for name in self.joint_names},
            },
        }
