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

from typing import TYPE_CHECKING, Any

from lerobot.configs.types import PipelineFeatureType

if TYPE_CHECKING:
    from lerobot.processor import RobotAction, RobotObservation


def create_initial_features(
    action: RobotAction | None = None, observation: RobotObservation | None = None
) -> dict[PipelineFeatureType, dict[str, Any]]:
    """
    Creates the initial features dict for the dataset from action and observation specs.

    Args:
        action: A dictionary of action feature names to their types/shapes.
        observation: A dictionary of observation feature names to their types/shapes.

    Returns:
        The initial features dictionary structured by PipelineFeatureType.
    """
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    if action:
        features[PipelineFeatureType.ACTION] = action
    if observation:
        features[PipelineFeatureType.OBSERVATION] = observation
    return features
