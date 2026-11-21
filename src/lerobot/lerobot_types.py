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

from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict, final

import numpy as np
import torch


@final
class TransitionKey:
    """Keys for accessing `EnvTransition` dictionary components (plain string constants, not an Enum)."""

    OBSERVATION: Final[Literal["observation"]] = "observation"
    ACTION: Final[Literal["action"]] = "action"
    PREDICTION: Final[Literal["prediction"]] = "prediction"
    REWARD: Final[Literal["reward"]] = "reward"
    DONE: Final[Literal["done"]] = "done"
    TRUNCATED: Final[Literal["truncated"]] = "truncated"
    INFO: Final[Literal["info"]] = "info"
    COMPLEMENTARY_DATA: Final[Literal["complementary_data"]] = "complementary_data"


# Kept as `TypeAlias` (not PEP 695 `type`): both are used in `isinstance()` checks.
PolicyAction: TypeAlias = torch.Tensor  # noqa: UP040
RobotAction = dict[str, Any]
EnvAction: TypeAlias = np.ndarray  # noqa: UP040
RobotObservation = dict[str, Any]
BatchType = dict[str, Any]


class Detection(TypedDict):
    """One labelled box of a bbox answer (annotation VQA schema)."""

    label: str
    bbox: list[float]  # [x1, y1, x2, y2] in image fractions (0 to 1)


class BboxAnswer(TypedDict):
    """A bbox answer as the annotation pipeline writes it: ``{"detections": [...]}``."""

    detections: list[Detection]


class PolicyPrediction(TypedDict, total=False):
    """What a policy expects or believes besides the action it takes, batched like the action.

    Every entry is keyed by what it refers to, so it maps to existing names, and holds one value
    per environment of the batch (see ``PreTrainedPolicy.select_action``).
    """

    # Predicted future observation, by observation key ("observation.images.top" -> [B, C, H, W]).
    observation: dict[str, torch.Tensor]
    # Predicted language, by annotation style ("subtask", "plan", "memory"), one string per env.
    language: dict[str, list[str]]
    # Boxes, by the observation image key they are drawn on, one answer per env.
    boxes: dict[str, list[BboxAnswer]]


class PolicyOutput(TypedDict):
    """What ``select_action`` / ``predict_action_chunk`` return when they predict more than actions.

    A batch keyed like the input batch: the action under ``"action"`` and the prediction under
    ``"prediction"``. `batch_to_transition` turns it into an `EnvTransition`.
    """

    action: torch.Tensor
    prediction: NotRequired[PolicyPrediction]


class EnvTransition(TypedDict):
    """A single environment transition, keyed by the `TransitionKey` constants.

    All keys are required; build transitions with `create_transition`.
    """

    observation: RobotObservation | None
    action: PolicyAction | RobotAction | EnvAction | None
    prediction: PolicyPrediction | None
    reward: float | torch.Tensor | None
    done: bool | torch.Tensor | None
    truncated: bool | torch.Tensor | None
    info: dict[str, Any] | None
    complementary_data: dict[str, Any] | None
