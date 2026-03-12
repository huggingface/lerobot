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

from enum import Enum
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch


class TransitionKey(str, Enum):
    """Keys for accessing EnvTransition dictionary components."""

    # TODO(Steven): Use consts
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction: TypeAlias = torch.Tensor
RobotAction: TypeAlias = dict[str, Any]
EnvAction: TypeAlias = np.ndarray
RobotObservation: TypeAlias = dict[str, Any]


EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: RobotObservation | None,
        TransitionKey.ACTION.value: PolicyAction | RobotAction | EnvAction | None,
        TransitionKey.REWARD.value: float | torch.Tensor | None,
        TransitionKey.DONE.value: bool | torch.Tensor | None,
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)
