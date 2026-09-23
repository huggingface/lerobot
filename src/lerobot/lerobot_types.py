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

from typing import Any, Final, Literal, TypeAlias, TypedDict, final

import numpy as np

from lerobot.utils.import_utils import LAZY_IMPORTS, lazy_getattr

# These need torch, so they are declared the first time one of them is used.
if LAZY_IMPORTS:
    import torch

    # Kept as `TypeAlias` (not PEP 695 `type`): both are used in `isinstance()` checks.
    PolicyAction: TypeAlias = torch.Tensor  # noqa: UP040

__getattr__ = lazy_getattr(__name__)


@final
class TransitionKey:
    """Keys for accessing `EnvTransition` dictionary components (plain string constants, not an Enum)."""

    OBSERVATION: Final[Literal["observation"]] = "observation"
    ACTION: Final[Literal["action"]] = "action"
    REWARD: Final[Literal["reward"]] = "reward"
    DONE: Final[Literal["done"]] = "done"
    TRUNCATED: Final[Literal["truncated"]] = "truncated"
    INFO: Final[Literal["info"]] = "info"
    COMPLEMENTARY_DATA: Final[Literal["complementary_data"]] = "complementary_data"


RobotAction = dict[str, Any]
EnvAction: TypeAlias = np.ndarray  # noqa: UP040
RobotObservation = dict[str, Any]
BatchType = dict[str, Any]


class EnvTransition(TypedDict):
    """A single environment transition, keyed by the `TransitionKey` constants.

    All keys are required; build transitions with `create_transition`.
    """

    observation: RobotObservation | None
    action: PolicyAction | RobotAction | EnvAction | None
    reward: float | torch.Tensor | None
    done: bool | torch.Tensor | None
    truncated: bool | torch.Tensor | None
    info: dict[str, Any] | None
    complementary_data: dict[str, Any] | None
