# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class PipelineFeatureType(str, Enum):
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class TextKind(str, Enum):
    """What the interactive language runtime can ask a policy's text head for.

    One vocabulary shared by every policy, so `generate_text` callers and
    implementations cannot drift. Values are lowercase to match the batched
    `generate_texts` APIs, which accept a wider set (caption, grounding, ...).
    """

    SUBTASK = "subtask"  # next low-level instruction to condition actions on
    VQA = "vqa"  # answer a question about the current view


@dataclass(frozen=True, slots=True)
class ActionChunkPrediction:
    """An action chunk together with the text belonging to it.

    Returned by `PreTrainedPolicy.predict_action_chunk_with_text`. Frozen so the
    pair cannot drift apart after the fact: a caller holding one of these knows
    the text describes exactly this chunk, which a mutable `last_*` attribute on
    the policy could not guarantee across concurrent inference calls.
    """

    action: Tensor
    text: str | None = None


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"
