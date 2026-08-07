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
from dataclasses import dataclass
from enum import Enum


class FeatureType(str, Enum):
    """The category of data a `PolicyFeature` represents.

    **Attributes**:
        - **STATE** -- A robot/environment proprioceptive state vector.
        - **VISUAL** -- An image or video feature.
        - **ENV** -- Environment-provided state, distinct from robot proprioception (e.g. simulation
          environment state).
        - **ACTION** -- An action vector.
        - **REWARD** -- A scalar reward.
        - **LANGUAGE** -- A natural-language feature (e.g. task instruction tokens).
    """

    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class PipelineFeatureType(str, Enum):
    """Which side of a processor pipeline a feature belongs to.

    **Attributes**:
        - **ACTION** -- The feature is part of the action space.
        - **OBSERVATION** -- The feature is part of the observation space.
    """

    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class NormalizationMode(str, Enum):
    """The normalization strategy applied to a feature by a `NormalizerProcessorStep`.

    **Attributes**:
        - **MIN_MAX** -- Scale to `[-1, 1]` using the feature's min/max statistics.
        - **MEAN_STD** -- Center and scale to unit variance using the feature's mean/std statistics.
        - **IDENTITY** -- Leave the feature unchanged.
        - **QUANTILES** -- Scale to `[-1, 1]` using the feature's 1st/99th percentile statistics.
        - **QUANTILE10** -- Scale to `[-1, 1]` using the feature's 10th/90th percentile statistics.
    """

    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    """Describes one entry of a policy's input/output feature space.

    Args:
        type (`FeatureType`): The category of the feature.
        shape (`tuple[int, ...]`): The feature's shape, excluding the batch dimension.
    """

    type: FeatureType
    shape: tuple[int, ...]


class RTCAttentionSchedule(str, Enum):
    """The prefix-attention weighting schedule used by the Real-Time Chunking (RTC) policy.

    Controls how much weight is given to the previous action chunk's prediction versus the new one,
    over the overlap region between consecutive chunks.

    **Attributes**:
        - **ZEROS** -- No prefix attention: weight is 1.0 before `start`, then 0.0.
        - **ONES** -- Full prefix attention: weight is 1.0 up to `end`, then 0.0.
        - **LINEAR** -- Linearly ramps the weight down from 1.0 to 0.0 between `start` and `end`.
        - **EXP** -- Like `LINEAR`, but with an exponential (rather than linear) decay curve.
    """

    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"
