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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lerobot.utils.constants import ACTION, OBS_ACTION, OBS_PREFIX, REWARD


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


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass(frozen=True)
class Deltas:
    """Frames, relative to the current one, that a batch holds for each feature.

    A declared feature arrives as ``[B, T, ...]`` with ``{key}_is_pad`` (true where the
    frame falls outside the episode); an undeclared observation arrives as the current
    frame alone, without a time axis. Offsets are in dataset frames: one frame per action
    interval at inference.

    Attributes:
        observation: Offsets per observation feature type: ``VISUAL`` (every camera),
            ``STATE`` (``observation.state`` and other vector observations), ``ENV``
            (``observation.environment_state``) and ``ACTION`` (``observation.action``, the
            commands executed at those frames, read from the dataset's ``action`` column).
            An observation key (``"observation.images.top"``) declares that feature alone and
            overrides the offsets of its type, e.g. a history for one camera only.
        action: Offsets of ``action``: the targets a policy is trained to predict.
        reward: Offsets of the reward.
    """

    observation: Mapping[FeatureType, Sequence[int]] | Mapping[str, Sequence[int]] = field(
        default_factory=dict
    )
    action: Sequence[int] | None = None
    reward: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Store offsets as tuples of integers."""
        allowed = {FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV, FeatureType.ACTION}
        for key in self.observation:
            if isinstance(key, FeatureType):
                if key not in allowed:
                    raise ValueError(f"Observation deltas cannot declare {key}.")
            elif not isinstance(key, str) or not key.startswith(OBS_PREFIX) or key == OBS_ACTION:
                raise ValueError(
                    f"Observation deltas take feature types or observation keys, got {key!r} "
                    "(past commands are FeatureType.ACTION)."
                )
        object.__setattr__(
            self, "observation", {kind: _offsets(kind, offsets) for kind, offsets in self.observation.items()}
        )
        for name in ("action", "reward"):
            if (offsets := getattr(self, name)) is not None:
                object.__setattr__(self, name, _offsets(name, offsets))

    def of(self, key: str, feature_type: FeatureType | None = None) -> tuple[int, ...] | None:
        """The offsets of feature ``key`` (of type ``feature_type``), or ``None`` when undeclared."""
        if key == ACTION:
            return self.action  # type: ignore[return-value]
        if key == REWARD:
            return self.reward  # type: ignore[return-value]
        observation: Mapping[Any, tuple[int, ...]] = self.observation  # type: ignore[assignment]
        if key == OBS_ACTION:
            return observation.get(FeatureType.ACTION)
        if key in observation:
            return observation[key]
        if key.startswith(OBS_PREFIX) and feature_type is not None:
            return observation.get(feature_type)
        return None


REMOVED_DELTA_PROPERTIES = (
    "observation_delta_indices",
    "image_observation_delta_indices",
    "state_observation_delta_indices",
    "action_delta_indices",
    "reward_delta_indices",
)


def reject_removed_delta_properties(cls: type) -> None:
    """Fail at class definition for configs still declaring the removed ``*_delta_indices``."""
    if removed := [name for name in REMOVED_DELTA_PROPERTIES if name in vars(cls)]:
        raise TypeError(
            f"{cls.__name__} defines {', '.join(removed)}, which were removed: declare the frames "
            "a batch holds with the `training_deltas` and `inference_deltas` properties "
            "(lerobot.configs.types.Deltas)."
        )


def _offsets(name: object, offsets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(offsets)
    if not values or any(type(v) is not int for v in values):
        raise ValueError(f"Deltas of {name} must be a nonempty sequence of integers; got {values}.")
    return values


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"
