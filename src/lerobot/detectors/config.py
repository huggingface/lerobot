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

"""Detector and supervisor configuration.

``DetectorConfig`` is a draccus ``ChoiceRegistry`` so the detector backend is
chosen with ``--<prefix>.detector.type=motion|red_cube_speed`` and only that
backend's fields are exposed. ``SupervisorConfig`` bundles the wiring (enable
flag, camera key, polling, cooldown) shared by every consumer, with the chosen
detector nested under it. Both the async-inference client and the RTC rollout
engine embed a single ``SupervisorConfig``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import draccus

from .motion import MotionDetector
from .red_cube_speed import RedCubeSpeedDetector


@dataclass
class DetectorConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for detector backends. Select with ``--....detector.type=<name>``."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def make(self):
        """Instantiate the detector callable this config describes."""
        raise NotImplementedError


@DetectorConfig.register_subclass("motion")
@dataclass
class MotionDetectorConfig(DetectorConfig):
    """Frame-difference motion detector (boolean trigger, no speed estimate)."""

    motion_threshold: float = 0.02

    def __post_init__(self):
        if not 0 < self.motion_threshold <= 1:
            raise ValueError(f"motion_threshold must be in (0, 1], got {self.motion_threshold}")

    def make(self) -> MotionDetector:
        return MotionDetector(motion_area_threshold=self.motion_threshold)


@DetectorConfig.register_subclass("red_cube_speed")
@dataclass
class RedCubeSpeedDetectorConfig(DetectorConfig):
    """Red-cube speed detector: maps image-plane speed to an adaptive replan threshold."""

    slow_speed_px_s: float = 40.0
    fast_speed_px_s: float = 200.0
    urgent_speed_px_s: float = 250.0
    min_chunk_size_threshold: float = 0.25
    max_chunk_size_threshold: float = 0.75
    hue_tolerance_deg: float = 20.0
    saturation_min: float = 0.45
    value_min: float = 0.25
    min_area_ratio: float = 0.001

    def __post_init__(self):
        if self.slow_speed_px_s < 0:
            raise ValueError(f"slow_speed_px_s must be non-negative, got {self.slow_speed_px_s}")
        if self.fast_speed_px_s <= self.slow_speed_px_s:
            raise ValueError(
                "fast_speed_px_s must be greater than slow_speed_px_s, "
                f"got {self.fast_speed_px_s} <= {self.slow_speed_px_s}"
            )
        if self.urgent_speed_px_s < self.slow_speed_px_s:
            raise ValueError(
                "urgent_speed_px_s must be at least slow_speed_px_s, "
                f"got {self.urgent_speed_px_s} < {self.slow_speed_px_s}"
            )
        if not 0 <= self.min_chunk_size_threshold <= 1:
            raise ValueError(
                f"min_chunk_size_threshold must be between 0 and 1, got {self.min_chunk_size_threshold}"
            )
        if not 0 <= self.max_chunk_size_threshold <= 1:
            raise ValueError(
                f"max_chunk_size_threshold must be between 0 and 1, got {self.max_chunk_size_threshold}"
            )
        if self.min_chunk_size_threshold > self.max_chunk_size_threshold:
            raise ValueError(
                "min_chunk_size_threshold must be <= max_chunk_size_threshold, "
                f"got {self.min_chunk_size_threshold} > {self.max_chunk_size_threshold}"
            )
        if not 0 <= self.hue_tolerance_deg <= 180:
            raise ValueError(f"hue_tolerance_deg must be between 0 and 180, got {self.hue_tolerance_deg}")
        if not 0 <= self.saturation_min <= 1:
            raise ValueError(f"saturation_min must be between 0 and 1, got {self.saturation_min}")
        if not 0 <= self.value_min <= 1:
            raise ValueError(f"value_min must be between 0 and 1, got {self.value_min}")
        if not 0 < self.min_area_ratio <= 1:
            raise ValueError(f"min_area_ratio must be in (0, 1], got {self.min_area_ratio}")

    def make(self) -> RedCubeSpeedDetector:
        return RedCubeSpeedDetector(
            slow_speed_px_s=self.slow_speed_px_s,
            fast_speed_px_s=self.fast_speed_px_s,
            min_chunk_size_threshold=self.min_chunk_size_threshold,
            max_chunk_size_threshold=self.max_chunk_size_threshold,
            urgent_speed_px_s=self.urgent_speed_px_s,
            hue_tolerance_deg=self.hue_tolerance_deg,
            saturation_min=self.saturation_min,
            value_min=self.value_min,
            min_area_ratio=self.min_area_ratio,
        )


@dataclass
class SupervisorConfig:
    """Event-triggered / speed-adaptive replanning supervisor.

    Disabled by default so existing behavior is unchanged. When ``enabled``, the
    chosen ``detector`` watches ``camera`` and can fire an early replan or, for
    speed-adaptive detectors, raise the effective replan threshold.
    """

    enabled: bool = False
    camera: str = "overall"
    poll_fps: int = 20
    cooldown_s: float = 1.0
    require_target_visible: bool = False
    detector: DetectorConfig = field(default_factory=MotionDetectorConfig)

    def __post_init__(self):
        if self.enabled:
            if self.poll_fps <= 0:
                raise ValueError(f"poll_fps must be positive, got {self.poll_fps}")
            if self.cooldown_s < 0:
                raise ValueError(f"cooldown_s must be non-negative, got {self.cooldown_s}")
            if not self.camera:
                raise ValueError("camera must be set when the supervisor is enabled")


def make_detector(config: DetectorConfig):
    """Instantiate the detector callable described by ``config``."""
    return config.make()
