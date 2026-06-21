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

"""Camera-frame detectors for event-triggered and speed-adaptive replanning.

These detectors are pure (``frame -> bool | DetectorOutput``) and transport
agnostic, so the same code backs both the async-inference client and the RTC
rollout engine. See :mod:`lerobot.detectors.config` for the draccus configs and
``make_detector`` factory used to build them from the CLI.
"""

from .base import DetectorOutput, normalize_detector_output
from .config import (
    DetectorConfig,
    MotionDetectorConfig,
    RedCubeSpeedDetectorConfig,
    SupervisorConfig,
    make_detector,
)
from .motion import MotionDetector
from .red_cube_speed import RedCubeSpeedDetector

__all__ = [
    "DetectorConfig",
    "DetectorOutput",
    "MotionDetector",
    "MotionDetectorConfig",
    "RedCubeSpeedDetector",
    "RedCubeSpeedDetectorConfig",
    "SupervisorConfig",
    "make_detector",
    "normalize_detector_output",
]
