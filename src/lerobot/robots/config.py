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

import abc
import json
from dataclasses import dataclass
from pathlib import Path

import draccus


def parse_max_relative_target_cli(raw: str) -> float | dict[str, float] | None:
    """Convert CLI string (empty, scalar, or JSON object) into runtime ``max_relative_target``."""
    if not isinstance(raw, str):
        raise TypeError(f"max_relative_target must be str, got {type(raw)}")
    s = raw.strip()
    if not s:
        return None
    if s.startswith("{"):
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError(
                "robot.max_relative_target must be a float string or a JSON object mapping motor names to floats."
            )
        return {str(k): float(v) for k, v in parsed.items()}
    return float(s)


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different robots of the same type.
    # Use empty string to mean "no id" for CLI/Draccus compatibility.
    id: str = ""
    # Directory to store calibration file.
    # Kept as string for CLI/Draccus compatibility and converted to Path at runtime.
    calibration_dir: str = ""

    def __post_init__(self):
        # Convert CLI string to Path object if provided
        if self.calibration_dir:
            self.calibration_dir = Path(self.calibration_dir)
        else:
            # Normalise empty string to falsy value downstream
            self.calibration_dir = ""  # type: ignore[assignment]

        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
