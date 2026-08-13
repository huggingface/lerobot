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
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration shared by every robot.

    Concrete robots subclass this and register themselves with
    `@RobotConfig.register_subclass("name")`, which is what makes `--robot.type=name` work on the command
    line. Subclasses inherit the two fields below and must document them alongside their own.

    Args:
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several robots of the same type. It
            also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-robot directory under the
            LeRobot calibration home.
    """

    # Allows to distinguish between different robots of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    def __post_init__(self):
        """Validate that every configured camera specifies the fields a robot requires.

        Raises:
            ValueError: If a camera does not set `width`, `height` and `fps`. A robot records frames at a
                fixed shape, so these cannot be left to the driver's defaults.
        """
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )

    @property
    def type(self) -> str:
        """The registered name of this robot type.

        Returns:
            `str`: The name passed to `@RobotConfig.register_subclass`, e.g. `"so101_follower"`. This is
            what `make_robot_from_config` dispatches on and what a user writes as `--robot.type=...`.
        """
        return self.get_choice_name(self.__class__)
