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
class TeleoperatorConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration shared by every teleoperator.

    Concrete teleoperators subclass this and register themselves with
    `@TeleoperatorConfig.register_subclass("name")`, which is what makes `--teleop.type=name` work on the
    command line. Subclasses inherit the two fields below and must document them alongside their own.

    Args:
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    # Allows to distinguish between different teleoperators of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    @property
    def type(self) -> str:
        """Return the registered name this config was registered under.

        Returns:
            `str`: The name passed to `@TeleoperatorConfig.register_subclass`, e.g. `"so101_leader"`.
        """
        return self.get_choice_name(self.__class__)
