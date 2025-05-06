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
from pathlib import Path
from typing import Any

import draccus

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.common.motors.motors_bus import MotorCalibration

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """The main LeRobot class for implementing teleoperation devices."""

    # Set these in ALL subclasses
    config_class: TeleoperatorConfig
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """Connects to the teleoperator."""
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Calibrates the teleoperator."""
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        """Gets the action to send to a teleoperator."""
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Sends feedback captured from a robot to the teleoperator."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the teleoperator."""
        pass
