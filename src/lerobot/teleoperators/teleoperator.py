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
import builtins
from pathlib import Path
from typing import Any

import draccus

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible teleoperation devices.

    This class provides a standardized interface for interacting with physical teleoperators.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this teleoperator.
        name (str): The unique name used to identify this teleoperator type.
    """

    # Set these in ALL subclasses
    config_class: builtins.type[TeleoperatorConfig]
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

    def __enter__(self):
        """
        Context manager entry.
        Automatically connects to the camera.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit.
        Automatically disconnects, ensuring resources are released even on error.
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        Destructor safety net.
        Attempts to disconnect if the object is garbage collected without cleanup.
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions produced by the teleoperator. Its
        structure (keys) should match the structure of what is returned by :pymeth:`get_action`. Values for
        the dict should be the type of the value if it's a simple value, e.g. `float` for single
        proprioceptive value (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        """
        A dictionary describing the structure and types of the feedback actions expected by the robot. Its
        structure (keys) should match the structure of what is passed to :pymeth:`send_feedback`. Values for
        the dict should be the type of the value if it's a simple value, e.g. `float` for single
        proprioceptive value (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not. If `False`, calling :pymeth:`get_action`
        or :pymeth:`send_feedback` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the teleoperator.

        Args:
            calibrate (bool): If True, automatically calibrate the teleoperator after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        Calibrate the teleoperator if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to load calibration data from the specified file.

        Args:
            fpath (Path | None): Optional path to the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to save calibration data to the specified file.

        Args:
            fpath (Path | None): Optional path to save the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    @abc.abstractmethod
    def get_action(self) -> RobotAction:
        """
        Retrieve the current action from the teleoperator.

        Returns:
            RobotAction: A flat dictionary representing the teleoperator's current actions. Its
                structure should match :pymeth:`observation_features`.
        """
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Send a feedback action command to the teleoperator.

        Args:
            feedback (dict[str, Any]): Dictionary representing the desired feedback. Its structure should match
                :pymeth:`feedback_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the teleoperator and perform any necessary cleanup."""
        pass
