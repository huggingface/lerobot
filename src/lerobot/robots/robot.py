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

import draccus

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """The base abstract class for all LeRobot-compatible robots.

    This class provides a standardized interface for interacting with physical robots. Subclasses must
    implement all abstract methods and properties to be usable.

    Used as a context manager, a robot connects on entry and disconnects on exit even if the body raises:

    ```python
    >>> with SO101Follower(config) as robot:  # doctest: +SKIP
    ...     obs = robot.get_observation()
    ...     robot.send_action(action)
    ```

    **Attributes**:
        - **config_class** (`type[RobotConfig]`) -- The expected configuration class for this robot.
        - **name** (`str`) -- The unique robot name used to identify this robot type.
    """

    # Set these in ALL subclasses
    config_class: builtins.type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        """Set up identity and calibration paths, loading an existing calibration file if there is one.

        Args:
            config (`RobotConfig`):
                The robot's configuration. Its `id` and `calibration_dir` decide where calibration is
                read from and written to.
        """
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        """Return this robot's id and class name, e.g. `"my_arm SO101Follower"`.

        Returns:
            `str`: A short identifier used in log messages.
        """
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):
        """Context manager entry. Automatically connects to the robot."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit. Disconnects, ensuring resources are released even on error."""
        self.disconnect()

    def __del__(self) -> None:
        """Destructor safety net. Disconnects if the object is garbage collected without cleanup."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """A dictionary describing the structure and types of the observations produced by the robot.

        Its keys should match the structure of what is returned by [`~robots.Robot.get_observation`]. Values
        should either be:

        - the type of the value if it's a simple value, e.g. `float` for a single proprioceptive value
          (a joint's position or velocity)
        - a tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for
          images

        > [!NOTE]
        > This property must be callable regardless of whether the robot is connected.

        Returns:
            `dict`: Observation names mapped to their type or shape.
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """A dictionary describing the structure and types of the actions expected by the robot.

        Its keys should match the structure of what is passed to [`~robots.Robot.send_action`]. Values should
        be the type of the value if it's a simple value, e.g. `float` for a single proprioceptive value
        (a joint's goal position or velocity).

        > [!NOTE]
        > This property must be callable regardless of whether the robot is connected.

        Returns:
            `dict`: Action names mapped to their type or shape.
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the robot is currently connected.

        If `False`, calling [`~robots.Robot.get_observation`] or [`~robots.Robot.send_action`] should raise
        an error.

        Returns:
            `bool`: `True` if communication with the robot is established.
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """Establish communication with the robot.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to automatically calibrate the robot after connecting, if it is not calibrated or
                needs recalibration. Whether calibration is needed is hardware-dependent.
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated.

        Returns:
            `bool`: `True` if the robot is calibrated. Always `True` for robots where calibration does not
            apply.
        """
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g. motor offsets) and update the `calibration`
        dictionary accordingly.
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """Helper to load calibration data from the specified file.

        Args:
            fpath (`Path`, *optional*):
                Path to the calibration file. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """Helper to save calibration data to the specified file.

        Args:
            fpath (`Path`, *optional*):
                Path to save the calibration file to. Defaults to `self.calibration_fpath`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """Apply any one-time or runtime configuration to the robot.

        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> RobotObservation:
        """Retrieve the current observation from the robot.

        Returns:
            `dict[str, Any]`: A flat dictionary representing the robot's current sensory state. Its structure
            should match [`~robots.Robot.observation_features`].

        Raises:
            DeviceNotConnectedError: If [`~robots.Robot.connect`] has not been called.
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send an action command to the robot.

        Args:
            action (`dict[str, Any]`):
                The desired action. Its structure should match [`~robots.Robot.action_features`].

        Returns:
            `dict[str, Any]`: The action actually sent to the motors, potentially clipped or modified, e.g.
            by safety limits on velocity. Prefer this over the requested action when logging or recording.

        Raises:
            DeviceNotConnectedError: If [`~robots.Robot.connect`] has not been called.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass
