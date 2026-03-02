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

from lerobot.configs.types import PipelineFeatureType
from lerobot.motors import MotorCalibration
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.factory import _make_identity_observation_pipeline, _make_identity_robot_action_pipeline
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible robots.

    This class provides a standardized interface for interacting with physical robots.
    Subclasses must implement all abstract methods and properties to be usable.

    Pipelines are first-class citizens: every robot carries an optional output pipeline
    (applied in get_observation()) and an optional input pipeline (applied in send_action()).
    Both default to identity (no-op), so existing robots work without any changes.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this robot.
        name (str): The unique robot name used to identify this robot type.
    """

    # Set these in ALL subclasses
    config_class: builtins.type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
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

        # Pipeline interface — default to identity (no-op), swap via set_output/input_pipeline()
        self._output_pipeline: RobotProcessorPipeline = _make_identity_observation_pipeline()
        self._input_pipeline: RobotProcessorPipeline = _make_identity_robot_action_pipeline()
        # Cache of most recent raw observation; used by input_pipeline for IK initial guess
        self._last_raw_obs: RobotObservation = {}

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

    # ── Pipeline interface ────────────────────────────────────────────────────

    def output_pipeline(self) -> RobotProcessorPipeline:
        """
        Pipeline applied inside get_observation() to transform raw hardware observations.
        Default: identity (no-op). Override via set_output_pipeline() or subclassing.

        Example: set a forward-kinematics pipeline to convert joint positions to EE pose.
        """
        return self._output_pipeline

    def input_pipeline(self) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
        """
        Pipeline applied inside send_action() to transform incoming actions before hardware write.
        Default: identity (no-op). Override via set_input_pipeline() or subclassing.

        The pipeline receives a (action, last_raw_obs) tuple so IK solvers can use the
        current joint configuration as an initial guess.

        Example: set an inverse-kinematics pipeline to convert EE commands to joint positions.
        """
        return self._input_pipeline

    def set_output_pipeline(self, pipeline: RobotProcessorPipeline) -> None:
        """Set the observation output pipeline (applied in get_observation())."""
        self._output_pipeline = pipeline

    def set_input_pipeline(self, pipeline: RobotProcessorPipeline) -> None:
        """Set the action input pipeline (applied in send_action())."""
        self._input_pipeline = pipeline

    # ── Feature properties ────────────────────────────────────────────────────

    @property
    def observation_features(self) -> dict:
        """
        Pipeline-transformed observation features.

        Applies output_pipeline().transform_features() to raw_observation_features so the
        returned dict matches what get_observation() actually returns to callers.

        Use raw_observation_features to inspect hardware-level feature shapes.

        Note: this property should be able to be called regardless of whether the robot
        is connected or not.
        """
        from lerobot.datasets.pipeline_features import create_initial_features  # lazy import

        initial = create_initial_features(observation=self.raw_observation_features)
        transformed = self.output_pipeline().transform_features(initial)
        return transformed.get(PipelineFeatureType.OBSERVATION, {})

    @property
    @abc.abstractmethod
    def raw_observation_features(self) -> dict:
        """
        Hardware-level observation features (before any pipeline transformation).

        A dictionary describing the structure and types of the observations produced
        directly by the robot hardware. Its structure (keys) should match the structure
        of what is returned by :pymeth:`_get_observation`. Values should be:
            - The type if it's a simple value, e.g. ``float`` for joint position
            - A tuple representing the shape for array values, e.g. ``(H, W, C)`` for images

        Note: this property should be able to be called regardless of whether the robot
        is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        Its structure (keys) should match the structure of what is passed to
        :pymeth:`send_action`. Values for the dict should be the type of the value if it's
        a simple value, e.g. ``float`` for single proprioceptive value
        (a joint's goal position/velocity).

        For simple robots (no input pipeline), this returns motor-level features.
        For robots with an IK pipeline, override this to return the pipeline's input spec
        (e.g., EE features) so that compatibility checks work correctly.

        Note: this property should be able to be called regardless of whether the robot
        is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If ``False``, calling
        :pymeth:`get_observation` or :pymeth:`send_action` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always ``True`` if not applicable"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to load calibration data from the specified file.

        Args:
            fpath (Path | None): Optional path to the calibration file. Defaults to ``self.calibration_fpath``.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Helper to save calibration data to the specified file.

        Args:
            fpath (Path | None): Optional path to save the calibration file. Defaults to ``self.calibration_fpath``.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    # ── Template methods (concrete, call pipeline internally) ─────────────────

    def get_observation(self) -> RobotObservation:
        """
        Retrieve the current observation from the robot and apply the output pipeline.

        Calls :pymeth:`_get_observation` to get raw hardware data, caches it for use as
        IK initial guess in :pymeth:`send_action`, then applies :pymeth:`output_pipeline`.

        Returns:
            RobotObservation: Pipeline-transformed observation. With the default identity
                pipeline this equals the raw observation from :pymeth:`_get_observation`.
        """
        raw = self._get_observation()
        self._last_raw_obs = raw
        return self.output_pipeline()(raw)

    @abc.abstractmethod
    def _get_observation(self) -> RobotObservation:
        """
        Retrieve the raw observation directly from robot hardware.

        Returns:
            RobotObservation: A flat dictionary representing the robot's current sensory
                state. Its structure should match :pymeth:`raw_observation_features`.
        """
        pass

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Apply the input pipeline and send the resulting action to robot hardware.

        The input pipeline receives ``(action, last_raw_obs)`` so IK solvers can use the
        cached joint configuration as an initial guess. With the default identity pipeline,
        the action is forwarded unchanged.

        Args:
            action (RobotAction): Dictionary representing the desired action. Its structure
                should match :pymeth:`action_features`.

        Returns:
            RobotAction: The action actually sent to the motors, potentially clipped or
                modified by the pipeline or hardware safety limits.
        """
        transformed = self.input_pipeline()((action, self._last_raw_obs))
        return self._send_action(transformed)

    @abc.abstractmethod
    def _send_action(self, action: RobotAction) -> RobotAction:
        """
        Send an action command directly to robot hardware.

        Args:
            action (RobotAction): Dictionary of motor-level commands. Its structure should
                match what the hardware expects (typically motor positions/velocities).

        Returns:
            RobotAction: The action actually sent, potentially clipped by safety limits.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass
