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

from __future__ import annotations

import abc
import builtins
from pathlib import Path
from typing import TYPE_CHECKING, Any

import draccus

from lerobot.configs.types import PipelineFeatureType
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

if TYPE_CHECKING:
    from lerobot.processor.core import RobotAction
    from lerobot.processor.pipeline import RobotProcessorPipeline

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible teleoperation devices.

    This class provides a standardized interface for interacting with physical teleoperators.
    Subclasses must implement all abstract methods and properties to be usable.

    Pipelines are first-class citizens: every teleoperator carries an optional output pipeline
    (applied in get_action()) and an optional input pipeline (applied in send_feedback()).
    Both default to identity (no-op), so existing teleoperators work without any changes.

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

        # Pipeline interface — default to identity (no-op), swap via set_output/input_pipeline()
        # Lazy import: factory is in lerobot.processor which loads after teleoperators at module init time,
        # but __init__ runs at instance-creation time when lerobot.processor is fully loaded.
        from lerobot.processor.factory import _make_identity_feedback_pipeline, _make_identity_teleop_action_pipeline

        self._output_pipeline: RobotProcessorPipeline = _make_identity_teleop_action_pipeline()
        self._input_pipeline: RobotProcessorPipeline = _make_identity_feedback_pipeline()

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
        Pipeline applied inside get_action() to transform raw hardware actions.
        Default: identity (no-op). Override via set_output_pipeline() or subclassing.

        Example: set a forward-kinematics pipeline to convert leader joint positions to EE pose.
        """
        return self._output_pipeline

    def input_pipeline(self) -> RobotProcessorPipeline:
        """
        Pipeline applied inside send_feedback() to transform incoming feedback.
        Default: identity (no-op). Override via set_input_pipeline() or subclassing.
        """
        return self._input_pipeline

    def set_output_pipeline(self, pipeline: RobotProcessorPipeline) -> None:
        """Set the action output pipeline (applied in get_action())."""
        self._output_pipeline = pipeline

    def set_input_pipeline(self, pipeline: RobotProcessorPipeline) -> None:
        """Set the feedback input pipeline (applied in send_feedback())."""
        self._input_pipeline = pipeline

    # ── Feature properties ────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        """
        Pipeline-transformed action features.

        Applies output_pipeline().transform_features() to raw_action_features so the
        returned dict matches what get_action() actually produces for callers.

        Use raw_action_features to inspect hardware-level feature shapes.

        Note: this property should be able to be called regardless of whether the
        teleoperator is connected or not.
        """
        from lerobot.datasets.pipeline_features import create_initial_features  # lazy import

        initial = create_initial_features(action=self.raw_action_features)
        transformed = self.output_pipeline().transform_features(initial)
        return transformed.get(PipelineFeatureType.ACTION, {})

    @property
    @abc.abstractmethod
    def raw_action_features(self) -> dict:
        """
        Hardware-level action features (before any pipeline transformation).

        A dictionary describing the structure and types of the actions produced
        directly by the teleoperator hardware. Its structure (keys) should match
        the structure of what is returned by :pymeth:`_get_action`. Values should be
        the type of the value if it's a simple value, e.g. ``float`` for single
        proprioceptive value (a joint's goal position/velocity).

        Note: this property should be able to be called regardless of whether the
        teleoperator is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def raw_feedback_features(self) -> dict:
        """
        Hardware-level feedback features (before any pipeline transformation).

        A dictionary describing the structure and types of the feedback accepted directly
        by the teleoperator hardware (i.e. what :pymeth:`_send_feedback` receives). Its
        structure (keys) should match the structure of what is expected by
        :pymeth:`_send_feedback`. Values should be the type of the value if it's a simple
        value, e.g. ``float`` for single proprioceptive value.

        Return an empty dict if this teleoperator does not support feedback.

        Note: this property should be able to be called regardless of whether the
        teleoperator is connected or not.
        """
        pass

    @property
    def feedback_features(self) -> dict:
        """
        Pipeline-transformed feedback features.

        Applies input_pipeline().transform_features() to raw_feedback_features so the
        returned dict reflects what the input pipeline outputs to the teleoperator hardware.

        Use raw_feedback_features to inspect hardware-level feedback feature shapes.

        Note: this property should be able to be called regardless of whether the
        teleoperator is connected or not.
        """
        from lerobot.datasets.pipeline_features import create_initial_features  # lazy import

        initial = create_initial_features(observation=self.raw_feedback_features)
        transformed = self.input_pipeline().transform_features(initial)
        return transformed.get(PipelineFeatureType.OBSERVATION, {})

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not. If ``False``, calling
        :pymeth:`get_action` or :pymeth:`send_feedback` should raise an error.
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
        """Whether the teleoperator is currently calibrated or not. Should be always ``True`` if not applicable"""
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
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    # ── Template methods (concrete, call pipeline internally) ─────────────────

    def get_action(self) -> RobotAction:
        """
        Retrieve the current action from the teleoperator and apply the output pipeline.

        Calls :pymeth:`_get_action` to get raw hardware data, then applies
        :pymeth:`output_pipeline`.

        Returns:
            RobotAction: Pipeline-transformed action. With the default identity pipeline
                this equals the raw action from :pymeth:`_get_action`.
        """
        raw = self._get_action()
        return self.output_pipeline()(raw)

    @abc.abstractmethod
    def _get_action(self) -> RobotAction:
        """
        Retrieve the raw action directly from teleoperator hardware.

        Returns:
            RobotAction: A flat dictionary representing the teleoperator's current actions.
                Its structure should match :pymeth:`raw_action_features`.
        """
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Apply the input pipeline and send the resulting feedback to teleoperator hardware.

        Args:
            feedback (dict[str, Any]): Dictionary representing the desired feedback.
                Its structure should match :pymeth:`feedback_features`.
        """
        transformed = self.input_pipeline()(feedback)
        self._send_feedback(transformed)

    @abc.abstractmethod
    def _send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Send feedback directly to teleoperator hardware.

        Args:
            feedback (dict[str, Any]): Dictionary of hardware-level feedback commands.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the teleoperator and perform any necessary cleanup."""
        pass
