#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""State and action processors, in the same two places as ``lerobot-record`` puts its own.

- Teleop side (``make_command_pipeline``): what the scripted "operator" commands, before it is executed.
  ``CommandDrift`` and ``CommandJerk`` make the hand less steady; the simulated arm executes the result.
- Robot side (``make_recording_pipelines``): how a real follower would report states and commands.
  ``CalibratedJointReadings``/``CalibratedJointCommands`` map them through a ``VirtualCalibration``,
  sampled per episode the way a person calibrates an SO-101; ``JointReadingNoise`` adds encoder noise.

Steps that sample something per episode implement ``start_episode(rng)``; ``start_episode`` below calls
it on every step of a pipeline. The calibration shared by the two robot-side steps is sampled separately,
with ``VirtualCalibration.start_episode``.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)

from .config import CalibrationConfig, NoiseConfig
from .so101 import ARM_JOINTS, GRIPPER, GRIPPER_RANGE_DEG, JOINT_LIMITS_DEG, JOINT_NAMES, JOINTS

# STS3215: 12-bit encoder over a full turn. Calibration sets the homing pose to mid-scale.
MAX_RES = 4095
HALF_TURN = 2047
TICKS_PER_DEG = MAX_RES / 360
# Joints swept over a full turn during calibration, whose zero is the homing pose.
FULL_TURN = ("wrist_roll",)


def to_vector(values: dict[str, Any]) -> np.ndarray:
    return np.array([float(values[name]) for name in JOINT_NAMES])


def to_dict(vector: np.ndarray, into: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {} if into is None else into
    out.update({name: float(v) for name, v in zip(JOINT_NAMES, vector, strict=True)})
    return out


@dataclass
class VirtualCalibration:
    """One SO-101 calibration, sampled the way a person produces one.

    Calibrating puts the arm in a middle pose (``homing``, which reads mid-scale), then sweeps each
    joint between its stops (``range_min``/``range_max``). A follower then reports degrees from the middle
    of the sweep, and the gripper in percent of its sweep. Here physical angles are the simulator's, so a
    sampled calibration is how one real arm would report the simulated motion.
    """

    cfg: CalibrationConfig = field(default_factory=CalibrationConfig)
    # Physical joint angles, in degrees, of the homing pose and of the sweep's ends.
    homing: np.ndarray = field(default_factory=lambda: np.zeros(len(JOINTS)))
    low: np.ndarray = field(default_factory=lambda: np.array([lo for lo, _ in JOINT_LIMITS_DEG]))
    high: np.ndarray = field(default_factory=lambda: np.array([hi for _, hi in JOINT_LIMITS_DEG]))

    def __post_init__(self) -> None:
        self._update_ranges()

    def start_episode(self, rng: np.random.Generator) -> None:
        cfg = self.cfg
        limits = np.array(JOINT_LIMITS_DEG)
        self.homing = rng.normal(0, cfg.homing_std, len(JOINTS))
        self.low = limits[:, 0] + rng.normal(0, cfg.range_end_std, len(JOINTS))
        self.high = limits[:, 1] + rng.normal(0, cfg.range_end_std, len(JOINTS))
        self.low[GRIPPER] = GRIPPER_RANGE_DEG[0] + rng.normal(0, cfg.gripper_closed_std)
        self.high[GRIPPER] = GRIPPER_RANGE_DEG[1] + rng.normal(0, cfg.gripper_open_std)
        self._update_ranges()

    def _update_ranges(self) -> None:
        self.range_min, self.range_max = self.raw(self.low), self.raw(self.high)
        for i, joint in enumerate(JOINTS):
            if joint in FULL_TURN:
                self.range_min[i], self.range_max[i] = 0, MAX_RES

    def raw(self, angles: np.ndarray, noise_ticks: np.ndarray | float = 0.0) -> np.ndarray:
        """Encoder ticks for physical angles (degrees), with the homing pose at mid-scale."""
        return np.round(HALF_TURN + (angles - self.homing) * TICKS_PER_DEG + noise_ticks).clip(0, MAX_RES)

    def motor_calibration(self) -> dict[str, MotorCalibration]:
        """The calibration as ``lerobot`` stores it; the homing offset lives in the motor's firmware."""
        return {
            joint: MotorCalibration(
                id=i + 1,
                drive_mode=0,
                homing_offset=0,
                range_min=int(self.range_min[i]),
                range_max=int(self.range_max[i]),
            )
            for i, joint in enumerate(JOINTS)
        }

    def report(self, values: np.ndarray, noise_ticks: np.ndarray | float = 0.0) -> np.ndarray:
        """What the follower reports for nominal joint values (degrees, gripper 0-100).

        The arithmetic of ``MotorsBus._normalize`` on whole encoder ticks: DEGREES mode for the arm,
        RANGE_0_100 for the gripper.
        """
        angles = np.array(values, dtype=np.float64)
        lo, hi = GRIPPER_RANGE_DEG
        angles[GRIPPER] = lo + values[GRIPPER] / 100 * (hi - lo)
        raw = self.raw(angles, noise_ticks)
        out = (raw - (self.range_min + self.range_max) / 2) * 360 / MAX_RES
        g_min, g_max = self.range_min[GRIPPER], self.range_max[GRIPPER]
        out[GRIPPER] = (min(g_max, max(g_min, raw[GRIPPER])) - g_min) / (g_max - g_min) * 100
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "homing": self.homing.tolist(),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "motors": {joint: vars(cal) for joint, cal in self.motor_calibration().items()},
        }


@ProcessorStepRegistry.register("so101_datagen_calibrated_joint_readings")
@dataclass
class CalibratedJointReadings(ObservationProcessorStep):
    """Joint state as a calibrated follower reads it, with encoder noise in ticks."""

    calibration: VirtualCalibration = field(default_factory=VirtualCalibration)
    noise_ticks: float = 0.0
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, init=False, repr=False)

    def start_episode(self, rng: np.random.Generator) -> None:
        self._rng = np.random.default_rng(rng.integers(2**63))

    def observation(self, observation: RobotObservation) -> RobotObservation:
        noise = self._rng.normal(0, self.noise_ticks, len(JOINTS)) if self.noise_ticks else 0.0
        return to_dict(self.calibration.report(to_vector(observation), noise), observation)

    def get_config(self) -> dict[str, Any]:
        return {"noise_ticks": self.noise_ticks}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("so101_datagen_calibrated_joint_commands")
@dataclass
class CalibratedJointCommands(RobotActionProcessorStep):
    """Joint commands in the units of a calibrated follower (the same calibration as the readings)."""

    calibration: VirtualCalibration = field(default_factory=VirtualCalibration)

    def action(self, action: RobotAction) -> RobotAction:
        return to_dict(self.calibration.report(to_vector(action)), action)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("so101_datagen_command_drift")
@dataclass
class CommandDrift(RobotActionProcessorStep):
    """A slow Ornstein-Uhlenbeck drift on every joint command, like an unsteady hand on a leader arm."""

    std: float = 0.6
    tau: float = 0.3
    fps: int = 30
    _state: np.ndarray = field(default_factory=lambda: np.zeros(len(JOINTS)), init=False, repr=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, init=False, repr=False)

    def start_episode(self, rng: np.random.Generator) -> None:
        self._rng = np.random.default_rng(rng.integers(2**63))
        self._state = self._rng.normal(0, self.std, len(JOINTS))

    def action(self, action: RobotAction) -> RobotAction:
        dt = 1 / self.fps
        self._state += -self._state * dt / self.tau + self.std * np.sqrt(
            2 * dt / self.tau
        ) * self._rng.normal(size=len(JOINTS))
        return to_dict(to_vector(action) + self._state, action)

    def get_config(self) -> dict[str, Any]:
        return {"std": self.std, "tau": self.tau, "fps": self.fps}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("so101_datagen_command_jerk")
@dataclass
class CommandJerk(RobotActionProcessorStep):
    """Rare short bumps on one arm joint's command (half a sine over a few frames)."""

    rate: float = 0.2
    amplitude: tuple[float, float] = (2.0, 4.0)
    frames: tuple[int, int] = (2, 5)
    fps: int = 30
    _bump: list[tuple[int, np.ndarray]] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, init=False, repr=False)

    def start_episode(self, rng: np.random.Generator) -> None:
        self._rng = np.random.default_rng(rng.integers(2**63))
        self._bump = []

    def action(self, action: RobotAction) -> RobotAction:
        rng = self._rng
        if not self._bump and rng.random() < self.rate / self.fps:
            n = int(rng.integers(self.frames[0], self.frames[1] + 1))
            joint = int(rng.integers(ARM_JOINTS))
            amplitude = rng.uniform(*self.amplitude) * rng.choice([-1.0, 1.0])
            for k, s in enumerate(np.sin(np.pi * (np.arange(n) + 1) / (n + 1))):
                delta = np.zeros(len(JOINTS))
                delta[joint] = amplitude * s
                self._bump.append((k, delta))
        if not self._bump:
            return action
        _, delta = self._bump.pop(0)
        return to_dict(to_vector(action) + delta, action)

    def get_config(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "amplitude": list(self.amplitude),
            "frames": list(self.frames),
            "fps": self.fps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_command_pipeline(cfg: NoiseConfig, fps: int) -> RobotProcessorPipeline:
    """Teleop side: planned command (and current observation) to executed command."""
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            CommandDrift(std=cfg.drift_std, tau=cfg.drift_tau, fps=fps),
            CommandJerk(rate=cfg.jerk_rate, amplitude=cfg.jerk_amplitude, frames=cfg.jerk_frames, fps=fps),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="so101_datagen_command",
    )


def make_recording_pipelines(
    cfg: CalibrationConfig,
) -> tuple[RobotProcessorPipeline, RobotProcessorPipeline, VirtualCalibration]:
    """Robot side: (observation pipeline, action pipeline) sharing one calibration."""
    calibration = VirtualCalibration(cfg)
    observation = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[CalibratedJointReadings(calibration=calibration, noise_ticks=cfg.reading_noise_ticks)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
        name="so101_datagen_robot_observation",
    )
    action = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[CalibratedJointCommands(calibration=calibration)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="so101_datagen_robot_action",
    )
    return observation, action, calibration


def start_episode(rng: np.random.Generator, *pipelines: RobotProcessorPipeline) -> None:
    """Reset every step, then let the ones that sample per episode draw from ``rng``."""
    for pipeline in pipelines:
        pipeline.reset()
        for step in pipeline.steps:
            if hasattr(step, "start_episode"):
                step.start_episode(rng)
