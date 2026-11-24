# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.rotation import Rotation

from .configuration_sana_robot import SanaRobotConfig
from .sana_robot_core.robot80 import (
    ROBOT_FIELDS,
    field_slots,
    gripper_closedness,
    rot6d_to_rotation,
    rotation_to_rot6d,
)
from .sana_robot_core.video import byte_frames, frames_to_model_range, resize_frames

FeatureMap = dict[PipelineFeatureType, dict[str, PolicyFeature]]

LIBERO_JOINT_POS = f"{OBS_STR}.joint_pos"
LIBERO_BASE_POS = f"{OBS_STR}.base_pos"
LIBERO_BASE_QUAT = f"{OBS_STR}.base_quat"
LIBERO_STATE_FIELDS = ("right_joints", "right_eef_position", "right_eef_rotation", "right_gripper")
LIBERO_ACTION_FIELDS = ("right_eef_position", "right_eef_rotation", "right_gripper")
# The distance between the two fingers of LIBERO's gripper, in metres, when it is open and when it is closed.
LIBERO_GRIPPER_OPEN, LIBERO_GRIPPER_CLOSED = 0.08, 0.0
# The simulated Panda of LIBERO. The checkpoint's poses are those of the flange: the frame of the last joint,
# 0.107 m along its axis, with the axes below. The hand body sits 0.1065 m along that axis and the grip site
# 0.097 m after the hand.
PANDA_FLANGE_HEIGHT = 0.107
PANDA_FLANGE_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
PANDA_HAND_HEIGHT, PANDA_HAND_QUATERNION = 0.1065, (0.0, 0.0, -0.383, 0.924)
PANDA_GRIP_SITE_HEIGHT, PANDA_GRIP_SITE_TURN = 0.097, -math.pi / 2
# What LIBERO-plus appends to the instruction of a task variant, from the variant's file name.
LIBERO_PLUS_TAG = re.compile(r" (?:(?:view|initstate|table|tb|light|noise|add) [-+]?\d|(?:moved )?level ?\d)")


@ProcessorStepRegistry.register(name="sana_robot_observations")
@dataclass
class SanaRobotObservations(ObservationProcessorStep):
    """Bring every camera frame to its view size as the training frames were resized, in [-1, 1]."""

    camera_keys: list[str]
    view_sizes: list[list[int]]
    resize: str

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for key, size in zip(self.camera_keys, self.view_sizes, strict=True):
            image = observation[key]
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{key} must be [B, 3, H, W]")
            if not image.is_floating_point():
                raise ValueError("Camera images must be floating point in [0, 1]")
            # The training frames were resized as float values and rounded back to 8 bits.
            frames = byte_frames(resize_frames(image.to(torch.float32, copy=True), size, self.resize))
            observation[key] = frames_to_model_range(frames)
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"camera_keys": self.camera_keys, "view_sizes": self.view_sizes, "resize": self.resize}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        features = {kind: dict(values) for kind, values in features.items()}
        for key, size in zip(self.camera_keys, self.view_sizes, strict=True):
            feature = features[PipelineFeatureType.OBSERVATION][key]
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=feature.type, shape=(3, *size)
            )
        return features


def make_sana_robot_pre_post_processors(
    config: SanaRobotConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build the processors of SANA-Robot. ``dataset_stats`` holds, under ``mean`` and ``std``, the center and
    scale of the robot slots of `observation.state` and of `action`."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    pre = [
        steps.rename_observations,
        steps.add_batch_dim,
        SanaRobotObservations(config.camera_keys, config.view_sizes, config.resize),
        steps.normalize,
        steps.to_device,
    ]
    return make_policy_processor_pipelines(input_steps=pre, output_steps=[steps.unnormalize, steps.to_cpu])


def robot_slots(fields: tuple[str, ...]) -> list[int]:
    """Return the slots of the 80-slot robot row that the named fields fill, in order."""
    return [slot for name in fields for slot in field_slots(name)]


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return the [4, 4] pose of a position [3] and a rotation matrix [3, 3]."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def panda_hand_pose() -> np.ndarray:
    """Return the fixed [4, 4] pose of the hand of LIBERO's Panda, the body whose rotation LIBERO observes, in the
    flange frame."""
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_quat(np.asarray(PANDA_HAND_QUATERNION)).as_matrix()
    hand[2, 3] = PANDA_HAND_HEIGHT - PANDA_FLANGE_HEIGHT
    axes = np.eye(4)
    axes[:3, :3] = PANDA_FLANGE_AXES
    return axes.T @ hand


def panda_grip_site_pose() -> np.ndarray:
    """Return the fixed [4, 4] pose of the grip site of LIBERO's Panda, the point LIBERO's controller moves, in
    the flange frame."""
    site = np.eye(4)
    site[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, PANDA_GRIP_SITE_TURN])).as_matrix()
    site[2, 3] = PANDA_GRIP_SITE_HEIGHT
    return panda_hand_pose() @ site


@dataclass
class LiberoFrames:
    """The pose [4, 4] of the robot base in the world of the running LIBERO episode, which the observation step
    writes and the action step reads."""

    world_from_base: np.ndarray | None = None


@ProcessorStepRegistry.register(name="sana_robot_libero_observation")
@dataclass
class SanaRobotLiberoObservation(ObservationProcessorStep):
    """Turn a LIBERO observation into the state of the SANA-Robot LIBERO checkpoint.

    LIBERO gives the position of the grip site and the rotation of the hand in the world, the two finger
    positions and, with ``--env.obs_type=pixels_agent_pos_joints``, the joint positions and the pose of the robot
    base in the world. The checkpoint reads the joint positions, the pose of the flange in the frame of the base
    and the gripper closedness.
    """

    frames: LiberoFrames = field(default_factory=LiberoFrames)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        missing = [
            key for key in (LIBERO_JOINT_POS, LIBERO_BASE_POS, LIBERO_BASE_QUAT) if key not in observation
        ]
        if missing:
            raise ValueError(
                f"SANA-Robot needs {missing} of LIBERO: use --env.obs_type=pixels_agent_pos_joints"
            )
        state = observation[OBS_STATE]
        if state.ndim != 2 or tuple(state.shape) != (1, 8):
            raise ValueError("SANA-Robot runs one LIBERO environment at a time: use --eval.batch_size=1")
        values = state[0].double().cpu().numpy()
        joints = observation.pop(LIBERO_JOINT_POS)[0].double().cpu().numpy()
        base_position = observation.pop(LIBERO_BASE_POS)[0].double().cpu().numpy()
        base_rotation = Rotation.from_quat(
            observation.pop(LIBERO_BASE_QUAT)[0].double().cpu().numpy()
        ).as_matrix()
        self.frames.world_from_base = pose_matrix(base_position, base_rotation)
        # The observed rotation is the hand's and the observed position the grip site's.
        rotation = (
            base_rotation.T @ Rotation.from_rotvec(values[3:6]).as_matrix() @ panda_hand_pose()[:3, :3].T
        )
        position = base_rotation.T @ (values[:3] - base_position) - rotation @ panda_grip_site_pose()[:3, 3]
        closedness = gripper_closedness(values[6] - values[7], LIBERO_GRIPPER_OPEN, LIBERO_GRIPPER_CLOSED)
        row = np.concatenate((joints, position, rotation_to_rot6d(rotation), [closedness]))
        observation[OBS_STATE] = torch.from_numpy(row.astype(np.float32))[None].to(state)
        return observation

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        features = {kind: dict(values) for kind, values in features.items()}
        observations = features.get(PipelineFeatureType.OBSERVATION, {})
        for key in (LIBERO_JOINT_POS, LIBERO_BASE_POS, LIBERO_BASE_QUAT):
            observations.pop(key, None)
        if OBS_STATE in observations:
            width = len(robot_slots(LIBERO_STATE_FIELDS))
            observations[OBS_STATE] = replace(observations[OBS_STATE], shape=(width,))
        return features


@ProcessorStepRegistry.register(name="sana_robot_libero_instruction")
class SanaRobotLiberoInstruction(ComplementaryDataProcessorStep):
    """Remove the perturbation tags of LIBERO-plus from the instruction.

    LIBERO-plus names the instruction of a task variant after its file, so "put the bowl on the plate" comes as
    "put the bowl on the plate view 0 0 100 0 0 initstate 1". The checkpoint is evaluated on the instruction
    without the tags; the instructions of LIBERO and the rewritten instructions of LIBERO-plus have none.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get("task"):
            return transition
        return super().__call__(transition)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks = complementary_data["task"]
        cleaned = [libero_instruction(task) for task in ([tasks] if isinstance(tasks, str) else tasks)]
        return {**complementary_data, "task": cleaned[0] if isinstance(tasks, str) else cleaned}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        return features


def libero_instruction(instruction: str) -> str:
    """Return a LIBERO instruction without the perturbation tags LIBERO-plus appends to it."""
    tag = LIBERO_PLUS_TAG.search(instruction)
    return instruction[: tag.start()].rstrip() if tag else instruction


@ProcessorStepRegistry.register(name="sana_robot_libero_action")
@dataclass
class SanaRobotLiberoAction(PolicyActionProcessorStep):
    """Turn an action of the SANA-Robot LIBERO checkpoint into the command of LIBERO's absolute control mode.

    The action is the target pose of the flange in the frame of the robot base, as a position and a 6D rotation,
    and the gripper closedness. The command is the position and the rotation vector of the grip site in the world, and
    -1 to open the gripper or 1 to close it. Run LIBERO with ``--env.control_mode=absolute``.
    """

    frames: LiberoFrames = field(default_factory=LiberoFrames)

    def action(self, action: PolicyAction) -> PolicyAction:
        if action.ndim != 2 or tuple(action.shape) != (1, 10):
            raise ValueError(
                f"Expected one action of the SANA-Robot LIBERO checkpoint [1, 10], got {tuple(action.shape)}"
            )
        if self.frames.world_from_base is None:
            raise ValueError("The LIBERO observation step has to run before the action step")
        values = action[0].double().cpu().numpy()
        row = np.zeros(80)
        row[robot_slots(LIBERO_ACTION_FIELDS)] = values
        goal = pose_matrix(
            row[ROBOT_FIELDS["right_eef_position"]],
            rot6d_to_rotation(row[ROBOT_FIELDS["right_eef_rotation"]]),
        )
        goal = self.frames.world_from_base @ goal @ panda_grip_site_pose()
        closedness = float(np.clip(row[ROBOT_FIELDS["right_gripper"]][0], 0.0, 1.0))
        command = np.concatenate(
            (
                goal[:3, 3],
                Rotation.from_matrix(goal[:3, :3]).as_rotvec(),
                [-1.0 if 2.0 * closedness - 1.0 < 0 else 1.0],
            )
        )
        return torch.from_numpy(command.astype(np.float32))[None].to(action.device)

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        actions = features.get(PipelineFeatureType.ACTION, {})
        if ACTION not in actions:
            return features
        return {
            **features,
            PipelineFeatureType.ACTION: {**actions, ACTION: replace(actions[ACTION], shape=(7,))},
        }


def make_sana_robot_libero_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Env processors of LIBERO for the SANA-Robot LIBERO checkpoint: the observation becomes the checkpoint's
    state, the instruction loses the tags of LIBERO-plus, and the predicted action becomes the command of LIBERO's
    absolute control mode."""
    frames = LiberoFrames()
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[SanaRobotLiberoObservation(frames), SanaRobotLiberoInstruction()]
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](steps=[SanaRobotLiberoAction(frames)]),
    )
