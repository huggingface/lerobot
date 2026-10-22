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

"""Native simulator fixtures and a shared recipe using the normal LeRobot entry points."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.lerobot_types import RobotAction, RobotObservation, TransitionKey
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.robots.sim import SimRobot, SimRobotConfig

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


PANDA_STATE_NAMES = (
    *(f"joint_{i}.pos" for i in range(1, 8)),
    *(f"joint_{i}.vel" for i in range(1, 8)),
    "gripper.command",
    "eef.x",
    "eef.y",
    "eef.z",
)
PANDA_ACTION_NAMES = (
    "eef.delta.x",
    "eef.delta.y",
    "eef.delta.z",
    "eef.delta.rx",
    "eef.delta.ry",
    "eef.delta.rz",
    "gripper.delta",
)


@EnvConfig.register_subclass("gym_hil_test")
@dataclass
class GymHilEnvConfig(EnvConfig):
    """gym-hil 0.1.13 Panda: world-frame translation deltas in metres; positive gripper delta closes."""

    task: str | None = "pick_cube"
    fps: int = 10
    episode_length: int = 200

    @property
    def gym_id(self) -> str:
        return "gym_hil/PandaPickCubeBase-v0"

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        return {
            "image_obs": True,
            "control_dt": 1 / self.fps,
            "physics_dt": 0.002,
            "max_episode_steps": self.episode_length,
        }

    def sim_settings(self) -> dict[str, Any]:
        return {
            "type": "gym",
            "task": self.task,
            "fps": self.fps,
            "control": "delta",
            "semantics": "gym-hil-panda-translation-v1",
            "state_names": PANDA_STATE_NAMES,
            "action_names": PANDA_ACTION_NAMES,
            "robot_type": "panda",
            "kwargs": {
                "gym_id": self.gym_id,
                "gym_kwargs": self.gym_kwargs,
                "packages": ("gym-hil", "mujoco"),
            },
        }


@dataclass
class KeyboardToPandaAction(RobotActionProcessorStep):
    """Convert scripted keyboard input to Panda controller commands for native tests."""

    translation_step: float = 0.005
    gripper_step: float = 0.1

    def action(self, action: RobotAction) -> RobotAction:
        result = dict.fromkeys(PANDA_ACTION_NAMES, 0.0)
        for axis, positive, negative in (("x", "d", "a"), ("y", "w", "s"), ("z", "e", "q")):
            result[f"eef.delta.{axis}"] = (
                (positive in action) - (negative in action)
            ) * self.translation_step
        # gym-hil 0.1.13 ignores rotation slots. Positive gripper delta closes; negative opens.
        result["gripper.delta"] = (("c" in action) - ("o" in action)) * self.gripper_step
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Keyboard controller commands must be finite")
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def panda_keyboard_processor() -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    processor = make_default_teleop_action_processor()
    processor.steps = [KeyboardToPandaAction()]
    return processor


@dataclass
class KeyboardToAlohaAction(RobotActionProcessorStep):
    """Joint targets in radians, except grippers 6/13: normalized position, 0 closed / 1 open."""

    joint_step: float = 0.01

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition[TransitionKey.OBSERVATION]
        if observation is None:
            raise ValueError("Aloha joint-target teleoperation requires the current observation")
        result = {f"joint_{i}.pos": float(observation[f"joint_{i}.pos"]) for i in range(14)}
        joint = next((i for i in range(6) if str(i + 1) in action), 0)
        result[f"joint_{joint}.pos"] += (("d" in action) - ("a" in action)) * self.joint_step
        result[f"joint_{joint + 7}.pos"] += (("l" in action) - ("j" in action)) * self.joint_step
        for index, close_key, open_key in ((6, "c", "v"), (13, "n", "m")):
            if close_key in action or open_key in action:
                result[f"joint_{index}.pos"] = float(open_key in action)
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def aloha_keyboard_processor() -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    processor = make_default_teleop_action_processor()
    processor.steps = [KeyboardToAlohaAction()]
    return processor


def train_and_rollout(
    dataset: LeRobotDataset,
    config: EnvConfig,
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One CPU update proves plumbing, not policy quality; execute that checkpoint in both locations."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.scripts.lerobot_rollout import rollout
    from lerobot.scripts.lerobot_train import train

    output = tmp_path / "train"
    train(
        TrainPipelineConfig(
            dataset=DatasetConfig(repo_id=dataset.repo_id, root=str(dataset.root)),
            policy=ACTConfig(
                device="cpu",
                push_to_hub=False,
                chunk_size=1,
                n_action_steps=1,
                dim_model=32,
                n_heads=4,
                dim_feedforward=64,
                n_encoder_layers=1,
                n_decoder_layers=1,
                use_vae=False,
                pretrained_backbone_weights=None,
            ),
            output_dir=output,
            steps=1,
            batch_size=2,
            num_workers=0,
            env_eval_freq=0,
            save_freq=1,
            log_freq=1,
        )
    )
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    assert (checkpoint / "model.safetensors").is_file()
    commands = []
    send_action = SimRobot.send_action

    def track_action(robot, action):
        result = send_action(robot, action)
        commands.append(robot.episode_status.step)
        return result

    monkeypatch.setattr(SimRobot, "send_action", track_action)
    for location in (None, endpoint):
        before = len(commands)
        policy = ACTConfig.from_pretrained(checkpoint)
        policy.pretrained_path = checkpoint
        rollout(
            RolloutConfig(
                robot=SimRobotConfig(
                    env=replace(config, endpoint=location),
                    calibration_dir=tmp_path / "calibration",
                ),
                policy=policy,
                device="cpu",
                fps=config.fps,
                duration=2,
                return_to_initial_position=False,
                play_sounds=False,
            )
        )
        assert len(commands) > before, "Rollout must execute a native simulator action"
