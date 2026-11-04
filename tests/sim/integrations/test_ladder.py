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

"""Unit tests for the LADDER benchmark integration on LocalVectorEnv (simulator-independent)."""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.env_server.server import EnvServerConfig
from lerobot.envs.configs import LadderEnv, LadderEnvConfig, check_served
from lerobot.envs.factory import make_env_config
from lerobot.envs.utils import (
    check_env_attributes_and_types,
    prepare_canonical_observation,
    validate_checkpoint,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.sim.config import SimConfig
from lerobot.sim.integrations.ladder import (
    ROBOLAB_TASKS,
    ROBOT_ARENA_TASKS,
    SIMPLER_TASKS,
    LadderVectorEnv,
    conventions,
    vector_env_fns,
)
from lerobot.sim.local_vector_env import LocalVectorEnv


@dataclass
class _FakeState:
    eef_pose: np.ndarray
    gripper_state: np.ndarray
    joints: np.ndarray | None = None


@dataclass
class _FakeObs:
    images: dict[str, np.ndarray]
    state: _FakeState
    metadata: dict[str, Any]


def test_ladder_imports_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "lerobot.sim.integrations.ladder", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    module = importlib.import_module("lerobot.sim.integrations.ladder")
    assert hasattr(module, "vector_env_fns") and hasattr(module, "conventions")


def test_ladder_config_registration_and_server_yaml() -> None:
    cfg = make_env_config("ladder")
    assert isinstance(cfg, LadderEnvConfig)
    assert LadderEnv is LadderEnvConfig
    assert cfg.type == "ladder"
    assert cfg.action_mode == "delta_ee"
    assert len(SIMPLER_TASKS) == 54
    assert len(ROBOT_ARENA_TASKS) == 47
    assert len(ROBOLAB_TASKS) == 44
    assert len(SIMPLER_TASKS) + len(ROBOT_ARENA_TASKS) + len(ROBOLAB_TASKS) == 145

    simpler_conv = conventions({"suite": "simpler", "action_mode": "delta_ee"})
    assert (simpler_conv["fps"], simpler_conv["control_mode"], simpler_conv["robot_type"]) == (
        5,
        "delta",
        "widowx",
    )

    robolab_cfg = LadderEnvConfig(suite="robolab", task="comp_bin_1")
    assert robolab_cfg.action_mode == "joint_pos"
    assert robolab_cfg.fps == 15
    robolab_conv = conventions({"suite": "robolab"})
    assert (robolab_conv["fps"], robolab_conv["control_mode"], robolab_conv["robot_type"]) == (
        15,
        "position",
        "Franka",
    )

    robolab_delta_cfg = LadderEnvConfig(suite="robolab", task="comp_bin_1", action_mode="delta_ee")
    assert robolab_delta_cfg.action_mode == "delta_ee"
    robolab_delta_conv = conventions({"suite": "robolab", "action_mode": "delta_ee"})
    assert robolab_delta_conv["control_mode"] == "delta"
    assert len(robolab_delta_conv["action_names"]) == 7

    yaml_path = Path(__file__).parents[3] / "docker" / "sim" / "configs" / "ladder.yaml"
    server_cfg = draccus.parse(config_class=EnvServerConfig, config_path=yaml_path, args=[])
    served_sim = asdict(SimConfig.resolve(asdict(server_cfg.sim)))
    client_expected = LadderEnvConfig(task="all").sim_settings()
    check_served(served_sim, client_expected, "tcp/127.0.0.1:7448")

    with pytest.raises(ValueError, match="Unknown LADDER suite"):
        LadderEnvConfig(suite="unknown")
    with pytest.raises(ValueError, match="Unknown LADDER tasks"):
        vector_env_fns(suite="simpler", task="nonexistent")


def test_ladder_tasks_run_as_lerobot_local_vector_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_reset_backend(self: LadderVectorEnv, episode_ids: list[int]) -> list[_FakeObs]:
        assert episode_ids == list(range(self.num_envs))
        return [
            _FakeObs(
                images={"main": np.zeros((480, 640, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([1.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": False,
                    "truncated": False,
                    "num_subgoals_complete": 0,
                    "oracle": {"subtasks": ["put carrot", "put spoon"], "achieved": [False, False]},
                },
            )
            for _ in episode_ids
        ]

    def fake_step_backend(self: LadderVectorEnv, actions: list[Any]) -> list[_FakeObs]:
        assert len(actions) == 2
        return [
            _FakeObs(
                images={"main": np.zeros((480, 640, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([0.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": True,
                    "truncated": False,
                    "num_subgoals_complete": 2,
                    "oracle": {"subtasks": ["put carrot", "put spoon"], "achieved": [True, True]},
                },
            ),
            _FakeObs(
                images={"main": np.zeros((480, 640, 3), dtype=np.uint8)},
                state=_FakeState(np.eye(4, dtype=np.float32), np.array([1.0], dtype=np.float32)),
                metadata={
                    "instruction": "put the carrot and the spoon on the plate",
                    "done": False,
                    "truncated": True,
                    "num_subgoals_complete": 1,
                    "oracle": {"subtasks": ["put carrot", "put spoon"], "achieved": [True, False]},
                },
            ),
        ]

    monkeypatch.setattr(LadderVectorEnv, "_reset_backend", fake_reset_backend)
    monkeypatch.setattr(LadderVectorEnv, "_step_backend", fake_step_backend)
    monkeypatch.setattr(LadderVectorEnv, "_decode_action", lambda self, env_i, raw: raw)

    cfg = LadderEnvConfig(suite="simpler", task="all1_plate,all2_plate")
    envs = cfg.create_envs(n_envs=2)
    assert set(envs.keys()) == {"all1_plate", "all2_plate"}
    vec_env = envs["all1_plate"][0]
    assert isinstance(vec_env, LocalVectorEnv)
    try:
        descriptor = vec_env.descriptor
        assert descriptor.tasks == {"all1_plate": [0], "all2_plate": [0]}
        assert (descriptor.fps, descriptor.control_mode, descriptor.robot_type) == (5, "delta", "widowx")
        assert {f.name: f.shape for f in descriptor.features} == {
            "observation.images.main": (480, 640, 3),
            "observation.state": (7,),
        }
        assert descriptor.action_feature.shape == (7,)
        assert vec_env.call("_max_episode_steps") == (1000, 1000)

        obs, info = vec_env.reset(seed=0)
        check_env_attributes_and_types(vec_env)
        assert vec_env.call("task_description") == (
            "put the carrot and the spoon on the plate",
            "put the carrot and the spoon on the plate",
        )
        assert info["is_success"].tolist() == [False, False]
        prepared = prepare_canonical_observation(obs)
        assert prepared["observation.images.main"].shape == (2, 3, 480, 640)
        assert prepared["observation.state"].shape == (2, 7)

        obs1, reward, terminated, truncated, info1 = vec_env.step(np.zeros((2, 7), dtype=np.float32))
        assert reward.tolist() == pytest.approx([1.0, 0.5])
        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, True]
        assert info1["is_success"].tolist() == [True, False]
        assert len(vec_env.call("render")) == 2

        robot = SimRobot(SimRobotConfig(env=LadderEnvConfig(suite="simpler", task="all1_plate")))
        try:
            assert robot.robot_type == "widowx"
            robot.connect()
            robot_obs = robot.get_observation()
            assert "main" in robot_obs and "x" in robot_obs
        finally:
            if robot.is_connected:
                robot.disconnect()

        @PreTrainedConfig.register_subclass("_ladder_dummy_policy")
        @dataclass
        class _DummyConfig(PreTrainedConfig):
            def get_optimizer_preset(self):
                return None

            def get_scheduler_preset(self):
                return None

            def validate_features(self):
                pass

            @property
            def observation_delta_indices(self):
                return None

            @property
            def action_delta_indices(self):
                return None

            @property
            def reward_delta_indices(self):
                return None

        class _DummyPolicy(PreTrainedPolicy):
            config_class = _DummyConfig
            name = "_ladder_dummy_policy"

            def __init__(self, cfg: _DummyConfig):
                super().__init__(cfg)
                self._p = nn.Parameter(torch.zeros(1))

            def get_optim_params(self):
                return self.parameters()

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def reset(self):
                pass

            def predict_action_chunk(self, batch):
                b = batch["observation.state"].shape[0]
                return torch.zeros((b, 1, 7), dtype=torch.float32)

            def select_action(self, batch):
                b = batch["observation.state"].shape[0]
                return torch.zeros((b, 7), dtype=torch.float32)

        policy_cfg = _DummyConfig(
            input_features={
                "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )
        rename_map = {"observation.images.main": "observation.images.camera1"}
        validate_checkpoint(descriptor, policy_cfg, rename_map)
        identity = PolicyProcessorPipeline(steps=[])
        action_identity = PolicyProcessorPipeline(
            steps=[],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        eval_info = eval_policy_all(
            envs=LadderEnvConfig(suite="simpler", task="all1_plate", episode_length=2).create_envs(n_envs=2),
            policy=_DummyPolicy(policy_cfg),
            env_preprocessor=identity,
            env_postprocessor=identity,
            preprocessor=identity,
            postprocessor=action_identity,
            n_episodes=2,
        )
        assert eval_info["overall"]["pc_success"] == pytest.approx(50.0)
        assert eval_info["overall"]["avg_max_reward"] == pytest.approx(0.75)
    finally:
        for group in envs.values():
            for env in group.values():
                env.close()
