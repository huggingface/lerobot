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

"""LIBERO integration observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.sim import SimSettings
from tests.utils import import_without_simulator, resolve_without_simulator


def test_libero_numpy_adapter_parity(monkeypatch):
    canonical_observation = import_without_simulator("libero", monkeypatch).canonical_observation

    native = {
        "pixels": {"camera1": np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)},
        "robot_state": {
            "eef": {"pos": np.array([1, 2, 3], np.float32), "quat": np.array([0, 0, 0.6, 0.8], np.float32)},
            "gripper": {"qpos": np.array([0.1, 0.2], np.float32)},
        },
    }

    def batch(value):
        return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else value[None]

    expected = LiberoProcessorStep().observation(preprocess_observation(batch(native)))
    actual = canonical_observation(native)
    np.testing.assert_allclose(
        actual["observation.state"], expected["observation.state"][0].numpy(), atol=1e-6
    )
    np.testing.assert_array_equal(
        actual["observation.images.camera1"],
        (expected["observation.images.camera1"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
    )


def test_libero_declares_the_control_mode_it_runs(monkeypatch):
    relative = resolve_without_simulator(
        SimSettings(**LiberoEnvConfig(task="libero_spatial").sim_settings()), monkeypatch, "libero_spatial"
    )
    assert (relative.control_mode, relative.semantics) == ("delta", "libero-delta-v1")
    assert relative.action_names[0] == "eef.delta.x"
    # LIBERO's absolute mode commands end-effector poses, not deltas.
    absolute = resolve_without_simulator(
        SimSettings(**LiberoEnvConfig(task="libero_spatial", control_mode="absolute").sim_settings()),
        monkeypatch,
        "libero_spatial",
    )
    assert (absolute.control_mode, absolute.semantics) == ("eef_pose", "libero-eef-pose-v1")
    assert absolute.action_names[0] == "eef.x"


def _libero_env(monkeypatch, obs_type):
    """A `LiberoEnv` on a stand-in simulator: a robot at rest, with its base turned about the vertical axis."""
    libero = import_without_simulator("libero", monkeypatch)
    monkeypatch.setattr(libero, "get_libero_path", lambda name: "bddl_files")
    task = SimpleNamespace(name="task", language="do the task", problem_folder="suite", bddl_file="task.bddl")
    env = libero.LiberoEnv(
        SimpleNamespace(get_task=lambda task_id: task),
        0,
        "libero_spatial",
        obs_type=obs_type,
        init_states=False,
        observation_height=4,
        observation_width=5,
    )
    robot = SimpleNamespace(
        controller=SimpleNamespace(ee_ori_mat=np.eye(3)), robot_model=SimpleNamespace(root_body="robot0_base")
    )
    data = SimpleNamespace(
        get_body_xpos=lambda name: np.array([-0.66, 0.0, 0.912]),
        get_body_xquat=lambda name: np.array([0.8, 0.0, 0.0, 0.6]),  # MuJoCo's wxyz
    )
    env._env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=data))
    raw = {camera: np.zeros((4, 5, 3), dtype=np.uint8) for camera in env.camera_name}
    raw |= {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "robot0_gripper_qvel": np.zeros(2),
        "robot0_joint_pos": np.linspace(-1.0, 1.0, 7),
        "robot0_joint_vel": np.zeros(7),
    }
    return libero, env, raw


def test_libero_joints_obs_type_adds_joint_positions_and_base_pose(monkeypatch):
    libero, env, raw = _libero_env(monkeypatch, "pixels_agent_pos_joints")

    native = env._format_raw_obs(raw)
    assert env.observation_space.contains(native)
    observation = libero.canonical_observation(native)

    assert observation["observation.state"].shape == (8,)
    np.testing.assert_allclose(observation["observation.joint_pos"], np.linspace(-1.0, 1.0, 7), atol=1e-7)
    np.testing.assert_allclose(observation["observation.base_pos"], [-0.66, 0.0, 0.912], atol=1e-7)
    # xyzw, as LIBERO's `robot0_eef_quat`.
    np.testing.assert_allclose(observation["observation.base_quat"], [0.0, 0.0, 0.6, 0.8], atol=1e-7)
    assert all(observation[key].dtype == np.float32 for key in observation if "images" not in key)


def test_libero_default_obs_type_is_unchanged(monkeypatch):
    libero, env, raw = _libero_env(monkeypatch, "pixels_agent_pos")

    native = env._format_raw_obs(raw)
    assert env.observation_space.contains(native)
    assert set(libero.canonical_observation(native)) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }


def test_libero_config_accepts_the_joints_obs_type():
    assert (
        LiberoEnvConfig(obs_type="pixels_agent_pos_joints").gym_kwargs["obs_type"]
        == "pixels_agent_pos_joints"
    )
    with pytest.raises(ValueError, match="Unsupported obs_type"):
        LiberoEnvConfig(obs_type="joints")
