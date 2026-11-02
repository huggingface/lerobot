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

"""The RoboLab integration on a stand-in for RoboLab's Isaac Lab env: Isaac Sim needs an RTX GPU."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

CAMERAS = ("over_shoulder_left_camera", "wrist_cam")


class FakeWorld:
    """RoboLab's ``RobolabEnv``: one world of envs that freeze when they finish, recording whether they succeeded.

    Env 0 succeeds on the second step and env 1 times out on the third.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"
        self.max_episode_length = 5
        self.observation_manager = SimpleNamespace(active_terms={"image_obs": list(CAMERAS)})
        self.scene = {name: SimpleNamespace(cfg=SimpleNamespace(height=4, width=6)) for name in CAMERAS}
        self._frozen_envs = torch.zeros(num_envs, dtype=torch.bool)
        self._env_results: dict[int, bool] = {}
        self.resets = 0
        self.t = 0

    def _obs(self):
        images = {name: torch.full((self.num_envs, 4, 6, 3), self.t, dtype=torch.uint8) for name in CAMERAS}
        joints = torch.full((self.num_envs, 7), float(self.t))
        return {
            "image_obs": images,
            "proprio_obs": {"arm_joint_pos": joints, "gripper_pos": torch.ones(self.num_envs, 1)},
        }

    def reset_eval_state(self):
        self._frozen_envs[:] = False
        self._env_results.clear()

    def reset(self, seed=None):
        self.resets += 1
        self.t = 0
        return self._obs(), {}

    def step(self, actions):
        assert actions.shape == (self.num_envs, 8)
        self.t += 1
        for env_id, (step, success) in {0: (2, True), 1: (3, False)}.items():
            if self.t == step:
                self._frozen_envs[env_id] = True
                self._env_results[env_id] = success
        zeros = torch.zeros(self.num_envs)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}

    def close(self):
        pass


@pytest.fixture
def robolab(tmp_path, monkeypatch):
    """Stand in for RoboLab's package; return the worlds it creates."""
    metadata = tmp_path / "_metadata" / "task_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps([{"task_name": "BananaInBowlTask"}, {"task_name": "RubiksCubeAndBananaTask"}])
    )
    worlds = []

    def create_env(env_cfg, instruction_type):
        worlds.append(FakeWorld(env_cfg.num_envs))
        return worlds[-1], SimpleNamespace(instruction={instruction_type: "Put the banana in the bowl"})

    attributes = {
        "robolab.constants": {"TASK_DIR": str(tmp_path), "set_output_dir": lambda path: None},
        "robolab.core.environments.config": {
            "parse_env_cfg": lambda task, **kwargs: SimpleNamespace(
                num_envs=kwargs["num_envs"], recorders=object()
            )
        },
        "robolab.core.environments.runtime": {"create_env": create_env},
        "robolab.core.task.instruction": {"instruction_text": lambda fields: next(iter(fields.values()))},
        "robolab.registrations.droid.camera_presets": {"WRIST_LEFT": list(CAMERAS)},
        "robolab.registrations.droid.auto_env_registrations_jointpos": {
            "auto_register_droid_envs": lambda **_: None
        },
    }
    for name, values in attributes.items():
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            module_name = ".".join(parts[:depth])
            if module_name not in sys.modules or depth == len(parts):
                monkeypatch.setitem(
                    sys.modules, module_name, sys.modules.get(module_name, ModuleType(module_name))
                )
            if depth > 1:
                monkeypatch.setattr(
                    sys.modules[".".join(parts[: depth - 1])],
                    parts[depth - 1],
                    sys.modules[module_name],
                    raising=False,
                )
        for key, value in values.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)

    from lerobot.sim.integrations import robolab as integration

    monkeypatch.setattr(integration, "_launch_app", lambda: None)
    return worlds


def test_robolab_tasks_run_as_lerobot_vector_envs(robolab):
    from lerobot.envs.configs import RoboLabEnvConfig
    from lerobot.sim.local_vector_env import LocalVectorEnv

    with pytest.raises(ValueError, match="Unknown RoboLab tasks"):
        LocalVectorEnv(RoboLabEnvConfig(task="NoSuchTask").sim_config())
    env = LocalVectorEnv(RoboLabEnvConfig(task="BananaInBowlTask").sim_config(), num_envs=2)
    try:
        descriptor = env.descriptor
        assert descriptor.tasks == {"BananaInBowlTask": [0]}
        assert (descriptor.fps, descriptor.control_mode, descriptor.robot_type) == (15, "position", "Franka")
        assert {f.name: f.shape for f in descriptor.features} == {
            "observation.images.over_shoulder_left_camera": (4, 6, 3),
            "observation.images.wrist_cam": (4, 6, 3),
            "observation.state": (8,),
        }
        assert descriptor.action_feature.names[-1] == "gripper.pos" and env.call("_max_episode_steps") == (
            5,
            5,
        )

        obs, info = env.reset(seed=0)
        world = robolab[-1]
        assert world.resets == 2  # as RoboLab's own runner
        assert info["task"] == ("Put the banana in the bowl",) * 2
        np.testing.assert_array_equal(obs["observation.state"][0], [0] * 7 + [1])

        actions = np.zeros((2, 8), np.float32)
        flags = []
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(actions)
            flags.append((terminated.tolist(), truncated.tolist(), info["is_success"].tolist()))
        # Success terminates; a time out is a truncation, never a success.
        assert flags == [
            ([False, False], [False, False], [False, False]),
            ([True, False], [False, False], [True, False]),
            ([True, False], [False, True], [True, False]),
        ]

        env.reset(seed=1)
        assert world.resets == 4 and not world._frozen_envs.any()
    finally:
        env.close()
