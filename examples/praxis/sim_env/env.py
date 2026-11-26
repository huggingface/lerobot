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

"""The praxis-sim MuJoCo rig of Praxis-1 as an EnvHub environment.

`pip install praxis-sim` provides the simulator: a Franka arm with a Robotiq 2F-85 gripper, an
exterior and a wrist camera (512 x 288), 15 Hz control, and the fixed evaluation suites `droid`,
`droid_guard` and `hard`. The observation and the action are what the `praxis` policy reads and
emits: `[x, y, z, roll, pitch, yaw, gripper]`, the flange pose in the robot base frame (metres,
extrinsic x-y-z Euler angles in radians, the DROID convention) and the gripper in [0, 1]; the
action is an absolute target.

This folder is the hub path, as for any EnvHub repository (see "Simulation" in
`docs/source/praxis.mdx`):

    --env.type=envhub --env.hub_path=examples/praxis/sim_env --env.trust_remote_code=true
    --robot.type=sim --robot.env.type=envhub --robot.env.hub_path=examples/praxis/sim_env ...

`make_env_kwargs`: `suite` is a suite name or a comma list of task names; `first_scene` is the scene
of the first unseeded episode; the others go to `praxis_sim.gym_env.PraxisSimGymEnv`
(`after_success_s`, `time_limit_s`, `prompt`, `render_overview`).
"""

from functools import partial
from typing import Any

import gymnasium as gym
from praxis_sim.gym_env import PraxisSimGymEnv, resolve_tasks

POSE = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

LEROBOT_SIM = {
    "fps": 15,
    "control_mode": "eef_pose",
    "semantics": "praxis-sim-eef-pose-v1",
    "state_names": POSE,
    "action_names": POSE,
    # The body of the DROID platform, as `lerobot/droid_1.0.1` records it.
    "robot_type": "Franka",
}


class PraxisSimEnv(PraxisSimGymEnv):
    """One praxis-sim task. The reset seed is the scene seed; an unseeded reset takes the next scene.

    `lerobot-eval` seeds every episode. A sim robot (`lerobot-rollout`, `lerobot-record`) resets
    without a seed, and the simulator would then repeat its last scene: every recorded episode would
    be the same one. Here the first unseeded reset is scene `first_scene`, and one that follows an
    episode with at least one step is the scene after it. A reset that follows no step repeats the
    scene, so the reset a sim robot makes when it connects and the one that starts its first episode
    are one scene.
    """

    def __init__(self, task: str, *, first_scene: int = 0, **kwargs: Any) -> None:
        super().__init__(task, **kwargs)
        self._scene = int(first_scene)
        self._stepped = False

    def reset(self, seed: int | None = None, options: dict | None = None, **kwargs: Any):
        if seed is None:
            seed = self._scene + int(self._stepped)
        self._scene = int(seed)
        self._stepped = False
        return super().reset(seed=self._scene, options=options, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self._stepped = True
        # Reported per episode by lerobot-eval: the task's progress and the tick of success (-1: none).
        info["metrics"] = {"progress": info["progress"], "success_tick": float(info["success_tick"])}
        return observation, reward, terminated, truncated, info


def make_env(n_envs: int = 1, use_async_envs: bool = False, suite: str = "droid_guard", **kwargs: Any):
    """The EnvHub entry point: `{suite: {task_index: vector env}}` for a suite or a comma list of tasks."""
    name, tasks = resolve_tasks(suite)
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return {
        name: {
            index: vector_cls([partial(PraxisSimEnv, task, **kwargs)] * n_envs)
            for index, task in enumerate(tasks)
        }
    }
