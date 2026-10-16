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

"""Batched lockstep worlds as a Gymnasium vector env, in this process or served; no native imports."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from lerobot.env_server.client import EnvClient
from lerobot.env_server.contracts import EnvDescriptor, StepResult
from lerobot.env_server.in_process import InProcessEnvClient


class BatchedEnv(gym.vector.VectorEnv):
    """``num_envs`` worlds of one task, stepped together in lockstep, with canonical observations.

    ``client`` runs them in this process (``InProcessEnvClient``) or on an env server (``EnvClient``).
    """

    canonical_observations = True

    def __init__(
        self,
        client: EnvClient | InProcessEnvClient,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int = 0,
        descriptor: EnvDescriptor | None = None,
        async_worlds: bool = False,
    ):
        self.client = client
        self.async_worlds = async_worlds
        self.descriptor = descriptor or self.client.describe()
        self.num_envs = num_envs
        self.task_group = task_group or next(iter(self.descriptor.tasks))
        self.task_id = task_id
        self.metadata = {"render_fps": self.descriptor.fps, "autoreset_mode": "Disabled"}
        self.render_mode = "rgb_array"
        self.single_observation_space = gym.spaces.Dict(
            {
                f.name: gym.spaces.Box(0, 255, f.shape, np.uint8)
                if f.kind == "rgb"
                else gym.spaces.Box(-np.inf, np.inf, f.shape, np.dtype(f.dtype))
                for f in self.descriptor.features
            }
        )
        self.single_action_space = gym.spaces.Box(
            -np.inf, np.inf, self.descriptor.action_feature.shape, np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.latest: StepResult | None = None
        self.closed = False

    def _open_session(self, seeds=None):
        # Not named `_ensure`: eval_policy_all prefetches envs exposing `_ensure`, and an
        # unseeded prefetch open would reset the worlds before the seeded rollout reset.
        if not self.client.session:
            self.latest = self.client.open(
                self.num_envs, "lockstep", self.task_group, self.task_id, seeds, self.async_worlds
            )
            assert self.client.descriptor is not None
            self.descriptor = self.client.descriptor
            return True
        return False

    def _result(self, body):
        self.latest = StepResult.from_dict(body["result"])
        return self.latest

    def reset(self, *, seed=None, options=None):
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        result = (
            self.latest
            if self._open_session(seed)
            else self._result(self.client.request("reset", seeds=seed))
        )
        assert result is not None
        return result.obs, {"is_success": result.is_success}

    def step(self, actions):
        self._open_session()
        result = self._result(self.client.request("step", actions=np.asarray(actions, dtype=np.float32)))
        return (
            result.obs,
            result.reward,
            result.terminated,
            result.truncated,
            {"is_success": result.is_success},
        )

    def call(self, name, *args, **kwargs):
        if name == "_max_episode_steps":
            return (self.descriptor.max_episode_steps,) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        self._open_session()
        if name == "task_description":
            assert self.latest is not None
            return self.latest.task
        if name == "render":
            return tuple(self.client.request("render")["frames"])
        raise AttributeError(f"Simulator vector attribute is not exposed: {name}")

    def get_attr(self, name):
        return self.call(name)

    def render(self):
        return self.call("render")

    def close_extras(self, **kwargs):
        self.client.close()

    def close(self, **kwargs):
        self.client.close()
        self.closed = True


def prepare_canonical_observation(observations):
    """Generic dataset-value conversion, once, on the policy side."""
    result = {}
    for name, value in observations.items():
        tensor = torch.from_numpy(value)
        if value.dtype == np.uint8 and value.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2).contiguous().float() / 255
        elif tensor.is_floating_point():
            tensor = tensor.float()
        result[name] = tensor
    return result
