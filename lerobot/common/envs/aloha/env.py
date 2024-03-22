import importlib
import logging
from collections import deque
from typing import Optional

import einops
import numpy as np
import torch
from dm_control import mujoco
from dm_control.rl import control
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from lerobot.common.envs.abstract import AbstractEnv
from lerobot.common.envs.aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from lerobot.common.envs.aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from lerobot.common.envs.aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from lerobot.common.envs.aloha.utils import sample_box_pose, sample_insertion_pose
from lerobot.common.utils import set_seed

_has_gym = importlib.util.find_spec("gym") is not None


class AlohaEnv(AbstractEnv):
    _reset_warning_issued = False

    def __init__(
        self,
        task,
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
        num_prev_obs=1,
        num_prev_action=0,
    ):
        super().__init__(
            task=task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            image_size=image_size,
            seed=seed,
            device=device,
            num_prev_obs=num_prev_obs,
            num_prev_action=num_prev_action,
        )

    def _make_env(self):
        if not _has_gym:
            raise ImportError("Cannot import gym.")

        if not self.from_pixels:
            raise NotImplementedError()

        self._env = self._make_env_task(self.task)

    def render(self, mode="rgb_array", width=640, height=480):
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if "sim_transfer_cube" in task_name:
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask(random=False)
        elif "sim_insertion" in task_name:
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask(random=False)
        elif "sim_end_effector_transfer_cube" in task_name:
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask(random=False)
        elif "sim_end_effector_insertion" in task_name:
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask(random=False)
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            image = torch.from_numpy(raw_obs["images"]["top"].copy())
            image = einops.rearrange(image, "h w c -> c h w")
            assert image.dtype == torch.uint8
            obs = {"image": {"top": image}}

            if not self.pixels_only:
                obs["state"] = torch.from_numpy(raw_obs["qpos"]).type(torch.float32)
        else:
            # TODO(rcadene):
            raise NotImplementedError()
            # obs = {"state": torch.from_numpy(raw_obs["observation"]).type(torch.float32)}

        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        if tensordict is not None and not AlohaEnv._reset_warning_issued:
            logging.warning(f"{self.__class__.__name__}._reset ignores the provided tensordict.")
            AlohaEnv._reset_warning_issued = True

        # Seed the environment and update the seed to be used for the next reset.
        self._next_seed = self.set_seed(self._next_seed)

        # TODO(rcadene): do not use global variable for this
        if "sim_transfer_cube" in self.task:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in self.task:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        raw_obs = self._env.reset()

        obs = self._format_raw_obs(raw_obs.observation)

        if self.num_prev_obs > 0:
            stacked_obs = {}
            if "image" in obs:
                self._prev_obs_image_queue = deque(
                    [obs["image"]["top"]] * (self.num_prev_obs + 1), maxlen=(self.num_prev_obs + 1)
                )
                stacked_obs["image"] = {"top": torch.stack(list(self._prev_obs_image_queue))}
            if "state" in obs:
                self._prev_obs_state_queue = deque(
                    [obs["state"]] * (self.num_prev_obs + 1), maxlen=(self.num_prev_obs + 1)
                )
                stacked_obs["state"] = torch.stack(list(self._prev_obs_state_queue))
            obs = stacked_obs

        td = TensorDict(
            {
                "observation": TensorDict(obs, batch_size=[]),
                "done": torch.tensor([False], dtype=torch.bool),
            },
            batch_size=[],
        )

        return td

    def _step(self, tensordict: TensorDict):
        td = tensordict
        action = td["action"].numpy()
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        success = done = reward == 4
        obs = self._format_raw_obs(raw_obs)

        if self.num_prev_obs > 0:
            stacked_obs = {}
            if "image" in obs:
                self._prev_obs_image_queue.append(obs["image"]["top"])
                stacked_obs["image"] = {"top": torch.stack(list(self._prev_obs_image_queue))}
            if "state" in obs:
                self._prev_obs_state_queue.append(obs["state"])
                stacked_obs["state"] = torch.stack(list(self._prev_obs_state_queue))
            obs = stacked_obs

        td = TensorDict(
            {
                "observation": TensorDict(obs, batch_size=[]),
                "reward": torch.tensor([reward], dtype=torch.float32),
                # success and done are true when coverage > self.success_threshold in env
                "done": torch.tensor([done], dtype=torch.bool),
                "success": torch.tensor([success], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td

    def _make_spec(self):
        obs = {}
        from omegaconf import OmegaConf

        if self.from_pixels:
            if isinstance(self.image_size, int):
                image_shape = (3, self.image_size, self.image_size)
            elif OmegaConf.is_list(self.image_size):
                assert len(self.image_size) == 3  # c h w
                assert self.image_size[0] == 3  # c is RGB
                image_shape = tuple(self.image_size)
            else:
                raise ValueError(self.image_size)
            if self.num_prev_obs > 0:
                image_shape = (self.num_prev_obs + 1, *image_shape)

            obs["image"] = {
                "top": BoundedTensorSpec(
                    low=0,
                    high=255,
                    shape=image_shape,
                    dtype=torch.uint8,
                    device=self.device,
                )
            }
            if not self.pixels_only:
                state_shape = (len(JOINTS),)
                if self.num_prev_obs > 0:
                    state_shape = (self.num_prev_obs + 1, *state_shape)

                obs["state"] = UnboundedContinuousTensorSpec(
                    # TODO: add low and high bounds
                    shape=state_shape,
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            # TODO(rcadene): add observation_space achieved_goal and desired_goal?
            state_shape = (len(JOINTS),)
            if self.num_prev_obs > 0:
                state_shape = (self.num_prev_obs + 1, *state_shape)

            obs["state"] = UnboundedContinuousTensorSpec(
                # TODO: add low and high bounds
                shape=state_shape,
                dtype=torch.float32,
                device=self.device,
            )
        self.observation_spec = CompositeSpec({"observation": obs})

        # TODO(rcadene): valid when controling end effector?
        # action_space = self._env.action_spec()
        # self.action_spec = BoundedTensorSpec(
        #     low=action_space.minimum,
        #     high=action_space.maximum,
        #     shape=action_space.shape,
        #     dtype=torch.float32,
        #     device=self.device,
        # )

        # TODO(rcaene): add bounds (where are they????)
        self.action_spec = BoundedTensorSpec(
            shape=(len(ACTIONS)),
            low=-1,
            high=1,
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2,
                    shape=(1,),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "success": DiscreteTensorSpec(
                    2,
                    shape=(1,),
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
        )

    def _set_seed(self, seed: Optional[int]):
        set_seed(seed)
        # TODO(rcadene): seed the env
        # self._env.seed(seed)
        logging.warning("Aloha env is not seeded")
