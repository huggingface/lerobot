import collections
import importlib
import logging
from collections import deque
from typing import Optional

import einops
import numpy as np
import torch
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from lerobot.common.utils import set_seed

from .constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
    PUPPET_GRIPPER_POSITION_CLOSE,
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)
from .utils import sample_box_pose, sample_insertion_pose

_has_gym = importlib.util.find_spec("gym") is not None


# def make_ee_sim_env(task_name):
#     """
#     Environment for simulated robot bi-manual manipulation, with end-effector control.
#     Action space:      [left_arm_pose (7),             # position and quaternion for end effector
#                         left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
#                         right_arm_pose (7),            # position and quaternion for end effector
#                         right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

#     Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
#                                         left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
#                                         right_arm_qpos (6),         # absolute joint position
#                                         right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
#                         "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
#                                         left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
#                                         right_arm_qvel (6),         # absolute joint velocity (rad)
#                                         right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
#                         "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
#     """
#     if "sim_transfer_cube" in task_name:
#         xml_path = ASSETS_DIR / "bimanual_viperx_ee_transfer_cube.xml"
#         physics = mujoco.Physics.from_xml_path(xml_path)
#         task = TransferCubeEETask(random=False)
#         env = control.Environment(
#             physics, task, time_limit=20, control_timestep=DT, n_sub_steps=None, flat_observation=False
#         )
#     elif "sim_insertion" in task_name:
#         xml_path = ASSETS_DIR / "bimanual_viperx_ee_insertion.xml"
#         physics = mujoco.Physics.from_xml_path(xml_path)
#         task = InsertionEETask(random=False)
#         env = control.Environment(
#             physics, task, time_limit=20, control_timestep=DT, n_sub_steps=None, flat_observation=False
#         )
#     else:
#         raise NotImplementedError
#     return env


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = unnormalize_puppet_gripper_position(action_left[7])
        g_right_ctrl = unnormalize_puppet_gripper_position(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array(
            [
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ]
        )
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")
        # used in scripted policy to obtain starting pose
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()

        # used when replaying joint trajectory
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()

        def id2index(j_id):
            return 16 + (j_id - 16) * 7  # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id("red_peg_joint", "joint")
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id("blue_socket_joint", "joint")
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


class AlohaEnv(EnvBase):
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
        super().__init__(device=device, batch_size=[])
        self.frame_skip = frame_skip
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.image_size = image_size
        self.num_prev_obs = num_prev_obs
        self.num_prev_action = num_prev_action

        if pixels_only:
            assert from_pixels
        if from_pixels:
            assert image_size

        if not _has_gym:
            raise ImportError("Cannot import gym.")

        if not from_pixels:
            raise NotImplementedError()

        # time limit is controlled by StepCounter in factory
        time_limit = float("inf")

        if "sim_transfer_cube" in task:
            xml_path = ASSETS_DIR / "bimanual_viperx_ee_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEETask(random=False)
            env = control.Environment(
                physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
            )
        elif "sim_insertion" in task:
            xml_path = ASSETS_DIR / "bimanual_viperx_ee_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEETask(random=False)
            env = control.Environment(
                physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
            )
        else:
            raise NotImplementedError

        self._env = env

        self._make_spec()
        self._current_seed = self.set_seed(seed)

        if self.num_prev_obs > 0:
            self._prev_obs_image_queue = deque(maxlen=self.num_prev_obs)
            self._prev_obs_state_queue = deque(maxlen=self.num_prev_obs)
        if self.num_prev_action > 0:
            raise NotImplementedError()
            # self._prev_action_queue = deque(maxlen=self.num_prev_action)

    def render(self, mode="rgb_array", width=640, height=480):
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            image = torch.from_numpy(raw_obs["images"]["top"].copy())
            image = einops.rearrange(image, "h w c -> c h w")
            obs = {"image": image.type(torch.float32) / 255.0}

            if not self.pixels_only:
                obs["state"] = torch.from_numpy(raw_obs["qpos"]).type(torch.float32)
        else:
            # TODO(rcadene):
            raise NotImplementedError()
            # obs = {"state": torch.from_numpy(raw_obs["observation"]).type(torch.float32)}

        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        td = tensordict
        if td is None or td.is_empty():
            # we need to handle seed iteration, since self._env.reset() rely an internal _seed.
            self._current_seed += 1
            self.set_seed(self._current_seed)
            raw_obs = self._env.reset()
            # TODO(rcadene): add assert
            # assert self._current_seed == self._env._seed

            obs = self._format_raw_obs(raw_obs.observation)

            if self.num_prev_obs > 0:
                stacked_obs = {}
                if "image" in obs:
                    self._prev_obs_image_queue = deque(
                        [obs["image"]] * (self.num_prev_obs + 1), maxlen=(self.num_prev_obs + 1)
                    )
                    stacked_obs["image"] = torch.stack(list(self._prev_obs_image_queue))
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
        else:
            raise NotImplementedError()
        return td

    def _step(self, tensordict: TensorDict):
        td = tensordict
        action = td["action"].numpy()
        # step expects shape=(4,) so we pad if necessary
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        sum_reward = 0

        if action.ndim == 1:
            action = einops.repeat(action, "c -> t c", t=self.frame_skip)
        else:
            if self.frame_skip > 1:
                raise NotImplementedError()

        num_action_steps = action.shape[0]
        for i in range(num_action_steps):
            _, reward, discount, raw_obs = self._env.step(action[i])
            del discount  # not used

            # TOOD(rcadene): add an enum
            success = done = reward == 4
            sum_reward += reward
            obs = self._format_raw_obs(raw_obs)

            if self.num_prev_obs > 0:
                stacked_obs = {}
                if "image" in obs:
                    self._prev_obs_image_queue.append(obs["image"])
                    stacked_obs["image"] = torch.stack(list(self._prev_obs_image_queue))
                if "state" in obs:
                    self._prev_obs_state_queue.append(obs["state"])
                    stacked_obs["state"] = torch.stack(list(self._prev_obs_state_queue))
                obs = stacked_obs

        td = TensorDict(
            {
                "observation": TensorDict(obs, batch_size=[]),
                "reward": torch.tensor([sum_reward], dtype=torch.float32),
                # succes and done are true when coverage > self.success_threshold in env
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

            obs["image"] = BoundedTensorSpec(
                low=0,
                high=1,
                shape=image_shape,
                dtype=torch.float32,
                device=self.device,
            )
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
