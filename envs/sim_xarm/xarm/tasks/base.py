import os

import mujoco
import numpy as np
from gymnasium_robotics.envs import robot_env

from xarm.tasks import mocap


class Base(robot_env.MujocoRobotEnv):
    """
    Superclass for all simxarm environments.
    Args:
            xml_name (str): name of the xml environment file
            gripper_rotation (list): initial rotation of the gripper (given as a quaternion)
    """

    def __init__(self, xml_name, gripper_rotation=None):
        if gripper_rotation is None:
            gripper_rotation = [0, 1, 0, 0]
        self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
        self.center_of_table = np.array([1.655, 0.3, 0.63625])
        self.max_z = 1.2
        self.min_z = 0.2
        super().__init__(
            model_path=os.path.join(os.path.dirname(__file__), "assets", xml_name + ".xml"),
            n_substeps=20,
            n_actions=4,
            initial_qpos={},
        )

    @property
    def dt(self):
        return self.n_substeps * self.model.opt.timestep

    @property
    def eef(self):
        return self._utils.get_site_xpos(self.model, self.data, "grasp")

    @property
    def obj(self):
        return self._utils.get_site_xpos(self.model, self.data, "object_site")

    @property
    def robot_state(self):
        gripper_angle = self._utils.get_joint_qpos(self.model, self.data, "right_outer_knuckle_joint")
        return np.concatenate([self.eef, gripper_angle])

    def is_success(self):
        return NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def _sample_goal(self):
        raise NotImplementedError()

    def get_obs(self):
        return self._get_obs()

    def _step_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _limit_gripper(self, gripper_pos, pos_ctrl):
        if gripper_pos[0] > self.center_of_table[0] - 0.105 + 0.15:
            pos_ctrl[0] = min(pos_ctrl[0], 0)
        if gripper_pos[0] < self.center_of_table[0] - 0.105 - 0.3:
            pos_ctrl[0] = max(pos_ctrl[0], 0)
        if gripper_pos[1] > self.center_of_table[1] + 0.3:
            pos_ctrl[1] = min(pos_ctrl[1], 0)
        if gripper_pos[1] < self.center_of_table[1] - 0.3:
            pos_ctrl[1] = max(pos_ctrl[1], 0)
        if gripper_pos[2] > self.max_z:
            pos_ctrl[2] = min(pos_ctrl[2], 0)
        if gripper_pos[2] < self.min_z:
            pos_ctrl[2] = max(pos_ctrl[2], 0)
        return pos_ctrl

    def _apply_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl = self._limit_gripper(
            self._utils.get_site_xpos(self.model, self.data, "grasp"), pos_ctrl
        ) * (1 / self.n_substeps)
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        mocap.apply_action(
            self.model,
            self._model_names,
            self.data,
            np.concatenate([pos_ctrl, self.gripper_rotation, gripper_ctrl]),
        )

    def _render_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self._sample_goal()
        self._mujoco.mj_step(self.model, self.data, nstep=10)
        return True

    def _set_gripper(self, gripper_pos, gripper_rotation):
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_pos)
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", gripper_rotation)
        self._utils.set_joint_qpos(self.model, self.data, "right_outer_knuckle_joint", 0)
        self.data.qpos[10] = 0.0
        self.data.qpos[12] = 0.0

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        mocap.reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._sample_goal()
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        self._reset_sim()
        return self._get_obs()

    def step(self, action):
        assert action.shape == (4,)
        assert self.action_space.contains(action), "{!r} ({}) invalid".format(action, type(action))
        self._apply_action(action)
        self._mujoco.mj_step(self.model, self.data, nstep=2)
        self._step_callback()
        obs = self._get_obs()
        reward = self.get_reward()
        done = False
        info = {"is_success": self.is_success(), "success": self.is_success()}
        return obs, reward, done, info

    def render(self, mode="rgb_array", width=384, height=384):
        self._render_callback()
        # HACK
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        return self.mujoco_renderer.render(mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
