from pathlib import Path
from typing import Any, Literal, Tuple, Dict

# import gym
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

import logging
import torch


try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None
# from mujoco.glfw import glfw

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPushCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        cfg,
        use_delta_action_space: bool = True,
        delta: float | None = None,
        action_scale: np.ndarray = np.asarray([0.05, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
    ):
        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self._action_scale = action_scale
        self.reward_type = reward_type
        self.render_mode = render_mode
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)
        self.image_obs = image_obs
        
        # Caching.
        self.panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self.panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )

        self.gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.initial_follower_position = self.data.qpos[self.panda_dof_ids].astype(np.float32)

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.cfg = cfg

        self.delta = delta
        self.use_delta_action_space = use_delta_action_space
        self.current_joint_positions = self.data.qpos[self.panda_dof_ids].astype(np.float32)

        # Retrieve the size of the joint position interval bound.
        self.relative_bounds_size = (
            (
                self.cfg.robot.joint_position_relative_bounds["max"]
                - self.cfg.robot.joint_position_relative_bounds["min"]
            )
            if self.cfg.robot.joint_position_relative_bounds is not None
            else None
        )

        self.cfg.robot.max_relative_target = (
            self.relative_bounds_size.float()
            if self.relative_bounds_size is not None
            else None
        )
        

        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32
                ),
                "observation.images.front": spaces.Box(
                    low=0,
                    high=255,
                    shape=(render_spec.height, render_spec.width, 3),
                    dtype=np.uint8,
                ),
                "observation.images.wrist": spaces.Box(
                    low=0,
                    high=255,
                    shape=(render_spec.height, render_spec.width, 3),
                    dtype=np.uint8,
                ),
            }
        )

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 7
        if self.use_delta_action_space:
            bounds = (
                self.relative_bounds_size
                if self.relative_bounds_size is not None
                else np.ones(action_dim) * 1000
            )
            action_space_robot = gym.spaces.Box(
                low=-bounds,
                high=bounds,
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            bounds_min = (
                self.cfg.robot.joint_position_relative_bounds["min"].cpu().numpy()
                if self.cfg.robot.joint_position_relative_bounds is not None
                else np.ones(action_dim) * -1000
            )
            bounds_max = (
                self.cfg.robot.joint_position_relative_bounds["max"].cpu().numpy()
                if self.cfg.robot.joint_position_relative_bounds is not None
                else np.ones(action_dim) * 1000
            )
            action_space_robot = gym.spaces.Box(
                low=bounds_min,
                high=bounds_max,
                shape=(action_dim,),
                dtype=np.float32,
            )

        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),
            ),
        )

        self._viewer = mujoco.Renderer(
            self.model,
            height=render_spec.height,
            width=render_spec.width
        )
        self._viewer.render()

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # mujoco.mj_resetData(self._model, self._data)

        # # Reset arm to home position.
        # # self._data.qpos[self.panda_dof_ids] = _PANDA_HOME
        # self._data.qpos[self.panda_dof_ids] = np.asarray(self.cfg.env.wrapper.fixed_reset_joint_positions)
        # # Gripper
        # self._data.ctrl[self._gripper_ctrl_id] = 255
        # mujoco.mj_forward(self._model, self._data)

        # # Reset mocap body to home position.
        # tcp_pos = self._data.sensor("2f85/pinch_pos").data
        # self._data.mocap_pos[0] = tcp_pos

        # # Sample a new block position.
        # # block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        # block_xy = np.array([0.5, 0.0])
        # self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        # mujoco.mj_forward(self._model, self._data)

        # # Cache the initial block height.
        # self._z_init = self._data.sensor("block_pos").data[2]
        # self._z_success = self._z_init + 0.2

        # # Sample a new target position
        # # target_region_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        # target_region_xy = np.array([0.5, 0.10])
        # self._model.geom("target_region").pos = (*target_region_xy, 0.005)
        # mujoco.mj_forward(self._model, self._data)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """

        policy_action, intervention_bool = action
        teleop_action = None
        self.current_joint_positions = self.data.qpos[self.panda_dof_ids].astype(np.float32)
        if isinstance(policy_action, torch.Tensor):
            policy_action = policy_action.cpu().numpy()
            policy_action = np.clip(
                policy_action, self.action_space[0].low, self.action_space[0].high
            )
            logging.info(f"Before clipping policy action: {policy_action}")

        if not intervention_bool:
            if self.use_delta_action_space:
                target_joint_positions = (
                    self.current_joint_positions + self.delta * policy_action
                )
            else:
                target_joint_positions = policy_action

            self._data.ctrl[self.panda_ctrl_ids] = target_joint_positions
            mujoco.mj_step(self._model, self._data)

            observation = self._compute_observation()
        else:
            # TODO (lilkm) : handle teleoperation
            pass
            # observation, teleop_action = self.robot.teleop_step(record_data=True)
            # teleop_action = teleop_action[
            #     "action"
            # ]  # Convert tensor to appropriate format

            # # When applying the delta action space, convert teleop absolute values to relative differences.
            # if self.use_delta_action_space:
            #     logging.info(f"Absolute teleop action: {teleop_action}")
            #     teleop_action = (
            #         teleop_action - self.current_joint_positions
            #     ) / self.delta
            #     logging.info(f"Relative teleop action: {teleop_action}")
            #     if self.relative_bounds_size is not None and (
            #         torch.any(teleop_action < -self.relative_bounds_size)
            #         and torch.any(teleop_action > self.relative_bounds_size)
            #     ):
            #         logging.debug(
            #             f"Relative teleop delta exceeded bounds {self.relative_bounds_size}, teleop_action {teleop_action}\n"
            #             f"lower bounds condition {teleop_action < -self.relative_bounds_size}\n"
            #             f"upper bounds condition {teleop_action > self.relative_bounds_size}"
            #         )

            #         teleop_action = torch.clamp(
            #             teleop_action,
            #             -self.relative_bounds_size,
            #             self.relative_bounds_size,
            #         )
            # # NOTE: To mimic the shape of a neural network output, we add a batch dimension to the teleop action.
            # if teleop_action.dim() == 1:
            #     teleop_action = teleop_action.unsqueeze(0)

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "action_intervention": teleop_action,
                "is_intervention": teleop_action is not None,
            },
        )


    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(
                self._viewer.render()
            )
        return rendered_frames


    def _compute_observation(self) -> dict:
        obs = {}
        obs["observation.state"] = {}

        qpos = self.data.qpos[self.panda_dof_ids].astype(np.float32)
        obs["observation.state"] = torch.from_numpy(qpos)

        front, wrist = self.render()
        obs["observation.images.front"], obs["observation.images.wrist"] = torch.from_numpy(front), torch.from_numpy(wrist)

        return obs


if __name__ == "__main__":
    env = PandaPushCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
