import math
import os
from collections import defaultdict
from collections.abc import Callable
from itertools import chain
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] = None,
    camera_name: str = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable = None,
    multitask_eval: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Here n_envs is per task and equal to the number of rollouts.
    Returns:
        dict[str, dict[str, list[LiberoEnv]]]: keys are task_suite and values are list of LiberoEnv envs.
    """
    print("num envs", n_envs)
    print("multitask_eval", multitask_eval)
    print("gym_kwargs", gym_kwargs)
    if gym_kwargs is None:
        gym_kwargs = {}

    if not multitask_eval:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task]()  # can also choose libero_spatial, libero_object, libero_10 etc.
        tasks_id = list(range(len(task_suite.tasks)))
        episode_indices = [0 for i in range(len(tasks_id))]
        if len(tasks_id) == 1:
            tasks_id = [tasks_id[0] for _ in range(n_envs)]
            episode_indices = list(range(n_envs))
        elif len(tasks_id) < n_envs and n_envs % len(tasks_id) == 0:
            n_repeat = n_envs // len(tasks_id)
            print("n_repeat", n_repeat)
            episode_indices = []
            for _ in range(len(tasks_id)):
                episode_indices.extend(list(range(n_repeat)))
            tasks_id = list(chain.from_iterable([[item] * n_repeat for item in tasks_id]))
        elif n_envs < len(tasks_id):
            tasks_id = tasks_id[:n_envs]
            episode_indices = list(range(n_envs))[:n_envs]
            print(f"WARNING: n_envs < len(tasks_id), evaluating only on {tasks_id}")
        print(f"Creating Libero envs with task ids {tasks_id} from suite {task}")
        assert n_envs == len(tasks_id), (
            f"len(n_envs) and tasks_id should be the same, got {n_envs} and {len(tasks_id)}"
        )
        return env_cls(
            [
                lambda i=i: LiberoEnv(
                    task_suite=task_suite,
                    task_id=tasks_id[i],
                    task_suite_name=task,
                    camera_name=camera_name,
                    init_states=init_states,
                    episode_index=episode_indices[i],
                    **gym_kwargs,
                )
                for i in range(n_envs)
            ]
        )
    else:
        envs = defaultdict(dict)
        benchmark_dict = benchmark.get_benchmark_dict()
        task = task.split(",")
        for _task in task:
            task_suite = benchmark_dict[
                _task
            ]()  # can also choose libero_spatial, libero_object, libero_10 etc.
            tasks_ids = list(range(len(task_suite.tasks)))
            # tasks_ids = [0] # FIXME(mshukor): debug
            for tasks_id in tasks_ids:
                episode_indices = list(range(n_envs))
                print(
                    f"Creating Libero envs with task ids {tasks_id} from suite {_task}, episode_indices: {episode_indices}"
                )
                envs_list = [
                    (
                        lambda i=i,
                        task_suite=task_suite,
                        tasks_id=tasks_id,
                        _task=_task,
                        episode_indices=episode_indices: LiberoEnv(
                            task_suite=task_suite,
                            task_id=tasks_id,
                            task_suite_name=_task,
                            camera_name=camera_name,
                            init_states=init_states,
                            episode_index=episode_indices[i],
                            **gym_kwargs,
                        )
                    )
                    for i in range(n_envs)
                ]
                envs[_task][tasks_id] = env_cls(envs_list)
        return envs


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_task_init_states(task_suite, i):
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        task_suite.tasks[i].problem_folder,
        task_suite.tasks[i].init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite,
        task_id,
        task_suite_name,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=256,
        observation_height=256,
        visualization_width=640,
        visualization_height=480,
        init_states=True,
        episode_index=0,
    ):
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = camera_name.split(
            ","
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)
        # TODO: jadechoghari, check mapping
        self.camera_name_mapping = {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        }

        self.num_steps_wait = (
            10  # Do nothing for the first few timesteps to wait for the simulator drops objects
        )
        self.episode_index = episode_index

        self._env = self._make_envs_task(task_suite, self.task_id)
        if task_suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        self._max_episode_steps = max_steps

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(8,),  # TODO: jadechoghari, check compatible
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def render(self):
        raw_obs = self._env.env._get_observations()
        formatted = self._format_raw_obs(raw_obs)
        # grab the "main" camera
        return formatted["pixels"]["image"]

    def _make_envs_task(self, task_suite, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        if self.init_states:
            init_states = get_task_init_states(
                task_suite, task_id
            )  # for benchmarking purpose, we fix the a set of initial states FIXME(mshukor): should be in the reset()?
            init_state_id = self.episode_index  # episode index
            env.set_init_state(init_states[init_state_id])

        return env

    def _format_raw_obs(self, raw_obs):
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            image = image[::-1, ::-1]  # rotate 180 degrees
            images[self.camera_name_mapping[camera_name]] = image
        # images = image if len(images) == 1 else images
        state = np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        )
        agent_pos = state
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"pixels": images.copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": images.copy(),
                "agent_pos": agent_pos,
            }
        return obs

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self._env.seed(seed)
        raw_obs = self._env.reset()
        # Do nothing for the first few timesteps to wait for the simulator drops objects
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        terminated = done or is_success
        info["is_success"] = is_success
        observation = self._format_raw_obs(raw_obs)
        truncated = False
        # note if it is unable to complete get libero error after many steps
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()
