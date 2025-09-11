from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from collections.abc import Callable
from itertools import chain
from typing import Any, Dict, List
from collections.abc import Callable, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

logger = logging.getLogger(__name__)

# ---- Helpers -----------------------------------------------------------------


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str):
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    LiberoEnv: type,  # injected to avoid forward ref issues if needed
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""
    joined_cams = ",".join(camera_names)  # keep backward-compat: downstream expects a string
    fns: list[Callable[[], LiberoEnv]] = []
    for i in range(n_envs):

        def _mk(
            i=i,
            suite=suite,
            task_id=task_id,
            suite_name=suite_name,
            joined_cams=joined_cams,
            init_states=init_states,
            gym_kwargs=dict(gym_kwargs),
        ):
            return LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=suite_name,
                camera_name=joined_cams,
                init_states=init_states,
                episode_index=i,
                **gym_kwargs,
            )

        fns.append(_mk)
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    multitask_eval: bool = True,  # kept for signature compatibility; return type is consistent regardless
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    # Avoid circular import/type issues: assume LiberoEnv is defined in this module
    try:
        LiberoEnv  # type: ignore[name-defined]
    except NameError:
        # If LiberoEnv is in the same file, this won't run. If it's elsewhere, import here.
        exit()
        # from .libero_env import LiberoEnv  # adjust if your class lives in another module

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    logger.info(
        "Creating LIBERO envs | suites=%s | n_envs(per task)=%d | init_states=%s | multitask_eval=%s",
        suite_names,
        n_envs,
        init_states,
        bool(multitask_eval),
    )
    if task_ids_filter is not None:
        logger.info("Restricting to task_ids=%s", task_ids_filter)

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)

        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                LiberoEnv=LiberoEnv,
            )
            out[suite_name][tid] = env_cls(fns)
            logger.debug("Built vec env | suite=%s | task_id=%d | n_envs=%d", suite_name, tid, n_envs)

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}


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
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


OBS_STATE_DIM = 8
ACTION_DIM = 7


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

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        self.camera_name_mapping = {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        }

        self.num_steps_wait = (
            10  # Do nothing for the first few timesteps to wait for the simulator drops objects
        )
        self.episode_index = episode_index

        self._env = self._make_envs_task(task_suite, self.task_id)
        TASK_SUITE_MAX_STEPS: dict[str, int] = {
            "libero_spatial": 220,  # longest training demo has 193 steps
            "libero_object": 280,  # longest training demo has 254 steps
            "libero_goal": 300,  # longest training demo has 270 steps
            "libero_10": 520,  # longest training demo has 505 steps
            "libero_90": 400,  # longest training demo has 373 steps
        }
        default_steps = 500
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

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
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32)

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        return image

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
        state = np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        )
        agent_pos = state
        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )
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
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info["is_success"] = done  # is_success

        observation = self._format_raw_obs(raw_obs)
        if done:
            self.reset()
            print(self.task, self.task_id, done, is_success)
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def create_libero_envs1(
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
