from collections import defaultdict
from typing import Any, Callable, Union
from itertools import chain
import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces
import json
from huggingface_hub import hf_hub_download
import metaworld.policies as policies

# download the JSON
path = hf_hub_download(
    repo_id="jadechoghari/smolvla-metaworld-keys",
    filename="tasks_metaworld.json"
)

# load JSON
with open(path, "r") as f:
    data = json.load(f)

# extract dicts
TASK_DESCRIPTIONS = data["TASK_DESCRIPTIONS"]
TASK_NAME_TO_ID = data["TASK_NAME_TO_ID"]
DIFFICULTY_TO_TASKS = data["DIFFICULTY_TO_TASKS"]

# this convert policy strings to real classes
TASK_POLICY_MAPPING = {
    k: getattr(policies, v) for k, v in data["TASK_POLICY_MAPPING"].items()
}

def create_metaworld_envs(task: str, n_envs: int, gym_kwargs: dict[str, Any] = None, env_cls: Callable = None, multitask_eval: bool = True)  -> Union[gym.vector.VectorEnv, dict[str, dict[str, gym.vector.VectorEnv]]]:
    if gym_kwargs is None:
        gym_kwargs = {}
    if not multitask_eval:
        tasks = task.split(",")
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(n_envs)]
        elif len(tasks) < n_envs and n_envs % len(tasks) == 0:
            n_repeat = n_envs // len(tasks)
            tasks = list(chain.from_iterable([[item] * n_repeat for item in tasks]))
        elif n_envs < len(tasks):
            tasks = tasks[:n_envs]
        assert n_envs == len(tasks), "n_envs and len(tasks) must be the same!"
        print(f"Creating Meta-World envs with tasks {tasks}")
        return env_cls([lambda i=i: MetaworldEnv(task=tasks[i], **gym_kwargs) for i in range(n_envs)])
    else:
        envs = defaultdict(dict)
        tasks = task.split(",")
        if tasks[0] not in DIFFICULTY_TO_TASKS: # evaluation on individual tasks
            task_groups = ["all"]
        else:
            task_groups = tasks
        for task_group in task_groups:
            _tasks = DIFFICULTY_TO_TASKS.get(task_group, task_groups)
            for _task in _tasks:
                print(f"Creating Meta-World envs with task {_task} from task group {task_group}")
                envs_list = [lambda i=i: MetaworldEnv(task=_task, **gym_kwargs) for i in range(n_envs)]
                envs[task_group][_task] = env_cls(envs_list)
        return envs


class MetaworldEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = 480
        self.observation_height = 480
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name

        self._env = self._make_envs_task(self.task)
        self._max_episode_steps = self._env.max_path_length
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(4,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def render(self):
        image = self._env.render()
        if self.camera_name == "corner2":
            image = np.flip(image, (0, 1))  # images for some reason are flipped
        return image

    def _make_envs_task(self, env_name: str):
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name=self.camera_name)
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [
                0.75,
                0.075,
                0.7,
            ]  # corner2 position, similar to https://arxiv.org/pdf/2206.14244
        env.reset()
        env._freeze_rand_vec = False  # otherwise no randomization
        return env

    def _format_raw_obs(self, raw_obs, env=None):
        image = None
        if env is not None:
            image = env.render()
            if self.camera_name == "corner2":
                image = np.flip(image, (0, 1))  # images for some reason are flipped
        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"pixels": image.copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": image.copy(),
                "agent_pos": agent_pos,
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        raw_obs, info = self._env.reset(seed=seed)

        observation = self._format_raw_obs(raw_obs, env=self._env)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        raw_obs, reward, done, truncated, info = self._env.step(action)

        terminated = is_success = int(info["success"]) == 1
        info["is_success"] = is_success

        observation = self._format_raw_obs(raw_obs, env=self._env)

        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()