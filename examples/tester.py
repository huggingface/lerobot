# here we make sure that the libero envs follow the gymnasium API and are reproducible
from lerobot.envs.libero import create_libero_envs, get_task_init_states
from gymnasium.utils.env_checker import check_env
import gymnasium as gym
import numpy as np
from lerobot.utils.random_utils import set_seed
import torch
import random


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def validate_libero_env(task: str, seed: int = 123, n_steps: int = 1):
    set_seed(seed)
    control_seed(seed)
    # --- Create env ---
    envs = create_libero_envs(
        task=task,
        n_envs=1,
        env_cls=gym.vector.SyncVectorEnv,
    )
    suite_name = next(iter(envs))
    task_id = next(iter(envs[suite_name]))
    vec_env = envs[suite_name][task_id]

    # api check
    raw_env = vec_env.env_fns[0]()
    check_env(raw_env, warn=True)
    # reset reproducibility
    obs1, _ = vec_env.reset(seed=seed)
    obs2, _ = vec_env.reset(seed=seed)
    for k1 in obs1:
        v1, v2 = obs1[k1], obs2[k1]
        if isinstance(v1, dict):
            for k2 in v1:
                if isinstance(v1[k2], np.ndarray):
                    assert np.array_equal(v1[k2], v2[k2]), f"Mismatch at reset in {k1}.{k2}"
                else:
                    assert v1[k2] == v2[k2], f"Mismatch at reset in {k1}.{k2}"
        elif isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f"Mismatch at reset in {k1}"
        else:
            assert v1 == v2, f"Mismatch at reset in {k1}"

    # step reproducibility
    obs1, _ = vec_env.reset(seed=seed)
    obs2, _ = vec_env.reset(seed=seed)
    for step in range(n_steps):
        dummy_action = np.zeros((vec_env.num_envs, vec_env.single_action_space.shape[0]), dtype=np.float32)
        obs1, _, _, _, _ = vec_env.step(dummy_action)
        obs2, _, _, _, _ = vec_env.step(dummy_action)
        for k1 in obs1:
            v1, v2 = obs1[k1], obs2[k1]
            if isinstance(v1, dict):
                for k2 in v1:
                    if isinstance(v1[k2], np.ndarray):
                        assert np.array_equal(v1[k2], v2[k2]), f"Mismatch at step{step} in {k1}.{k2}"
                    else:
                        assert v1[k2] == v2[k2], f"Mismatch at step{step} in {k1}.{k2}"
            elif isinstance(v1, np.ndarray):
                assert np.array_equal(v1, v2), f"Mismatch at step{step} in {k1}"
            else:
                assert v1 == v2, f"Mismatch at step{step} in {k1}"

    print(f"✅ {task} passes API and reproducibility check")

import numpy as np
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# setup
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)

# fix init state
init_states = task_suite.get_task_init_states(task_id)
init_state_id = 0

def one_step_pixels(seed):
    env.seed(seed)
    env.reset()
    env.set_init_state(init_states[init_state_id])
    dummy_action = [0.] * 7
    obs, reward, done, info = env.step(dummy_action)
    return obs["agentview_image"].copy()

# run twice with the same seed
pix1 = one_step_pixels(123)
pix2 = one_step_pixels(123)

print("pixels reproducible:", np.array_equal(pix1, pix2))

env.close()


# run test
validate_libero_env(task="libero_spatial")
