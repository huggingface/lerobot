"""
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_datasets_per_env)
        print(lerobot.available_real_world_datasets)
        print(lerobot.available_policies)
        print(lerobot.available_policies_per_env)
    ```

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `lerobot/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

from lerobot.__version__ import __version__  # noqa: F401

available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
}
available_envs = list(available_tasks_per_env.keys())

available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
    ],
    "pusht": ["lerobot/pusht"],
    "xarm": [
        "lerobot/xarm_lift_medium",
        "lerobot/xarm_lift_medium_replay",
        "lerobot/xarm_push_medium",
        "lerobot/xarm_push_medium_replay",
    ],
}

available_real_world_datasets = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_coffee_new",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_pro_pencil",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_vinh_cup",
    "lerobot/aloha_static_vinh_cup_left",
    "lerobot/aloha_static_ziploc_slide",
    "lerobot/umi_cup_in_the_wild",
]

available_datasets = list(
    itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets)
)

available_policies = [
    "act",
    "diffusion",
    "tdmpc",
]

available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion"],
    "xarm": ["tdmpc"],
}

env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
