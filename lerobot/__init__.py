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
available_datasets = [dataset for datasets in available_datasets_per_env.values() for dataset in datasets]

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
