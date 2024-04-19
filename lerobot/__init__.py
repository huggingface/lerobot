"""
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_policies)
        print(lerobot.available_policies_per_env)
    ```

When implementing a new dataset class (e.g. `AlohaDataset`) follow these steps:
- Update `available_datasets` in `lerobot/__init__.py`
- Set the required `available_datasets` class attribute using the previously updated `lerobot.available_datasets`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_envs`, `available_tasks_per_env` and `available_datasets` in `lerobot/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` in `lerobot/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

from lerobot.__version__ import __version__  # noqa: F401

available_envs = [
    "aloha",
    "pusht",
    "xarm",
]

available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
}

available_datasets = {
    "aloha": [
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
    ],
    "pusht": ["pusht"],
    "xarm": ["xarm_lift_medium"],
}

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
env_dataset_pairs = [(env, dataset) for env, datasets in available_datasets.items() for dataset in datasets]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
