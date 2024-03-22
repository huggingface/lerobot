"""
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_policies)
    ```

Note:
    When implementing a concrete class (e.g. `AlohaDataset`, `PushtEnv`, `DiffusionPolicy`), you need to:
        1. set the required class attributes:
            - for classes inheriting from `AbstractDataset`: `available_datasets`
            - for classes inheriting from `AbstractEnv`: `name`, `available_tasks`
            - for classes inheriting from `AbstractPolicy`: `name`
        2. update variables in `lerobot/__init__.py` (e.g. `available_envs`, `available_datasets_per_envs`, `available_policies`)
        3. update variables in `tests/test_available.py` by importing your new class
"""

from lerobot.__version__ import __version__  # noqa: F401

available_envs = [
    "aloha",
    "pusht",
    "simxarm",
]

available_tasks_per_env = {
    "aloha": [
        "sim_insertion",
        "sim_transfer_cube",
    ],
    "pusht": ["pusht"],
    "simxarm": ["lift"],
}

available_datasets_per_env = {
    "aloha": [
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
    ],
    "pusht": ["pusht"],
    "simxarm": ["xarm_lift_medium"],
}

available_datasets = [dataset for env in available_envs for dataset in available_datasets_per_env[env]]

available_policies = [
    "act",
    "diffusion",
    "tdmpc",
]
