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

When implementing a new dataset (e.g. `AlohaDataset`), policy (e.g. `DiffusionPolicy`), or environment, follow these steps:
- Set the required class attributes: `available_datasets`.
- Set the required class attributes: `name`.
- Update variables in `lerobot/__init__.py` (e.g. `available_envs`, `available_datasets_per_envs`, `available_policies`)
- Update variables in `tests/test_available.py` by importing your new class
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

available_datasets_per_env = {
    "aloha": [
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
    ],
    "pusht": ["pusht"],
    "xarm": ["xarm_lift_medium"],
}

available_datasets = [dataset for env in available_envs for dataset in available_datasets_per_env[env]]

available_policies = [
    "act",
    "diffusion",
    "tdmpc",
]
