import logging
from pprint import pformat
from dataclasses import dataclass
from dataclasses import asdict
import torch
import tqdm
import argparse
from typing import Any
import importlib


from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot import envs

"""Zero action rollout example for Isaac Lab Arena.

Usage examples:

Run a zero action rollout for 500 steps.

ARGS:
- env.environment: environment to use
    expected to be "path.to.env.module.ClassName"
    "ClassName" must include a get_env method that returns an `IsaacLabArenaEnvironment` instance
- env.embodiment: embodiment to use
- env.object: object to use

OPTIONAL ARGS: In case you want to override the default values in the config file.
- env.n_envs: number of environments to run
- env.seed: seed for the random number generator
- More args are supported by the environment class, see the IsaaclabArenaEnv config class for more details.


```
python -m lerobot.envs.arena \
    --env.type=isaaclab_arena \
    --env.environment="isaaclab_arena.examples.example_environments.gr1_open_microwave_environment.Gr1OpenMicrowaveEnvironment" \
    --env.embodiment=gr1_pink \
    --env.object=cracker_box \
    --env.num_envs=4 \
    --env.seed=1000
```
"""


@dataclass
class ArenaConfig:
    env: envs.EnvConfig


def config_to_namespace(config: dict[str, Any]) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace.
    This is for compatibility with ArenaEnvBuilder.
    """
    return argparse.Namespace(**config)


@parser.wrap()
def arena_main(cfg: ArenaConfig):
    logging.info(pformat(asdict(cfg)))

    from isaaclab.app import AppLauncher

    if cfg.env.enable_pinocchio:
        import pinocchio  # noqa: F401

    print("Launching simulation app")
    _simulation_app = AppLauncher()  # noqa: F841

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Discover the environment module and class from the environment string
    module_path, class_name = cfg.env.environment.rsplit(".", 1)
    environment_module = importlib.import_module(module_path)
    environment_class = getattr(environment_module, class_name)()

    as_isaaclab_argparse = argparse.Namespace(**asdict(cfg.env))
    env_builder = ArenaEnvBuilder(environment_class.get_env(as_isaaclab_argparse), as_isaaclab_argparse)
    env = env_builder.make_registered()
    
    env.reset()

    # Run zero action rollout for the episode length
    for _ in tqdm.tqdm(range(cfg.env.episode_length)):
        with torch.inference_mode():
            action_shape = env.action_space.shape
            actions = torch.zeros(action_shape, device=env.unwrapped.device)
            env.step(actions)


def main():
    init_logging()
    arena_main()


if __name__ == "__main__":
    main()
