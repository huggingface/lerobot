import argparse
import importlib
import logging
import os
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np
import torch
import tqdm

from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot import envs

# Base module path for Isaac Lab Arena example environments
# This can be overridden via environment variable for custom installations
ISAACLAB_ARENA_ENV_MODULE = os.environ.get(
    "ISAACLAB_ARENA_ENV_MODULE",
    "isaaclab_arena.examples.example_environments"
)

# Environment aliases for leaner CLI commands
# Maps short names to full module paths: "module.path.ClassName"
ENVIRONMENT_ALIASES: dict[str, str] = {
    # GR1 environments
    "gr1_microwave": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.gr1_open_microwave_environment"
        ".Gr1OpenMicrowaveEnvironment"
    ),
    # Galileo environments
    "galileo_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.galileo_pick_and_place_environment"
        ".GalileoPickAndPlaceEnvironment"
    ),
    "g1_locomanip_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}"
        ".galileo_g1_locomanip_pick_and_place_environment"
        ".GalileoG1LocomanipPickAndPlaceEnvironment"
    ),
    # Kitchen environments
    "kitchen_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.kitchen_pick_and_place_environment"
        ".KitchenPickAndPlaceEnvironment"
    ),
    # Other environments
    "press_button": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.press_button_environment"
        ".PressButtonEnvironment"
    ),
}


def resolve_environment_alias(environment: str) -> str:
    """Resolve an environment alias to its full module path.

    Args:
        environment: Either an alias or a full module path.

    Returns:
        The full module path for the environment.
    """
    return ENVIRONMENT_ALIASES.get(environment, environment)


class IsaacLabVectorEnvWrapper:
    """Wrapper to make IsaacLab environments compatible with gym.vector.VectorEnv interface.

    IsaacLab environments handle multiple parallel envs internally but don't follow
    the gym.vector.VectorEnv API. This wrapper adapts the interface for compatibility
    with lerobot evaluation scripts.

    Note: We don't inherit from gym.vector.VectorEnv since it's an abstract generic
    class. Instead, we implement the required interface directly.
    """

    def __init__(self, env, episode_length: int = 500, task: str | None = None):
        self._env = env
        self._num_envs = env.num_envs
        self._episode_length = episode_length

        # Copy spaces from underlying env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.task = task

        # Metadata for video recording
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

        # Track step count per environment for max_episode_steps
        self._step_counts = np.zeros(self._num_envs, dtype=np.int32)

    @property
    def unwrapped(self):
        """Return the base unwrapped environment."""
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def _max_episode_steps(self) -> int:
        """Return max episode steps for compatibility with lerobot_eval."""
        return self._episode_length

    def reset(self, seed=None, options=None):
        """Reset the environment(s).

        Args:
            seed: Either a single int or a list of ints. IsaacLab expects a single int,
                  so we use the first seed if a list is provided.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info)
        """
        # IsaacLab expects a single seed, not a list
        if isinstance(seed, (list, tuple, range)):
            seed = seed[0] if len(seed) > 0 else None

        self._step_counts = np.zeros(self._num_envs, dtype=np.int32)
        obs, info = self._env.reset(seed=seed, options=options)

        # Ensure info has the expected structure
        if "final_info" not in info:
            info["final_info"] = {"is_success": np.zeros(self._num_envs, dtype=bool)}

        return obs, info

    def step(self, actions):
        """Step the environment(s).

        Args:
            actions: Actions to apply (numpy array or torch tensor).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert numpy to torch if needed (IsaacLab expects torch tensors)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        obs, reward, terminated, truncated, info = self._env.step(actions)

        self._step_counts += 1

        # Convert torch tensors to numpy for gym compatibility
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()

        # Ensure info has the expected final_info structure for VectorEnv
        # Gymnasium >= 1.0 expects final_info to be a dict with numpy arrays
        if "final_info" not in info:
            # Check for success in the info dict
            is_success = info.get("is_success", np.zeros(self._num_envs, dtype=bool))
            if isinstance(is_success, torch.Tensor):
                is_success = is_success.cpu().numpy()
            elif not isinstance(is_success, np.ndarray):
                is_success = np.array([is_success] * self._num_envs, dtype=bool)
            info["final_info"] = {"is_success": is_success}
        elif isinstance(info["final_info"], dict):
            # Ensure is_success is a numpy array
            if "is_success" in info["final_info"]:
                is_success = info["final_info"]["is_success"]
                if isinstance(is_success, torch.Tensor):
                    info["final_info"]["is_success"] = is_success.cpu().numpy()
                elif not isinstance(is_success, np.ndarray):
                    info["final_info"]["is_success"] = np.array(
                        [is_success] * self._num_envs, dtype=bool
                    )

        return obs, reward, terminated, truncated, info

    def call(self, method_name: str, *args, **kwargs):
        """Call a method on the underlying environment(s).

        This mimics gym.vector.VectorEnv.call() which returns a list of results.
        """
        if method_name == "_max_episode_steps":
            return [self._episode_length] * self._num_envs
        elif method_name == "task":
            return [self.task] * self._num_envs
        elif method_name == "render":
            # Return rendered frames for each environment
            return self._render_all()
        elif hasattr(self._env, method_name):
            result = getattr(self._env, method_name)(*args, **kwargs)
            # Wrap single result in list for VectorEnv compatibility
            if not isinstance(result, list):
                return [result] * self._num_envs
            return result
        else:
            raise AttributeError(f"Environment has no method '{method_name}'")

    def _render_all(self):
        """Render all environments and return list of frames."""
        # IsaacLab renders all envs at once, we need to split by env
        if hasattr(self._env, "render"):
            frames = self._env.render()
            if frames is not None:
                if isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                # If single frame, replicate for all envs
                if frames.ndim == 3:  # (H, W, C)
                    return [frames] * self._num_envs
                elif frames.ndim == 4:  # (N, H, W, C)
                    return [frames[i] for i in range(min(len(frames), self._num_envs))]
        return [np.zeros((480, 640, 3), dtype=np.uint8)] * self._num_envs

    def render(self):
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def envs(self):
        """Return list of sub-environments for SyncVectorEnv compatibility."""
        # Return self wrapped in a list to mimic SyncVectorEnv.envs
        return [self] * self._num_envs

    @property
    def device(self):
        """Return the device of the underlying environment."""
        return self._env.device if hasattr(self._env, "device") else "cpu"


"""Zero action rollout example for Isaac Lab Arena.

Usage examples:

Run a zero action rollout for 500 steps.

ARGS:
- env.environment: environment alias or full module path
    Supported aliases: gr1_microwave, galileo_pnp, g1_locomanip_pnp, kitchen_pnp, press_button
    Or full path: "module.path.ClassName" where ClassName has get_env() returning IsaacLabArenaEnvironment
- env.embodiment: embodiment to use (e.g., gr1_pink, gr1_joint)
- env.object: object to use (e.g., mustard_bottle, cracker_box)

OPTIONAL ARGS: In case you want to override the default values in the config file.
- env.num_envs: number of environments to run
- env.seed: seed for the random number generator
- More args are supported by the environment class, see the IsaaclabArenaEnv config class for more details.


```
python -m lerobot.envs.arena \
    --env.type=isaaclab_arena \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=cracker_box \
    --env.num_envs=4 \
    --env.seed=1000
```


```
lerobot-eval \
    --policy.path=nvkartik/smolvla-arena-gr1-microwave-test \
    --env.type=isaaclab_arena \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.num_envs=1 \
    --env.headless=true \
    --policy.device=cuda \
    --env.enable_cameras=true
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


def create_isaaclab_arena_envs(
    cfg: envs.EnvConfig,
):  # -> isaaclab.envs.manager_based_env.ManagerBasedEnv
    """Create IsaacLab Arena environments wrapped for gym.vector.VectorEnv compatibility.

    Args:
        cfg: Environment configuration containing environment path, embodiment, etc.

    Returns:
        Dict mapping environment name to task_id to wrapped VectorEnv.
    """
    from isaaclab.app import AppLauncher

    if cfg.enable_pinocchio:
        import pinocchio  # noqa: F401

    as_isaaclab_argparse = argparse.Namespace(**asdict(cfg))

    print("Launching simulation app")
    _simulation_app = AppLauncher(as_isaaclab_argparse)  # noqa: F841

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Resolve alias to full module path if needed
    environment_path = resolve_environment_alias(cfg.environment)

    # Discover the environment module and class from the environment string
    module_path, class_name = environment_path.rsplit(".", 1)
    environment_module = importlib.import_module(module_path)
    environment_class = getattr(environment_module, class_name)()

    env_builder = ArenaEnvBuilder(
        environment_class.get_env(as_isaaclab_argparse), as_isaaclab_argparse
    )
    raw_env = env_builder.make_registered()

    # Wrap the IsaacLab env to be compatible with gym.vector.VectorEnv interface
    episode_length = getattr(cfg, "episode_length", 500)
    wrapped_env = IsaacLabVectorEnvWrapper(
        raw_env, episode_length=episode_length, task=cfg.task
    )

    return {cfg.environment: {0: wrapped_env}}


@parser.wrap()
def arena_main(cfg: ArenaConfig):
    logging.info(pformat(asdict(cfg)))

    envs = create_isaaclab_arena_envs(cfg.env)
    env = next(iter(envs.values()))[0]

    env.reset()

    # Run zero action rollout for the episode length
    for _ in tqdm.tqdm(range(cfg.env.episode_length)):
        with torch.inference_mode():
            # Action shape is (num_envs, action_dim) for batched environments
            action_dim = env.action_space.shape[-1]
            actions = torch.zeros((env.num_envs, action_dim), device=env.device)
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(obs.keys())


def main():
    init_logging()
    arena_main()


if __name__ == "__main__":
    main()
