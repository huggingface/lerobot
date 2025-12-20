"""IsaacLab Arena environment integration for LeRobot.

IsaacLab environments are GPU-accelerated batched environments that handle
multiple parallel environments internally. The wrapper adapts this to
gym.vector.VectorEnv interface for LeRobot compatibility.

Usage:
    from lerobot.envs.factory import make_env
    envs = make_env(cfg, n_envs=4)
"""

from __future__ import annotations

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

from lerobot import envs
from lerobot.configs import parser
from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.utils.utils import init_logging

ISAACLAB_ARENA_ENV_MODULE = os.environ.get("ISAACLAB_ARENA_ENV_MODULE", "isaaclab_arena_environments")

# Environment aliases for common configurations
ENVIRONMENT_ALIASES: dict[str, str] = {
    "gr1_microwave": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.gr1_open_microwave_environment.Gr1OpenMicrowaveEnvironment"
    ),
    "galileo_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.galileo_pick_and_place_environment.GalileoPickAndPlaceEnvironment"
    ),
    "g1_locomanip_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}"
        ".galileo_g1_locomanip_pick_and_place_environment"
        ".GalileoG1LocomanipPickAndPlaceEnvironment"
    ),
    "kitchen_pnp": (
        f"{ISAACLAB_ARENA_ENV_MODULE}.kitchen_pick_and_place_environment.KitchenPickAndPlaceEnvironment"
    ),
    "press_button": (f"{ISAACLAB_ARENA_ENV_MODULE}.press_button_environment.PressButtonEnvironment"),
}


def resolve_environment_alias(environment: str) -> str:
    """Resolve an environment alias to its full module path."""
    return ENVIRONMENT_ALIASES.get(environment, environment)


class IsaacLabVectorEnvWrapper:
    """Wrapper adapting IsaacLab batched GPU env to VectorEnv interface.

    IsaacLab handles vectorization internally on GPU, unlike gym's
    SyncVectorEnv/AsyncVectorEnv. This provides the expected interface
    for LeRobot evaluation.

    Video Recording:
        Supports gymnasium.wrappers.RecordVideo (IsaacLab native approach).
        Requires enable_cameras=True in config when running headless.
        See: isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env,
        episode_length: int = 500,
        task: str | None = None,
        render_mode: str | None = "rgb_array",
    ):
        self._env = env
        self._num_envs = env.num_envs
        self._episode_length = episode_length
        self._closed = False
        self.render_mode = render_mode

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.task = task

        # Use env metadata if available
        if hasattr(env, "metadata") and env.metadata:
            self.metadata = {**self.metadata, **env.metadata}

    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_length

    @property
    def device(self) -> str:
        return getattr(self._env, "device", "cpu")

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all environments."""
        # IsaacLab expects a single seed
        if isinstance(seed, (list, tuple, range)):
            seed = seed[0] if len(seed) > 0 else None

        obs, info = self._env.reset(seed=seed, options=options)

        if "final_info" not in info:
            info["final_info"] = {"is_success": np.zeros(self._num_envs, dtype=bool)}

        return obs, info

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all environments."""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        obs, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to numpy for gym compatibility
        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)

        # Extract success status
        is_success = self._get_success(terminated, truncated)
        info["final_info"] = {"is_success": is_success}

        return obs, reward, terminated, truncated, info

    def _get_success(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        """Extract per-environment success status from termination manager."""
        is_success = np.zeros(self._num_envs, dtype=bool)

        term_manager = self._env.termination_manager
        success_tensor = term_manager.get_term("success")
        if isinstance(success_tensor, torch.Tensor):
            is_success = success_tensor.cpu().numpy().astype(bool)
        else:
            is_success = np.array(success_tensor, dtype=bool)

        return is_success & (terminated | truncated)

    def call(self, method_name: str, *args, **kwargs) -> list[Any]:
        """Call a method on the underlying environment(s)."""
        if method_name == "_max_episode_steps":
            return [self._episode_length] * self._num_envs
        if method_name == "task":
            return [self.task] * self._num_envs
        if method_name == "render":
            return self.render_all()

        if hasattr(self._env, method_name):
            attr = getattr(self._env, method_name)
            result = attr(*args, **kwargs) if callable(attr) else attr
            if isinstance(result, list):
                return result
            return [result] * self._num_envs

        raise AttributeError(f"Environment has no method/attribute '{method_name}'")

    def render_all(self) -> list[np.ndarray]:
        """Render all environments and return list of frames.

        Public method for LeRobot eval video recording.
        Returns a list of RGB frames, one per environment.
        """
        frames = self.render()
        if frames is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            return [placeholder] * self._num_envs

        if frames.ndim == 3:  # Single frame (H, W, C)
            return [frames] * self._num_envs
        if frames.ndim == 4:  # Batch (N, H, W, C)
            return [frames[i] for i in range(min(len(frames), self._num_envs))]

        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        return [placeholder] * self._num_envs

    def render(self) -> np.ndarray | None:
        """Render the environment (gymnasium RecordVideo compatible).

        Returns rgb_array for video recording per IsaacLab native approach.
        Requires enable_cameras=True in config when running headless.
        """
        if self.render_mode != "rgb_array":
            return None

        frames = self._env.render() if hasattr(self._env, "render") else None
        if frames is None:
            return None

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Return first env frame for RecordVideo compatibility
        if frames.ndim == 4:
            return frames[0]
        return frames

    def close(self) -> None:
        """Close the environment and release resources."""
        if not self._closed:
            if hasattr(self._env, "close"):
                self._env.close()
            self._closed = True

    @property
    def envs(self) -> list[IsaacLabVectorEnvWrapper]:
        """Return list of sub-environments for VectorEnv compatibility."""
        return [self] * self._num_envs

    def __del__(self):
        self.close()


def _validate_env_config(
    env,
    state_keys: tuple[str, ...],
    camera_keys: tuple[str, ...],
    cfg_state_dim: int,
    cfg_action_dim: int,
) -> None:
    """Validate observation keys and dimensions against IsaacLab managers."""
    obs_manager = env.observation_manager
    active_terms = obs_manager.active_terms
    policy_terms = set(active_terms.get("policy", []))
    camera_terms = set(active_terms.get("camera_obs", []))

    # Validate keys exist
    missing_state = [k for k in state_keys if k not in policy_terms]
    if missing_state:
        raise ValueError(f"Invalid state_keys: {missing_state}. Available: {sorted(policy_terms)}")

    missing_cam = [k for k in camera_keys if k not in camera_terms]
    if missing_cam:
        raise ValueError(f"Invalid camera_keys: {missing_cam}. Available: {sorted(camera_terms)}")

    # Validate dimensions
    env_action_dim = env.action_space.shape[-1]
    if cfg_action_dim != env_action_dim:
        raise ValueError(f"action_dim mismatch: config={cfg_action_dim}, env={env_action_dim}")

    # Compute expected state dimension
    policy_dims = obs_manager.group_obs_dim.get("policy", [])
    policy_names = active_terms.get("policy", [])
    term_dims = dict(zip(policy_names, policy_dims, strict=False))

    expected_state_dim = 0
    for key in state_keys:
        if key in term_dims:
            shape = term_dims[key]
            dim = 1
            for s in shape if isinstance(shape, (tuple, list)) else [shape]:
                dim *= s
            expected_state_dim += dim

    if cfg_state_dim != expected_state_dim:
        raise ValueError(
            f"state_dim mismatch: config={cfg_state_dim}, "
            f"computed={expected_state_dim}. "
            f"Term dims: {term_dims}"
        )

    logging.info(f"Validated: state_keys={state_keys}, camera_keys={camera_keys}")


def create_isaaclab_arena_envs(
    cfg: IsaaclabArenaEnv,
    n_envs: int | None = None,
    use_async_envs: bool = False,
) -> dict[str, dict[int, IsaacLabVectorEnvWrapper]]:
    """Create IsaacLab Arena envs wrapped for VectorEnv compatibility.

    Args:
        cfg: Environment configuration (IsaaclabArenaEnv).
        n_envs: Number of parallel environments. Overrides cfg.num_envs.
        use_async_envs: Ignored (IsaacLab uses GPU-based batched execution).

    Returns:
        Dict mapping environment name to task_id (0) to wrapped VectorEnv.
    """
    from isaaclab.app import AppLauncher

    if cfg.enable_pinocchio:
        import pinocchio  # noqa: F401

    # Override num_envs if n_envs is provided
    cfg_dict = asdict(cfg)
    if n_envs is not None:
        cfg_dict["num_envs"] = n_envs
        logging.info(f"Overriding num_envs from {cfg.num_envs} to {n_envs}")

    as_isaaclab_argparse = argparse.Namespace(**cfg_dict)

    logging.info("Launching IsaacLab simulation app...")
    _simulation_app = AppLauncher(as_isaaclab_argparse)  # noqa: F841

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    if cfg.environment is None:
        raise ValueError("cfg.environment must be specified")

    # Resolve alias and create environment
    environment_path = resolve_environment_alias(cfg.environment)
    logging.info(f"Creating environment: {environment_path}")

    module_path, class_name = environment_path.rsplit(".", 1)
    environment_module = importlib.import_module(module_path)
    environment_class = getattr(environment_module, class_name)()

    env_builder = ArenaEnvBuilder(environment_class.get_env(as_isaaclab_argparse), as_isaaclab_argparse)
    # Determine render_mode before creating env
    # render_mode="rgb_array" enables video recording via RecordVideo
    render_mode = "rgb_array" if cfg.enable_cameras else None

    raw_env = env_builder.make_registered()

    # Set render_mode on underlying env (not passed through gym.make by ArenaEnvBuilder)
    # This is required for IsaacLab's render() method to return rgb_array data
    if render_mode and hasattr(raw_env, "render_mode"):
        raw_env.render_mode = render_mode
        logging.info(f"Set render_mode={render_mode} on underlying IsaacLab env")

    # Validate config
    state_keys = tuple(k.strip() for k in cfg.state_keys.split(",") if k.strip())
    camera_keys = tuple(k.strip() for k in cfg.camera_keys.split(",") if k.strip())
    _validate_env_config(raw_env, state_keys, camera_keys, cfg.state_dim, cfg.action_dim)

    # Wrap and return
    wrapped_env = IsaacLabVectorEnvWrapper(
        raw_env,
        episode_length=cfg.episode_length,
        task=cfg.task,
        render_mode=render_mode,
    )
    logging.info(f"Created: {cfg.environment} with {wrapped_env.num_envs} envs, render_mode={render_mode}")

    return {cfg.environment: {0: wrapped_env}}


# ---- CLI Entry Point ----


@dataclass
class ArenaConfig:
    env: envs.EnvConfig


@parser.wrap()
def arena_main(cfg: ArenaConfig):
    """Run zero action rollout for IsaacLab Arena environment."""
    logging.info(pformat(asdict(cfg)))

    env_dict = create_isaaclab_arena_envs(cfg.env)
    env = next(iter(env_dict.values()))[0]
    env.reset()

    for _ in tqdm.tqdm(range(cfg.env.episode_length)):
        with torch.inference_mode():
            action_dim = env.action_space.shape[-1]
            actions = torch.zeros((env.num_envs, action_dim), device=env.device)
            obs, rewards, terminated, truncated, info = env.step(actions)


def main():
    init_logging()
    arena_main()


if __name__ == "__main__":
    main()
