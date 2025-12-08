# env.py - Upload this to your Hub repository
# Example: huggingface.co/your-username/test-kwargs-env

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


def make_env(
    n_envs=1,
    use_async_envs=False,
    config_path=None,
    config_overrides=None,
    **kwargs,
):
    """
    Create vectorized CartPole environments with configurable options.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv
        config_path: Optional path to a config file (for demonstration)
        config_overrides: Optional dict of config overrides
        **kwargs: Additional configuration options

    Returns:
        dict mapping suite name to task environments
    """
    # Merge all config sources for demonstration
    config = {}
    if config_overrides:
        config.update(config_overrides)
    config.update(kwargs)

    # Store config in a way the test can verify
    # In a real env, you'd use these to configure the simulation
    stored_config = {
        "config_path": config_path,
        "config_overrides": config_overrides,
        "extra_kwargs": kwargs,
    }

    def _mk():
        env = gym.make("CartPole-v1")
        # Attach config to env for test verification
        env.hub_config = stored_config
        return env

    Vec = gym.vector.AsyncVectorEnv if use_async_envs else SyncVectorEnv
    vec_env = Vec([_mk for _ in range(n_envs)])

    # Also attach to vector env for easy access in tests
    vec_env.hub_config = stored_config

    return {"cartpole_suite": {0: vec_env}}

