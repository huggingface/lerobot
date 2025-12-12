import argparse
from typing import Any


def get_arena_builder_from_cli(args_cli: argparse.Namespace):  # -> tuple[ManagerBasedRLEnvCfg, str]:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Get the example environment
    # assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    # assert (
    #     args_cli.example_environment in ExampleEnvironments
    # ), f"Example environment type {args_cli.example_environment} not supported"
    # example_env = ExampleEnvironments[args_cli.example_environment]()
    _env

    # Compile the environment
    env_builder = ArenaEnvBuilder(_env.get_env(args_cli), args_cli)
    return env_builder



def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    arena_args_cli: argparse.Namespace | None = None,
) -> dict[str, dict[int, Any]]:

    from isaaclab.app import AppLauncher

    if arena_args_cli.enable_pinocchio:
        import pinocchio  # noqa: F401

    print("Launching simulation app")
    _simulation_app = AppLauncher()  # noqa: F841

    # Import only the builder, not the CLI parser
    # from isaaclab_arena.examples.example_environments.cli import (
    #     get_arena_builder_from_cli,
    # )

    # Convert JSON config to Namespace (replaces CLI parsing)
    arena_builder = get_arena_builder_from_cli(arena_args_cli)
    env = arena_builder.make_registered()

    # Return in LeRobot factory format to avoid SyncVectorEnv wrapping.
    # IsaacLab envs are already GPU-vectorized and return CUDA tensors,
    # which are incompatible with Gymnasium's numpy-based SyncVectorEnv.
    suite_name = arena_args_cli.example_environment
    return {suite_name: {0: env}}