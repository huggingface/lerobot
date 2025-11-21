import time
from typing import Any, Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from .base_sim import BaseSimulator


def init_channel(config: Dict[str, Any]) -> None:
    """
    Initialize the communication channel for simulator/robot communication.

    Args:
        config: Configuration dictionary containing DOMAIN_ID and optionally INTERFACE
    """
    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])


class SimulatorFactory:
    """Factory class for creating different types of simulators."""

    @staticmethod
    def create_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """
        Create a simulator based on the configuration.

        Args:
            config: Configuration dictionary containing SIMULATOR type
            env_name: Environment name
            **kwargs: Additional keyword arguments for specific simulators
        """
        simulator_type = config.get("SIMULATOR", "mujoco")
        if simulator_type == "mujoco":
            return SimulatorFactory._create_mujoco_simulator(config, env_name, **kwargs)
        elif simulator_type == "robocasa":
            return SimulatorFactory._create_robocasa_simulator(config, env_name, **kwargs)
        else:
            print(
                f"Warning: Invalid simulator type: {simulator_type}. "
                "If you are using run_sim_loop, please ignore this warning."
            )
            return None

    @staticmethod
    def _create_mujoco_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """Create a MuJoCo simulator instance."""
        camera_configs = kwargs.pop("camera_configs", {})
        if len(camera_configs) > 0:
            print(f"Debug: SimulatorFactory received {len(camera_configs)} camera config(s)")
        
        env_kwargs = dict(
            onscreen=kwargs.pop("onscreen", True),
            offscreen=kwargs.pop("offscreen", False),
            camera_configs=camera_configs,
        )
        return BaseSimulator(config=config, env_name=env_name, **env_kwargs)

    @staticmethod
    def _create_robocasa_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """Create a RoboCasa simulator instance."""
        from gr00t_wbc.control.envs.g1.sim.robocasa_sim import RoboCasaG1EnvServer
        from gr00t_wbc.control.envs.robocasa.utils.controller_utils import (
            update_robosuite_controller_configs,
        )
        from gr00t_wbc.control.envs.robocasa.utils.sim_utils import change_simulation_timestep

        change_simulation_timestep(config["SIMULATE_DT"])

        # Use default environment if not specified
        if env_name == "default":
            env_name = "GroundOnly"

        # Get or create controller configurations
        controller_configs = kwargs.get("controller_configs")
        if controller_configs is None:
            wbc_version = kwargs.get("wbc_version", "gear_wbc")
            controller_configs = update_robosuite_controller_configs("G1", wbc_version)

        # Build environment kwargs
        env_kwargs = dict(
            onscreen=kwargs.pop("onscreen", True),
            offscreen=kwargs.pop("offscreen", False),
            camera_names=kwargs.pop("camera_names", None),
            camera_heights=kwargs.pop("camera_heights", None),
            camera_widths=kwargs.pop("camera_widths", None),
            control_freq=kwargs.pop("control_freq", 50),
            controller_configs=controller_configs,
            ik_indicator=kwargs.pop("ik_indicator", False),
            randomize_cameras=kwargs.pop("randomize_cameras", True),
        )

        kwargs.update(
            {
                "verbose": config.pop("verbose", False),
                "sim_freq": 1 / config.pop("SIMULATE_DT"),
            }
        )

        return RoboCasaG1EnvServer(
            env_name=env_name,
            wbc_config=config,
            env_kwargs=env_kwargs,
            **kwargs,
        )

    @staticmethod
    def start_simulator(
        simulator,
        as_thread: bool = True,
        enable_image_publish: bool = False,
        mp_start_method: str = "spawn",
        camera_port: int = 5555,
    ):
        """
        Start the simulator either as a thread or as a separate process.

        Args:
            simulator: The simulator instance to start
            config: Configuration dictionary
            as_thread: If True, start as thread; if False, start as subprocess
            enable_offscreen: If True and not as_thread, start image publishing
        """

        if as_thread:
            simulator.start_as_thread()
        else:
            # Wrap in try-except to make sure simulator is properly closed upon exit.
            try:
                if enable_image_publish:
                    simulator.start_image_publish_subprocess(
                        start_method=mp_start_method,
                        camera_port=camera_port,
                    )
                    time.sleep(1)
                simulator.start()
            except KeyboardInterrupt:
                print("+++++Simulator interrupted by user.")
            except Exception as e:
                print(f"++++error in simulator: {e} ++++")
            finally:
                print("++++closing simulator ++++")
                simulator.close()

        # Allow simulator to initialize
        time.sleep(1)
