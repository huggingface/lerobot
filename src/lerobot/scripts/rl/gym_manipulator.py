# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import (
    DeviceProcessor,
    IdentityProcessor,
    RobotProcessor,
    ToBatchProcessor,
    VanillaObservationProcessor,
)
from lerobot.processor.hil_processor import (
    GripperPenaltyProcessor,
    ImageCropResizeProcessor,
    InterventionActionProcessor,
    RewardClassifierProcessor,
    TimeLimitProcessor,
)
from lerobot.processor.pipeline import TransitionKey
from lerobot.processor.robot_processor import (
    InverseKinematicsProcessor,
    JointVelocityProcessor,
    MotorCurrentProcessor,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.robots.robot import Robot
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO)


class DummyTeleopDevice:
    """
    A dummy teleop device for simulation environments that provides the same interface
    as real teleop devices but returns neutral/empty values.
    """

    @property
    def action_features(self):
        """Return action features for dataset recording."""
        return {"dtype": "float32", "shape": (4,), "names": ["delta_x", "delta_y", "delta_z", "gripper"]}

    def get_action(self):
        """Return neutral action values."""
        return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

    def get_teleop_events(self):
        """Return empty events."""
        return {}


@dataclass
class DatasetConfig:
    repo_id: str
    dataset_root: str
    task: str
    num_episodes: int
    episode: int
    push_to_hub: bool


@dataclass
class GymManipulatorConfig:
    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def create_transition(
    observation=None, action=None, reward=0.0, done=False, truncated=False, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray):
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


class RobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.
    """

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ):
        """
        Initialize the RobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            display_cameras: If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._setup_spaces()

    def _get_observation(self) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Helper to convert a dictionary from bus.sync_read to an ordered numpy array."""
        obs_dict = self.robot.get_observation()
        joint_positions = np.array([obs_dict[name] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}
        return {"agent_pos": joint_positions, "pixels": images}

    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Box space representing joint position commands. It is defined as relative (delta)
              or absolute, based on the configuration.
        """
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = "observation.images"
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            observation_spaces["observation.state"] = gym.spaces.Box(
                low=0,
                high=10,
                shape=current_observation["agent_pos"].shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
            seed: A seed for random number generation to ensure reproducibility.
            options: Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "is_intervention".
        """
        # Reset the robot
        # self.robot.reset()
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        return obs, {
            "is_intervention": False,
            "raw_joint_positions": obs["agent_pos"],
        }

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {"is_intervention": False, "raw_joint_positions": obs["agent_pos"]},
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """
    Factory function to create a robot environment.

    This function builds a robot environment with all necessary wrappers
    based on the provided configuration.

    Args:
        cfg: Configuration object containing environment parameters.

    Returns:
        A tuple containing:
            - A gym environment with all necessary wrappers applied.
            - The teleoperation device for use in action processors.
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gymnasium as gym
        
        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0
        
        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, DummyTeleopDevice()

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"
    
    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None
    
    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device


def make_processors(env: gym.Env, cfg: HILSerlRobotEnvConfig, device: str = "cpu"):
    """
    Factory function to create environment and action processors.

    Args:
        env: The robot environment (RobotEnv or gym.Env for GymHIL)
        cfg: Configuration object containing processor parameters

    Returns:
        tuple: (env_processor, action_processor)
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        # Minimal processor pipeline for GymHIL simulation
        env_pipeline_steps = [
            VanillaObservationProcessor(),
            ToBatchProcessor(),
            DeviceProcessor(device=device),
        ]

        # Use IdentityProcessor for GymHIL as actions are handled by the environment
        action_pipeline_steps = [IdentityProcessor()]

        return RobotProcessor(steps=env_pipeline_steps), RobotProcessor(steps=action_pipeline_steps)

    # Full processor pipeline for real robot environment
    env_pipeline_steps = [
        VanillaObservationProcessor(),
    ]
    
    # Add observation-based processors if observation config exists
    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessor(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessor(env=env))
    
    # Add image preprocessing if config exists
    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessor(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )
    
    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessor(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )
    
    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessor(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    # Add reward classifier processor if configured
    if (cfg.processor.reward_classifier is not None and 
        cfg.processor.reward_classifier.pretrained_path is not None):
        env_pipeline_steps.append(
            RewardClassifierProcessor(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
            )
        )

    env_pipeline_steps.append(ToBatchProcessor())
    env_pipeline_steps.append(DeviceProcessor(device=device))


    action_pipeline_steps = [
        InterventionActionProcessor(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
        ),
    ]
    
    if cfg.processor.inverse_kinematics is not None:
        action_pipeline_steps.append(
            InverseKinematicsProcessor(
                urdf_path=cfg.processor.inverse_kinematics.urdf_path,
                target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    return RobotProcessor(steps=env_pipeline_steps), RobotProcessor(steps=action_pipeline_steps)


def step_env_and_process_transition(
    env,
    transition,
    action,
    teleop_device,
    env_processor,
    action_processor,
):
    """
    Execute one step with processors handling intervention and observation processing.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute (will be replaced by neutral action in gym_manipulator mode)
        teleop_device: Teleoperator device for getting intervention signals (DummyTeleopDevice for GymHIL)
        env_processor: Environment processor for observations
        action_processor: Action processor for handling interventions

    Returns:
        tuple: (new_transition, terminate_episode)
    """
    # Get teleoperation action and events (DummyTeleopDevice for GymHIL, real device for robots)
    teleop_action = teleop_device.get_action()
    teleop_events = teleop_device.get_teleop_events() if hasattr(teleop_device, "get_teleop_events") else {}

    # Create action transition
    action_transition = dict(transition)
    action_transition[TransitionKey.ACTION] = action

    # Add teleoperation data to complementary data
    action_complementary_data = action_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).copy()
    action_complementary_data["teleop_action"] = teleop_action
    action_complementary_data.update(teleop_events)
    action_transition[TransitionKey.COMPLEMENTARY_DATA] = action_complementary_data

    # Process action through action pipeline (IdentityProcessor for GymHIL, full pipeline for robots)
    processed_action_transition = action_processor(action_transition)

    # Extract processed action and metadata
    processed_action = processed_action_transition[TransitionKey.ACTION]
    terminate_episode = processed_action_transition.get(TransitionKey.DONE, False)

    # Step environment with processed action
    obs, reward, terminated, truncated, info = env.step(processed_action)

    # Combine rewards from environment and action processor
    reward = reward + processed_action_transition[TransitionKey.REWARD]

    # Process new observation - handle raw_joint_positions if it exists
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    if "raw_joint_positions" in info:
        complementary_data["raw_joint_positions"] = info.pop("raw_joint_positions")

    info.update(processed_action_transition[TransitionKey.INFO])

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated or terminate_episode,
        truncated=truncated,
        info=info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition, terminate_episode


def control_loop(env, env_processor, action_processor, teleop_device, cfg: GymManipulatorConfig):
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    complementary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(transition)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    
    dataset = None
    if cfg.mode == "record":
        action_features = teleop_device.action_features
        features = {
            "action": action_features,
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == "observation.state":
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.dataset_root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    while episode_idx < cfg.dataset.num_episodes:
        step_start_time = time.perf_counter()

        # Create a neutral action (no movement)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Use the new step function
        transition, terminate_episode = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            teleop_device=teleop_device,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record":
            observations = {k: v.squeeze(0).cpu() for k, v in transition[TransitionKey.OBSERVATION].items()}
            # Use teleop_action if available, otherwise use the action from the transition
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                "action": action_to_record.cpu(),
                "next.reward": np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                "next.done": np.array([terminated or truncated], dtype=bool),
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)
            if dataset is not None:
                dataset.add_frame(frame, task=cfg.dataset.task)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated or terminate_episode:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                if transition[TransitionKey.INFO].get("rerecord_episode", False):
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            complementary_data = (
                {"raw_joint_positions": info.pop("raw_joint_positions")}
                if "raw_joint_positions" in info
                else {}
            )
            env_processor.reset()
            action_processor.reset()

            transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
            transition = env_processor(transition)

        # Maintain fps timing
        busy_wait(dt - (time.perf_counter() - step_start_time))

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(env, action_processor, cfg: GymManipulatorConfig):
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.dataset_root,
        episodes=[cfg.dataset.episode],
        download_videos=False,
    )
    dataset_actions = dataset.hf_dataset.select_columns(["action"])
    _, info = env.reset()

    for action_data in dataset_actions:
        start_time = time.perf_counter()
        transition = create_transition(
            action=action_data["action"], complementary_data={"raw_joint_positions": info["raw_joint_positions"]}
        )
        transition = action_processor(transition)
        _, _, _, _, info = env.step(transition[TransitionKey.ACTION])
        busy_wait(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig):
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
