# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Pure inference script for robot control with policy, without dataset recording overhead.

This script allows you to:
- Start/stop inference with keyboard controls
- Switch between policy control and teleoperation
- Run inference without video recording or dataset management

Example:

```shell
lerobot-infer \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --policy.path=${HF_USER}/my_policy \
    --display_data=true \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```

Keyboard Controls:
- 'p' or Space: Start/Stop policy inference
- 't': Toggle between policy and teleoperation mode
- 'r': Reset the episode
- Esc: Exit the script
"""

import logging
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


class ControlMode(Enum):
    """Control mode for the robot."""

    IDLE = "idle"
    POLICY = "policy"
    TELEOP = "teleop"


@dataclass
class InferConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    # Optional teleoperator for manual control
    teleop: TeleoperatorConfig | None = None
    # Task description for the policy
    task: str | None = None
    # Control frequency in Hz
    fps: int = 30
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events
    play_sounds: bool = True
    # Temporary dataset directory for storing stats (not for recording)
    temp_dataset_dir: str | Path = ".cache/lerobot_infer"
    # Rename map for observations
    rename_map: dict[str, str] | None = None
    # Temporal ensembling: Number of recent actions to average (1 = disabled, >1 = enabled)
    temporal_ensemble_k: int = 1
    # Temporal ensembling: Exponential weights decay factor (1.0 = uniform weights, <1.0 = exponential decay)
    temporal_ensemble_exp: float = 1.0

    def __post_init__(self):
        # Parse policy path from CLI args
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Policy is required. Use --policy.path=<path_to_policy>")

        if self.rename_map is None:
            self.rename_map = {}

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class TemporalEnsembler:
    """
    Temporal ensembling for smooth action prediction.

    This class maintains a buffer of recent actions and computes a weighted average
    to reduce jitter and create smoother robot control.

    Args:
        k: Number of recent actions to keep in the buffer (window size)
        exp: Exponential decay factor for weights:
            - 1.0: uniform weights (simple moving average)
            - <1.0: exponential decay (recent actions weighted more)
    """

    def __init__(self, k: int = 1, exp: float = 1.0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if exp <= 0:
            raise ValueError(f"exp must be > 0, got {exp}")

        self.k = k
        self.exp = exp
        self.enabled = k > 1
        self.action_buffer: deque = deque(maxlen=k)

        # Precompute weights for efficiency
        if self.enabled:
            self._compute_weights()

    def _compute_weights(self):
        """Compute normalized exponential weights."""
        # Generate weights: [exp^(k-1), exp^(k-2), ..., exp^1, exp^0]
        # Most recent action has weight exp^0 = 1.0
        weights = [self.exp**i for i in range(self.k - 1, -1, -1)]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def reset(self):
        """Clear the action buffer."""
        self.action_buffer.clear()

    def update(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Update buffer with new action and return smoothed action.

        Args:
            action: Dictionary of action values (e.g., motor positions)

        Returns:
            Smoothed action dictionary
        """
        if not self.enabled:
            return action

        # Add new action to buffer
        self.action_buffer.append(action)

        # If buffer not full yet, return the current action
        if len(self.action_buffer) < self.k:
            return action

        # Compute weighted average
        smoothed_action = {}

        # Get all keys from the most recent action
        for key in action:
            values = []
            weights_to_use = []

            # Collect values from buffer (some old actions might not have all keys)
            for i, buffered_action in enumerate(self.action_buffer):
                if key in buffered_action:
                    values.append(buffered_action[key])
                    weights_to_use.append(self.weights[i])

            if not values:
                # Key not found in any buffered action
                smoothed_action[key] = action[key]
                continue

            # Normalize weights for available values
            weight_sum = sum(weights_to_use)
            normalized_weights = [w / weight_sum for w in weights_to_use]

            # Compute weighted average
            if isinstance(values[0], (int, float)):
                # Scalar value
                smoothed_action[key] = sum(v * w for v, w in zip(values, normalized_weights, strict=True))
            else:
                smoothed_value = sum(v * w for v, w in zip(values, normalized_weights, strict=True))
                smoothed_action[key] = smoothed_value

        return smoothed_action


def init_inference_keyboard_listener(
    play_sounds: bool = True, has_teleop: bool = False
) -> tuple[Any | None, dict]:
    """
    Initializes a keyboard listener for inference control.

    Args:
        play_sounds: Whether to use TTS for announcements.
        has_teleop: Whether a teleoperator is configured.

    Returns:
        A tuple containing:
        - The pynput.keyboard.Listener instance, or None if in a headless environment.
        - A dictionary of control state flags.
    """
    state = {
        "control_mode": ControlMode.IDLE,
        "previous_mode": ControlMode.IDLE,
        "exit": False,
        "reset_episode": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard controls will not be available. "
            "The robot will start in POLICY mode by default."
        )
        state["control_mode"] = ControlMode.POLICY
        state["previous_mode"] = ControlMode.POLICY
        return None, state

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            # Start/Stop policy inference
            if key == keyboard.Key.space or (hasattr(key, "char") and key.char == "p"):
                if state["control_mode"] == ControlMode.POLICY:
                    log_say("Pausing inference", play_sounds)
                    state["control_mode"] = ControlMode.IDLE
                else:
                    log_say("Starting inference", play_sounds)
                    state["control_mode"] = ControlMode.POLICY

            # Toggle teleoperation mode
            elif hasattr(key, "char") and key.char == "t":
                if not has_teleop:
                    log_say("No teleoperator configured", play_sounds)
                elif state["control_mode"] == ControlMode.TELEOP:
                    log_say("Switching to idle mode", play_sounds)
                    state["control_mode"] = ControlMode.IDLE
                else:
                    log_say("Switching to manual control", play_sounds)
                    state["control_mode"] = ControlMode.TELEOP

            # Reset episode (go back to starting position)
            elif hasattr(key, "char") and key.char == "r":
                log_say("Resetting episode", play_sounds)
                state["reset_episode"] = True

            # Exit
            elif key == keyboard.Key.esc:
                log_say("Exiting", play_sounds)
                state["exit"] = True

        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, state


def _handle_mode_transition(
    state: dict,
    teleop: Teleoperator | list[Teleoperator] | None,
    teleop_arm: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    temporal_ensembler: TemporalEnsembler | None = None,
) -> None:
    """
    Handle transitions between control modes, managing teleoperator motor torque and policy state.

    Args:
        state: Current state dictionary with control_mode and previous_mode
        teleop: Single teleoperator or list of teleoperators
        teleop_arm: Arm teleoperator (for multi-teleop setup)
        policy: Policy instance to reset when entering TELEOP mode
        preprocessor: Preprocessor pipeline to reset when entering TELEOP mode
        postprocessor: Postprocessor pipeline to reset when entering TELEOP mode
        temporal_ensembler: Temporal ensembler to reset when entering TELEOP mode
    """
    current_mode = state["control_mode"]
    previous_mode = state["previous_mode"]

    if teleop is None:
        return

    try:
        # Entering TELEOP mode - disable torque so user can move freely and reset policy state
        if current_mode == ControlMode.TELEOP and previous_mode != ControlMode.TELEOP:
            if isinstance(teleop, list):
                if teleop_arm is not None and hasattr(teleop_arm, "bus"):
                    teleop_arm.bus.disable_torque()
                    logging.info("Teleoperator torque disabled for manual control")
            else:
                if hasattr(teleop, "bus"):
                    teleop.bus.disable_torque()
                    logging.info("Teleoperator torque disabled for manual control")

            # Reset policy state so it starts fresh from the new position
            if policy is not None:
                policy.reset()
            if preprocessor is not None:
                preprocessor.reset()
            if postprocessor is not None:
                postprocessor.reset()
            if temporal_ensembler is not None:
                temporal_ensembler.reset()
            logging.info("Policy state reset for teleoperation - will start fresh from new position")

        # Leaving TELEOP mode - enable torque for syncing
        elif previous_mode == ControlMode.TELEOP and current_mode != ControlMode.TELEOP:
            if isinstance(teleop, list):
                if teleop_arm is not None and hasattr(teleop_arm, "bus"):
                    teleop_arm.bus.enable_torque()
                    logging.info("Teleoperator torque enabled for syncing")
            else:
                if hasattr(teleop, "bus"):
                    teleop.bus.enable_torque()
                    logging.info("Teleoperator torque enabled for syncing")

    except Exception as e:
        logging.warning(f"Failed to handle mode transition: {e}")


def _sync_teleop_with_robot(
    teleop: Teleoperator | list[Teleoperator],
    robot_obs: RobotObservation,
    teleop_arm: Teleoperator | None = None,
) -> None:
    """
    Synchronize teleoperator position with robot observation.
    This allows smooth takeover when switching from policy to teleoperation mode.

    Args:
        teleop: Single teleoperator or list of teleoperators
        robot_obs: Current robot observation containing position data
        teleop_arm: Arm teleoperator (for multi-teleop setup)
    """
    try:
        if isinstance(teleop, list):
            # Multi-teleop case (e.g., LeKiwi with arm + keyboard)
            if teleop_arm is not None and hasattr(teleop_arm, "bus"):
                # Extract arm positions from robot observation
                arm_positions = {
                    motor: robot_obs.get(f"arm_{motor}.pos")
                    for motor in teleop_arm.bus.motors
                    if f"arm_{motor}.pos" in robot_obs
                }
                if arm_positions:
                    # Filter out None values
                    arm_positions = {k: v for k, v in arm_positions.items() if v is not None}
                    if arm_positions:
                        teleop_arm.bus.sync_write("Goal_Position", arm_positions, normalize=True)
        else:
            # Single teleoperator case
            if hasattr(teleop, "bus"):
                # Extract positions matching teleoperator's motors
                positions = {
                    motor: robot_obs.get(f"{motor}.pos")
                    for motor in teleop.bus.motors
                    if f"{motor}.pos" in robot_obs
                }
                if positions:
                    # Filter out None values
                    positions = {k: v for k, v in positions.items() if v is not None}
                    if positions:
                        teleop.bus.sync_write("Goal_Position", positions, normalize=True)
    except Exception as e:
        # Log but don't crash - syncing is a convenience feature
        logging.debug(f"Failed to sync teleoperator with robot: {e}")


def inference_loop(
    robot: Robot,
    state: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    dataset_features: dict,
    temporal_ensembler: TemporalEnsembler,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    task: str | None = None,
    display_data: bool = False,
    play_sounds: bool = True,
) -> None:
    """Main inference loop that switches between policy and teleoperation control."""

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (t for t in teleop if not isinstance(t, KeyboardTeleop)),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. "
                "Currently only supported for LeKiwi robot."
            )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    temporal_ensembler.reset()

    if temporal_ensembler.enabled:
        logging.info(f"Temporal ensembling enabled: k={temporal_ensembler.k}, exp={temporal_ensembler.exp}")

    log_say(
        "Inference ready. Press 'p' or Space to start policy control, 't' for teleoperation, 'r' to reset, Esc to exit.",
        play_sounds=play_sounds,
    )

    while not state["exit"]:
        start_loop_t = time.perf_counter()

        # Handle control mode transitions
        if state["control_mode"] != state["previous_mode"]:
            _handle_mode_transition(
                state, teleop, teleop_arm, policy, preprocessor, postprocessor, temporal_ensembler
            )
            state["previous_mode"] = state["control_mode"]

        # Handle episode reset
        if state["reset_episode"]:
            state["reset_episode"] = False
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            temporal_ensembler.reset()
            logging.info("Episode reset - policy, processors, and temporal ensembler cleared")

        # Get robot observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # Prepare observation frame for policy inference
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        action_values = None
        act_processed_teleop = None
        robot_action_to_send = None

        # Get action based on current control mode
        if state["control_mode"] == ControlMode.POLICY:
            # Policy inference
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            act_processed: RobotAction = make_robot_action(action_values, dataset_features)

            # Apply temporal ensembling for smooth actions
            act_processed_smoothed = temporal_ensembler.update(act_processed)
            robot_action_to_send = robot_action_processor((act_processed_smoothed, obs))

            # Sync teleoperator with robot position during policy control
            # This allows smooth takeover when switching to teleoperation mode
            if teleop is not None:
                _sync_teleop_with_robot(teleop, obs, teleop_arm)

        elif state["control_mode"] == ControlMode.TELEOP and teleop is not None:
            # Teleoperation control
            if isinstance(teleop, Teleoperator):
                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
            elif isinstance(teleop, list):
                arm_action = teleop_arm.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                # Convert keyboard action to base action (only for robots with base, like LeKiwi)
                if hasattr(robot, "_from_keyboard_to_base_action"):
                    base_action = robot._from_keyboard_to_base_action(keyboard_action)
                    act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                else:
                    act = arm_action
                act_processed_teleop = teleop_action_processor((act, obs))

            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif state["control_mode"] == ControlMode.IDLE:
            # Idle mode - no action sent to robot
            # Note: Teleoperator is not synced in IDLE mode, allowing manual positioning
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(1 / fps - dt_s)
            continue

        # Send action to robot if we have one
        if action_values is not None:
            robot.send_action(robot_action_to_send)

            if display_data:
                # Log the robot action (dictionary format) instead of raw tensor
                log_rerun_data(observation=obs_processed, action=robot_action_to_send)

        # Maintain target FPS
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)


@parser.wrap()
def infer(cfg: InferConfig) -> None:
    """Main inference function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="inference")

    # Initialize robot and teleoperator
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    robot.connect()
    if teleop is not None:
        teleop.connect()

    # Create processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build dataset features for policy (needed for observation frame building)
    # We need image features to be included in the dataset_features for proper frame building
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,  # Need image features for policy inference
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,  # Need image features for policy inference
        ),
    )

    # We need dataset metadata for the policy, so create a minimal in-memory dataset structure
    # Clean up any existing temp directory first
    temp_dataset_path = Path(cfg.temp_dataset_dir)
    if temp_dataset_path.exists():
        logging.info(f"Cleaning up existing temporary dataset directory: {temp_dataset_path}")
        shutil.rmtree(temp_dataset_path)

    # Create temporary dataset just to get metadata for policy
    temp_dataset = LeRobotDataset.create(
        repo_id="temp/inference_dataset",
        fps=cfg.fps,
        root=cfg.temp_dataset_dir,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,  # Must match dataset_features having video features
    )

    # Load policy with dataset metadata
    policy = make_policy(cfg.policy, ds_meta=temp_dataset.meta, rename_map=cfg.rename_map)

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(temp_dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # Initialize temporal ensembler for action smoothing
    temporal_ensembler = TemporalEnsembler(
        k=cfg.temporal_ensemble_k,
        exp=cfg.temporal_ensemble_exp,
    )

    # Initialize keyboard listener
    listener, state = init_inference_keyboard_listener(
        play_sounds=cfg.play_sounds, has_teleop=teleop is not None
    )

    try:
        # Run inference loop
        inference_loop(
            robot=robot,
            state=state,
            fps=cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            temporal_ensembler=temporal_ensembler,
            teleop=teleop,
            task=cfg.task,
            display_data=cfg.display_data,
            play_sounds=cfg.play_sounds,
        )
    finally:
        # Cleanup
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if listener is not None:
            listener.stop()

        # Shutdown rerun visualization if enabled
        if cfg.display_data:
            import rerun as rr

            rr.rerun_shutdown()

        # Clean up temporary dataset directory
        temp_dataset_path = Path(cfg.temp_dataset_dir)
        if temp_dataset_path.exists():
            logging.info(f"Cleaning up temporary dataset directory: {temp_dataset_path}")
            shutil.rmtree(temp_dataset_path)

        log_say("Inference stopped", cfg.play_sounds)


def main() -> None:
    """Run the inference script."""
    register_third_party_plugins()
    infer()


if __name__ == "__main__":
    main()
