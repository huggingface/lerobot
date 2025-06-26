import time
from dataclasses import dataclass
from pprint import pformat
import draccus
import rerun as rr
import os
from pathlib import Path

from examples.lekiwi.utils import display_data
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader
from lerobot.common.utils.utils import init_logging, log_say
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.common.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.common.utils.robot_utils import busy_wait


@dataclass
class RecordConfig:
    num_episodes: int = 1
    nb_cycles: int = 9000
    repo_id: str = "local/lekiwi_001_tape_a"
    fps: int = 30
    reset_time_s: int | float = 10
    warmup_time_s: int | float = 5
    task_description: str = "Grab the tape and put it in the cup"
    robot_ip: str = "192.168.1.204"
    robot_id: str = "lekiwi"
    leader_arm_port: str = "/dev/ttyUSB0"
    leader_arm_id: str = "lekiwi_teleop"
    keyboard_id: str = "my_laptop_keyboard"
    display_data: bool = False
    rerun_session_name: str = "lekiwi_teleoperation"
    play_sounds: bool = True


def record_loop(
    robot,
    leader_arm,
    keyboard,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    task_description: str | None = None,
    should_display_data: bool = False,
):
    """Record loop that handles keyboard events and data collection."""
    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # Check for keyboard events
        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()
        action = leader_arm.get_action()
        keyboard_keys = keyboard.get_action()

        action_sent = robot.send_action(action)

        # Create frame and add to dataset only if dataset is provided
        if dataset is not None and task_description is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, action_sent, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=task_description)

        if should_display_data:
            display_data(observation, action)

        # Maintain timing
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


@draccus.wrap()
def record(cfg: RecordConfig):
    if cfg.display_data:
        _init_rerun(session_name=cfg.rerun_session_name)

    # Initialize robot and teleop devices
    robot_config = LeKiwiClientConfig(
        remote_ip=cfg.robot_ip,
        id=cfg.robot_id
    )
    leader_arm_config = SO100LeaderConfig(
        port=cfg.leader_arm_port,
        id=cfg.leader_arm_id
    )
    keyboard_config = KeyboardTeleopConfig(id=cfg.keyboard_id)

    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect to all devices
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Check connection status
    if not all([robot.is_connected, leader_arm.is_connected, keyboard.is_connected]):
        print("Failed to connect to one or more devices:")
        print(f"  Robot: {robot.is_connected}")
        print(f"  Leader Arm: {leader_arm.is_connected}")
        print(f"  Keyboard: {keyboard.is_connected}")
        return

    # Setup dataset
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        features=dataset_features,
        robot_type=robot.name,
    )

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    print(f"Starting LeKiwi recording for {cfg.num_episodes} episodes")
    print(f"Dataset will be saved to: {cfg.repo_id}")
    print("Keyboard controls:")
    print("  Right arrow: Save current episode and continue")
    print("  Left arrow: Re-record current episode")
    print("  Escape: Stop recording entirely")
    print("  Ctrl+C: Emergency stop")

    try:
        # Calculate control time based on nb_cycles and fps
        control_time_s = cfg.nb_cycles / cfg.fps

        # Warm-up period before first episode
        if cfg.warmup_time_s > 0:
            log_say(f"Warming up for {cfg.warmup_time_s} seconds...", cfg.play_sounds)
            record_loop(
                robot=robot,
                leader_arm=leader_arm,
                keyboard=keyboard,
                events=events,
                fps=cfg.fps,
                dataset=None,  # No dataset during warm-up
                control_time_s=cfg.warmup_time_s,
                task_description=None,
                should_display_data=cfg.display_data,
            )

        recorded_episodes = 0
        while recorded_episodes < cfg.num_episodes:
            # Audio feedback for episode start (using dataset.num_episodes like main record)
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            print(f"\nRecording episode {recorded_episodes + 1}/{cfg.num_episodes}")

            # Reset events for new episode
            events["exit_early"] = False
            events["rerecord_episode"] = False

            record_loop(
                robot=robot,
                leader_arm=leader_arm,
                keyboard=keyboard,
                events=events,
                fps=cfg.fps,
                dataset=dataset,
                control_time_s=control_time_s,
                task_description=cfg.task_description,
                should_display_data=cfg.display_data,
            )

            # Handle re-record episode event
            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()

                # Run reset time to allow manual environment reset
                log_say("Episode discarded, reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    leader_arm=leader_arm,
                    keyboard=keyboard,
                    events=events,
                    fps=cfg.fps,
                    dataset=None,  # No dataset during reset time
                    control_time_s=cfg.reset_time_s,
                    task_description=None,
                    should_display_data=cfg.display_data,
                )
                continue  # Re-run the while loop for this episode

            # Save the episode
            dataset.save_episode()
            print(f"Episode {recorded_episodes + 1} saved")

            # Execute reset time without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if recorded_episodes < cfg.num_episodes - 1:
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    leader_arm=leader_arm,
                    keyboard=keyboard,
                    events=events,
                    fps=cfg.fps,
                    dataset=None,  # No dataset during reset time
                    control_time_s=cfg.reset_time_s,
                    task_description=None,
                    should_display_data=cfg.display_data,
                )

            recorded_episodes += 1

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    finally:
        # Cleanup connections
        print("Disconnecting devices...")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()

        # Cleanup keyboard listener
        if not is_headless() and listener is not None:
            listener.stop()

        # Save and upload dataset only if there are frames in the buffer
        if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
            print("Saving dataset...")
            dataset.save_episode()

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        log_say("Exiting", cfg.play_sounds)

        # Todo: 6/20/2025: Will push to hub when proper data structure is implemented
        # dataset.push_to_hub()
        # print(f"Dataset successfully uploaded to: {repo_id}")


if __name__ == "__main__":
    record()
