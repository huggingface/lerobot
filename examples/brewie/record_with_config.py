#!/usr/bin/env python

"""
Example of recording a dataset for Brewie robot using configuration file.

This script uses record_config.py to configure all recording parameters.
Modify settings in record_config.py before running this script.

FEATURES:
- Automatic detection of existing datasets
- Ability to continue recording in existing dataset (adding new episodes)
- Interactive recording mode selection
- Support for all standard LeRobot functions
- Secure HuggingFace token retrieval from environment variables

Usage:
    # With environment variable
    export HUGGINGFACE_TOKEN=your_token_here
    python examples/brewie/record_with_config.py
    
    # With command line argument
    python examples/brewie/record_with_config.py --hf-token your_token_here
    
    # Interactive input (token will be requested at startup)
    python examples/brewie/record_with_config.py

Operating modes:
1. Creating new dataset (default)
2. Continuing recording in existing dataset (automatically offered when detected)
3. Forced continuation of recording (via resume_existing_dataset=True in config)
"""

import os
import shutil
import sys
from pathlib import Path

# Add path to lerobot modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.brewie.config_Brewie import BrewieConfig
from lerobot.robots.brewie.Brewie_base import BrewieBase
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say as _log_say
from lerobot.utils.visualization_utils import _init_rerun

# Import configuration
from record_config import config

def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    """
    Wrapper for log_say that duplicates information to console.
    
    Args:
        text: Text to output
        play_sounds: Whether to play sound (passed to original log_say)
        blocking: Blocking mode (passed to original log_say)
    """
    # Output to console
    print(f"[LOG] {text}")
    
    # Call original log_say function
    _log_say(text, play_sounds, blocking)

def check_dataset_exists(dataset_repo_id: str) -> bool:
    """Check if dataset exists locally or on Hub."""
    try:
        # Attempt to load existing dataset
        existing_dataset = LeRobotDataset(dataset_repo_id)
        return True
    except Exception:
        return False

def get_existing_dataset_info(dataset_repo_id: str) -> dict:
    """Get information about existing dataset."""
    try:
        existing_dataset = LeRobotDataset(dataset_repo_id)
        return {
            "exists": True,
            "num_episodes": existing_dataset.num_episodes,
            "fps": existing_dataset.fps,
            "robot_type": existing_dataset.meta.robot_type,
            "features": list(existing_dataset.features.keys())
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }

def validate_config():
    """Check configuration correctness."""
    errors = []
    
    if config.hf_username == "your_username":
        errors.append("Need to specify your HuggingFace username in config.hf_username")
    
    # Check token through new method
    try:
        hf_token = config.get_hf_token()
        if not hf_token or hf_token.strip() == "":
            errors.append("Failed to get HuggingFace token")
    except ValueError as e:
        errors.append(f"Error getting HuggingFace token: {e}")
    
    if not config.dataset_name:
        errors.append("Need to specify dataset name in config.dataset_name")
    
    if config.num_episodes <= 0:
        errors.append("Number of episodes must be greater than 0")
    
    if config.episode_time_sec <= 0:
        errors.append("Episode duration must be greater than 0")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nFix errors in record_config.py file and run script again.")
        print("\nTo set HuggingFace token use:")
        print("  export HUGGINGFACE_TOKEN=your_token_here")
        print("or pass token via command line argument:")
        print("  python record_with_config.py --hf-token your_token_here")
        return False
    
    return True

def print_config_summary(dataset_repo_id: str, existing_dataset_info: dict = None):
    """Print configuration summary."""
    print("=" * 60)
    print("BREWIE DATASET RECORDING CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {dataset_repo_id}")
    
    if existing_dataset_info and existing_dataset_info.get("exists"):
        print(f"MODE: Continue recording in existing dataset")
        print(f"Existing episodes: {existing_dataset_info['num_episodes']}")
        print(f"Episodes to add: {config.num_episodes}")
        print(f"Total episodes: {existing_dataset_info['num_episodes'] + config.num_episodes}")
    else:
        print(f"MODE: Create new dataset")
        print(f"Episodes: {config.num_episodes}")
    
    print(f"Task: {config.task_description}")
    print(f"Category: {config.task_category}")
    print(f"Difficulty: {config.difficulty_level}")
    print(f"Episode duration: {config.episode_time_sec}s")
    print(f"Reset time: {config.reset_time_sec}s")
    print(f"Recording frequency: {config.fps} FPS")
    print(f"ROS Master: {config.ros_master_ip}:{config.ros_master_port}")
    print("=" * 60)

def main():
    """Main dataset recording function."""
    
    # Check configuration
    if not validate_config():
        return
    
    # Create dataset ID
    dataset_repo_id = f"{config.hf_username}/{config.dataset_name}"
    
    # Check dataset existence
    existing_dataset_info = get_existing_dataset_info(dataset_repo_id)
    
    # Automatic resume mode detection if not explicitly set
    should_resume = config.resume_existing_dataset
    if existing_dataset_info.get("exists") and not config.resume_existing_dataset:
        print(f"\nFound existing dataset: {dataset_repo_id}")
        print(f"Existing episodes: {existing_dataset_info['num_episodes']}")
        print(f"FPS: {existing_dataset_info['fps']}")
        print(f"Robot type: {existing_dataset_info['robot_type']}")
        
        response = input("\nContinue recording in existing dataset? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            should_resume = True
            print("Mode: Continue recording in existing dataset")
        else:
            print("Mode: Create new dataset (existing will be overwritten)")
    
    # Print configuration summary
    if should_resume and existing_dataset_info.get("exists"):
        print_config_summary(dataset_repo_id, existing_dataset_info)
    else:
        print_config_summary(dataset_repo_id)
    
    # Confirmation to start
    response = input("Continue recording with these settings? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Recording cancelled.")
        return
    
    # =============================================================================
    # CREATE CONFIGURATIONS
    # =============================================================================
    
    # Robot configuration
    robot_config = BrewieConfig(
        master_ip=config.ros_master_ip,
        master_port=config.ros_master_port,
        servo_duration=config.servo_duration,
        max_relative_target=config.max_relative_target,
    )
    
    # Teleoperator configuration
    keyboard_config = KeyboardTeleopConfig()
    
    # =============================================================================
    # DEVICE INITIALIZATION
    # =============================================================================
    
    robot = BrewieBase(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)
    
    # =============================================================================
    # DATASET SETUP
    # =============================================================================
    
    # Dataset features configuration
    #action_features = hw_to_dataset_features(robot.action_features, "action") 
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**obs_features}
    #**action_features,
    
    # Create or load dataset
    if should_resume and existing_dataset_info.get("exists"):
        log_say("Loading existing dataset to continue recording...")
        dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            batch_encoding_size=1,  # Use default value
        )
        
        # Start image writer for existing dataset
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,  # Use default value
                num_threads=config.image_writer_threads,
            )
        
        log_say(f"Dataset loaded. Existing episodes: {dataset.num_episodes}")
    else:
        log_say("Creating new dataset...")
        try:
            dataset = LeRobotDataset.create(
                repo_id=dataset_repo_id,
                fps=config.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=config.use_videos,
                image_writer_threads=config.image_writer_threads,
            )
        except FileExistsError:
            log_say("Dataset folder already exists. Removing existing folder and creating new dataset...")
            # Remove existing dataset folder
            dataset_root = Path.home() / ".cache" / "huggingface" / "lerobot" / dataset_repo_id
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            # Create new dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_repo_id,
                fps=config.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=config.use_videos,
                image_writer_threads=config.image_writer_threads,
            )
    
    # =============================================================================
    # DEVICE CONNECTION
    # =============================================================================
    
    log_say("Connecting to Brewie robot...")
    try:
        robot.connect()
    except Exception as e:
        log_say(f"Error connecting to robot: {e}")
        return
    
    log_say("Connecting to teleoperator...")
    try:
        keyboard.connect()
    except Exception as e:
        log_say(f"Error connecting to teleoperator: {e}")
        robot.disconnect()
        return
    
    # Initialize visualization
    _init_rerun(session_name=config.session_name)
    
    # Initialize keyboard listener
    listener, events = init_keyboard_listener()
    
    # Check connections
    if not robot.is_connected:
        log_say("ERROR: Brewie robot not connected!")
        keyboard.disconnect()
        listener.stop()
        return
        
    if not keyboard.is_connected:
        log_say("ERROR: Teleoperator not connected!")
        robot.disconnect()
        listener.stop()
        return
    
    log_say("All devices connected successfully!")
    log_say("Controls:")
    log_say("  - ENTER: Start/continue episode recording")
    log_say("  - ESC: Stop recording")
    log_say("  - R: Rewrite current episode")
    
    # =============================================================================
    # EPISODE RECORDING LOOP
    # =============================================================================
    
    # Determine starting episode number
    start_episode = dataset.num_episodes if should_resume else 0
    total_episodes_to_record = config.num_episodes
    recorded_episodes = 0
    
    log_say(f"Starting recording. Will record {total_episodes_to_record} episodes")
    if should_resume:
        log_say(f"Continuing from episode {start_episode}")
    
    try:
        while recorded_episodes < total_episodes_to_record and not events["stop_recording"]:
            current_episode_num = start_episode + recorded_episodes + 1
            log_say(f"Recording episode {current_episode_num} ({recorded_episodes + 1}/{total_episodes_to_record})")
            log_say("Press ENTER to start episode recording...")
            input()
            
            # Start recording loop
            record_loop(
                robot=robot,
                events=events,
                fps=config.fps,
                dataset=dataset,
                teleop=keyboard,
                control_time_s=config.episode_time_sec,
                single_task=config.task_description,
                display_data=config.display_data,
            )
            
            # Environment reset logic
            if not events["stop_recording"] and (
                (recorded_episodes < total_episodes_to_record - 1) or events["rerecord_episode"]
            ):
                log_say("Resetting environment...")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=config.fps,
                    teleop=keyboard,
                    control_time_s=config.reset_time_sec,
                    single_task="Robot position reset",
                    display_data=config.display_data,
                )
            
            # Handle episode rewrite
            if events["rerecord_episode"]:
                log_say("Rewriting episode...")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
            current_episode_num = start_episode + recorded_episodes
            log_say(f"Episode {current_episode_num} saved")
            
    except KeyboardInterrupt:
        log_say("Recording interrupted by user")
    except Exception as e:
        log_say(f"Error during recording: {e}")
    
    # =============================================================================
    # COMPLETION AND HUB UPLOAD
    # =============================================================================
    
    total_episodes_in_dataset = dataset.num_episodes
    if should_resume:
        log_say(f"Recording completed! Added {recorded_episodes} new episodes")
        log_say(f"Total episodes in dataset: {total_episodes_in_dataset}")
    else:
        log_say(f"Recording completed! Saved {recorded_episodes} episodes")
    
    if recorded_episodes > 0 and config.auto_push_to_hub:
        log_say("Uploading dataset to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            log_say(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{dataset_repo_id}")
        except Exception as e:
            log_say(f"Error uploading to Hub: {e}")
            log_say("Dataset saved locally")
    elif recorded_episodes > 0:
        log_say("Dataset saved locally (auto_push_to_hub = False)")
    
    # Disconnect devices
    log_say("Disconnecting devices...")
    try:
        robot.disconnect()
        keyboard.disconnect()
        listener.stop()
    except Exception as e:
        log_say(f"Error during disconnection: {e}")
    
    log_say("Dataset recording completed!")

if __name__ == "__main__":
    main()
