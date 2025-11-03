"""
OpenArms Dataset Recording with Gravity + Friction Compensation

Records a dataset using OpenArms follower robot with leader teleoperator.
Leader arms have gravity and friction compensation for weightless, easy movement.
Includes 3 cameras: left wrist, right wrist, and base camera.

Uses the same compensation approach as teleop_with_compensation.py
"""

import shutil
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Recording parameters
NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 600
RESET_TIME_SEC = 120
TASK_DESCRIPTION = "OpenArms task description"

# Friction compensation scale factor (1.0 = full, 0.3 = 30% for stability)
FRICTION_SCALE = 1.0

def record_loop_with_compensation(
    robot,
    leader,
    events,
    fps,
    dataset,
    dataset_features,
    control_time_s,
    single_task,
    display_data=True,
):
    """
    Custom record loop that applies gravity + friction compensation to leader.
    Based on record_loop but with integrated compensation.
    """
    dt = 1 / fps
    episode_start_time = time.perf_counter()
    
    # All joints (both arms)
    all_joints = []
    for motor in leader.bus_right.motors:
        all_joints.append(f"right_{motor}")
    for motor in leader.bus_left.motors:
        all_joints.append(f"left_{motor}")
    
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - episode_start_time
        
        # Check if we should exit
        if elapsed >= control_time_s or events["exit_early"] or events["stop_recording"]:
            break
        
        # Get leader state
        leader_action = leader.get_action()
        
        # Extract positions and velocities in degrees
        leader_positions_deg = {}
        leader_velocities_deg_per_sec = {}
        
        for motor in leader.bus_right.motors:
            pos_key = f"right_{motor}.pos"
            vel_key = f"right_{motor}.vel"
            if pos_key in leader_action:
                leader_positions_deg[f"right_{motor}"] = leader_action[pos_key]
            if vel_key in leader_action:
                leader_velocities_deg_per_sec[f"right_{motor}"] = leader_action[vel_key]
        
        for motor in leader.bus_left.motors:
            pos_key = f"left_{motor}.pos"
            vel_key = f"left_{motor}.vel"
            if pos_key in leader_action:
                leader_positions_deg[f"left_{motor}"] = leader_action[pos_key]
            if vel_key in leader_action:
                leader_velocities_deg_per_sec[f"left_{motor}"] = leader_action[vel_key]
        
        # Calculate gravity torques for leader using built-in method
        leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
        leader_gravity_torques_nm = leader._gravity_from_q(leader_positions_rad)
        
        # Calculate friction torques for leader using built-in method
        leader_velocities_rad_per_sec = {k: np.deg2rad(v) for k, v in leader_velocities_deg_per_sec.items()}
        leader_friction_torques_nm = leader._friction_from_velocity(
            leader_velocities_rad_per_sec,
            friction_scale=FRICTION_SCALE
        )
        
        # Combine gravity + friction torques
        leader_total_torques_nm = {}
        for motor_name in leader_gravity_torques_nm:
            gravity = leader_gravity_torques_nm.get(motor_name, 0.0)
            friction = leader_friction_torques_nm.get(motor_name, 0.0)
            leader_total_torques_nm[motor_name] = gravity + friction
        
        # Apply gravity + friction compensation to leader RIGHT arm (all joints including gripper)
        for motor in leader.bus_right.motors:
            full_name = f"right_{motor}"
            position = leader_positions_deg.get(full_name, 0.0)
            torque = leader_total_torques_nm.get(full_name, 0.0)
            
            # Get damping gain for stability
            kd = leader.get_damping_kd(motor)
            
            leader.bus_right._mit_control(
                motor=motor,
                kp=0.0,
                kd=kd,  # Add damping for stability
                position_degrees=position,
                velocity_deg_per_sec=0.0,
                torque=torque,
            )
        
        # Apply gravity + friction compensation to leader LEFT arm (all joints including gripper)
        for motor in leader.bus_left.motors:
            full_name = f"left_{motor}"
            position = leader_positions_deg.get(full_name, 0.0)
            torque = leader_total_torques_nm.get(full_name, 0.0)
            
            # Get damping gain for stability
            kd = leader.get_damping_kd(motor)
            
            leader.bus_left._mit_control(
                motor=motor,
                kp=0.0,
                kd=kd,  # Add damping for stability
                position_degrees=position,
                velocity_deg_per_sec=0.0,
                torque=torque,
            )
        
        # Send leader positions to follower (both arms)
        follower_action = {}
        for joint in all_joints:
            pos_key = f"{joint}.pos"
            if pos_key in leader_action:
                follower_action[pos_key] = leader_action[pos_key]
        
        # Send action to robot
        if follower_action:
            robot.send_action(follower_action)
        
        # Get observation from robot (includes camera images)
        observation = robot.get_observation()
        
        # Add to dataset if we have a dataset
        if dataset is not None:
            # Build properly formatted observation frame
            obs_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
            
            # Build properly formatted action frame (keep .pos suffix - it matches the feature names)
            action_frame = build_dataset_frame(dataset_features, follower_action, prefix="action")
            
            # Combine into single frame
            frame = {**obs_frame, **action_frame}
            
            # Add metadata (task is required, timestamp will be auto-calculated by add_frame)
            frame["task"] = single_task
            
            dataset.add_frame(frame)
        
        # Display data if requested
        if display_data:
            log_rerun_data(observation=observation, action=follower_action)
        
        # Maintain loop rate
        loop_duration = time.perf_counter() - loop_start
        sleep_time = dt - loop_duration
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    """Main recording loop with gravity compensation."""
    
    print("=" * 70)
    print("OpenArms Dataset Recording with Compensation")
    print("=" * 70)
    
    # Create camera configurations (3 cameras: left wrist, right wrist, base)
    # Using actual device paths found by lerobot-find-cameras opencv
    camera_config = {
        "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=FPS),
        "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=FPS),
        "base": OpenCVCameraConfig(index_or_path="/dev/video7", width=640, height=480, fps=FPS),
    }
    
    # Configure follower robot with cameras
    follower_config = OpenArmsFollowerConfig(
        port_left="can2",
        port_right="can3",
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
        cameras=camera_config,
    )
    
    # Configure leader teleoperator (no cameras needed)
    leader_config = OpenArmsLeaderConfig(
        port_left="can0",
        port_right="can1",
        can_interface="socketcan",
        id="openarms_leader",
        manual_control=False,  # Enable torque control for gravity compensation
    )
    
    # Initialize robot and teleoperator
    print("\nInitializing devices...")
    follower = OpenArmsFollower(follower_config)
    leader = OpenArmsLeader(leader_config)
    
    # Connect devices
    print("Connecting and calibrating...")
    follower.connect(calibrate=True)
    leader.connect(calibrate=True)
    
    # Verify URDF is loaded for gravity compensation
    if leader.pin_robot is None:
        raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")
    
    # Configure the dataset features
    # For actions, we only want to record positions (not velocity or torque)
    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value
    
    action_features = hw_to_dataset_features(action_features_hw, "action")
    obs_features = hw_to_dataset_features(follower.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Create the dataset
    print("\nCreating dataset...")
    repo_id = "<hf_username>/<dataset_repo_id>"  # TODO: Replace with your Hugging Face repo
    
    # Check if dataset already exists and prompt user
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    while dataset_path.exists():
        print(f"\nDataset already exists at: {dataset_path}")
        print("\nOptions:")
        print("  1. Overwrite existing dataset")
        print("  2. Use a different name")
        print("  3. Abort")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print(f"Removing existing dataset...")
            shutil.rmtree(dataset_path)
            print("✓ Existing dataset removed")
            break
        elif choice == '2':
            print("\nCurrent repo_id:", repo_id)
            new_repo_id = input("Enter new repo_id (format: <username>/<dataset_name>): ").strip()
            if new_repo_id and '/' in new_repo_id:
                repo_id = new_repo_id
                dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                print(f"✓ Using new repo_id: {repo_id}")
                # Loop will continue if this new path also exists
            else:
                print("Invalid repo_id format. Please use format: <username>/<dataset_name>")
        elif choice == '3':
            print("Aborting. Please remove the existing dataset manually or restart with a different repo_id.")
            follower.disconnect()
            leader.disconnect()
            return
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Initialize keyboard listener and visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="openarms_recording")
    
    # Enable motors on both leader arms for gravity compensation
    leader.bus_right.enable_torque()
    leader.bus_left.enable_torque()
    time.sleep(0.1)
    
    print("\n" + "=" * 70)
    print(f"Recording {NUM_EPISODES} episodes")
    print(f"Task: {TASK_DESCRIPTION}")
    print("=" * 70)
    print("\nLeader BOTH arms: Gravity + Friction comp | Follower BOTH arms: Teleop")
    print("\nKeyboard controls:")
    print("  - Press 'q' to stop recording")
    print("  - Press 'r' to re-record current episode")
    print("=" * 70)
    
    episode_idx = 0
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
            
            # Record episode with compensation active
            record_loop_with_compensation(
                robot=follower,
                leader=leader,
                events=events,
                fps=FPS,
                dataset=dataset,
                dataset_features=dataset_features,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            
            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
                log_say("Reset the environment")
                record_loop_with_compensation(
                    robot=follower,
                    leader=leader,
                    events=events,
                    fps=FPS,
                    dataset=None,  # Don't save reset period
                    dataset_features=dataset_features,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )
            
            # Handle re-recording
            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Only save episode if frames were recorded
            if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                dataset.save_episode()
                episode_idx += 1
            else:
                log_say("No frames recorded, skipping episode save")
                # Clear the empty buffer
                dataset.episode_buffer = None
    
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    
    finally:
        # Clean up
        log_say("Stop recording")
        try:
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
            leader.disconnect()
            follower.disconnect()
            print("✓ Shutdown complete")
        except Exception as e:
            print(f"Shutdown error: {e}")
        
        # Upload dataset
        print("\nUploading dataset to Hugging Face Hub...")
        try:
            dataset.push_to_hub()
            print("✓ Dataset uploaded successfully")
        except Exception as e:
            print(f"Warning: Failed to upload dataset: {e}")
            print("You can manually upload later using: dataset.push_to_hub()")
        
        print("✓ Recording complete!")


if __name__ == "__main__":
    main()
