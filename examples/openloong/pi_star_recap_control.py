#!/usr/bin/env python
"""
Example of using π*₀.₆ RECAP policy with OpenLoong robot.

This example demonstrates how to:
1. Load a trained π*₀.₆ RECAP policy
2. Use the policy to control OpenLoong robot
3. Handle RECAP data types (demo, auto, intervention)
"""

import time
import torch
import numpy as np

from lerobot.robots.openloong import OpenLoong, OpenLoongConfig
from lerobot.policies.pi_star_recap import PiStarRecapPolicy, PiStarRecapConfig


def prepare_observation_for_policy(obs: dict, device: str = "cuda") -> dict:
    """
    Convert OpenLoong observation to policy input format.
    
    Args:
        obs: Observation from robot
        device: Device to place tensors on
        
    Returns:
        Dictionary with tensors ready for policy
    """
    # Extract proprioceptive data
    # Concatenate joint positions, velocities, and IMU data
    joint_pos = []
    joint_vel = []
    
    from lerobot.robots.openloong import OpenLoongJointIndex
    
    for joint in OpenLoongJointIndex:
        joint_pos.append(obs.get(f"{joint.name}.q", 0.0))
        joint_vel.append(obs.get(f"{joint.name}.dq", 0.0))
    
    joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
    joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
    
    # IMU data
    imu_data = torch.tensor([
        obs.get("imu.gyro.x", 0.0),
        obs.get("imu.gyro.y", 0.0),
        obs.get("imu.gyro.z", 0.0),
        obs.get("imu.accel.x", 0.0),
        obs.get("imu.accel.y", 0.0),
        obs.get("imu.accel.z", 0.0),
        obs.get("imu.rpy.roll", 0.0),
        obs.get("imu.rpy.pitch", 0.0),
        obs.get("imu.rpy.yaw", 0.0),
    ], dtype=torch.float32, device=device)
    
    # Base position
    base_pos = torch.tensor([
        obs.get("base.pos.x", 0.0),
        obs.get("base.pos.y", 0.0),
        obs.get("base.pos.z", 0.0),
    ], dtype=torch.float32, device=device)
    
    # Combine into state vector
    state = torch.cat([joint_pos, joint_vel, imu_data, base_pos])
    
    # Handle camera image if available
    image = None
    for key in obs:
        if "cam" in key.lower() or "image" in key.lower():
            image = obs[key]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).to(device)
            break
    
    return {
        "state": state.unsqueeze(0),  # Add batch dimension
        "image": image.unsqueeze(0) if image is not None else None,
    }


def policy_action_to_robot_action(
    policy_output: torch.Tensor,
) -> dict:
    """
    Convert policy output to robot action format.
    
    Args:
        policy_output: Action tensor from policy (batch, action_dim)
        
    Returns:
        Dictionary with joint position commands
    """
    action_dict = {}
    action_np = policy_output.squeeze(0).cpu().numpy()
    
    from lerobot.robots.openloong import OpenLoongJointIndex
    
    # Map policy output to joint commands
    # First 29 dimensions are joint positions
    for i, joint in enumerate(OpenLoongJointIndex):
        if i < len(action_np):
            action_dict[f"{joint.name}.q"] = float(action_np[i])
    
    return action_dict


def run_policy_control(
    robot: OpenLoong,
    policy: PiStarRecapPolicy,
    num_steps: int = 1000,
    device: str = "cuda",
):
    """
    Run policy control loop.
    
    Args:
        robot: OpenLoong robot instance
        policy: π*₀.₆ RECAP policy
        num_steps: Number of control steps
        device: Device for policy inference
    """
    print(f"Running policy control for {num_steps} steps...")
    
    policy.eval()
    
    with torch.no_grad():
        for step in range(num_steps):
            start_time = time.time()
            
            # Get observation from robot
            obs = robot.get_observation()
            
            # Prepare for policy
            policy_input = prepare_observation_for_policy(obs, device=device)
            
            # Run policy inference
            # Note: actual policy interface may differ
            policy_output = policy.select_action(policy_input)
            
            # Convert to robot action
            action = policy_action_to_robot_action(policy_output)
            
            # Send action to robot
            robot.send_action(action)
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, robot.control_dt - elapsed)
            time.sleep(sleep_time)
            
            if step % 100 == 0:
                print(f"  Step {step}/{num_steps}")


def main():
    """Run π*₀.₆ RECAP policy control example."""
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create robot configuration
    robot_config = OpenLoongConfig(
        is_simulation=True,
        control_dt=1.0 / 50.0,  # 50Hz for policy control
    )
    
    # Initialize robot
    print("Initializing OpenLoong robot...")
    robot = OpenLoong(robot_config)
    robot.connect(calibrate=True)
    
    try:
        # Create policy configuration
        policy_config = PiStarRecapConfig(
            freeze_vlm=True,
            iql_expectile=0.7,
            iql_temperature=0.5,
        )
        
        # Initialize policy
        print("Initializing π*₀.₆ RECAP policy...")
        policy = PiStarRecapPolicy(policy_config)
        policy.to(device)
        
        # In practice, you would load pretrained weights here
        # policy.load_state_dict(torch.load("path/to/checkpoint.pt"))
        
        # Run policy control
        run_policy_control(robot, policy, num_steps=500, device=device)
        
        # Reset robot
        print("Resetting robot...")
        robot.reset()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Disconnect
        print("Disconnecting...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
