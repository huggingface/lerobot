#!/usr/bin/env python3
"""Test script for Koch screwdriver follower MuJoCo model."""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

def test_basic_model():
    """Test loading and visualizing the simplified model."""
    
    # Load model
    model_path = Path("koch_screwdriver_simple.xml")
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found!")
        return
    
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Print model information
    print(f"\nModel loaded successfully!")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of sensors: {model.nsensor}")
    print(f"Timestep: {model.opt.timestep}")
    
    # Print joint names
    print("\nJoints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"  {i}: {joint_name}")
    
    # Print actuator info
    print("\nActuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")
    
    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - Space: Pause/unpause simulation")
    print("  - Scroll: Zoom in/out")
    print("  - Left click + drag: Rotate camera")
    print("  - Right click + drag: Pan camera")
    print("  - Esc: Exit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial joint positions
        data.qpos[0] = 0.0    # shoulder_pan
        data.qpos[1] = -0.5   # shoulder_lift
        data.qpos[2] = 0.8    # elbow_flex
        data.qpos[3] = -0.3   # wrist_flex
        data.qpos[4] = 0.0    # wrist_roll
        data.qpos[5] = 0.0    # screwdriver
        
        # Initialize simulation
        mujoco.mj_forward(model, data)
        
        # Simulation loop
        t = 0
        while viewer.is_running():
            # Simple sinusoidal motion for testing
            t += model.opt.timestep
            
            # Position control for joints 1-5
            data.ctrl[0] = 0.5 * np.sin(0.5 * t)      # shoulder_pan
            data.ctrl[1] = -0.5 + 0.3 * np.sin(0.7 * t)  # shoulder_lift
            data.ctrl[2] = 0.8 + 0.2 * np.sin(0.9 * t)   # elbow_flex
            data.ctrl[3] = -0.3 + 0.2 * np.sin(1.1 * t)  # wrist_flex
            data.ctrl[4] = 0.3 * np.sin(1.3 * t)      # wrist_roll
            
            # Velocity control for screwdriver
            data.ctrl[5] = 5.0 * np.sin(2.0 * t)  # screwdriver velocity
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Print some sensor data periodically
            if int(t * 10) % 10 == 0:  # Every second
                screw_vel = data.sensordata[model.sensor('screwdriver_vel').id]
                screw_torque = data.sensordata[model.sensor('screwdriver_torque').id]
                print(f"t={t:.1f}s: Screwdriver vel={screw_vel:.2f}, torque={screw_torque:.3f}")

def test_screw_insertion():
    """Test screw insertion task simulation."""
    
    model_path = Path("koch_screwdriver_simple.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    print("\nTesting screw insertion task...")
    
    # Get relevant body IDs
    screwdriver_body = model.body("screwdriver_mount").id
    screw_body = model.body("screw_1").id
    hole_body = model.body("screw_hole_1").id
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set robot to approach position
        data.qpos[0] = 0.0    # shoulder_pan
        data.qpos[1] = -0.7   # shoulder_lift
        data.qpos[2] = 1.2    # elbow_flex
        data.qpos[3] = -0.5   # wrist_flex
        data.qpos[4] = 0.0    # wrist_roll
        data.qpos[5] = 0.0    # screwdriver
        
        mujoco.mj_forward(model, data)
        
        # Control loop
        step = 0
        phase = "approach"  # approach, align, insert
        
        while viewer.is_running():
            # Get current positions
            screw_pos = data.xpos[screwdriver_body].copy()
            hole_pos = data.xpos[hole_body].copy()
            
            # Simple P controller for approach
            error = hole_pos - screw_pos
            distance = np.linalg.norm(error)
            
            if phase == "approach" and distance < 0.05:
                phase = "align"
                print("Phase: Align")
            elif phase == "align" and distance < 0.02:
                phase = "insert"
                print("Phase: Insert")
            
            # Control based on phase
            if phase == "approach":
                # Move towards hole
                data.ctrl[0] = np.clip(error[0] * 2.0, -1.0, 1.0)
                data.ctrl[1] = np.clip(error[2] * 2.0 - 0.7, -1.57, 0)
                
            elif phase == "align":
                # Fine alignment
                data.ctrl[0] = np.clip(error[0] * 5.0, -0.5, 0.5)
                data.ctrl[1] = np.clip(error[2] * 5.0 - 0.7, -1.57, 0)
                
            elif phase == "insert":
                # Rotate screwdriver
                data.ctrl[5] = 5.0  # Constant rotation
                
                # Apply slight downward pressure
                data.ctrl[1] = np.clip(data.qpos[1] - 0.01, -1.57, 0)
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print status
            if step % 50 == 0:
                torque = data.sensordata[model.sensor('screwdriver_torque').id]
                print(f"Step {step}: Distance={distance:.3f}m, Torque={torque:.3f}")
            
            step += 1

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "insertion":
        test_screw_insertion()
    else:
        test_basic_model() 