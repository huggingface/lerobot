# Koch Screwdriver Follower MuJoCo Simulation

This directory contains the plan and initial implementation for a MuJoCo simulation environment for the Koch screwdriver follower robot.

## Files Created

1. **`mujoco_koch_sim_plan.md`** - Comprehensive implementation plan
   - Robot model development (MJCF)
   - Environment implementation
   - Integration with LeRobot
   - Real2Sim calibration strategy
   - Timeline and resources needed

2. **`koch_screwdriver_simple.xml`** - Simplified MJCF model
   - Uses primitive shapes (cylinders, capsules, spheres)
   - 6 DOF: 5 position-controlled joints + 1 velocity-controlled screwdriver
   - Includes workpiece with screw holes
   - Three cameras: side, top, and wrist-mounted
   - Torque and touch sensors

3. **`test_koch_mujoco.py`** - Test script for the model
   - Basic visualization with sinusoidal motion
   - Screw insertion task demonstration
   - Sensor data printing

## Quick Start

### Prerequisites
```bash
pip install mujoco
pip install numpy
```

### Test the Model
```bash
# Basic visualization with joint motion
python test_koch_mujoco.py

# Test screw insertion task
python test_koch_mujoco.py insertion
```

## Key Features

### Mixed Control Modes
- **Joints 1-5**: Position control (matching real robot)
- **Joint 6 (Screwdriver)**: Velocity control

### Software Clutch Simulation
The plan includes simulating the clutch behavior:
- Monitor torque on screwdriver
- Cut velocity to 0 when torque exceeds threshold
- Cooldown period before re-enabling

### Tasks Implemented
1. **Screw Insertion**: Navigate to hole, align, and insert while rotating
2. **Screwdriver Rotation**: Pure rotation tasks

## Next Steps

1. **Obtain Real Robot Measurements**
   - Joint lengths and offsets
   - Mass properties
   - Gear ratios

2. **Create Accurate Meshes**
   - Check Koch robot GitHub for CAD files
   - Or measure and model in CAD software

3. **Implement Full Environment Class**
   - Copy the environment code from plan
   - Integrate with LeRobot's gym interface

4. **Calibration**
   - Record real robot trajectories
   - Run system identification
   - Fine-tune simulation parameters

5. **Training**
   - Use same ACT policy architecture
   - Train in simulation
   - Test sim2real transfer

## Integration with LeRobot

The simulation is designed to match the real robot's interface:
```python
# Same observation/action structure as real robot
obs = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "screwdriver.vel": float,
    "wrist_camera": np.array,  # (H, W, 3)
    "side_camera": np.array    # (H, W, 3)
}
```

## Simulation Parameters to Tune

Based on real robot data:
- Joint damping coefficients
- Actuator gains (kp, kd, kv)
- Friction parameters
- Current-to-torque conversion for clutch
- Camera positions and FOV

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Koch Robot GitHub](https://github.com/AlexanderKoch-Koch/low_cost_robot)
- [LeRobot Environment Guide](https://github.com/huggingface/lerobot) 