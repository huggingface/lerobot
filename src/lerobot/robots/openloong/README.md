# OpenLoong "Qinglong" Humanoid Robot Support

This package provides LeRobot integration for the OpenLoong (青龙/Qinglong) humanoid robot from the Shanghai Humanoid Robotics Innovation Center.

## Overview

OpenLoong is a humanoid robot that uses:
- **MPC (Model Predictive Control)** for locomotion planning
- **WBC (Whole-Body Control)** for task prioritization and coordination

Reference: [OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control)

## Features

- ✅ MuJoCo simulation support
- ✅ 29-DOF joint control (legs, waist, arms)
- ✅ IMU feedback (quaternion, gyroscope, accelerometer)
- ✅ PD position control with configurable gains
- ✅ Camera support (ZMQ-based)
- ✅ Compatible with LeRobot training pipelines
- ✅ π*₀.₆ RECAP policy integration

## Installation

### Prerequisites

```bash
# MuJoCo (for simulation)
pip install mujoco

# LeRobot
pip install -e .
```

### Optional: OpenLoong Model

Download the OpenLoong URDF/MJCF from the official repository:
```bash
git clone https://github.com/loongOpen/OpenLoong-Dyn-Control.git
```

Set the path in your configuration:
```python
config = OpenLoongConfig(
    mjcf_path="path/to/openloong.xml"
)
```

## Quick Start

### Basic Usage

```python
from lerobot.robots.openloong import OpenLoong, OpenLoongConfig

# Create configuration
config = OpenLoongConfig(
    is_simulation=True,
    control_dt=1.0 / 500.0,  # 500Hz
)

# Initialize and connect
robot = OpenLoong(config)
robot.connect()

# Get observation
obs = robot.get_observation()
print(f"Base height: {obs['base.pos.z']:.3f} m")

# Send action (joint positions)
action = {
    "kLeftKnee.q": 0.5,
    "kRightKnee.q": 0.5,
    "kLeftHipPitch.q": -0.3,
    "kRightHipPitch.q": -0.3,
}
robot.send_action(action)

# Reset to default position
robot.reset()

# Disconnect
robot.disconnect()
```

### Joint Configuration

OpenLoong has 29 joints:

| Body Part | Joints | Description |
|-----------|--------|-------------|
| Left Leg | 6 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll |
| Right Leg | 6 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll |
| Waist | 3 | yaw, roll, pitch |
| Left Arm | 7 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw |
| Right Arm | 7 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw |

Access joint indices:
```python
from lerobot.robots.openloong import OpenLoongJointIndex

for joint in OpenLoongJointIndex:
    print(f"{joint.name}: index {joint.value}")
```

### Observation Space

```python
obs = robot.get_observation()

# Joint states
joint_pos = obs["kLeftHipPitch.q"]  # Joint position (rad)
joint_vel = obs["kLeftHipPitch.dq"]  # Joint velocity (rad/s)
joint_torque = obs["kLeftHipPitch.tau"]  # Joint torque (Nm)

# IMU data
gyro_x = obs["imu.gyro.x"]  # Angular velocity
accel_x = obs["imu.accel.x"]  # Linear acceleration
roll = obs["imu.rpy.roll"]  # Roll angle

# Base position
base_x = obs["base.pos.x"]
base_y = obs["base.pos.y"]
base_z = obs["base.pos.z"]

# Camera images
camera_image = obs["camera"]
```

### Control Gains

Default PD gains are configured for each body part:

```python
from lerobot.robots.openloong import OPENLOONG_DEFAULT_GAINS

print(OPENLOONG_DEFAULT_GAINS)
# {
#     "left_leg": {"kp": [150, 150, 150, 300, 40, 40], "kd": [2, 2, 2, 4, 2, 2]},
#     "right_leg": {...},
#     "waist": {"kp": [250, 250, 250], "kd": [5, 5, 5]},
#     "left_arm": {"kp": [80, 80, 80, 80, 40, 40, 40], "kd": [3, 3, 3, 3, 1.5, 1.5, 1.5]},
#     "right_arm": {...},
# }
```

Customize gains in configuration:
```python
config = OpenLoongConfig(
    kp=[...],  # 29 values
    kd=[...],  # 29 values
)
```

## MPC/WBC Parameters

Configure MPC and WBC parameters:

```python
config = OpenLoongConfig(
    # MPC parameters
    mpc_horizon=1.0,  # Prediction horizon (seconds)
    mpc_dt=0.01,  # MPC timestep
    mpc_u_weight=0.001,  # Input weight
    mpc_L_diag=[100.0, ...],  # State error weights
    mpc_K_diag=[0.1, ...],  # Input weights
    
    # WBC parameters
    wbc_qp_weight_Q1=[1.0, ...],  # Contact force error weight
    wbc_qp_weight_Q2=[1.0, ...],  # Joint acceleration error weight
    
    # Gait parameters
    gait_period=0.4,  # Walking period
    step_height=0.08,  # Step height (meters)
)
```

## Training with π*₀.₆ RECAP

### 1. Prepare RECAP Dataset

```python
from lerobot.examples.openloong.recap_dataset_example import prepare_recap_dataset

dataset_path = prepare_recap_dataset(
    num_demo_episodes=50,
    num_auto_episodes=200,
    num_intervention_episodes=30,
    output_dir="./openloong_recap_dataset",
)
```

### 2. Train Policy

```bash
python -m lerobot.train \
    policy=pi_star_recap \
    dataset.path=./openloong_recap_dataset \
    robot=openloong \
    training.num_epochs=100
```

### 3. Run Trained Policy

```python
from lerobot.examples.openloong.pi_star_recap_control import run_policy_control

run_policy_control(robot, policy, num_steps=1000)
```

## Examples

See `examples/openloong/` for complete examples:

- `basic_control.py` - Basic robot control
- `pi_star_recap_control.py` - Policy inference
- `recap_dataset_example.py` - Dataset preparation

## Architecture

```
lerobot/robots/openloong/
├── __init__.py              # Module exports
├── config_openloong.py      # Configuration classes
├── openloong.py             # Main robot class
├── openloong_utils.py       # Utilities and joint definitions
└── README.md                # This file
```

## Physical Robot Connection

⚠️ **Physical robot control is not yet implemented.**

To connect to the physical OpenLoong robot, you would need to implement the communication protocol with the robot's controller. The interface is defined in `openloong.py`:

```python
def _init_physical_robot(self) -> None:
    # Implement connection to physical robot
    # This would communicate with OpenLoong's control system
    pass
```

## Troubleshooting

### Simulation Issues

**Error: MuJoCo not found**
```bash
pip install mujoco
```

**Robot falls over**
- Check default positions in configuration
- Verify PD gains are appropriate
- Increase simulation timestep stability

### Policy Issues

**Action space mismatch**
- Ensure policy output dimension matches robot action space (29 joints)
- Check observation preprocessing in `prepare_observation_for_policy()`

## References

1. [OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control)
2. [LeRobot Documentation](https://github.com/huggingface/lerobot)
3. [π*₀.₆ RECAP Paper](https://arxiv.org/abs/...)  # Add when available

## License

Apache 2.0 - See LeRobot repository for details.
