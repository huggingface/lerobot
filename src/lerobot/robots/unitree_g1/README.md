# Unitree G1 Dex3 - LeRobot Integration

This module provides LeRobot-compatible drivers for the Unitree G1 humanoid robot with Dex3 dexterous hands.

## Installation

```bash
pip install lerobot[unitree_g1_dex3,televuer]
```

## Quick Start

```bash
# VR Teleoperation + Recording
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --teleop.type=televuer \
  --dataset.repo_id=your_user/g1_demo

# Train a policy
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=your_user/g1_demo

# Deploy
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --policy.path=outputs/train/checkpoints/last \
  --dataset.repo_id=your_user/g1_eval
```

## Components

| File | Description |
|------|-------------|
| `unitree_g1.py` | Base G1 robot driver (DDS + ZMQ) |
| `unitree_g1_dex3.py` | G1 + Dex3 hands extension |
| `g1_utils.py` | Joint constants, limits, and DDS topics |
| `robot_control/` | Arm IK solver + hand retargeting |
| `assets/` | URDF files for IK and retargeting |

## Python API

```python
from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config

config = UnitreeG1Dex3Config(id="g1")
robot = UnitreeG1Dex3(config)
robot.connect()

obs = robot.get_observation()
print(f"Arm joints: {len([k for k in obs if 'shoulder' in k])}")
print(f"Hand joints: {len([k for k in obs if 'hand' in k])}")

robot.disconnect()
```

## Joint Configuration

| Group | Joints | DOF |
|-------|--------|-----|
| Left arm | shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw | 7 |
| Right arm | shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw | 7 |
| Left hand | thumb (3), index (2), middle (2) | 7 |
| Right hand | thumb (3), index (2), middle (2) | 7 |
| **Total** | | **28** |

## Network Setup

The G1 robot uses Ethernet (default subnet: `192.168.123.x`):

```bash
# Set static IP on your machine
sudo ip addr add 192.168.123.100/24 dev eth0

# Test connection
ping 192.168.123.161
```

## Dependencies

- `unitree-sdk2py` - DDS communication (requires Python < 3.12)
- `pinocchio` + `casadi` - Arm IK solver  
- `dex-retargeting` - Hand retargeting from VR keypoints
- `vuer` - VR interface for TeleVuer teleoperator
