# G1 Dex3 VR Teleoperation

This module provides VR teleoperation for the Unitree G1 Dex3 robot using Meta Quest 3.

## Requirements

> **Important**: This module requires `pinocchio` with CasADi support, which is only available via conda-forge.

## Installation

```bash
# Create conda environment with VR teleop dependencies
cd lerobot
conda env create -f environment_vr_teleop.yml
conda activate lerobot_vr
```

## Components

| File | Purpose |
|------|---------|
| `robot_control/g1_arm_ik.py` | Pinocchio + CasADi IK solver (14 DOF arms) |
| `robot_control/hand_retargeting.py` | dex_retargeting wrapper for Dex3 hands |
| `robot_control/weighted_moving_filter.py` | Smoothing filter for IK output |

## Usage

```python
from lerobot.robots.unitree_g1.robot_control import G1_29_ArmIK, HandRetargeting, HandType

# Initialize IK solver
ik = G1_29_ArmIK()

# Solve for arm joints given wrist poses (4x4 matrices)
arm_q, tau = ik.solve_ik(left_wrist_pose, right_wrist_pose)

# Initialize hand retargeting
retargeting = HandRetargeting(HandType.UNITREE_DEX3)

# Retarget VR hand keypoints (21x3) to Dex3 joints
left_hand_q = retargeting.retarget_left(left_hand_keypoints)
right_hand_q = retargeting.retarget_right(right_hand_keypoints)
```

## Recording with VR

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --teleop.type=televuer \
  --dataset.repo_id=my_user/vr_recording
```
