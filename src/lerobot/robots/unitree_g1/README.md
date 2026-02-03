# Unitree G1 Dex3 - LeRobot Integration

Complete guide for VR teleoperation, data collection, policy training, and deployment with the Unitree G1 humanoid robot with Dex3 hands.

## Quick Start

```bash
# 1. Setup environment
cd lerobot
conda env create -f environment_vr_teleop.yml
conda activate lerobot_vr

# 2. Collect demo data with VR
lerobot-record --robot.type=unitree_g1_dex3 --teleop.type=televuer \
  --dataset.repo_id=my_user/g1_pick_kettle

# 3. Train a policy
lerobot-train --policy.type=diffusion \
  --dataset.repo_id=my_user/g1_pick_kettle

# 4. Deploy on robot
lerobot-record --robot.type=unitree_g1_dex3 \
  --policy.path=outputs/train/diffusion/checkpoints/last \
  --dataset.repo_id=my_user/g1_eval
```

---

## 1. Environment Setup

### Prerequisites
- Ubuntu 22.04+
- Python 3.10
- Meta Quest 3 (for VR teleoperation)
- Unitree G1 robot with Dex3 hands

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/prometheus-vla.git
cd prometheus-vla/lerobot

# Create conda environment (required for pinocchio+CasADi)
conda env create -f environment_vr_teleop.yml
conda activate lerobot_vr
```

### Network Setup

The G1 robot uses Ethernet connection:

```bash
# Set static IP on robot's network interface
# Robot default: 192.168.123.xxx
sudo ip addr add 192.168.123.100/24 dev eth0
```

---

## 2. Robot Configuration

### Test Connection

```python
from lerobot.robots import make_robot_from_config
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Dex3Config

robot = make_robot_from_config(UnitreeG1Dex3Config())
robot.connect()

# Get current state
obs = robot.get_observation()
print(f"Left arm: {obs['left_shoulder_pitch_joint.q']:.3f}")

robot.disconnect()
```

### Calibration

```bash
lerobot-calibrate --robot.type=unitree_g1_dex3
```

---

## 3. Data Collection

### VR Teleoperation

Use Meta Quest 3 for demonstration collection:

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --teleop.type=televuer \
  --teleop.use_hand_tracking=true \
  --dataset.repo_id=my_user/g1_pick_kettle \
  --dataset.single_task="Pick up the kettle"
```

**Controls:**
- Hand tracking: Move hands to control robot arms
- Pinch gesture: Control gripper
- Voice: "Stop" to pause recording

### Keyboard Teleoperation (Debug)

```bash
lerobot-teleoperate --robot.type=unitree_g1_dex3
```

### Resume Recording

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --teleop.type=televuer \
  --dataset.repo_id=my_user/g1_pick_kettle \
  --resume=true
```

---

## 4. Policy Training

### Diffusion Policy

```bash
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=my_user/g1_pick_kettle \
  --training.num_epochs=100 \
  --training.batch_size=32 \
  --output_dir=outputs/train/g1_diffusion
```

### ACT Policy

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=my_user/g1_pick_kettle \
  --training.num_epochs=50 \
  --output_dir=outputs/train/g1_act
```

### Monitor Training

```bash
# View in Weights & Biases
wandb login
# Training automatically logs to W&B
```

---

## 5. Deployment

### Local Inference

Run policy on the same machine as robot:

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --policy.path=outputs/train/g1_diffusion/checkpoints/last \
  --dataset.repo_id=my_user/g1_eval \
  --dataset.single_task="Pick up the kettle"
```

### Remote Inference (GPU Server → Robot)

For running inference on a separate GPU machine:

**On GPU server:**
```bash
python -m lerobot.async_inference.policy_server \
  --policy.path=outputs/train/g1_diffusion/checkpoints/last \
  --port=50051
```

**On robot:**
```bash
python -m lerobot.async_inference.robot_client \
  --robot.type=unitree_g1_dex3 \
  --server_address=GPU_SERVER_IP:50051 \
  --task="Pick up the kettle"
```

---

## 6. Visualization

### Live Dashboard

```bash
python -m lerobot.scripts.visualization.visualize_g1_dashboard
```

### 3D Visualization

```bash
python -m lerobot.scripts.visualization.visualize_g1_3d
```

### Dataset Visualization

```bash
lerobot-dataset-viz --dataset.repo_id=my_user/g1_pick_kettle
```

---

## Configuration Reference

### Robot Config

```python
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Dex3Config

config = UnitreeG1Dex3Config(
    robot_address="192.168.123.161",  # Robot IP
    cameras={
        "head": {"type": "opencv", "index": 0},
    }
)
```

### Joint Groups

| Group | Joints | DOF |
|-------|--------|-----|
| Left arm | shoulder_pitch/roll/yaw, elbow, wrist | 7 |
| Right arm | shoulder_pitch/roll/yaw, elbow, wrist | 7 |
| Left hand | thumb (3), index (2), middle (2) | 7 |
| Right hand | thumb (3), index (2), middle (2) | 7 |
| **Total** | | **28** |

---

## 7. Simulation Testing

### Mock Robot

Test the full pipeline without hardware using the mock robot:

```bash
# Record with mock robot
lerobot-record \
  --robot.type=mock_unitree_g1_dex3 \
  --teleop.type=keyboard \
  --dataset.repo_id=my_user/g1_sim_test

# Train on simulated data
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=my_user/g1_sim_test

# Evaluate with mock robot
lerobot-record \
  --robot.type=mock_unitree_g1_dex3 \
  --policy.path=outputs/train/diffusion/checkpoints/last \
  --dataset.repo_id=my_user/g1_sim_eval
```

### Python API

```python
from lerobot.robots.unitree_g1.mock_unitree_g1_dex3 import (
    MockUnitreeG1Dex3, MockUnitreeG1Dex3Config
)

robot = MockUnitreeG1Dex3(MockUnitreeG1Dex3Config())
robot.connect()

# Get observation
obs = robot.get_observation()
print(f"Left shoulder: {obs['left_shoulder_pitch_joint.q']:.3f}")

# Send action
action = {f"{name}.q": 0.0 for name in robot.action_features}
action["left_shoulder_pitch_joint.q"] = 0.5
robot.send_action(action)

robot.disconnect()
```

### Visualization with Mock Robot

```bash
# Run 3D visualization with mock robot
python -c "
from lerobot.robots.unitree_g1.mock_unitree_g1_dex3 import *
robot = MockUnitreeG1Dex3(MockUnitreeG1Dex3Config())
robot.connect()
# Now use with visualization scripts
"
```

---

## Troubleshooting

### Robot not responding
```bash
# Check network
ping 192.168.123.161

# Restart DDS
sudo systemctl restart unitree_dds
```

### VR not connecting
```bash
# Ensure Quest and PC on same network
# Check firewall allows ports 8012, 8013
```

### IK solver errors
```bash
# Verify pinocchio installed from conda-forge
python -c "from pinocchio import casadi as cpin; print('OK')"
```
