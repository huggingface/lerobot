# Bimanual Parcelot Robot

The Bimanual Parcelot robot is a dual-arm manipulation system consisting of:
- Two SO-101 Follower Arms (left and right)
- Two SO-101 Leader Arms for teleoperation
- Three cameras: top-view and two wrist-mounted cameras

## Hardware Components

### Arms
- **Left Arm**: SO-101 Follower connected to specified USB port
- **Right Arm**: SO-101 Follower connected to specified USB port
- Each arm has 6 degrees of freedom:
  - Shoulder Pan
  - Shoulder Lift
  - Elbow Flex
  - Wrist Flex
  - Wrist Roll
  - Gripper

### Cameras
- **Top Camera**: Overhead view of the workspace
- **Left Wrist Camera**: Mounted on the left arm for close-up manipulation
- **Right Wrist Camera**: Mounted on the right arm for close-up manipulation

## Configuration

Example configuration for the bimanual robot:

```python
from lerobot.common.robots.bimanual_parcelot import BimanualParcelotConfig
from lerobot.common.cameras import OpenCVCameraConfig

config = BimanualParcelotConfig(
    id="parcelot_bimanual_01",
    left_arm_port="/dev/tty.usbmodem5A460815201",
    right_arm_port="/dev/tty.usbmodem5A680104701",
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=1,
            fps=30,
            width=640,
            height=480,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=2,
            fps=30,
            width=640,
            height=480,
        ),
    },
    max_relative_target=5,  # Safety limit for joint movements
)
```

## Calibration

Before using the robot, both arms need to be calibrated:

```bash
# Calibrate the bimanual robot
python -m lerobot.calibrate \
    --robot.type=bimanual_parcelot \
    --robot.left_arm_port=/dev/tty.usbmodem5A460815201 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680104701 \
    --robot.id=parcelot_bimanual_01
```

The calibration process will:
1. Prompt you to move the LEFT arm to its middle position
2. Record the range of motion for the LEFT arm
3. Prompt you to move the RIGHT arm to its middle position  
4. Record the range of motion for the RIGHT arm
5. Save calibration data for both arms

## Teleoperation

For teleoperation with two leader arms, use the `bimanual_teleop` teleoperator:

```bash
# Teleoperate bimanual robot
python -m lerobot.teleoperate \
    --teleop.type=bimanual_teleop \
    --teleop.left_port=/dev/tty.usbmodem5A460826891 \
    --teleop.right_port=/dev/tty.usbmodem58FD0163901 \
    --teleop.id=teleop_bimanual_01 \
    --robot.type=bimanual_parcelot \
    --robot.left_arm_port=/dev/tty.usbmodem5A460815201 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680104701 \
    --robot.id=parcelot_bimanual_01
```

## Safety Features

- **Torque Disable**: Arms automatically disable torque on disconnect
- **Relative Target Limiting**: Movement magnitude is limited for safety
- **Per-arm Configuration**: Independent safety limits for each arm
- **Connection Monitoring**: Continuous monitoring of both arms and cameras

## Usage in Code

```python
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.bimanual_parcelot import BimanualParcelotConfig

# Create and connect the robot
config = BimanualParcelotConfig(
    id="my_bimanual_robot",
    left_arm_port="/dev/ttyUSB0",
    right_arm_port="/dev/ttyUSB1",
    # ... camera configurations
)

robot = make_robot_from_config(config)
robot.connect()

# Get observations from both arms and all cameras
obs = robot.get_observation()
print("Left arm positions:", {k: v for k, v in obs.items() if k.startswith("left_")})
print("Right arm positions:", {k: v for k, v in obs.items() if k.startswith("right_")})

# Send actions to both arms
action = {
    "left_shoulder_pan.pos": 0.1,
    "left_gripper.pos": 50.0,
    "right_shoulder_pan.pos": -0.1,
    "right_gripper.pos": 30.0,
    # ... other joint positions
}
robot.send_action(action)

robot.disconnect()
```

## Observation Space

The robot provides observations from:
- 12 joint positions (6 per arm): `{left|right}_{joint_name}.pos`
- 3 cameras: `top`, `left_wrist`, `right_wrist` as (H, W, 3) arrays

## Action Space

Actions control 12 joint positions:
- Left arm: `left_shoulder_pan.pos`, `left_shoulder_lift.pos`, etc.
- Right arm: `right_shoulder_pan.pos`, `right_shoulder_lift.pos`, etc.
- Grippers: `left_gripper.pos`, `right_gripper.pos` (0-100 range) 