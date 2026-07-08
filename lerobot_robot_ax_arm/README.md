# lerobot_robot_ax_arm

A third-party [LeRobot](https://github.com/huggingface/lerobot) robot: a 4-DoF arm driven by Dynamixel
AX-series servos (e.g. AX-12A) over **Protocol 1.0**.

| Motor ID | Joint           | Normalization |
| -------- | --------------- | ------------- |
| 1        | `shoulder_pan`  | [-100, 100]   |
| 2        | `shoulder_lift` | [-100, 100]   |
| 3        | `elbow_flex`    | [-100, 100]   |
| 4        | `gripper`       | [0, 100]      |

## Protocol 1.0 notes

AX-series motors differ from the X-series (Protocol 2.0) used elsewhere in LeRobot:

- **No Sync Read** — positions are read sequentially (`get_observation` / calibration). Sync Write is still
  used for `Goal_Position`.
- **No `Operating_Mode` / PID registers** — `configure()` only lowers the return delay time.
- **No homing offset register** — calibration records the range of motion only (`homing_offset = 0`) and is
  stored via the CW/CCW angle limits.

## Install

```bash
pip install -e .
```

This is discovered automatically by LeRobot thanks to the `lerobot_robot_` package prefix.

## Usage

```bash
lerobot-record \
  --robot.type=ax_arm \
  --robot.port=/dev/tty.usbserial-AL02L1E0 \
  # ... other arguments
```

Or from Python:

```python
from lerobot_robot_ax_arm import AXArm, AXArmConfig

robot = AXArm(AXArmConfig(port="/dev/tty.usbserial-AL02L1E0"))
robot.connect()
obs = robot.get_observation()
robot.disconnect()
```
