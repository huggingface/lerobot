# reBot B601-DM

Native LeRobot integration for the [Seeed Studio reBot B601-DM](https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/)
robot arm — a 6-DOF arm plus gripper driven by Damiao CAN motors — together with
the **StarArm102 / reBot Arm 102** leader arm used to teleoperate it.

This page covers single-arm and bimanual setups for both the follower (robot)
and the leader (teleoperator).

## Install LeRobot 🤗

Follow the [Installation Guide](./installation), then install the reBot extra:

```bash
pip install -e ".[rebot]"
```

This pulls in `motorbridge` (CAN motor control for the B601-DM follower) and
`motorbridge-smart-servo` (FashionStar UART servos for the reBot Arm 102 leader).

> On Linux, remove `brltty` (`sudo apt remove brltty`) so it does not hold the
> leader's USB serial port.

## Calibration

Neither arm stores a persistent hardware calibration: at every connection the
motors are re-zeroed against the pose the arm is physically holding. When
prompted, move the arm to its zero pose (the default sit-down position, gripper
closed) and press ENTER.

## Single-arm teleoperation

The follower talks to its CAN bus through a Damiao serial bridge
(`can_adapter=damiao`, the default) or a SocketCAN adapter (`can_adapter=socketcan`).

```bash
lerobot-teleoperate \
  --robot.type=rebot_b601_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=rebot_102_leader \
  --teleop.port=/dev/ttyUSB0
```

## Bimanual teleoperation

The bimanual robot and teleoperator reuse the single-arm classes; each arm is
configured through a nested `left_arm_config` / `right_arm_config`, and its
observation/action keys are namespaced with a `left_` / `right_` prefix.

```bash
lerobot-teleoperate \
  --robot.type=bi_rebot_b601_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --teleop.type=bi_rebot_102_leader \
  --teleop.left_arm_config.port=/dev/ttyUSB0 \
  --teleop.right_arm_config.port=/dev/ttyUSB1
```

## Recording datasets

Swap `lerobot-teleoperate` for `lerobot-record` (with the same `--robot.*` /
`--teleop.*` arguments) to record demonstrations for training.

See the [Seeed Studio wiki](https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/)
for hardware assembly and wiring details.
