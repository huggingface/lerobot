# Fork Notes — DS4 Teleoperator for SO-101

This fork adds native DualShock 4 controller support for the SO-101 follower arm,
enabling teleoperation and episode recording without a leader arm.

**Added:** `src/lerobot/teleoperators/ds4_arm/` — see [PR #3907](https://github.com/huggingface/lerobot/pull/3907)

---

# DS4 Teleoperator for SO-101

Enables teleoperation and episode recording with a DualShock 4 controller,
without needing a leader arm.

## Controller Mapping

| Input       | Joint                                   |
| ----------- | --------------------------------------- |
| LS X        | shoulder_pan (left/right)               |
| LS Y        | shoulder_lift (push up = arm rises)     |
| RS Y        | elbow_flex (push up = elbow folds down) |
| RS X        | wrist_roll (twist left/right)           |
| L1 / R1     | wrist_flex (pitch down / up)            |
| D-pad ↑ / ↓ | gripper (open / close)                  |

## Speed Modes

| Button  | Mode   | Speed  |
| ------- | ------ | ------ |
| Square  | Slow   | 40 °/s |
| Default | Normal | 60 °/s |
| Circle  | Fast   | 90 °/s |

## Usage

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A7B2890981 \
    --robot.id=my_arm \
    --teleop.type=ds4_arm \
    --teleop.id=my_ds4
```

For recording episodes:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A7B2890981 \
    --robot.id=my_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}" \
    --teleop.type=ds4_arm \
    --teleop.id=my_ds4 \
    --dataset.repo_id=your_hf_username/your_dataset \
    --dataset.num_episodes=50 \
    --dataset.single_task="describe your task here"
```

## Notes

- Works with both USB and Bluetooth connection
- Tested on macOS with SO-101 follower arm
- EMA smoothing applied to analog axes to eliminate stick jitter
- Digital buttons (L1/R1, D-pad) have no smoothing for crisp response
