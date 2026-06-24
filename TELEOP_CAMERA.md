# Camera-Based Teleoperation

Teleoperate the SO-101 follower arm with the SO-101 leader arm, while streaming
the front camera feed.

## Command

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=dk \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=bose \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

## Configuration

| Component | Setting        | Value         |
|-----------|----------------|---------------|
| Robot     | type           | so101_follower |
| Robot     | port           | /dev/ttyACM0  |
| Robot     | id             | dk            |
| Teleop    | type           | so101_leader  |
| Teleop    | port           | /dev/ttyACM1  |
| Teleop    | id             | bose          |
| Camera    | name           | front         |
| Camera    | index_or_path  | /dev/video2   |
| Camera    | resolution     | 640x480       |
| Camera    | fps            | 30            |

`--display_data=true` opens a live visualization of the robot state and camera
feed during teleoperation.

See [`SETUP_INSTRUCTIONS.md`](./SETUP_INSTRUCTIONS.md) for the full detected
hardware configuration.
