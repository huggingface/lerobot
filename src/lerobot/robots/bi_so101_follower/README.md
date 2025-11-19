# BiSO-101 Follower Robot

The `bi_so101_follower` robot lets LeRobot control two SO-101 follower arms (left/right) as a single synchronized robot. It follows the same design as our SO-100 bimanual implementation but upgrades all joints and calibration paths to the SO-101 hardware designed by [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100).

- Wraps two `SO101Follower` instances and exposes their joints with `left_*/right_*` prefixes so policies and teleoperators can address both arms with a single action dictionary (see `action_features` in `bi_so101_follower.py`).
- Shares camera streams between the arms. The config accepts arbitrary `CameraConfig` entries and they are automatically wired in `observation_features`.
- Fully compatible with `lerobot-record`, `lerobot-replay`, and `lerobot-teleoperate`.
