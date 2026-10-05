# Example kinematics models

These self-contained URDFs support the Cartesian agent examples without a separate model download. Run from the repository root, or make the YAML `urdf_path` values absolute. LeRobot resolves model paths from the working directory.

| File                   | Source                                                                                                                                                                                                         | License                           | Controlled link      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------- |
| `so101_new_calib.urdf` | [TheRobotStudio/SO-ARM100, revision 5f6d2b876a53a4872e405b991dd925556c9e38a4](https://github.com/TheRobotStudio/SO-ARM100/blob/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101/so101_new_calib.urdf) | Apache-2.0, [copy](LICENSE-SO101) | `gripper_frame_link` |
| `yam.urdf`             | [I2RT Robotics, revision 120c3c81400171174604e503943f8d1ebc891058](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/arm/yam/v1/yam.urdf)                  | MIT, [copy](LICENSE-YAM)          | `gripper` (flange)   |

Modification: all `visual` and `collision` elements were removed from the source files. Links, joints, origins, axes, limits, inertial parameters and other source content are preserved. No external meshes are referenced. Model provenance and modification notices also appear in each URDF.

These models provide FK/IK only. For rendering or simulation with geometry, obtain the original URDF and its adjacent `assets` directory from the source links. No collision geometry, camera calibration or fingertip TCP calibration is supplied here. SO-101 uses the new calibration convention and five arm joints; each YAM model describes one arm in its own base frame. See [the rollout guide](../../../docs/source/agent_rollout.mdx) for units, mappings and workspace configuration.
