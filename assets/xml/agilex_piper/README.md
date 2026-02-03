## AgileX PiPER Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 2.3.4 or later.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

### Overview

This package contains a simplified robot description (MJCF) of the [AgileX PiPER](https://global.agilex.ai/products/piper). It is derived from the publicly available [model](https://github.com/agilexrobotics/Piper_ros/tree/ros-noetic-no-aloha/src/piper_description/urdf).

<p float="left">
  <img src="piper.png" width="400">
</p>

### Derivation steps

1.  Added `<mujoco> <compiler balanceinertia="true" discardvisual="false"/> </mujoco>` to the URDF's
   `<robot>` clause in order to preserve visual geometries.
2. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
3. Converted the the .objs to .xmls using [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) and replaced the original stls with them (since each obj in mujoco can have 1 color).
4. Merged similar materials between the .objs
5. Created a `<default>` section to define common properties for joints, actuators, and geoms.
6. Added an equality constraint so that the right finger mimics the position of the left finger.
7. Manually designed box collision geoms for the gripper.
8. Added `exclude` clause to prevent collisions between `base_link` and `link1`.
9. Added position controlled actuators.
10. Added `impratio=10` and `cone=elliptic` for better noslip.
11. Added `scene.xml` which includes the robot, with a textured groundplane, skybox, and haze.

## License

This model is released under an [MIT License](LICENSE).

## Acknowledgement

This model was graciously contributed by [Omar Rayyan](https://orayyan.com/).
