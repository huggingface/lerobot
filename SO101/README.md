# SO101 Robot - URDF and MuJoCo Description

This directory is intended to hold URDF and MuJoCo (MJCF) files for the SO101 robot.

## Getting the URDF

The URDF is not bundled in this repo. Download it from the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) project and place it here:

- **Recommended**: [so101_new_calib.urdf](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf) — save as `SO101/so101_new_calib.urdf`

Then run scripts with: `--urdf=./SO101/so101_new_calib.urdf` (or `--urdf=./SO101` if the file is present).

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

The MuJoCo file `scene.xml` supports two differenly calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

---

Feel free to open an issue or contribute improvements!
