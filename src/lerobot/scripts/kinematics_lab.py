#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Interactive forward/inverse kinematics lab for SO100/SO101-style follower arms.

By default this script only computes FK/IK from a URDF. Pass `--connect --port ...`
to read joints from, and send IK results to, a real follower arm.

Real robot usage example
------------------------

0. Install the optional kinematics dependency.

   From the LeRobot repository root, inside the same Python environment used
   to run this script:

   PowerShell:

   ```powershell
   python -m pip install -e ".[kinematics]"
   ```

   On Windows, if pip tries to build `placo` from source and fails with an
   `nmake` or C/C++ compiler error, install `placo` from conda-forge instead:

   ```powershell
   conda install -c conda-forge placo
   python -m pip install -e .
   ```

   Bash/zsh:

   ```shell
   python -m pip install -e '.[kinematics]'
   ```

1. Find the motor bus serial port.

   Windows ports usually look like `COM5`; Linux/macOS ports usually look like
   `/dev/ttyACM0`, `/dev/ttyUSB0`, or `/dev/tty.usbmodem*`.

   ```shell
   lerobot-find-port
   ```

2. Prepare the URDF and calibration id.

   The URDF describes the arm geometry and joint order used by FK/IK. The
   LeRobot repository does not ship an SO100/SO101 URDF file; for SO101-style
   arms, use the calibrated URDF from the SO-ARM100 repository:

   https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf

   Download the whole `Simulation/SO101` folder, not only the `.urdf` file.
   The URDF references mesh files such as
   `assets/base_motor_holder_so101_v1.stl`; those STL files must stay next to
   the URDF with the same relative layout:

   ```text
   assets/urdf/so101/
   |-- so101_new_calib.urdf
   `-- assets/
       |-- base_motor_holder_so101_v1.stl
       |-- base_so101_v2.stl
       `-- ...
   ```

   One convenient way is to sparse-clone only that folder:

   ```powershell
   git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git external\SO-ARM100
   cd external\SO-ARM100
   git sparse-checkout set Simulation/SO101
   cd ..\..
   ```

   Then pass:

   ```powershell
   --urdf-path external\SO-ARM100\Simulation\SO101\so101_new_calib.urdf
   ```

   `kinematics_lab` does not generate this file from the LeRobot calibration
   JSON. The LeRobot calibration JSON maps raw motor positions to calibrated
   joint angles; the URDF provides the physical link lengths, joint axes, and
   end-effector frame. For real-arm runs, use the same `--id` that you used
   when running LeRobot calibration commands so motor readings and commands use
   the correct calibration file.

   To inspect the frame and joint names in the URDF:

   ```powershell
   Select-String -Path external\SO-ARM100\Simulation\SO101\so101_new_calib.urdf -Pattern '<joint name=','<link name='
   ```

3. Do a dry run before powering motion.

   This loads the URDF, lets you type FK/IK commands, and prints joint targets
   without moving hardware.

   PowerShell:

   ```powershell
   python -m lerobot.scripts.kinematics_lab `
       --urdf-path path\to\so101_new_calib.urdf `
       --target-frame-name gripper_frame_link `
       --units mm
   ```

   Bash/zsh:

   ```shell
   python -m lerobot.scripts.kinematics_lab \
       --urdf-path path/to/so101_new_calib.urdf \
       --target-frame-name gripper_frame_link \
       --units mm
   ```

   Example dry-run session:

   ```text
   kinematics> joints 0 -45 45 0 0 30
   kinematics> fk
   kinematics> ik 180 0 120
   kinematics> quit
   ```

4. Connect to the real follower arm.

   Keep one hand near the power switch, make sure the workspace is clear, and
   start with a small `--max-relative-target` so each command is limited to a
   modest joint change.

   PowerShell:

   ```powershell
   python -m lerobot.scripts.kinematics_lab `
       --urdf-path path\to\so101_new_calib.urdf `
       --connect `
       --robot-type so101_follower `
       --port COM5 `
       --id classroom_so101 `
       --target-frame-name gripper_frame_link `
       --units mm `
       --max-relative-target 5
   ```

   Bash/zsh:

   ```shell
   python -m lerobot.scripts.kinematics_lab \
       --urdf-path path/to/so101_new_calib.urdf \
       --connect \
       --robot-type so101_follower \
       --port /dev/ttyACM0 \
       --id classroom_so101 \
       --target-frame-name gripper_frame_link \
       --units mm \
       --max-relative-target 5
   ```

5. Recommended first real-arm session.

   ```text
   kinematics> read
   kinematics> fk
   kinematics> ik 180 0 120
   kinematics> move 180 0 120
   kinematics> move 185 0 120
   kinematics> move 185 5 120 30
   kinematics> read
   kinematics> quit
   ```

   `read` synchronizes the current joint guess with the real robot. `ik`
   computes a target without moving. `move` computes IK and sends the target to
   hardware. The optional fourth value in `move x y z gripper` commands the
   gripper joint in degrees when the robot exposes one.

Command reference
-----------------

```text
read
    Read real robot joints and print FK. Requires --connect.
joints q1 q2 q3 q4 q5 [gripper]
    Set the current joint guess in degrees and print FK.
fk
    Print FK for the current joint guess.
ik x y z
    Solve IK for an absolute end-effector position in the selected units;
    keeps the current end-effector orientation.
move x y z [gripper]
    Solve IK, then send the joint target to the robot. Requires --connect.
send
    Send the current joint guess to the robot. Requires --connect.
help
    Show the interactive command help.
quit
    Disconnect and exit.
```

Useful options
--------------

`--robot-type`
    `so100_follower` or `so101_follower`; choose the implementation matching
    the physical follower arm.
`--units`
    Unit used by interactive Cartesian commands: `m`, `cm`, or `mm`.
`--target-frame-name`
    URDF frame used as the end-effector target. The default is
    `gripper_frame_link`.
`--joint-names`
    Optional explicit list of URDF joint names if the URDF contains extra
    joints or a nonstandard order.
`--visualize`
    Open a Rerun viewer and update a 3D link-frame skeleton during dry-run or
    real-arm commands.
`--calibration-dir`
    Optional directory containing calibration files. If omitted, LeRobot uses
    its normal calibration lookup path for the given `--id`.
`--no-calibrate`
    Skip calibration on connect. Use only when the arm is already calibrated
    and you intentionally do not want the startup calibration flow.
`--max-relative-target`
    Maximum per-command relative joint target used by the follower robot
    safety layer. Smaller values are safer for first tests.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np

from lerobot.model.kinematics import RobotKinematics

if TYPE_CHECKING:
    from lerobot.robots import Robot


ARM_JOINT_COUNT = 5


def _link_chain_from_urdf(urdf_path: Path, target_frame_name: str) -> list[str]:
    root = ElementTree.parse(urdf_path).getroot()
    parent_by_child = {}

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if parent_name and child_name:
            parent_by_child[child_name] = parent_name

    chain = [target_frame_name]
    while chain[-1] in parent_by_child:
        chain.append(parent_by_child[chain[-1]])

    return list(reversed(chain))


class RerunKinematicsVisualizer:
    def __init__(self, kinematics: RobotKinematics, urdf_path: Path, target_frame_name: str):
        try:
            import rerun as rr

            from lerobot.utils.visualization_utils import _init_rerun
        except ImportError as e:
            raise ImportError(
                "rerun-sdk is required for --visualize. Install LeRobot's normal dependencies."
            ) from e

        self.rr = rr
        self.kinematics = kinematics
        self.frame_names = [
            frame_name
            for frame_name in _link_chain_from_urdf(urdf_path, target_frame_name)
            if frame_name in set(kinematics.robot.frame_names())
        ]
        if len(self.frame_names) < 2:
            self.frame_names = [
                frame_name
                for frame_name in kinematics.robot.frame_names()
                if frame_name not in {"universe", "root_joint"}
            ]

        self.step = 0
        _init_rerun(session_name="kinematics_lab")
        print("Rerun visualization started. The viewer should open in a separate window.")

    def update(self, joints_deg: np.ndarray, label: str) -> None:
        self.kinematics.forward_kinematics(joints_deg)
        positions = np.array(
            [self.kinematics.robot.get_T_world_frame(frame_name)[:3, 3] for frame_name in self.frame_names],
            dtype=float,
        )
        target_pose = self.kinematics.robot.get_T_world_frame(self.kinematics.target_frame_name)
        target_xyz = target_pose[:3, 3]

        self.rr.set_time_sequence("step", self.step)
        self.rr.log(
            "kinematics/arm",
            self.rr.LineStrips3D([positions], colors=[0, 170, 255], radii=0.004),
        )
        self.rr.log(
            "kinematics/frames",
            self.rr.Points3D(
                positions,
                labels=self.frame_names,
                show_labels=True,
                colors=[255, 255, 255],
                radii=0.01,
            ),
        )
        self.rr.log(
            "kinematics/end_effector",
            self.rr.Points3D([target_xyz], labels=[self.kinematics.target_frame_name], colors=[255, 80, 80], radii=0.018),
        )
        self.rr.log("kinematics/status", self.rr.TextLog(label))
        self.step += 1


def _parse_floats(values: Sequence[str], expected: int | None = None) -> np.ndarray:
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}.")
    return np.array([float(value) for value in values], dtype=float)


def _format_vector(values: Sequence[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _unit_scale(units: str) -> float:
    if units == "m":
        return 1.0
    if units == "cm":
        return 0.01
    if units == "mm":
        return 0.001
    raise ValueError(f"Unsupported units: {units}")


def _print_pose(pose: np.ndarray, units: str) -> None:
    scale = 1.0 / _unit_scale(units)
    xyz = pose[:3, 3] * scale
    print(f"End-effector xyz ({units}): {_format_vector(xyz, precision=2 if units != 'm' else 4)}")
    print("End-effector transform:")
    print(np.array2string(pose, precision=4, suppress_small=True))


def _read_robot_joints(robot: "Robot") -> np.ndarray:
    observation = robot.get_observation()
    return np.array([observation[f"{motor}.pos"] for motor in robot.bus.motors], dtype=float)


def _joint_action(robot: "Robot", joints_deg: np.ndarray) -> dict[str, float]:
    return {f"{motor}.pos": float(joints_deg[i]) for i, motor in enumerate(robot.bus.motors)}


def _make_follower_robot(args: argparse.Namespace) -> "Robot":
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    if args.port is None:
        raise ValueError("--port is required when using --connect.")

    config_kwargs = {
        "port": args.port,
        "id": args.id,
        "calibration_dir": args.calibration_dir,
        "max_relative_target": args.max_relative_target,
        "use_degrees": True,
    }
    if args.robot_type == "so100_follower":
        robot_config = SO100FollowerConfig(**config_kwargs)
    elif args.robot_type == "so101_follower":
        robot_config = SO101FollowerConfig(**config_kwargs)
    else:
        raise ValueError(f"Unsupported robot type for kinematics_lab: {args.robot_type}")

    return make_robot_from_config(robot_config)


def _print_help(units: str) -> None:
    print(
        f"""
Commands:
  read
      Read the real robot joints and print FK. Requires --connect.
  joints q1 q2 q3 q4 q5 [gripper]
      Set the current joint guess in degrees and print FK.
  fk
      Print FK for the current joint guess.
  ik x y z
      Solve IK for an absolute end-effector position in {units}; keeps current orientation.
  move x y z [gripper]
      Solve IK, then send the joint target to the robot. Requires --connect.
  send
      Send the current joint guess to the robot. Requires --connect.
  help
      Show this help.
  quit
      Exit.
"""
    )


def run_interactive_lab(args: argparse.Namespace) -> None:
    scale = _unit_scale(args.units)
    try:
        kinematics = RobotKinematics(
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            joint_names=args.joint_names,
        )
    except ImportError as e:
        original_error = e.__cause__ or e
        raise SystemExit(
            "Missing optional kinematics dependency `placo`.\n"
            f"Python executable: {sys.executable}\n"
            f"Original import error: {original_error}\n"
            "Install it from the LeRobot repository root with:\n"
            '  python -m pip install -e ".[kinematics]"\n'
            "On Windows, if pip fails while building placo, try:\n"
            "  conda install -c conda-forge placo\n"
            "  python -m pip install -e .\n"
            "To verify the active environment, run:\n"
            '  python -c "import sys, placo; print(sys.executable); print(placo.__file__)"\n'
            "Then run kinematics_lab again."
        ) from e
    except ValueError as e:
        if "Mesh" in str(e) and "could not be found" in str(e):
            raise SystemExit(
                f"Could not load URDF mesh asset: {e}\n"
                "The SO101 URDF references STL files in a sibling `assets/` directory.\n"
                "Download the whole SO-ARM100 `Simulation/SO101` folder, not only the `.urdf` file, then run:\n"
                "  python -m lerobot.scripts.kinematics_lab "
                "--urdf-path external\\SO-ARM100\\Simulation\\SO101\\so101_new_calib.urdf --units mm"
            ) from e
        raise

    robot = None
    if args.connect:
        robot = _make_follower_robot(args)
        robot.connect(calibrate=not args.no_calibrate)
        current_joints = _read_robot_joints(robot)
        print("Connected to robot. Current joints (deg):", _format_vector(current_joints))
    else:
        current_joints = np.zeros(max(len(kinematics.joint_names), ARM_JOINT_COUNT), dtype=float)
        print("Dry-run mode: computing FK/IK only. Pass --connect to command a real arm.")

    print(f"Using target frame: {args.target_frame_name}")
    print(f"Kinematic joints: {kinematics.joint_names}")

    visualizer = None
    if args.visualize:
        try:
            visualizer = RerunKinematicsVisualizer(kinematics, args.urdf_path, args.target_frame_name)
            visualizer.update(current_joints, "initial")
        except ImportError as e:
            raise SystemExit(str(e)) from e

    _print_help(args.units)

    try:
        while True:
            raw = input("kinematics> ").strip()
            if not raw:
                continue

            parts = raw.split()
            command = parts[0].lower()
            values = parts[1:]

            try:
                if command in {"quit", "exit", "q"}:
                    break

                if command == "help":
                    _print_help(args.units)
                    continue

                if command == "read":
                    if robot is None:
                        print("read requires --connect.")
                        continue
                    current_joints = _read_robot_joints(robot)
                    print("Current joints (deg):", _format_vector(current_joints))
                    _print_pose(kinematics.forward_kinematics(current_joints), args.units)
                    if visualizer is not None:
                        visualizer.update(current_joints, "read")
                    continue

                if command == "joints":
                    joints = _parse_floats(values)
                    if len(joints) < len(kinematics.joint_names):
                        raise ValueError(
                            f"Need at least {len(kinematics.joint_names)} joint values for this URDF."
                        )
                    current_joints = joints
                    print("Current joint guess (deg):", _format_vector(current_joints))
                    _print_pose(kinematics.forward_kinematics(current_joints), args.units)
                    if visualizer is not None:
                        visualizer.update(current_joints, "joints")
                    continue

                if command == "fk":
                    print("Current joint guess (deg):", _format_vector(current_joints))
                    _print_pose(kinematics.forward_kinematics(current_joints), args.units)
                    if visualizer is not None:
                        visualizer.update(current_joints, "fk")
                    continue

                if command in {"ik", "move"}:
                    xyz = _parse_floats(values[:3], expected=3) * scale
                    current_pose = kinematics.forward_kinematics(current_joints)
                    target_pose = current_pose.copy()
                    target_pose[:3, 3] = xyz
                    target_joints = kinematics.inverse_kinematics(current_joints, target_pose)

                    if command == "move" and len(values) >= 4 and len(target_joints) > ARM_JOINT_COUNT:
                        target_joints[ARM_JOINT_COUNT] = float(values[3])

                    current_joints = target_joints
                    print("IK solution joints (deg):", _format_vector(current_joints))
                    _print_pose(kinematics.forward_kinematics(current_joints), args.units)
                    if visualizer is not None:
                        visualizer.update(current_joints, command)

                    if command == "move":
                        if robot is None:
                            print("move computed IK but did not command hardware because --connect was not set.")
                            continue
                        sent = robot.send_action(_joint_action(robot, current_joints))
                        print("Sent action:", sent)
                    continue

                if command == "send":
                    if robot is None:
                        print("send requires --connect.")
                        continue
                    sent = robot.send_action(_joint_action(robot, current_joints))
                    print("Sent action:", sent)
                    continue

                print(f"Unknown command: {command}. Type `help` for commands.")
            except ValueError as e:
                print(f"Invalid input: {e}")

    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf-path", type=Path, required=True, help="Path to the robot URDF file.")
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--units", choices=["m", "cm", "mm"], default="m")
    parser.add_argument("--visualize", action="store_true", help="Open a Rerun 3D viewer for FK/IK dry runs.")

    parser.add_argument("--connect", action="store_true", help="Connect to a real follower arm.")
    parser.add_argument(
        "--robot-type",
        choices=["so100_follower", "so101_follower"],
        default="so100_follower",
        help="Follower robot implementation to control when --connect is set.",
    )
    parser.add_argument("--port", help="Motor bus serial port, for example COM5 or /dev/ttyACM0.")
    parser.add_argument("--id", default=None, help="Robot id used for calibration files.")
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration on connect.")

    run_interactive_lab(parser.parse_args())


if __name__ == "__main__":
    main()
