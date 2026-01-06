#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
OpenArms End-Effector Replay Example with Visualization

Replays a dataset recorded with absolute joint positions by:
1. Converting joint positions to EE poses using FK
2. Converting EE poses back to joint positions using IK
3. Sending joint commands to the robot OR visualizing in simulation

Supports three modes:
- real: Send commands to physical robot
- sim: Visualize in simulation only (no robot required)
- both: Real robot + visualization

Example usage:
    python examples/openarms/replay_ee.py --mode sim
    python examples/openarms/replay_ee.py --mode real
    python examples/openarms/replay_ee.py --mode both --visualizer meshcat
"""

import argparse
import time
from os.path import dirname, expanduser

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.openarms.robot_kinematic_processor import (
    BimanualEEBoundsAndSafety,
    BimanualForwardKinematicsJointsToEE,
    BimanualInverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep


# Default configuration
DEFAULT_EPISODE_IDX = 0
DEFAULT_DATASET = "lerobot-data-collection/rac_blackf0"
DEFAULT_URDF = "src/lerobot/robots/openarms/urdf/openarm_bimanual_pybullet.urdf"
DEFAULT_LEFT_EE_FRAME = "openarm_left_hand_tcp"
DEFAULT_RIGHT_EE_FRAME = "openarm_right_hand_tcp"

# Motor names as used in the dataset actions (e.g., left_joint_1.pos)
MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]

# URDF joint names (no underscore between "joint" and number)
LEFT_URDF_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_URDF_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]


class MeshcatVisualizer:
    """Lightweight URDF visualizer using pinocchio + meshcat."""
    
    def __init__(self, urdf_path: str):
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer as PinMeshcat
        
        urdf_dir = dirname(urdf_path)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, urdf_dir, pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
        
        self.viz = PinMeshcat(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        
        # Build joint name mapping: dataset name -> pinocchio joint index
        # Dataset uses: left_joint_1, right_joint_2, etc.
        # URDF uses: openarm_left_joint1, openarm_right_joint2, etc.
        self.joint_map = {}
        for jid in range(1, self.model.njoints):
            urdf_name = self.model.names[jid]  # e.g., "openarm_left_joint1"
            # Extract side and number
            if "left_joint" in urdf_name:
                num = urdf_name.split("joint")[-1]  # "1"
                dataset_name = f"left_joint_{num}"
                self.joint_map[dataset_name] = jid
            elif "right_joint" in urdf_name:
                num = urdf_name.split("joint")[-1]
                dataset_name = f"right_joint_{num}"
                self.joint_map[dataset_name] = jid
        
        print(f"  Meshcat viewer opened (mapped {len(self.joint_map)} joints)")
        print(f"  Joint map: {list(self.joint_map.keys())[:4]}...")
        print("  Waiting for meshcat to load...")
        time.sleep(3)  # Give meshcat time to load meshes
        self._first_update = True
    
    def update(self, joint_positions: dict[str, float]):
        """Update visualization with new joint positions."""
        if self._first_update:
            pos_keys = [k for k in joint_positions.keys() if k.endswith(".pos")]
            print(f"  First update keys: {pos_keys[:4]}...")
            # Print sample values
            for k in pos_keys[:2]:
                print(f"    {k} = {joint_positions[k]:.2f}")
        
        # Build configuration vector (base pose + joints)
        # Free flyer base: [x, y, z, qx, qy, qz, qw]
        q = np.zeros(self.model.nq)
        q[3:7] = [0, 0, 0, 1]  # Identity quaternion
        
        matched = 0
        # Map joint positions using pre-built mapping
        for name, pos in joint_positions.items():
            if not name.endswith(".pos"):
                continue
            joint_name = name.removesuffix(".pos")  # e.g., "left_joint_1"
            
            jid = self.joint_map.get(joint_name)
            if jid is not None:
                idx = self.model.idx_qs[jid]
                if idx < len(q):
                    q[idx] = np.deg2rad(pos)
                    matched += 1
        
        if self._first_update:
            print(f"  Matched {matched} joints, q[7:14] = {q[7:14]}")
            self._first_update = False
        
        self.viz.display(q)


class RerunVisualizer:
    """Rerun-based visualizer for plots and EE trajectories."""
    
    def __init__(self, urdf_path: str = None, session_name: str = "openarms_replay"):
        import rerun as rr
        self.rr = rr
        rr.init(session_name)
        rr.spawn(memory_limit="10%")
        print("  Rerun viewer spawned (plots only, use --visualizer meshcat for 3D robot)")
    
    def update(self, joint_positions: dict[str, float], ee_poses: dict[str, float], frame_idx: int):
        """Log joint positions and EE poses."""
        self.rr.set_time("frame", sequence=frame_idx)
        
        # Log EE positions as colored spheres
        for prefix, color in [("left", [255, 100, 100]), ("right", [100, 100, 255])]:
            x, y, z = ee_poses.get(f"{prefix}_ee.x"), ee_poses.get(f"{prefix}_ee.y"), ee_poses.get(f"{prefix}_ee.z")
            if None not in (x, y, z):
                self.rr.log(f"ee/{prefix}", self.rr.Points3D([[x, y, z]], colors=[color], radii=[0.02]))
        
        # Log joint positions as time series
        for name, pos in joint_positions.items():
            if name.endswith(".pos"):
                self.rr.log(f"joints/{name}", self.rr.Scalars(pos))
        
        # Log EE poses as time series
        for name, val in ee_poses.items():
            self.rr.log(f"ee_plots/{name}", self.rr.Scalars(val))


def parse_args():
    parser = argparse.ArgumentParser(description="OpenArms EE Replay with Visualization")
    parser.add_argument("--mode", choices=["real", "sim", "both"], default="sim",
                        help="Execution mode: real (robot), sim (visualization), both")
    parser.add_argument("--visualizer", choices=["meshcat", "rerun", "none"], default="meshcat",
                        help="Visualization backend (meshcat shows 3D robot, rerun shows plots)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Dataset repo ID")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE_IDX,
                        help="Episode index to replay")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF,
                        help="Path to URDF file")
    parser.add_argument("--left-ee-frame", type=str, default=DEFAULT_LEFT_EE_FRAME,
                        help="Left arm end-effector frame name in URDF")
    parser.add_argument("--right-ee-frame", type=str, default=DEFAULT_RIGHT_EE_FRAME,
                        help="Right arm end-effector frame name in URDF")
    parser.add_argument("--port-left", type=str, default="can2",
                        help="CAN port for left arm")
    parser.add_argument("--port-right", type=str, default="can3",
                        help="CAN port for right arm")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    return parser.parse_args()


def main():
    args = parse_args()
    use_robot = args.mode in ["real", "both"]
    use_viz = args.mode in ["sim", "both"] and args.visualizer != "none"
    
    print("=" * 70)
    print("OpenArms EE Replay (FK -> IK Pipeline)")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Visualizer: {args.visualizer}")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    print(f"Speed: {args.speed}x")
    print("=" * 70)

    robot = None
    viz = None
    
    # Resolve URDF path (handle relative and ~ paths)
    from pathlib import Path
    urdf_path = args.urdf
    if urdf_path.startswith("~"):
        urdf_path = expanduser(urdf_path)
    elif not Path(urdf_path).is_absolute():
        # Relative to workspace root
        urdf_path = str(Path(__file__).parent.parent.parent / urdf_path)
    
    # Initialize robot if needed
    if use_robot:
        from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
        from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
        
        print("\n[1/5] Initializing robot...")
        robot_config = OpenArmsFollowerConfig(
            port_left=args.port_left,
            port_right=args.port_right,
            can_interface="socketcan",
            id="openarms_follower",
            disable_torque_on_disconnect=True,
            max_relative_target=10.0,
        )
        robot = OpenArmsFollower(robot_config)
    else:
        print("\n[1/5] Skipping robot (sim mode)")

    # Initialize visualizer if needed
    if use_viz:
        print(f"\n[2/5] Initializing {args.visualizer} visualizer...")
        if args.visualizer == "meshcat":
            viz = MeshcatVisualizer(urdf_path)
        elif args.visualizer == "rerun":
            viz = RerunVisualizer(urdf_path)
    else:
        print("\n[2/5] Skipping visualization")

    # Initialize kinematics with URDF joint names
    print("\n[3/5] Initializing kinematics solvers...")
    
    left_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=args.left_ee_frame,
        joint_names=LEFT_URDF_JOINTS,
    )
    right_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=args.right_ee_frame,
        joint_names=RIGHT_URDF_JOINTS,
    )

    # Build pipelines - use motor names without gripper for the processor
    motor_names_no_gripper = [n for n in MOTOR_NAMES if n != "gripper"]
    
    joints_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            BimanualForwardKinematicsJointsToEE(
                left_kinematics=left_kinematics,
                right_kinematics=right_kinematics,
                motor_names=MOTOR_NAMES,
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            BimanualEEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            BimanualInverseKinematicsEEToJoints(
                left_kinematics=left_kinematics,
                right_kinematics=right_kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Load dataset
    print(f"\n[4/5] Loading dataset '{args.dataset}'...")
    dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == args.episode)
    
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {args.episode}")
    
    print(f"  Found {len(episode_frames)} frames at {dataset.fps} FPS")
    
    action_features = dataset.features.get(ACTION, {})
    action_names = action_features.get("names", [])
    actions = episode_frames.select_columns(ACTION)

    # Connect robot if needed
    if use_robot:
        print("\n[5/5] Connecting to robot...")
        robot.connect(calibrate=False)
        if not robot.is_connected:
            raise RuntimeError("Robot failed to connect!")
    else:
        print("\n[5/5] Skipping robot connection (sim mode)")

    print("\n" + "=" * 70)
    print(f"Ready to replay! Mode: {args.mode}")
    print("=" * 70)
    
    if use_robot:
        input("\nPress ENTER to start...")
    else:
        print("\nStarting visualization playback...")
        time.sleep(1)

    # Simulated observation for sim-only mode
    sim_obs = {f"{prefix}_{motor}.pos": 0.0 
               for prefix in ["left", "right"] 
               for motor in MOTOR_NAMES}

    try:
        for idx in range(len(episode_frames)):
            loop_start = time.perf_counter()

            # Get observation
            if use_robot:
                robot_obs = robot.get_observation()
            else:
                robot_obs = sim_obs.copy()

            # Build joint action from dataset
            action_array = actions[idx][ACTION]
            joint_action = {}
            for i, name in enumerate(action_names):
                if name.endswith(".pos"):
                    joint_action[name] = float(action_array[i])

            # Convert: joints -> EE (FK)
            ee_action = joints_to_ee(joint_action.copy())

            # Convert: EE -> joints (IK)
            final_joint_action = ee_to_joints((ee_action.copy(), robot_obs))

            # Update simulated observation for next iteration
            if not use_robot:
                sim_obs.update(final_joint_action)

            # Send to robot
            if use_robot:
                robot.send_action(final_joint_action)

            # Update visualization with ORIGINAL dataset trajectory
            if viz:
                if isinstance(viz, MeshcatVisualizer):
                    viz.update(joint_action)  # Use original, not FK->IK reconstructed
                elif isinstance(viz, RerunVisualizer):
                    viz.update(joint_action, ee_action, idx)

            # Maintain replay rate
            loop_duration = time.perf_counter() - loop_start
            dt_s = (1.0 / dataset.fps / args.speed) - loop_duration
            if dt_s > 0:
                precise_sleep(dt_s)

            if (idx + 1) % 100 == 0:
                progress = (idx + 1) / len(episode_frames) * 100
                print(f"Progress: {idx + 1}/{len(episode_frames)} ({progress:.1f}%)")

        print(f"\n✓ Replayed {len(episode_frames)} frames")

    except KeyboardInterrupt:
        print("\n\nReplay interrupted")
    finally:
        if use_robot and robot:
            print("\nDisconnecting robot...")
            robot.disconnect()
        print("✓ Done!")


if __name__ == "__main__":
    main()

