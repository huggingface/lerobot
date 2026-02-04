import math
import time
from pathlib import Path

import numpy as np
import rerun as rr

# Constants
NPZ_PATH = Path("my_scripts/action_chunks.npz")
OUTPUT_RRD = Path("my_scripts/action_chunks.rrd")


class PiperFK:
    def __init__(self):
        self.RADIAN = 180 / math.pi
        self.PI = math.pi
        # DH parameters from piper_fk.py (dh_is_offset=1) - Converted to Meters
        self._a = [x / 1000.0 for x in [0, 0, 285.03, -21.98, 0, 0]]
        self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
        # Offset override from code logic if dh_is_offset=1
        self._theta_offset = [0, -self.PI * 172.22 / 180, -102.78 / 180 * self.PI, 0, 0, 0]
        self._d = [x / 1000.0 for x in [123, 0, 0, 250.75, 0, 91]]

        # Load mesh paths
        self.mesh_dir = Path("my_scripts/piper_description/meshes")
        self.links = [
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "gripper_base",
            "link7",
            "link8",
        ]

    def _link_transform(self, alpha, a, theta, d):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array(
            [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]]
        )

    def get_transforms(self, joints, gripper_val=0):
        # joints: 6 angles
        # gripper_val: distance in meters
        transforms = {}
        T = np.eye(4)
        transforms["base_link"] = T.copy()

        # FK for 6 joints
        for i in range(6):
            theta = joints[i] + self._theta_offset[i]
            Ti = self._link_transform(self._alpha[i], self._a[i], theta, self._d[i])
            T = T @ Ti
            transforms[f"link{i + 1}"] = T.copy()

        # Gripper Base (Fixed to link6)
        T_gripper_base = T.copy()  # joint6_to_gripper_base origin is 0 0 0
        transforms["gripper_base"] = T_gripper_base

        # Gripper Fingers
        # Joint 7: origin 0 0 0.1358, rpy 1.5708 0 0. Prismatic z
        # Helper for RPY + XYZ fixed transform
        def get_fixed(xyz, rpy):
            cx, sx = math.cos(rpy[0]), math.sin(rpy[0])
            cy, sy = math.cos(rpy[1]), math.sin(rpy[1])
            cz, sz = math.cos(rpy[2]), math.sin(rpy[2])

            Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
            Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
            Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            Tr = np.eye(4)
            Tr[:3, 3] = xyz
            return Tr @ Rz @ Ry @ Rx

        # Link 7 (Left Finger)
        j7_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, 0])

        # User specified gripper_val is the total distance between fingers.
        half_dist = gripper_val / 2.0
        j7_pos = max(0, min(half_dist, 0.035))

        T_prismatic = np.eye(4)
        T_prismatic[2, 3] = j7_pos
        transforms["link7"] = T_gripper_base @ j7_origin @ T_prismatic

        # Link 8 (Right Finger)
        j8_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, -3.1416])
        j8_pos = -1.0 * j7_pos  # Symmetric

        T_prismatic8 = np.eye(4)
        T_prismatic8[2, 3] = j8_pos
        transforms["link8"] = T_gripper_base @ j8_origin @ T_prismatic8

        return transforms

    def log_initial_meshes(self, prefix="simulation"):
        # Log parsed meshes as Asset3D once
        for link in self.links:
            mesh_path = self.mesh_dir / f"{link}.STL"
            if mesh_path.exists():
                rr.log(f"{prefix}/{link}", rr.Asset3D(path=mesh_path), static=True)


def visualize_trajectory(fk, action_data, prefix, offset_y=0.0):
    """
    Logs a single step of the trajectory.
    """
    if len(action_data) >= 14:
        # Left Arm
        left_joints = action_data[0:6]
        left_grip = action_data[6]
        left_T = fk.get_transforms(left_joints, left_grip)

        T_left_base = np.eye(4)
        T_left_base[1, 3] = 0.3 + offset_y

        for link_name, T_local in left_T.items():
            T_global = T_left_base @ T_local
            rr.log(
                f"{prefix}/left_arm/{link_name}",
                rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
            )

        # Right Arm
        right_joints = action_data[7:13]
        right_grip = action_data[13]
        right_T = fk.get_transforms(right_joints, right_grip)

        T_right_base = np.eye(4)
        T_right_base[1, 3] = -0.3 + offset_y

        for link_name, T_local in right_T.items():
            T_global = T_right_base @ T_local
            rr.log(
                f"{prefix}/right_arm/{link_name}",
                rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
            )


def main():
    if not NPZ_PATH.exists():
        print(f"Error: {NPZ_PATH} not found.")
        return

    print(f"Loading {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    action_chunk = data["action"]
    smoothed_action_chunk = data["smoothed_action"]

    print(f"Action shape: {action_chunk.shape}")
    print(f"Smoothed action shape: {smoothed_action_chunk.shape}")

    # Initialize Rerun
    print("Initializing Rerun...")
    rr.init("NPZ Visualization", spawn=False)

    # Initialize Piper simulation
    print("Initializing Piper simulation...")
    fk = PiperFK()
    fk.log_initial_meshes("simulation/original/left_arm")
    fk.log_initial_meshes("simulation/original/right_arm")
    fk.log_initial_meshes("simulation/smoothed/left_arm")
    fk.log_initial_meshes("simulation/smoothed/right_arm")

    chunk_size = len(action_chunk)
    print(f"Starting simulation playback ({chunk_size} steps, 30Hz)...")

    # We'll offset the smoothed robot slightly in Y or Z to see both?
    # Or maybe just put them side by side?
    # Let's offset smoothed by +1.0 meter in X (forward) so they are in front of each other.

    rr.set_time_sequence("step", 0)

    for t in range(chunk_size):
        start_time = time.time()
        rr.set_time_sequence("step", t)

        # Log Original (at 0,0,0)
        visualize_trajectory(fk, action_chunk[t], "simulation/original", offset_y=0.0)

        # Log Smoothed (offset by 1.0m in Y to be side-by-side)
        visualize_trajectory(fk, smoothed_action_chunk[t], "simulation/smoothed", offset_y=1.0)

        # Sleep to match 30Hz simulation rate
        # Although for recording it doesn't matter much effectively,
        # but prevents loop from running instantly.
        # elapsed = time.time() - start_time
        # sleep_time = max(0, (1.0/30.0) - elapsed)
        # time.sleep(sleep_time)

    print("Playback complete.")

    # Save Rerun file
    print(f"Saving Rerun recording to {OUTPUT_RRD}...")
    rr.save(OUTPUT_RRD)
    print(
        "You can view this file at https://app.rerun.io/ or by running `rerun my_scripts/action_chunks.rrd` locally."
    )


if __name__ == "__main__":
    main()
