import argparse
import json
import math
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image


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
        # gripper_val: 0 to 1 (mapped to 0 to 0.035m)
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
        # Joint 8: origin 0 0 0.1358, rpy 1.5708 0 -3.1416. Prismatic z (negative axis in URDF but limit -0.035..0)

        # Helper for RPY + XYZ fixed transform
        def get_fixed(xyz, rpy):
            # Simple RPY to matrix (roll fixed X, pitch fixed Y, yaw fixed Z? URDF is extrinsic usually sxyz)
            # URDF RPY is fixed axis (extrinsic) X-Y-Z
            cx, sx = math.cos(rpy[0]), math.sin(rpy[0])
            cy, sy = math.cos(rpy[1]), math.sin(rpy[1])
            cz, sz = math.cos(rpy[2]), math.sin(rpy[2])

            # Rx * Ry * Rz
            Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
            Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
            Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            R = Rz @ Ry @ Rx  # Standard convention might satisfy URDF?
            # Actually ROS URDF uses Fixed Fixed Fixed (sxyz) order?
            # Convention is rot_z * rot_y * rot_x (intrinsic) OR rot_x * rot_y * rot_z (extrinsic) ?
            # Standard: Roll (X), Pitch (Y), Yaw (Z). T = Trans * Rz * Ry * Rx.

            Tr = np.eye(4)
            Tr[:3, 3] = xyz
            return Tr @ Rz @ Ry @ Rx

        # Link 7 (Left Finger)
        j7_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, 0])
        # Prismatic along Z of child frame
        j7_val = max(0, min(gripper_val, 0.035))  # Map is Meter to Meter
        T_prismatic = np.eye(4)
        T_prismatic[2, 3] = j7_val
        transforms["link7"] = T_gripper_base @ j7_origin @ T_prismatic

        # Link 8 (Right Finger)
        j8_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, -3.1416])
        # Axis 0 0 -1. Limit -0.035 to 0.
        # If gripper_val is 0 (closed) -> 0. 1 (open) -> -0.035?
        # Usually grippers mimic each other. Open = finger moves out.
        # Link 7 moves +Z. Link 8 moves -Z (due to axis -1) or +Z ?
        # Link 8 axis is 0 0 -1. Limit is negative?
        # Let's assume symmetric opening.
        j8_val = -1 * j7_val
        T_prismatic8 = np.eye(4)
        T_prismatic8[2, 3] = j8_val  # Axis check?
        # Actually URDF axis -1 means positive displacement moves in -Z direction.
        # But joint limits are -0.035 to 0.
        # So it moves from -0.035 to 0? Or 0 to -0.035?
        # To match visual symmetry, probably moves away from center.
        transforms["link8"] = T_gripper_base @ j8_origin @ T_prismatic8

        return transforms

    def log_initial_meshes(self, prefix="simulation"):
        # Log parsed meshes as Asset3D once
        for link in self.links:
            mesh_path = self.mesh_dir / f"{link}.STL"
            if mesh_path.exists():
                # We log it to the entity path, but Asset3D content is static.
                # To animate, we log Transform3D updates to the same entity.
                rr.log(f"{prefix}/{link}", rr.Asset3D(path=mesh_path), static=True)


def view_inspection(episode_path):
    episode_dir = Path(episode_path)
    if not episode_dir.exists():
        print(f"Error: Directory not found: {episode_dir}")
        return

    files = list(episode_dir.glob("step_*"))
    steps = sorted(list({int(f.name.split("_")[1]) for f in files if "step_" in f.name}))

    if not steps:
        print("No steps found.")
        return

    print(f"Found {len(steps)} steps. Initializing Rerun...")
    rr.init("LeRobot Inspection", spawn=True)

    fk = PiperFK()

    print("Logging initial meshes...")
    fk.log_initial_meshes("simulation/left_arm")
    fk.log_initial_meshes("simulation/right_arm")

    print("Logging data to Rerun...")

    for step_num in steps:
        step_prefix = f"step_{step_num:04d}"
        rr.set_time_sequence("step", step_num)

        vec_file = episode_dir / f"{step_prefix}_vectors.json"
        if vec_file.exists():
            with open(vec_file) as f:
                vectors = json.load(f)

            if "action" in vectors:
                act = np.array(vectors["action"])
                rr.log("vectors/action_tensor", rr.Tensor(act))

                # Update Transforms
                if len(act) >= 14:  # 2 arms + grippers
                    # Left Arm (0-6) -> 0-5 joints, 6 gripper
                    left_joints = act[0:6]
                    left_grip = act[6]
                    left_T = fk.get_transforms(left_joints, left_grip)

                    # Log Left Arm Transforms
                    # Base for left arm
                    T_left_base = np.eye(4)
                    T_left_base[1, 3] = 0.3  # Offset 30cm +Y

                    for link_name, T_local in left_T.items():
                        T_global = T_left_base @ T_local
                        rr.log(
                            f"simulation/left_arm/{link_name}",
                            rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
                        )

                    # Right Arm (7-13) -> 7-12 joints, 13 gripper
                    right_joints = act[7:13]
                    right_grip = act[13]
                    right_T = fk.get_transforms(right_joints, right_grip)

                    # Base for right arm
                    T_right_base = np.eye(4)
                    T_right_base[1, 3] = -0.3  # Offset 30cm -Y

                    for link_name, T_local in right_T.items():
                        T_global = T_right_base @ T_local
                        rr.log(
                            f"simulation/right_arm/{link_name}",
                            rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
                        )

            if "observation" in vectors:
                obs = np.array(vectors["observation"])
                rr.log("vectors/observation_text", rr.TextDocument(str(obs)))

        # Images...
        img_files = sorted(list(episode_dir.glob(f"{step_prefix}_cam_*.png")))
        for img_f in img_files:
            cam_name = img_f.stem.replace(f"{step_prefix}_cam_", "")
            try:
                img = Image.open(img_f)
                rr.log(f"cameras/{cam_name}", rr.Image(img))
            except:
                pass

    print("Done logging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View inspection data using Rerun.")
    parser.add_argument("episode_path", type=str, help="Path to the episode folder")
    args = parser.parse_args()
    view_inspection(args.episode_path)
