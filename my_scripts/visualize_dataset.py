import argparse
import math
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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


def visualize_dataset(repo_id, root=None):
    print(f"Loading dataset: {repo_id}")
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Ensure episodes metadata
    if dataset.meta.episodes is None:
        try:
            from lerobot.datasets.utils import load_episodes

            dataset.meta.episodes = load_episodes(dataset.root)
        except:
            pass

    total_episodes = len(dataset.meta.episodes)
    print(f"Total episodes available: {total_episodes}")

    print("Initializing Rerun...")
    rr.init("LeRobot Dataset Visualizer", spawn=True)

    fk = PiperFK()
    print("Logging initial meshes...")
    fk.log_initial_meshes("simulation/left_arm")
    fk.log_initial_meshes("simulation/right_arm")

    global_step = 0
    stride = 7  # 7x speed

    for episode_idx in range(total_episodes):
        print(f"Streaming Episode {episode_idx}/{total_episodes}...", end="\r")

        ep_meta = dataset.meta.episodes[episode_idx]
        from_idx = int(
            ep_meta["dataset_from_index"]
            if not isinstance(ep_meta["dataset_from_index"], list)
            else ep_meta["dataset_from_index"][0]
        )
        to_idx = int(
            ep_meta["dataset_to_index"]
            if not isinstance(ep_meta["dataset_to_index"], list)
            else ep_meta["dataset_to_index"][0]
        )

        # Log Episode ID Overlay
        # Using a separate log call at the start of the episode ensures it's visible even if we skip the first step?
        # But we log it continuously or just once? TextDocument is stateful.
        # Log it at the current global_step so it appears at the right time.

        for i in range(from_idx, to_idx, stride):
            relative_step = i - from_idx

            # Set timelines - ONLY global_step to ensure continuous playback
            rr.set_time_sequence("global_step", global_step)
            global_step += 1

            # Update Overlay (Redundant to log every step but ensures it's always there)
            rr.log("overlay/episode_id", rr.TextDocument(f"# Episode {episode_idx}"), static=False)

            try:
                item = dataset[i]
            except Exception as e:
                print(f"Error loading frame {i}: {e}")
                continue

            # 1. Action vector (Simulation)
            # Find action key
            act = None
            if "action" in item:
                val = item["action"]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                act = np.array(val)

            if act is not None:
                rr.log("vectors/action_tensor", rr.Tensor(act))

                if len(act) >= 14:
                    # Left Arm
                    left_joints = act[0:6]
                    left_grip = act[6]
                    left_T = fk.get_transforms(left_joints, left_grip)

                    T_left_base = np.eye(4)
                    T_left_base[1, 3] = 0.3

                    for link_name, T_local in left_T.items():
                        T_global = T_left_base @ T_local
                        rr.log(
                            f"simulation/left_arm/{link_name}",
                            rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
                        )

                    # Right Arm
                    right_joints = act[7:13]
                    right_grip = act[13]
                    right_T = fk.get_transforms(right_joints, right_grip)

                    T_right_base = np.eye(4)
                    T_right_base[1, 3] = -0.3

                    for link_name, T_local in right_T.items():
                        T_global = T_right_base @ T_local
                        rr.log(
                            f"simulation/right_arm/{link_name}",
                            rr.Transform3D(translation=T_global[:3, 3], mat3x3=T_global[:3, :3]),
                        )

            # 2. Images
            image_keys = [k for k in item.keys() if "image" in k]
            for img_key in image_keys:
                img_data = item[img_key]
                clean_key = img_key.replace("observation.images.", "")

                # Convert to numpy/image
                if isinstance(img_data, dict) and "bytes" in img_data:
                    import io

                    img = Image.open(io.BytesIO(img_data["bytes"]))
                    rr.log(f"cameras/{clean_key}", rr.Image(img))
                else:
                    arr = img_data
                    if hasattr(arr, "numpy"):
                        arr = arr.numpy()
                    if arr.ndim == 3:
                        if arr.shape[0] <= 4:
                            arr = np.transpose(arr, (1, 2, 0))

                    rr.log(f"cameras/{clean_key}", rr.Image(arr))

            # 3. Observation Text
            step_data = {}
            for key, value in item.items():
                if "observation.state" in key:
                    val = value
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    elif hasattr(val, "numpy"):
                        val = val.numpy().tolist()
                    step_data[key] = val
            if step_data:
                rr.log("vectors/observation_text", rr.TextDocument(str(step_data)))

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")

    args = parser.parse_args()
    visualize_dataset(args.repo_id, args.root)
