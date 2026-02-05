"""
Usage:
python tools/viz/simulate_episode.py \
    --pretrained_path /path/to/model \
    --repo_id lerobot_pick_and_place \
    --episode_index 0
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Defaults
# DEFAULT_PRETRAINED_MODEL_PATH = Path(
#     "/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_pick_and_place_50/checkpoints/last/pretrained_model"
# )
# DEFAULT_DATASET_ROOT = Path("/home/droplab/.cache/huggingface/lerobot/local/lerobot_pick_and_place")
# DEFAULT_DATASET_ID = "lerobot_pick_and_place"

DEFAULT_PRETRAINED_MODEL_PATH = Path(
    "/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_fold_towel_50chunksz/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/droplab/.cache/huggingface/lerobot/local/lerobot_fold_towel")
DEFAULT_DATASET_ID = "lerobot_fold_towel"

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
        self.mesh_dir = Path("assets/piper_description/meshes")
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
        # Colors (RGB)
        self.light_grey = [128, 128, 128]
        self.link_colors = {
            "base_link": self.light_grey,
            "link1": self.light_grey,
            "link2": self.light_grey,
            "link3": self.light_grey,
            "link4": self.light_grey,
            "link5": self.light_grey,
            "link6": self.light_grey,
            "gripper_base": self.light_grey,
            "link7": self.light_grey,
            "link8": self.light_grey,
        }

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
        transform = np.eye(4)
        transforms["base_link"] = transform.copy()

        # FK for 6 joints
        for i in range(6):
            theta = joints[i] + self._theta_offset[i]
            t_i = self._link_transform(self._alpha[i], self._a[i], theta, self._d[i])
            transform = transform @ t_i
            transforms[f"link{i + 1}"] = transform.copy()

        # Gripper Base (Fixed to link6)
        t_gripper_base = transform.copy()  # joint6_to_gripper_base origin is 0 0 0
        transforms["gripper_base"] = t_gripper_base

        # Gripper Fingers
        # Joint 7: origin 0 0 0.1358, rpy 1.5708 0 0. Prismatic z
        # Helper for RPY + XYZ fixed transform
        def get_fixed(xyz, rpy):
            cx, sx = math.cos(rpy[0]), math.sin(rpy[0])
            cy, sy = math.cos(rpy[1]), math.sin(rpy[1])
            cz, sz = math.cos(rpy[2]), math.sin(rpy[2])

            r_x = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
            r_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
            r_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            t_r = np.eye(4)
            t_r[:3, 3] = xyz
            return t_r @ r_z @ r_y @ r_x

        # Link 7 (Left Finger)
        j7_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, 0])

        # User specified gripper_val is the total distance between fingers.
        half_dist = gripper_val / 2.0
        j7_pos = max(0, min(half_dist, 0.035))

        t_prismatic = np.eye(4)
        t_prismatic[2, 3] = j7_pos
        transforms["link7"] = t_gripper_base @ j7_origin @ t_prismatic

        # Link 8 (Right Finger)
        j8_origin = get_fixed([0, 0, 0.1358], [1.5708, 0, -3.1416])
        j8_pos = j7_pos  # Symmetric movement in rotated frame

        t_prismatic8 = np.eye(4)
        t_prismatic8[2, 3] = j8_pos
        transforms["link8"] = t_gripper_base @ j8_origin @ t_prismatic8

        return transforms

    def log_initial_meshes(self, prefix="simulation"):
        # Log parsed meshes as Asset3D once
        for link in self.links:
            mesh_path = self.mesh_dir / f"{link}.STL"
            if mesh_path.exists():
                color = self.link_colors.get(link, [200, 200, 200])
                rr.log(f"{prefix}/{link}", rr.Asset3D(path=mesh_path, albedo_factor=color), static=True)


def visualize_trajectory(fk, action_data, prefix, offset_y=0.0):
    """
    Logs a single step of the trajectory.
    """
    if len(action_data) >= 14:
        # Left Arm
        left_joints = action_data[0:6]
        left_grip = action_data[6]
        left_transforms = fk.get_transforms(left_joints, left_grip)

        t_left_base = np.eye(4)
        t_left_base[1, 3] = 0.3 + offset_y

        for link_name, t_local in left_transforms.items():
            t_global = t_left_base @ t_local
            rr.log(
                f"{prefix}/left_arm/{link_name}",
                rr.Transform3D(translation=t_global[:3, 3], mat3x3=t_global[:3, :3]),
            )

        # Right Arm
        right_joints = action_data[7:13]
        right_grip = action_data[13]
        right_transforms = fk.get_transforms(right_joints, right_grip)

        t_right_base = np.eye(4)
        t_right_base[1, 3] = -0.3 + offset_y

        for link_name, t_local in right_transforms.items():
            t_global = t_right_base @ t_local
            rr.log(
                f"{prefix}/right_arm/{link_name}",
                rr.Transform3D(translation=t_global[:3, 3], mat3x3=t_global[:3, :3]),
            )


def main():
    parser = argparse.ArgumentParser(description="Simulate an entire episode using ACT policy chunks.")
    parser.add_argument(
        "--pretrained_path", type=Path, default=DEFAULT_PRETRAINED_MODEL_PATH, help="Path to pretrained model"
    )
    parser.add_argument(
        "--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root directory of the dataset"
    )
    parser.add_argument("--repo_id", type=str, default=DEFAULT_DATASET_ID, help="Dataset repository ID")
    parser.add_argument(
        "--episode_index", type=int, default=None, help="Index of the episode to inspect (default: random)"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.pretrained_path}...")
    try:
        policy = ACTPolicy.from_pretrained(args.pretrained_path)
        policy.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading dataset {args.repo_id} from {args.dataset_root}...")
    try:
        dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Pick episode
    num_episodes = len(dataset.meta.episodes)
    if args.episode_index is None:
        episode_idx = np.random.randint(num_episodes)
        print(f"Selected random episode index: {episode_idx}")
    else:
        episode_idx = args.episode_index
        if episode_idx < 0 or episode_idx >= num_episodes:
            print(f"Error: Episode index {episode_idx} out of range (0-{num_episodes - 1})")
            return
        print(f"Selected episode index: {episode_idx}")

    # Get episode info
    episode_data = dataset.meta.episodes[episode_idx]
    if "index" in episode_data:
        start_index = episode_data["index"]
    elif "dataset_from_index" in episode_data:
        val = episode_data["dataset_from_index"]
        # Handle cases where value might be tensor or array
        start_index = val.item() if hasattr(val, "item") else (val[0] if isinstance(val, (list, tuple, np.ndarray)) else val)
    else:
        raise KeyError("Could not find start index in episode data")

    length = episode_data["length"]
    end_index = start_index + length
    print(f"Episode {episode_idx} frames: {start_index} to {end_index-1} (Length: {length})")

    # Create processors
    print("Creating processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    # Initialize Rerun
    print("Initializing Rerun...")
    rr.init("Episode_Simulation", spawn=True)

    # Initialize Piper simulation
    print("Initializing Piper simulation...")
    fk = PiperFK()
    fk.log_initial_meshes("simulation/prediction/left_arm")
    fk.log_initial_meshes("simulation/prediction/left_arm")
    fk.log_initial_meshes("simulation/prediction/right_arm")

    # Output Directory
    output_dir = Path("outputs/visualize_episode")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(policy.parameters()).device
    current_frame_idx = start_index
    global_step = 0

    print("Starting simulation loop...")

    with torch.inference_mode():
        while current_frame_idx < end_index:
            print(f"Processing frame {current_frame_idx} (Step {global_step})...")
            
            # Load item
            item = dataset[current_frame_idx]

            # Log images at the start of the chunk
            image_keys = [k for k in item if "image" in k]
            for key in sorted(image_keys):
                img_tensor = item[key]
                # Convert (C, H, W) -> (H, W, C) numpy
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                        img_np = np.clip(img_np, 0, 1)
                    
                    clean_key = key.replace("observation.images.", "")
                    # Log to current global step
                    rr.set_time_sequence("step", global_step)
                    rr.log(f"cameras/{clean_key}", rr.Image(img_np))

            # Prepare batch
            batch = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(device)

            # Preprocess
            batch = preprocessor(batch)

            # Predict
            action_chunk = policy.predict_action_chunk(batch)
            
            # Postprocess
            action_chunk_unnormalized = postprocessor(action_chunk)
            action_data = action_chunk_unnormalized[0].cpu().numpy() # (chunk_size, dim)
            
            chunk_size = action_data.shape[0]

            # PLOTTING
            # Identify image keys
            image_keys = [k for k in item if "image" in k]
            num_images = len(image_keys)
            timestamp = item.get("timestamp", 0.0)

            if num_images > 0:
                # Create figure with 2 rows: images on top, trajectories below
                fig = plt.figure(figsize=(15, 12))
                gs = fig.add_gridspec(2, num_images, height_ratios=[1, 3])

                # Plot images
                for idx, key in enumerate(sorted(image_keys)):
                    ax_img = fig.add_subplot(gs[0, idx])
                    img_tensor = item[key]
                    # Convert (C, H, W) -> (H, W, C) numpy
                    if isinstance(img_tensor, torch.Tensor):
                        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                        # Normalize if needed (assuming float 0-1 or int 0-255)
                        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                            img_np = np.clip(img_np, 0, 1)
                        elif img_np.dtype == np.uint8:
                            pass  # Imshow handles it

                        ax_img.imshow(img_np)
                        ax_img.set_title(f"{key.replace('observation.images.', '')}\nFrame: {current_frame_idx}, t={timestamp:.2f}s")
                        ax_img.axis("off")

                # Plot Trajectories
                ax_plot = fig.add_subplot(gs[1, :])
            else:
                # Fallback to single plot if no images
                fig, ax_plot = plt.subplots(figsize=(15, 10))

            steps_plot = np.arange(chunk_size)
            _, action_dim = action_data.shape

            for i in range(action_dim):
                # Plot original
                ax_plot.plot(steps_plot, action_data[:, i], label=f"Dim {i}", color=f"C{i}")

            ax_plot.set_title(f"Action Chunk Trajectories (Episode {episode_idx}, Frame {current_frame_idx})")
            ax_plot.set_xlabel("Chunk Step")
            ax_plot.set_ylabel("Action Value")
            # Move legend outside
            ax_plot.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax_plot.grid(True)
            plt.tight_layout()

            output_plot_path = output_dir / f"episode_{episode_idx}_frame_{current_frame_idx}_plot.png"
            print(f"Saving plot to {output_plot_path}...")
            plt.savefig(output_plot_path)
            plt.close(fig) # Important to close to save memory!

            # Log trajectory for this chunk
            for t in range(chunk_size):
                rr.set_time_sequence("step", global_step)
                visualize_trajectory(fk, action_data[t], "simulation/prediction")
                global_step += 1

            # Advance by chunk size
            current_frame_idx += chunk_size
            
            # Optional: Visualize dataset observation as Reference? 
            # The prompt says "at the end of every chunk it should repeat inference... and paste everything together"
            # It doesn't strictly ask for ground truth comparison, but it's good practice. 
            # I will stick to the prompt strictly: "simulate episode".

    print(f"Simulation detailed. Reached frame {current_frame_idx}. Global steps logged: {global_step}")
    print("Rerun should be open.")

if __name__ == "__main__":
    main()
