"""
Script lerobot to h5.
# --repo-id     Your unique repo ID on Hugging Face Hub
# --output_dir  Save path to h5 file

python unitree_lerobot/utils/convert_lerobot_to_h5.py.py \
    --repo-id your_name/g1_grabcube_double_hand \
    --output_dir "$HOME/datasets/g1_grabcube_double_hand" 
"""

import os
import cv2
import h5py
import tyro
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotDataProcessor:
    def __init__(self, repo_id: str, root: str = None, image_dtype: str = "to_unit8") -> None:
        self.image_dtype = image_dtype
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)

    def process_episode(self, episode_index: int) -> dict:
        """Process a single episode to extract camera images, state, and action."""
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)

        for step_idx in tqdm(
            range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True
        ):
            step = self.dataset[step_idx]

            image_dict = {
                key.split(".")[2]: cv2.cvtColor(
                    np.transpose((value.numpy() * 255).astype(np.uint8), (1, 2, 0)), cv2.COLOR_BGR2RGB
                )
                for key, value in step.items()
                if key.startswith("observation.image") and len(key.split(".")) >= 3
            }

            for key, value in image_dict.items():
                if self.image_dtype == "to_unit8":
                    cameras[key].append(value)
                elif self.image_dtype == "to_bytes":
                    success, encoded_img = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if not success:
                        raise ValueError(f"Image encoding failed for key: {key}")
                    cameras[key].append(np.void(encoded_img.tobytes()))

            cam_height, cam_width = next(iter(image_dict.values())).shape[:2]
            episode["state"].append(step["observation.state"])
            episode["action"].append(step["action"])

        episode["cameras"] = cameras
        episode["task"] = step["task"]
        episode["episode_length"] = to_idx - from_idx

        # Data configuration for later use
        episode["data_cfg"] = {
            "camera_names": list(image_dict.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": np.squeeze(step["observation.state"].numpy().shape),
            "action_dim": np.squeeze(step["action"].numpy().shape),
        }
        episode["episode_index"] = episode_index

        return episode


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        """Write episode data to HDF5 file."""

        episode_length = episode["episode_length"]
        episode_index = episode["episode_index"]
        state = episode["state"]
        action = episode["action"]
        qvel = np.zeros_like(episode["state"])
        cameras = episode["cameras"]
        task = episode["task"]
        data_cfg = episode["data_cfg"]

        # Prepare data dictionary
        data_dict = {
            "/observations/qpos": [state],
            "/observations/qvel": [qvel],
            "/action": [action],
            **{f"/observations/images/{k}": [v] for k, v in cameras.items()},
        }

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
            # Set attributes
            root.attrs["sim"] = False

            # Create datasets
            obs = root.create_group("observations")
            image = obs.create_group("images")

            # Write camera images
            for cam_name, images in cameras.items():
                data_dtype = images[0].dtype
                shape = (
                    (episode_length, data_cfg["cam_height"], data_cfg["cam_width"], 3)
                    if data_dtype == "uint8"
                    else (episode_length,)
                )
                chunks = (1, data_cfg["cam_height"], data_cfg["cam_width"], 3) if data_dtype == "uint8" else (1,)
                image.create_dataset(cam_name, shape=shape, dtype=data_dtype, chunks=chunks, compression="gzip")
                # root[f'/observations/images/{cam_name}'][...] = images

            # Write state and action data
            obs.create_dataset("qpos", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("qvel", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("action", (episode_length, data_cfg["action_dim"]), dtype="float32", compression="gzip")

            # Write metadata
            root.create_dataset("is_edited", (1,), dtype="uint8")
            substep_reasonings = root.create_dataset(
                "substep_reasonings", (episode_length,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip"
            )
            root.create_dataset("language_raw", data=task)
            substep_reasonings[:] = [task] * episode_length

            # Write additional data
            for name, array in data_dict.items():
                root[name][...] = array


def lerobot_to_h5(repo_id: str, output_dir: Path, root: str = None) -> None:
    """Main function to process and write LeRobot data to HDF5 format."""

    # Initialize data processor and H5 writer
    data_processor = LeRobotDataProcessor(
        repo_id, root, image_dtype="to_unit8"
    )  # image_dtype Options: "to_unit8", "to_bytes"
    h5_writer = H5Writer(output_dir)

    # Process each episode
    for episode_index in tqdm(
        range(data_processor.dataset.num_episodes), desc="Episodes", position=0, dynamic_ncols=True
    ):
        episode = data_processor.process_episode(episode_index)
        h5_writer.write_to_h5(episode)


if __name__ == "__main__":
    tyro.cli(lerobot_to_h5)
