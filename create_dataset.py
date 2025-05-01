import json

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def extract_metadata(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        robot = env_meta["env_kwargs"]["robots"][0]  # noqa: F841
        camera_heights = env_meta["env_kwargs"]["camera_heights"]
        camera_widths = env_meta["env_kwargs"]["camera_widths"]
        env_name = env_meta["env_name"]

    return camera_heights, camera_widths, env_name


def process_episode(hdf5_path, dataset, env_name):
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(d[5:]) for d in demos])
        demos = [demos[i] for i in inds]

        for ep in demos:
            ep_data = f["data"][ep]
            actions = ep_data["actions"][()]
            dones = ep_data["dones"][()]
            rewards = ep_data["rewards"][()]
            states = ep_data["states"][()]  # noqa: F841

            obs = ep_data["obs"]
            observation_state = np.concatenate(
                [
                    obs["robot0_eef_pos"][()],
                    obs["robot0_eef_quat"][()],
                    obs["robot0_gripper_qpos"][()],
                ],
                axis=-1,
            )

            episode_length = actions.shape[0]
            for t in range(episode_length):
                dataset.add_frame(
                    {
                        "observation.images.agentview_image": obs["agentview_image"][t],
                        "observation.images.robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"][t],
                        "observation.state": observation_state[t].astype(np.float32),
                        "action": actions[t].astype(np.float32),
                        "next.reward": np.array([rewards[t]], dtype=np.float32),
                        "next.done": np.array([dones[t]], dtype=np.bool_),
                        "task": env_name,
                    }
                )
            dataset.save_episode()
            print(f"Episode {ep} processed and saved.")


def main(file_path):
    camera_height, camera_width, env_name = extract_metadata(file_path)

    features = {
        "observation.images.agentview_image": {
            "dtype": "image",
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.robot0_eye_in_hand_image": {
            "dtype": "image",
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper_1, gripper_2"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(repo_id=env_name.lower(), fps=10, features=features)

    process_episode(file_path, dataset, env_name)


if __name__ == "__main__":
    hdf5_path = "/sfs/weka/scratch/jqm9ba/mimicgen_datasets/core/stack_three_d1.hdf5"
    main(hdf5_path)
