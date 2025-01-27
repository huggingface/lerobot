from pathlib import Path

import cv2
import h5py
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

data_path = Path("/home/ccop/code/aloha_data")


def get_features(hdf5_file):
    topics = []
    features = {}
    hdf5_file.visititems(lambda name, obj: topics.append(name) if isinstance(obj, h5py.Dataset) else None)
    for topic in topics:
        # print(topic.replace('/', '.'))
        if "images" in topic.split("/"):
            features[topic.replace("/", ".")] = {
                "dtype": "image",
                "shape": cv2.imdecode(hdf5_file[topic][0], 1).transpose(2, 0, 1).shape,
                "names": None,
            }
        elif "compress_len" in topic.split("/"):
            continue
        else:
            features[topic.replace("/", ".")] = {
                "dtype": str(hdf5_file[topic][0].dtype),
                "shape": hdf5_file[topic][0].shape,
                "names": None,
            }

    return features


def extract_episode(episode_path, features, n_frames, dataset):
    with h5py.File(episode_path, "r") as file:
        # List all groups
        for frame_idx in range(n_frames):
            frame = {}
            for feature in features:
                if "images" in feature.split("."):
                    frame[feature] = torch.from_numpy(
                        cv2.imdecode(file[feature.replace(".", "/")][frame_idx], 1).transpose(2, 0, 1)
                    )
                else:
                    frame[feature] = torch.from_numpy(file[feature.replace(".", "/")][frame_idx])

            dataset.add_frame(frame)


def get_dataset_properties(raw_folder):
    from os import listdir

    episode_list = listdir(raw_folder)
    with h5py.File(raw_folder / episode_list[0], "r") as file:
        features = get_features(file)
        n_frames = file["observations/images/cam_high"][:].shape[0]
    return features, n_frames


if __name__ == "__main__":
    raw_folder = data_path.absolute() / "aloha_stationary_replay_test"
    episode_file = "episode_0.hdf5"

    features, n_frames = get_dataset_properties(raw_folder)

    dataset = LeRobotDataset.create(
        repo_id="ccop/aloha_stationary_replay_test_v3",
        fps=50,
        robot_type="aloha-stationary",
        features=features,
        image_writer_threads=4,
    )

    extract_episode(raw_folder / episode_file, features, n_frames, dataset)
    print("save episode!")
    dataset.save_episode(
        task="move_cube",
    )
    dataset.consolidate()
    dataset.push_to_hub()
