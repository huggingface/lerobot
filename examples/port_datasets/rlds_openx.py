import argparse
import shutil
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def tf_img_convert(img):
    if img.dtype == tf.string:
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    elif img.dtype != tf.uint8:
        raise ValueError(f"Unsupported image dtype: found with dtype {img.dtype}")
    return torch.from_numpy(img.numpy())


def get_type(dtype):
    if dtype == tf.uint8:
        return "uint8"
    elif dtype == tf.float32:
        return "float32"
    elif dtype == tf.float64:
        return "float64"
    elif dtype == tf.bool:
        return "bool"
    elif dtype == tf.string:
        return "str"


def _broadcast_metadata_rlds(i: tf.Tensor, traj: dict) -> dict:
    """
    In the RLDS format, each trajectory has some top-level metadata that is explicitly separated out, and a "steps"
    entry. This function moves the "steps" entry to the top level, broadcasting any metadata to the length of the
    trajectory. This function also adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.

    NOTE: adapted from DLimp library https://github.com/kvablack/dlimp/
    """
    steps = traj.pop("steps")

    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]

    # broadcast metadata to the length of the trajectory
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)

    # put steps back in
    assert "traj_metadata" not in steps
    traj = {**steps, "traj_metadata": metadata}

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    return traj


def load_raw_dataset(path: Path):
    ds_builder = tfds.builder_from_directory(str(path))
    dataset = ds_builder.as_dataset(
        split="all",
        decoders={"steps": tfds.decode.SkipDecoding()},
    )

    dataset_info = ds_builder.info
    print("dataset_info: ", dataset_info)
    ds_length = len(dataset)
    dataset = dataset.take(ds_length)
    # "flatten" the dataset as such we can apply trajectory level map() easily
    # each [obs][key] has a shape of (frame_size, ...)
    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)

    return dataset, dataset_info


def build_features_and_dataset_keys(dataset_info):
    features = {}
    image_keys = []
    state_keys = []
    other_keys = []
    for key, data_info in dataset_info.features["steps"].items():
        if "observation" in key:
            # check whether the key is for an image or a vector observation
            # only add rgb images, discard depth
            for k, info in data_info.items():
                if len(info.shape) == 3 and info.dtype == tf.uint8:
                    image_keys.append(k)
                    dtype = "video"
                    shape = info.shape
                    # TODO (michel_aractingi) add info[key].doc for feature description
                    features["observation.image." + k] = {"dtype": dtype, "shape": shape, "name": None}
                else:
                    state_keys.append(k)
                    dtype = get_type(info.dtype)
                    shape = info.shape
                    # TODO (michel_aractingi) add info[key].doc for feature description
                    features["observation.state." + k] = {"dtype": dtype, "shape": shape, "name": None}
        else:
            if type(data_info) is tfds.features.Tensor:
                # TODO extend features to take language instructions
                if "language_instruction" in key:
                    continue
                other_keys.append(key)
                dtype = get_type(data_info.dtype)
                shape = data_info.shape
                if len(shape) == 0:
                    shape = (1,)
                if key == "is_last":
                    features["next.done"] = {"dtype": dtype, "shape": shape, "name": None}
                elif key == "reward":
                    features["next.reward"] = {"dtype": dtype, "shape": shape, "name": None}
                else:
                    features[key] = {"dtype": dtype, "shape": shape, "name": None}
            # elif type(data_info) is tfds.features.FeaturesDict: TODO add dictionary based variables

    return features, image_keys, state_keys, other_keys


def to_lerobotdataset_with_save_episode(raw_dir: Path, repo_id: str, push_to_hub: bool = True, fps=30):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    dataset, dataset_info = load_raw_dataset(path=raw_dir)

    # Build features
    features, image_keys, state_keys, other_keys = build_features_and_dataset_keys(dataset_info)

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    it = iter(dataset)
    lang_instruction = None
    # The iterator it loops over each EPISODE in dataset (not frame-by-frame)
    # len(dataset) is the number of trajectories/episodes in a dataset
    for ep_idx in tqdm.tqdm(range(len(dataset))):
        episode = next(it)
        episode_data = {}
        num_frames = episode["action"].shape[0]
        lang_instruction = episode["language_instruction"].numpy()[0].decode("utf-8")

        for key in state_keys:
            episode_data["observation.state." + key] = tf_to_torch(episode["observation"][key])
        for key in image_keys:
            decoded_images = [tf_img_convert(img) for img in episode["observation"][key]]
            episode_data["observation.image." + key] = decoded_images

        for key in other_keys:
            if "language_instruction" in key:
                continue
                # Some openx dataset have multiple language commands
                # episode_data[key] = episode[key].numpy()[0].decode("utf-8")
            else:
                if key == "is_last":
                    episode_data["next.done"] = tf_to_torch(episode[key])
                elif key == "reward":
                    episode_data["next.reward"] = tf_to_torch(episode[key])
                else:
                    episode_data[key] = tf_to_torch(episode[key])

        episode_data["size"] = num_frames
        episode_data["episode_index"] = ep_idx  # torch.tensor([ep_idx] * num_frames)
        episode_data["frame_index"] = torch.arange(0, num_frames, 1)
        episode_data["timestamp"] = torch.arange(0, num_frames, 1) / fps
        episode_data["task_index"] = 0  # TODO calculate task index correctly
        episode_data["index"] = 0  # TODO figure out what index is for in DEFAULT_FEATURES

        lerobot_dataset.save_episode(task=lang_instruction, episode_data=episode_data)

    lerobot_dataset.consolidate()

    if push_to_hub:
        lerobot_dataset.push_to_hub()


def to_lerobotdataset_with_add_frame(raw_dir: Path, repo_id: str, push_to_hub: bool = True, fps=30):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    dataset, dataset_info = load_raw_dataset(path=raw_dir)

    # Build features, get keys
    features, image_keys, state_keys, other_keys = build_features_and_dataset_keys(dataset_info)

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    it = iter(dataset)
    lang_instruction = None
    # The iterator it loops over each EPISODE in dataset (not frame-by-frame)
    # len(dataset) is the number of trajectories/episodes in a dataset
    for _ep_idx in tqdm.tqdm(range(len(dataset))):
        episode = next(it)
        episode_data = {}
        num_frames = episode["action"].shape[0]
        lang_instruction = episode["language_instruction"].numpy()[0].decode("utf-8")

        for key in state_keys:
            episode_data["observation.state." + key] = tf_to_torch(episode["observation"][key])
        for key in image_keys:
            decoded_images = [tf_img_convert(img) for img in episode["observation"][key]]
            episode_data["observation.image." + key] = decoded_images

        for key in other_keys:
            if "language_instruction" in key:
                # Some openx dataset have multiple language commands
                # like droid has 1-3 language instructions for some trajectories
                episode_data[key] = episode[key].numpy()[0].decode("utf-8")
            else:
                if key == "is_last":
                    episode_data["next.done"] = tf_to_torch(episode[key])
                elif key == "reward":
                    episode_data["next.reward"] = tf_to_torch(episode[key])
                else:
                    episode_data[key] = tf_to_torch(episode[key])

        for i in range(num_frames):
            frame = {}
            for key in episode_data:
                if "language_instruction" in key:
                    frame[key] = episode_data[key]
                else:
                    frame[key] = episode_data[key][i]

            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode(task=lang_instruction)

    lerobot_dataset.consolidate()

    if push_to_hub:
        lerobot_dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to the directory of the raw dataset in rlds/openx format.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=0,
        help="Binary value to indicate whether you want to push the dataset to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frames per second, can be found the openx spreadsheet for openx datasets."
        "https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0",
    )

    args = parser.parse_args()

    to_lerobotdataset_with_add_frame(args.raw_dir, args.repo_id, args.push_to_hub, args.fps)
