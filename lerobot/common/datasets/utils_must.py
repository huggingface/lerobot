"""
Utils function by Mustafa to refactor
"""
import torch
import numpy as np
from lerobot.common.datasets.compute_stats import (
    aggregate_stats
)
from collections import defaultdict
OBS_IMAGE = "observation.image"
OBS_IMAGE_2 = "observation.image2"
OBS_IMAGE_3 = "observation.image3"

def reshape_features_to_max_dim(features: dict, reshape_dim: int = -1, keys_to_max_dim: dict = {}) -> dict:
    """Reshape features to have a maximum dimension of `max_dim`."""
    reshaped_features = {}
    for key in features:
        if key in keys_to_max_dim and keys_to_max_dim[key] is not None:
            reshaped_features[key] = features[key]
            shape = list(features[key]["shape"])
            if any([k in key for k in [OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3]]):  # Assume square images
                shape[-3] = keys_to_max_dim[key]
                shape[-2] = keys_to_max_dim[key]
            else:
                shape[reshape_dim] = keys_to_max_dim[key]
            reshaped_features[key]["shape"] = tuple(shape)
        else:
            reshaped_features[key] = features[key]
    return reshaped_features

def keep_datasets_with_valid_fps(
    ls_datasets: list, min_fps: int = 1, max_fps: int = 100
) -> list:
    print(f"Keeping datasets with fps between {min_fps} and {max_fps}. Considering {len(ls_datasets)} datasets.")
    for ds in ls_datasets:
        if ds.fps < min_fps or ds.fps > max_fps:
            print(f"Dataset {ds} has invalid fps: {ds.fps}. Removing it.")
            ls_datasets.remove(ds)
    print(f"Keeping {len(ls_datasets)} datasets with valid fps.")
    return ls_datasets

def keep_datasets_with_the_same_features_per_robot_type(
    ls_datasets: list
) -> list:
    """
    Filters datasets to only keep those with consistent feature shapes per robot type.

    Args:
        ls_datasets (List): List of datasets, each with a `meta.info['robot_type']`
            and `meta.episodes_stats` dictionary.

    Returns:
        List: Filtered list of datasets with consistent feature shapes.
    """
    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    datasets_to_remove = set()

    for robot_type in robot_types:
        # Collect all stats dicts for this robot type
        stats_list = [
            ep_stats
            for ds in ls_datasets if ds.meta.info["robot_type"] == robot_type
            for ep_stats in ds.meta.episodes_stats.values()
        ]
        if not stats_list:
            continue

        # Determine the most common shape for each key
        all_keys = {key for stats in stats_list for key in stats}
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] != robot_type:
                continue
            for key in all_keys:
                shape_counter = defaultdict(int)

                for stats in stats_list:
                    value = stats.get(key)
                    if value and "mean" in value and isinstance(value["mean"], (torch.Tensor, np.ndarray)): # FIXME(mshukor): check all stats; min, mean, max
                        shape_counter[value["mean"].shape] += 1
                if not shape_counter:
                    continue

                # Identify the most frequent shape
                main_shape = max(shape_counter, key=shape_counter.get)
                # Flag datasets that don't match the main shape
                # for ds in ls_datasets:
                first_ep_stats = next(iter(ds.meta.episodes_stats.values()), None)
                if not first_ep_stats:
                    continue
                value = first_ep_stats.get(key)
                if value and "mean" in value and isinstance(value["mean"], (torch.Tensor, np.ndarray)) and value["mean"].shape != main_shape:
                    datasets_to_remove.add(ds)
                    break

    # Filter out inconsistent datasets
    datasets_maks = [ds not in datasets_to_remove for ds in ls_datasets]
    filtered_datasets = [ds for ds in ls_datasets if ds not in datasets_to_remove]
    print(f"Keeping {len(filtered_datasets)} datasets. Removed {len(datasets_to_remove)} inconsistent ones. Inconsistent datasets:\n{datasets_to_remove}")
    return filtered_datasets, datasets_maks



def aggregate_stats_per_robot_type(ls_datasets) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate stats of multiple LeRobot datasets into multiple set of stats per robot type.

    The final stats will have the union of all data keys from each of the datasets.

    The final stats will have the union of all data keys from each of the datasets. For instance:
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_mean = (mean of all data)
    - new_std = (std of all data)
    """

    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    stats = {robot_type: {} for robot_type in robot_types}
    for robot_type in robot_types:
        robot_type_datasets = []
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] == robot_type:
                robot_type_datasets.extend(list(ds.meta.episodes_stats.values()))
        # robot_type_datasets = [list(ds.episodes_stats.values()) for ds in ls_datasets if ds.meta.info["robot_type"] == robot_type]
        stat = aggregate_stats(robot_type_datasets)
        stats[robot_type] = stat
    return stats

def str_to_torch_dtype(dtype_str):
    """Convert a dtype string to a torch dtype."""
    mapping = {
        "float32": torch.float32,
        "int64": torch.int64,
        "int16": torch.int16,
        "bool": torch.bool,
        "video": torch.float32,  # Assuming video is stored as uint8 images
    }
    return mapping.get(dtype_str, torch.float32)  # Default to float32

def create_padded_features(item: dict, features: dict = {}):
    for key, ft in features.items():
        if any([k in key for k in ["cam", "effort", "absolute"]]): # FIXME(mshukor): temporary hack
            continue
        shape = ft["shape"]
        if len(shape) == 3:  # images to torch format (C, H, W)
            shape = (shape[2], shape[0], shape[1])
        if len(shape) == 1 and shape[0] == 1:  # ft with shape are actually tensor(ele)
            shape = []
        if key not in item:
            dtype = str_to_torch_dtype(ft["dtype"])
            item[key] = torch.zeros(shape, dtype=dtype)
            item[f"{key}_padding_mask"] = torch.tensor(0, dtype=torch.int64)
            if "image" in key: # FIXME(mshukor): support other observations
                item[f"{key}_is_pad"] = torch.BoolTensor([False])
        else:
            item[f"{key}_padding_mask"] = torch.tensor(1, dtype=torch.int64)
    return item

ROBOT_TYPE_KEYS_MAPPING = {
    "lerobot/stanford_hydra_dataset": "static_single_arm",
    "lerobot/iamlab_cmu_pickup_insert": "static_single_arm",
    "lerobot/berkeley_fanuc_manipulation": "static_single_arm",
    "lerobot/toto": "static_single_arm",
    "lerobot/roboturk": "static_single_arm",
    "lerobot/jaco_play": "static_single_arm",
    "lerobot/taco_play": "static_single_arm_7statedim",
}

def pad_tensor(
    tensor: torch.Tensor, max_size: int, pad_dim: int = -1, pad_value: float = 0.0
) -> torch.Tensor:
    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        tensor = torch.tensor(tensor)
    pad = max_size - tensor.shape[pad_dim]
    if pad > 0:
        pad_sizes = (0, pad)  # pad right
        tensor = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
    return tensor.numpy() if is_numpy else tensor

def map_dict_keys(item: dict, feature_keys_mapping: dict, training_features: list = None, pad_key: str = "is_pad") -> dict:
    """Maps feature keys from the dataset to the keys used in the model."""
    if feature_keys_mapping is None:
        return item
    features = {}
    for key in item:
        if key in feature_keys_mapping:
            if feature_keys_mapping[key] is not None:
                if training_features is None or feature_keys_mapping[key] in training_features:
                    features[feature_keys_mapping[key]] = item[key]
        else:
            if training_features is None or key in training_features or pad_key in key:
                features[key] = item[key]
    return features

import yaml
import requests
def load_yaml_mapping(name: str) -> dict:
    """
    Loads a YAML mapping from a Hugging Face repo.
    Example: name='features' → https://huggingface.co/jadechoghari/smolvla-keys/resolve/main/features.yaml
    """
    url = f"https://huggingface.co/jadechoghari/smolvla-keys/resolve/main/{name}.yaml"
    response = requests.get(url)
    response.raise_for_status()  # raise if the download fails

    return yaml.safe_load(response.text)

# Example usage
TASKS_KEYS_MAPPING = load_yaml_mapping("features")
