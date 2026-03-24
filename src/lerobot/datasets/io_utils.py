#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas
import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import embed_table_storage
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.datasets.utils import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_EPISODES_PATH,
    DEFAULT_SUBTASKS_PATH,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    flatten_dict,
    serialize_dict,
    unflatten_dict,
)
from lerobot.utils.utils import SuppressProgressBars


def get_parquet_file_size_in_mb(parquet_path: str | Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            col_metadata = rg_metadata.column(column)
            total_uncompressed_size += col_metadata.total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def get_hf_dataset_size_in_mb(hf_ds: Dataset) -> int:
    return hf_ds.data.nbytes // (1024**2)


def load_nested_dataset(
    pq_dir: Path, features: datasets.Features | None = None, episodes: list[int] | None = None
) -> Dataset:
    """Find parquet files in provided directory {pq_dir}/chunk-xxx/file-xxx.parquet
    Convert parquet files to pyarrow memory mapped in a cache folder for efficient RAM usage
    Concatenate all pyarrow references to return HF Dataset format

    Args:
        pq_dir: Directory containing parquet files
        features: Optional features schema to ensure consistent loading of complex types like images
        episodes: Optional list of episode indices to filter. Uses PyArrow predicate pushdown for efficiency.
    """
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")

    with SuppressProgressBars():
        # We use .from_parquet() memory-mapped loading for efficiency
        filters = pa_ds.field("episode_index").isin(episodes) if episodes is not None else None
        return Dataset.from_parquet([str(path) for path in paths], filters=filters, features=features)


def get_parquet_num_frames(parquet_path: str | Path) -> int:
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def get_file_size_in_mb(file_path: Path) -> float:
    """Get file size on disk in megabytes.

    Args:
        file_path (Path): Path to the file.
    """
    file_size_bytes = file_path.stat().st_size
    return file_size_bytes / (1024**2)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    """Embed image bytes into the dataset table before saving to Parquet.

    This function prepares a Hugging Face dataset for serialization by converting
    image objects into an embedded format that can be stored in Arrow/Parquet.

    Args:
        dataset (datasets.Dataset): The input dataset, possibly containing image features.

    Returns:
        datasets.Dataset: The dataset with images embedded in the table storage.
    """
    # Embed image bytes into the table before saving to parquet
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def load_json(fpath: Path) -> Any:
    """Load data from a JSON file.

    Args:
        fpath (Path): Path to the JSON file.

    Returns:
        Any: The data loaded from the JSON file.
    """
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_info(info: dict, local_dir: Path) -> None:
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    """Load dataset info metadata from its standard file path.

    Also converts shape lists to tuples for consistency.

    Args:
        local_dir (Path): The root directory of the dataset.

    Returns:
        dict: The dataset information dictionary.
    """
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path) -> None:
    """Serialize and write dataset statistics to their standard file path.

    Args:
        stats (dict): The statistics dictionary (can contain tensors/numpy arrays).
        local_dir (Path): The root directory of the dataset.
    """
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    """Recursively cast numerical values in a stats dictionary to numpy arrays.

    Args:
        stats (dict): The statistics dictionary.

    Returns:
        dict: The statistics dictionary with values cast to numpy arrays.
    """
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    """Load dataset statistics and cast numerical values to numpy arrays.

    Returns None if the stats file doesn't exist.

    Args:
        local_dir (Path): The root directory of the dataset.

    Returns:
        A dictionary of statistics or None if the file is not found.
    """
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_tasks(tasks: pandas.DataFrame, local_dir: Path) -> None:
    path = local_dir / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def load_tasks(local_dir: Path) -> pandas.DataFrame:
    tasks = pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
    tasks.index.name = "task"
    return tasks


def load_subtasks(local_dir: Path) -> pandas.DataFrame | None:
    """Load subtasks from subtasks.parquet if it exists."""
    subtasks_path = local_dir / DEFAULT_SUBTASKS_PATH
    if subtasks_path.exists():
        return pd.read_parquet(subtasks_path)
    return None


def write_episodes(episodes: Dataset, local_dir: Path) -> None:
    """Write episode metadata to a parquet file in the LeRobot v3.0 format.
    This function writes episode-level metadata to a single parquet file.
    Used primarily during dataset conversion (v2.1 → v3.0) and in test fixtures.

    Args:
        episodes: HuggingFace Dataset containing episode metadata
        local_dir: Root directory where the dataset will be stored
    """
    episode_size_mb = get_hf_dataset_size_in_mb(episodes)
    if episode_size_mb > DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise NotImplementedError(
            f"Episodes dataset is too large ({episode_size_mb} MB) to write to a single file. "
            f"The current limit is {DEFAULT_DATA_FILE_SIZE_IN_MB} MB. "
            "This function only supports single-file episode metadata. "
        )

    fpath = local_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(fpath)


def load_episodes(local_dir: Path) -> datasets.Dataset:
    episodes = load_nested_dataset(local_dir / EPISODES_DIR)
    # Select episode features/columns containing references to episode data and videos
    # (e.g. tasks, dataset_from_index, dataset_to_index, data/chunk_index, data/file_index, etc.)
    # This is to speedup access to these data, instead of having to load episode stats.
    episodes = episodes.select_columns([key for key in episodes.features if not key.startswith("stats/")])
    return episodes


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    """Load an image from a file into a numpy array.

    Args:
        fpath (str | Path): Path to the image file.
        dtype (np.dtype): The desired data type of the output array. If floating,
            pixels are scaled to [0, 1].
        channel_first (bool): If True, converts the image to (C, H, W) format.
            Otherwise, it remains in (H, W, C) format.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:
    """Convert a batch from a Hugging Face dataset to torch tensors.

    This transform function converts items from Hugging Face dataset format (pyarrow)
    to torch tensors. Importantly, images are converted from PIL objects (H, W, C, uint8)
    to a torch image representation (C, H, W, float32) in the range [0, 1]. Other
    types are converted to torch.tensor.

    Args:
        items_dict (dict): A dictionary representing a batch of data from a
            Hugging Face dataset.

    Returns:
        dict: The batch with items converted to torch tensors.
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def to_parquet_with_hf_images(
    df: pandas.DataFrame, path: Path, features: datasets.Features | None = None
) -> None:
    """This function correctly writes to parquet a panda DataFrame that contains images encoded by HF dataset.
    This way, it can be loaded by HF dataset and correctly formatted images are returned.

    Args:
        df: DataFrame to write to parquet.
        path: Path to write the parquet file.
        features: Optional HuggingFace Features schema. If provided, ensures image columns
                  are properly typed as Image() in the parquet schema.
    """
    # TODO(qlhoest): replace this weird synthax by `df.to_parquet(path)` only
    ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=features)
    ds.to_parquet(path)


def item_to_torch(item: dict) -> dict:
    """Convert all items in a dictionary to PyTorch tensors where appropriate.

    This function is used to convert an item from a streaming dataset to PyTorch tensors.

    Args:
        item (dict): Dictionary of items from a dataset.

    Returns:
        dict: Dictionary with all tensor-like items converted to torch.Tensor.
    """
    for key, val in item.items():
        if isinstance(val, (np.ndarray | list)) and key not in ["task"]:
            # Convert numpy arrays and lists to torch tensors
            item[key] = torch.tensor(val)
    return item
