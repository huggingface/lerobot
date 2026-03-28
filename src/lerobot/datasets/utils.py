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
import contextlib
import importlib.resources
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import packaging.version
import torch
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError

V30_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v3.0 which is not backward compatible with v2.1.
Please, update your dataset to the new format using this command:
```
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

If you already have a converted version uploaded to the hub, then this error might be because of
an older version in your local cache. Consider deleting the cached version and retrying.

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError(
                "Contact the maintainer on [Discord](https://discord.com/invite/s3KuuzsPFb)."
            )
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


DEFAULT_CHUNK_SIZE = 1000  # Max number of files per chunk
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # Max size per file
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200  # Max size per file

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_SUBTASKS_PATH = "meta/subtasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def has_legacy_hub_download_metadata(root: Path) -> bool:
    """Return ``True`` when *root* looks like a legacy Hub ``local_dir`` mirror.

    ``snapshot_download(local_dir=...)`` stores lightweight metadata under
    ``<local_dir>/.cache/huggingface/download/``.  The presence of this
    directory is a reliable indicator that the dataset was downloaded with
    the old non-revision-safe ``local_dir`` mode and should be re-fetched
    through the snapshot cache instead.
    """
    return (root / ".cache" / "huggingface" / "download").exists()


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary by joining keys with a separator.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to prepend to the keys in this level.
        sep (str): The separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary.

    Example:
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): A dictionary with flattened keys.
        sep (str): The separator used in the keys.

    Returns:
        dict: A nested dictionary.
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """Serialize a dictionary containing tensors or numpy arrays to be JSON-compatible.

    Converts torch.Tensor, np.ndarray, and np.generic types to lists or native Python types.

    Args:
        stats (dict): A dictionary that may contain non-serializable numeric types.

    Returns:
        dict: A dictionary with all values converted to JSON-serializable types.

    Raises:
        NotImplementedError: If a value has an unsupported type.
    """
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor | np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int | float | list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int | float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def is_valid_version(version: str) -> bool:
    """Check if a string is a valid PEP 440 version.

    Args:
        version (str): The version string to check.

    Returns:
        bool: True if the version string is valid, False otherwise.
    """
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    """Check for version compatibility between a dataset and the current codebase.

    Args:
        repo_id (str): The repository ID for logging purposes.
        version_to_check (str | packaging.version.Version): The version of the dataset.
        current_version (str | packaging.version.Version): The current version of the codebase.
        enforce_breaking_major (bool): If True, raise an error on major version mismatch.

    Raises:
        BackwardCompatibilityError: If the dataset version is from a newer, incompatible
            major version of the codebase.
    """
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    """Return available valid versions (branches and tags) on a given Hub repo.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        list[packaging.version.Version]: A list of valid versions found.
    """
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    """Return the specified version if available on repo, or the latest compatible one.

    If the exact version is not found, it looks for the latest version with the
    same major version number that is less than or equal to the target minor version.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        version (str | packaging.version.Version): The target version.

    Returns:
        str: The safe version string (e.g., "v1.2.3") to use as a revision.

    Raises:
        RevisionNotFoundError: If the repo has no version tags.
        BackwardCompatibilityError: If only older major versions are available.
        ForwardCompatibilityError: If only newer major versions are available.
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.

    This is an equivalent of `itertools.cycle` but is safe for use with
    PyTorch DataLoaders with multiple workers.
    See https://github.com/pytorch/pytorch/issues/23900 for details.

    Args:
        iterable: The iterable to cycle over.

    Yields:
        Items from the iterable, restarting from the beginning when exhausted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:
    """Create a branch on an existing Hugging Face repo.

    Deletes the branch if it already exists before creating it.

    Args:
        repo_id (str): The ID of the repository.
        branch (str): The name of the branch to create.
        repo_type (str | None): The type of the repository (e.g., "dataset").
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """Create a `DatasetCard` for a LeRobot dataset.

    Keyword arguments are used to replace values in the card template.
    Note: If specified, `license` must be a valid license identifier from
    https://huggingface.co/docs/hub/repositories-licenses.

    Args:
        tags (list | None): A list of tags to add to the dataset card.
        dataset_info (dict | None): The dataset's info dictionary, which will
            be displayed on the card.
        **kwargs: Additional keyword arguments to populate the card template.

    Returns:
        DatasetCard: The generated dataset card object.
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


def is_float_in_list(target, float_list, threshold=1e-6):
    return any(abs(target - x) <= threshold for x in float_list)


def find_float_index(target, float_list, threshold=1e-6):
    for i, x in enumerate(float_list):
        if abs(target - x) <= threshold:
            return i
    return -1


def safe_shard(dataset: datasets.IterableDataset, index: int, num_shards: int) -> datasets.Dataset:
    """
    Safe shards the dataset.
    """
    shard_idx = min(dataset.num_shards, index + 1) - 1

    return dataset.shard(num_shards, index=shard_idx)
