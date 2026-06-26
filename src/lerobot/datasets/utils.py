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
import dataclasses
import importlib.resources
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import pandas
import torch
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from PIL import Image as PILImage
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.utils.utils import flatten_dict, unflatten_dict

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


logger = logging.getLogger(__name__)


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
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


@dataclass
class DatasetInfo:
    """Typed representation of the ``meta/info.json`` file for a LeRobot dataset.

    Replaces the previously untyped ``dict`` returned by ``load_info()`` and
    created by ``create_empty_dataset_info()``.  Using a dataclass provides
    explicit field definitions, IDE auto-completion, and validation at
    construction time.
    """

    codebase_version: str
    fps: int
    features: dict[str, dict]

    # Episode / frame counters — start at zero for new datasets
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0

    # Storage settings
    chunks_size: int = field(default=DEFAULT_CHUNK_SIZE)
    data_files_size_in_mb: int = field(default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    video_files_size_in_mb: int = field(default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)

    # File path templates
    data_path: str = field(default=DEFAULT_DATA_PATH)
    video_path: str | None = field(default=DEFAULT_VIDEO_PATH)

    # Optional metadata
    robot_type: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    # OpenAI-style tool schemas declared by the dataset. ``None`` means the
    # dataset doesn't declare any — readers fall back to ``DEFAULT_TOOLS``.
    tools: list[dict] | None = None

    def __post_init__(self) -> None:
        # Coerce feature shapes from list to tuple — JSON deserialisation
        # returns lists, but the rest of the codebase expects tuples.
        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {self.chunks_size}")
        if self.data_files_size_in_mb <= 0:
            raise ValueError(f"data_files_size_in_mb must be positive, got {self.data_files_size_in_mb}")
        if self.video_files_size_in_mb <= 0:
            raise ValueError(f"video_files_size_in_mb must be positive, got {self.video_files_size_in_mb}")

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Converts tuple shapes back to lists so ``json.dump`` can handle them.
        Drops ``tools`` when unset so existing datasets keep a clean
        ``info.json``.
        """
        d = dataclasses.asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        if d.get("tools") is None:
            d.pop("tools", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetInfo":
        """Construct from a raw dict (e.g. loaded directly from JSON).

        Unknown keys are ignored for forward compatibility with datasets that
        carry additional fields (e.g. ``total_videos`` from v2.x). A warning is
        logged when such fields are present.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            logger.warning(f"Unknown fields in DatasetInfo: {unknown}. These will be ignored.")
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------------------------
    # Temporary dict-style compatibility layer
    # Allows existing ``info["key"]`` call-sites to keep working without changes.
    # Once all callers have been migrated to attribute access, remove these.
    # ---------------------------------------------------------------------------
    def __getitem__(self, key: str):
        import warnings

        warnings.warn(
            f"Accessing DatasetInfo with dict-style syntax info['{key}'] is deprecated. "
            f"Use attribute access info.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value) -> None:
        import warnings

        warnings.warn(
            f"Setting DatasetInfo with dict-style syntax info['{key}'] = ... is deprecated. "
            f"Use attribute assignment info.{key} = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, key):
            raise KeyError(f"DatasetInfo has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists (dict-like interface)."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """Get attribute value with default fallback (dict-like interface)."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


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
    dataset_info: DatasetInfo | None = None,
    **kwargs,
) -> DatasetCard:
    """Create a `DatasetCard` for a LeRobot dataset.

    Keyword arguments are used to replace values in the card template.
    Note: If specified, `license` must be a valid license identifier from
    https://huggingface.co/docs/hub/repositories-licenses.

    Args:
        tags (list | None): A list of tags to add to the dataset card.
        dataset_info (DatasetInfo | None): The dataset's info object, which will
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
        dataset_structure += f"```json\n{json.dumps(dataset_info.to_dict(), indent=4)}\n```\n"
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


def validate_frame(frame: dict, features: dict) -> None:
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)

    # task is a special required field that's not part of regular features
    if "task" not in actual_features:
        raise ValueError("Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n")

    # Remove task from actual_features for regular feature validation
    actual_features_for_validation = actual_features - {"task"}

    error_message = validate_features_presence(actual_features_for_validation, expected_features)

    common_features = actual_features_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:
    """Check for missing or extra features in a frame.

    Args:
        actual_features (set[str]): The set of feature names present in the frame.
        expected_features (set[str]): The set of feature names expected in the frame.

    Returns:
        str: An error message string if there's a mismatch, otherwise an empty string.
    """
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:
    """Validate the dtype and shape of a single feature's value.

    Args:
        name (str): The name of the feature.
        feature (dict): The feature specification from the LeRobot features dictionary.
        value: The value of the feature to validate.

    Returns:
        str: An error message if validation fails, otherwise an empty string.

    Raises:
        NotImplementedError: If the feature dtype is not supported for validation.
    """
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:
    """Validate a feature that is expected to be a numpy array.

    Args:
        name (str): The name of the feature.
        expected_dtype (str): The expected numpy dtype as a string.
        expected_shape (list[int]): The expected shape.
        value (np.ndarray): The numpy array to validate.

    Returns:
        str: An error message if validation fails, otherwise an empty string.
    """
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:
    """Validate a feature that is expected to be an image or video frame.

    Accepts `np.ndarray` (channel-first or channel-last) or `PIL.Image.Image`.

    Args:
        name (str): The name of the feature.
        expected_shape (list[str]): The expected shape (C, H, W).
        value: The image data to validate.

    Returns:
        str: An error message if validation fails, otherwise an empty string.
    """
    # Note: The check of pixels range ([0,1] for float and [0,255] for uint8) is done by the image writer threads.
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str) -> str:
    """Validate a feature that is expected to be a string.

    Args:
        name (str): The name of the feature.
        value (str): The value to validate.

    Returns:
        str: An error message if validation fails, otherwise an empty string.
    """
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    """Validate the episode buffer before it's written to disk.

    Ensures the buffer has the required keys, contains at least one frame, and
    has features consistent with the dataset's specification.

    Args:
        episode_buffer (dict): The buffer containing data for a single episode.
        total_episodes (int): The current total number of episodes in the dataset.
        features (dict): The LeRobot features dictionary for the dataset.

    Raises:
        ValueError: If the buffer is invalid.
        NotImplementedError: If the episode index is manually set and doesn't match.
    """
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        # TODO(aliberts): Add option to use existing episode_index
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )


def to_parquet_with_hf_images(df: pandas.DataFrame, path: Path) -> None:
    """This function correctly writes to parquet a panda DataFrame that contains images encoded by HF dataset.
    This way, it can be loaded by HF dataset and correctly formatted images are returned.
    """
    # TODO(qlhoest): replace this weird synthax by `df.to_parquet(path)` only
    datasets.Dataset.from_dict(df.to_dict(orient="list")).to_parquet(path)


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
