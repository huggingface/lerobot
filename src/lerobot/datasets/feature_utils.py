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
from pprint import pformat

import datasets
import numpy as np
from PIL import Image as PILImage

from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.utils import is_valid_numpy_dtype_string

from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
)


def get_hf_features_from_features(features: dict) -> datasets.Features:
    """Convert a LeRobot features dictionary to a `datasets.Features` object.

    Args:
        features (dict): A LeRobot-style feature dictionary.

    Returns:
        datasets.Features: The corresponding Hugging Face `datasets.Features` object.

    Raises:
        ValueError: If a feature has an unsupported shape.
    """
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
) -> dict:
    """Create a template dictionary for a new dataset's `info.json`.

    Args:
        codebase_version (str): The version of the LeRobot codebase.
        fps (int): The frames per second of the data.
        features (dict): The LeRobot features dictionary for the dataset.
        use_videos (bool): Whether the dataset will store videos.
        robot_type (str | None): The type of robot used, if any.

    Returns:
        dict: A dictionary with the initial dataset metadata.
    """
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": chunks_size or DEFAULT_CHUNK_SIZE,
        "data_files_size_in_mb": data_files_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": video_files_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """Check if delta timestamps are multiples of 1/fps +/- tolerance.

    This ensures that adding these delta timestamps to any existing timestamp in
    the dataset will result in a value that aligns with the dataset's frame rate.

    Args:
        delta_timestamps (dict): A dictionary where values are lists of time
            deltas in seconds.
        fps (int): The frames per second of the dataset.
        tolerance_s (float): The allowed tolerance in seconds.
        raise_value_error (bool): If True, raises an error on failure.

    Returns:
        bool: True if all deltas are valid, False otherwise.

    Raises:
        ValueError: If any delta is outside the tolerance and `raise_value_error` is True.
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    """Convert delta timestamps in seconds to delta indices in frames.

    Args:
        delta_timestamps (dict): A dictionary of time deltas in seconds.
        fps (int): The frames per second of the dataset.

    Returns:
        dict: A dictionary of frame delta indices.
    """
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def validate_frame(frame: dict, features: dict) -> None:
    # DEFAULT_FEATURES (timestamp, frame_index, episode_index, index, task_index) are
    # auto-populated by the recording pipeline (add_frame / save_episode) and must not
    # be supplied by the caller. Excluding them here means any frame dict that contains
    # these keys will be rejected as extra features.
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
