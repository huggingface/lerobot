# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.utils.constants import OBS_IMAGES

from .constants import (
    IMAGE_COMPRESSION_JPEG,
    IMAGE_COMPRESSION_NONE,
    IMAGE_COMPRESSION_PNG,
    SUPPORTED_IMAGE_COMPRESSIONS,
)
from .helpers import RawObservation, TimedObservation


@dataclass(frozen=True)
class CompressedImage:
    codec: str
    data: bytes
    shape: tuple[int, ...]
    dtype: str
    uncompressed_nbytes: int


@dataclass(frozen=True)
class CompressionStats:
    image_count: int = 0
    uncompressed_nbytes: int = 0
    compressed_nbytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.uncompressed_nbytes == 0:
            return 1.0
        return self.compressed_nbytes / self.uncompressed_nbytes


def validate_image_compression(codec: str, quality: int) -> None:
    if codec not in SUPPORTED_IMAGE_COMPRESSIONS:
        raise ValueError(
            f"Unsupported observation image compression '{codec}'. "
            f"Expected one of {SUPPORTED_IMAGE_COMPRESSIONS}."
        )

    if quality < 1 or quality > 100:
        raise ValueError(f"observation_image_compression_quality must be between 1 and 100, got {quality}")


def image_keys_from_lerobot_features(lerobot_features: dict[str, dict]) -> set[str]:
    image_prefix = f"{OBS_IMAGES}."
    return {key.removeprefix(image_prefix) for key in lerobot_features if key.startswith(image_prefix)}


def _opencv_extension(codec: str) -> str:
    if codec == IMAGE_COMPRESSION_JPEG:
        return ".jpg"
    if codec == IMAGE_COMPRESSION_PNG:
        return ".png"
    raise ValueError(f"Unsupported image compression codec '{codec}'")


def _opencv_encode_params(codec: str, quality: int) -> list[int]:
    if codec == IMAGE_COMPRESSION_JPEG:
        return [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    if codec == IMAGE_COMPRESSION_PNG:
        # Interpret higher quality as less PNG compression work. PNG is still lossless.
        compression = round((100 - quality) * 9 / 99)
        return [int(cv2.IMWRITE_PNG_COMPRESSION), compression]

    raise ValueError(f"Unsupported image compression codec '{codec}'")


def _as_uint8_image(image: Any) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.dtype != np.uint8:
        raise ValueError(f"Only uint8 images can be compressed, got dtype {image_array.dtype}")

    if image_array.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D images can be compressed, got shape {image_array.shape}")

    if image_array.ndim == 3 and image_array.shape[2] not in (1, 3, 4):
        raise ValueError(
            "Only 1-, 3-, or 4-channel images can be compressed, "
            f"got shape {image_array.shape}"
        )

    return np.ascontiguousarray(image_array)


def compress_image(image: Any, codec: str, quality: int) -> CompressedImage:
    validate_image_compression(codec, quality)
    if codec == IMAGE_COMPRESSION_NONE:
        raise ValueError("'none' is not a valid codec for compress_image")

    image_array = _as_uint8_image(image)
    try:
        success, encoded = cv2.imencode(
            _opencv_extension(codec),
            image_array,
            _opencv_encode_params(codec, quality),
        )
    except cv2.error as exc:
        raise ValueError(f"Failed to encode image with codec '{codec}'") from exc

    if not success:
        raise ValueError(f"Failed to encode image with codec '{codec}'")

    return CompressedImage(
        codec=codec,
        data=encoded.tobytes(),
        shape=tuple(image_array.shape),
        dtype=str(image_array.dtype),
        uncompressed_nbytes=image_array.nbytes,
    )


def decompress_image(payload: CompressedImage) -> np.ndarray:
    if payload.codec not in SUPPORTED_IMAGE_COMPRESSIONS or payload.codec == IMAGE_COMPRESSION_NONE:
        raise ValueError(f"Unsupported compressed image codec '{payload.codec}'")

    encoded = np.frombuffer(payload.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image with codec '{payload.codec}'")

    if tuple(image.shape) != payload.shape:
        raise ValueError(f"Decoded image shape {image.shape} does not match expected {payload.shape}")

    expected_dtype = np.dtype(payload.dtype)
    if image.dtype != expected_dtype:
        raise ValueError(f"Decoded image dtype {image.dtype} does not match expected {expected_dtype}")

    return image


def compression_stats(raw_obs: RawObservation) -> CompressionStats:
    image_count = 0
    uncompressed_nbytes = 0
    compressed_nbytes = 0

    for value in raw_obs.values():
        if not isinstance(value, CompressedImage):
            continue
        image_count += 1
        uncompressed_nbytes += value.uncompressed_nbytes
        compressed_nbytes += len(value.data)

    return CompressionStats(
        image_count=image_count,
        uncompressed_nbytes=uncompressed_nbytes,
        compressed_nbytes=compressed_nbytes,
    )


def compress_raw_observation(
    raw_obs: RawObservation,
    image_keys: set[str],
    codec: str,
    quality: int,
) -> RawObservation:
    validate_image_compression(codec, quality)
    if codec == IMAGE_COMPRESSION_NONE:
        return raw_obs

    compressed_obs = dict(raw_obs)
    for key in image_keys:
        if key not in compressed_obs or isinstance(compressed_obs[key], CompressedImage):
            continue
        compressed_obs[key] = compress_image(compressed_obs[key], codec, quality)

    return compressed_obs


def decompress_raw_observation(raw_obs: RawObservation) -> RawObservation:
    if not any(isinstance(value, CompressedImage) for value in raw_obs.values()):
        return raw_obs

    decompressed_obs = dict(raw_obs)
    for key, value in raw_obs.items():
        if isinstance(value, CompressedImage):
            decompressed_obs[key] = decompress_image(value)

    return decompressed_obs


def compress_timed_observation(
    obs: TimedObservation,
    image_keys: set[str],
    codec: str,
    quality: int,
) -> TimedObservation:
    compressed_observation = compress_raw_observation(obs.get_observation(), image_keys, codec, quality)
    if compressed_observation is obs.get_observation():
        return obs

    return TimedObservation(
        timestamp=obs.get_timestamp(),
        timestep=obs.get_timestep(),
        observation=compressed_observation,
        must_go=obs.must_go,
    )


def decompress_timed_observation(obs: TimedObservation) -> TimedObservation:
    decompressed_observation = decompress_raw_observation(obs.get_observation())
    if decompressed_observation is obs.get_observation():
        return obs

    return TimedObservation(
        timestamp=obs.get_timestamp(),
        timestep=obs.get_timestep(),
        observation=decompressed_observation,
        must_go=obs.must_go,
    )
