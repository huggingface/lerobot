#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Storage helpers for the audio features of a :class:`LeRobotDataset`.

Audio is stored losslessly as 24-bit FLAC, so a sample read back is the sample that was recorded (up to
24-bit quantization) and sample indices stay exact: no encoder delay or padding shifts the audio relative to
the frames. Samples are handled as float32 arrays of shape ``(num_samples, num_channels)`` in ``[-1, 1]``,
the layout returned by :meth:`lerobot.microphones.Microphone.read`.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.import_utils import _soundfile_available, require_package

if TYPE_CHECKING or _soundfile_available:
    import soundfile as sf

logger = logging.getLogger(__name__)

AUDIO_CODEC = "flac"
DEFAULT_AUDIO_WINDOW_S = 0.5
AUDIO_SAMPLE_RATE_INFO_KEY = "audio.sample_rate"
_FLAC_SUBTYPE = "PCM_24"
_COPY_BLOCK_SIZE = 1 << 20


def get_audio_sample_rate(feature: dict) -> int:
    """Return the sample rate of an audio feature, read from its ``info`` dict."""
    return int(feature["info"][AUDIO_SAMPLE_RATE_INFO_KEY])


def validate_audio_feature(key: str, feature: dict) -> None:
    """Check that an audio feature declares its channels and its sample rate.

    Args:
        key: The feature name.
        feature: The feature specification, e.g.
            ``{"dtype": "audio", "shape": (2,), "names": ["left", "right"], "info": {"audio.sample_rate": 48000}}``.

    Raises:
        ValueError: If the shape is not ``(num_channels,)`` or the sample rate is missing or invalid.
    """
    shape = tuple(feature.get("shape") or ())
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError(f"Audio feature '{key}' must have a shape of (num_channels,), got {shape}.")
    sample_rate = (feature.get("info") or {}).get(AUDIO_SAMPLE_RATE_INFO_KEY)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"Audio feature '{key}' must declare a positive integer sample rate in "
            f"info['{AUDIO_SAMPLE_RATE_INFO_KEY}'], got {sample_rate!r}."
        )


def load_audio(audio: NDArray[Any] | str | Path) -> tuple[NDArray[np.float32], int | None]:
    """Load audio samples from an array or an audio file.

    Args:
        audio: Samples of shape ``(num_samples, num_channels)`` (or ``(num_samples,)`` for a single
            channel), or the path of an audio file readable by libsndfile (WAV, FLAC, ...).

    Returns:
        The float32 samples of shape ``(num_samples, num_channels)``, and the sample rate of the file
        (``None`` for an array). Samples outside ``[-1, 1]`` are clipped, with a warning.

    Raises:
        ValueError: If the samples are not finite.
    """
    if isinstance(audio, (str, Path)):
        require_package("soundfile", extra="audio")
        samples, file_sample_rate = sf.read(audio, dtype="float32", always_2d=True)
        sample_rate: int | None = int(file_sample_rate)
    else:
        samples = np.asarray(audio)
        if not np.issubdtype(samples.dtype, np.floating):
            raise TypeError(
                f"Audio samples must be floating point values in [-1, 1], got dtype {samples.dtype}."
            )
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2:
            raise ValueError(
                f"Audio samples must have a shape of (num_samples, num_channels), got {samples.shape}."
            )
        samples = samples.astype(np.float32)
        sample_rate = None

    if not np.isfinite(samples).all():
        raise ValueError("Audio samples must be finite, got NaN or infinite values.")
    # The FLAC encoder would clip them anyway: do it explicitly so that it is not silent.
    num_clipped = int(np.count_nonzero(np.abs(samples) > 1.0))
    if num_clipped > 0:
        logger.warning(
            f"Clipping {num_clipped} audio sample(s) outside [-1, 1] (peak {np.abs(samples).max():.3f})."
        )
        np.clip(samples, -1.0, 1.0, out=samples)
    return samples, sample_rate


def write_audio_file(path: Path, samples: NDArray[np.float32], sample_rate: int) -> None:
    """Write samples of shape ``(num_samples, num_channels)`` to a new FLAC file."""
    require_package("soundfile", extra="audio")
    sf.write(path, samples, sample_rate, format="FLAC", subtype=_FLAC_SUBTYPE)


def concatenate_audio_files(input_paths: list[Path], output_path: Path) -> None:
    """Concatenate FLAC files with the same sample rate and channels into ``output_path``.

    FLAC files cannot be appended to in place, so the inputs are copied block by block into a temporary
    file that then replaces ``output_path``. ``output_path`` can be one of the inputs.
    """
    require_package("soundfile", extra="audio")
    info = sf.info(input_paths[0])
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with sf.SoundFile(
            tmp_path, "w", info.samplerate, info.channels, format="FLAC", subtype=_FLAC_SUBTYPE
        ) as output:
            for input_path in input_paths:
                with sf.SoundFile(input_path) as input_file:
                    if (input_file.samplerate, input_file.channels) != (info.samplerate, info.channels):
                        raise ValueError(
                            f"Cannot concatenate {input_path} ({input_file.samplerate} Hz, "
                            f"{input_file.channels} channels) with {input_paths[0]} ({info.samplerate} Hz, "
                            f"{info.channels} channels)."
                        )
                    for block in input_file.blocks(
                        blocksize=_COPY_BLOCK_SIZE, dtype="float32", always_2d=True
                    ):
                        output.write(block)
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def get_audio_num_samples(path: Path) -> int:
    """Return the number of samples (per channel) in an audio file."""
    require_package("soundfile", extra="audio")
    return int(sf.info(path).frames)


def read_audio_file(path: Path, start: int, stop: int) -> NDArray[np.float32]:
    """Read the samples ``[start, stop)`` of an audio file, as an array of shape ``(stop - start, num_channels)``."""
    require_package("soundfile", extra="audio")
    samples, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    return samples
