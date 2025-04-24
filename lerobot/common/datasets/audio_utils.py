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
import logging
import subprocess
from collections import OrderedDict
from pathlib import Path

import torch
import torchaudio
import torchcodec
from numpy import ceil


def decode_audio(
    audio_path: Path | str,
    timestamps: list[float],
    duration: float,
    backend: str | None = "torchcodec",
) -> torch.Tensor:
    """
    Decodes audio using the specified backend.
    Args:
        audio_path (Path): Path to the audio file.
        timestamps (list[float]): List of (starting) timestamps to extract audio chunks.
        duration (float): Duration of the audio chunks in seconds.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec".

    Returns:
        torch.Tensor: Decoded audio chunks.

    Currently supports ffmpeg.
    """
    if backend == "torchcodec":
        return decode_audio_torchcodec(audio_path, timestamps, duration)
    elif backend == "torchaudio":
        return decode_audio_torchaudio(audio_path, timestamps, duration)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_audio_torchcodec(
    audio_path: Path | str,
    timestamps: list[float],
    duration: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    # TODO(CarolinePascal) : add channels selection
    audio_decoder = torchcodec.decoders.AudioDecoder(audio_path)

    audio_chunks = []
    for ts in timestamps:
        current_audio_chunk = audio_decoder.get_samples_played_in_range(
            start_seconds=ts - duration, stop_seconds=ts
        )

        if log_loaded_timestamps:
            logging.info(
                f"audio chunk loaded at starting timestamp={current_audio_chunk.pts_seconds:.4f} with duration={current_audio_chunk.duration_seconds:.4f}"
            )

        audio_chunks.append(current_audio_chunk.data)

    audio_chunks = torch.stack(audio_chunks)

    assert len(timestamps) == len(audio_chunks)
    return audio_chunks


def decode_audio_torchaudio(
    audio_path: Path | str,
    timestamps: list[float],
    duration: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    # TODO(CarolinePascal) : add channels selection
    audio_path = str(audio_path)

    reader = torchaudio.io.StreamReader(src=audio_path)
    audio_sample_rate = reader.get_src_stream_info(reader.default_audio_stream).sample_rate

    # TODO(CarolinePascal) : sort timestamps ?
    reader.add_basic_audio_stream(
        frames_per_chunk=int(ceil(duration * audio_sample_rate)),  # Too much is better than not enough
        buffer_chunk_size=-1,  # No dropping frames
        format="fltp",  # Format as float32
    )

    audio_chunks = []
    for ts in timestamps:
        reader.seek(ts - duration)  # Default to closest audio sample
        status = reader.fill_buffer()
        if status != 0:
            logging.warning("Audio stream reached end of recording before decoding desired timestamps.")

        current_audio_chunk = reader.pop_chunks()[0]

        if log_loaded_timestamps:
            logging.info(
                f"audio chunk loaded at starting timestamp={current_audio_chunk['pts']:.4f} with duration={len(current_audio_chunk) / audio_sample_rate:.4f}"
            )

        audio_chunks.append(current_audio_chunk)

    audio_chunks = torch.stack(audio_chunks)

    assert len(timestamps) == len(audio_chunks)
    return audio_chunks


def encode_audio(
    input_path: Path | str,
    output_path: Path | str,
    codec: str = "aac",  # TODO(CarolinePascal) : investigate Fraunhofer FDK AAC (libfdk_aac) codec and and constant (file size control) /variable (quality control) bitrate options
    log_level: str | None = "error",
    overwrite: bool = False,
) -> None:
    """Encodes an audio file using ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_args = OrderedDict(
        [
            ("-i", str(input_path)),
            ("-acodec", codec),
        ]
    )

    if log_level is not None:
        ffmpeg_args["-loglevel"] = str(log_level)

    ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
    if overwrite:
        ffmpeg_args.append("-y")

    ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(output_path)]

    # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
    subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

    if not output_path.exists():
        raise OSError(
            f"Audio encoding did not work. File not found: {output_path}. "
            f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
        )


def get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }
