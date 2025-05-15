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
import logging
from pathlib import Path

import av
import torch
import torchaudio
import torchcodec
from numpy import ceil

CHANNELS_LAYOUTS_MAPPING = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "3.1",
    5: "4.1",
    6: "5.1",
    7: "6.1",
    8: "7.1",
    16: "hexadecagonal",
    24: "22.2",
}


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
    audio_sample_rate = audio_decoder.metadata.sample_rate
    audio_channels = audio_decoder.metadata.num_channels
    # TODO(CarolinePascal) : assert ts < total record duration

    audio_chunks = []
    for ts in timestamps:
        current_audio_chunk = audio_decoder.get_samples_played_in_range(
            start_seconds=max(0.0, ts - duration), stop_seconds=ts
        )

        current_audio_chunk_data = current_audio_chunk.data

        # Case where the requested audio chunk starts before the beginning of the audio stream
        if ts - duration < 0:
            # No useful audio sample has been recorded
            if ts < 1 / audio_sample_rate:
                # TODO(CarolinePascal) : add low level white noise instead of zeros ?
                current_audio_chunk_data = torch.zeros(
                    (audio_channels, int(ceil(duration * audio_sample_rate)))
                )
            # At least one useful audio sample has been recorded
            else:
                # Pad the beginning of the audio chunk with zeros
                # TODO(CarolinePascal) : add low level white noise instead of zeros ?
                current_audio_chunk_data = torch.nn.functional.pad(
                    current_audio_chunk_data,
                    (int(ceil((duration - ts) * audio_sample_rate)), 0, 0, 0),  # left, right, top, bottom
                )

        if log_loaded_timestamps:
            logging.info(
                f"audio chunk loaded at timestamp={current_audio_chunk.pts_seconds:.4f} with duration={current_audio_chunk.duration_seconds:.4f}"
            )

        audio_chunks.append(current_audio_chunk_data)

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
    audio_channels = reader.get_src_stream_info(reader.default_audio_stream).num_channels
    # TODO(CarolinePascal) : assert ts < total record duration

    # TODO(CarolinePascal) : sort timestamps ?
    reader.add_basic_audio_stream(
        frames_per_chunk=int(ceil(duration * audio_sample_rate)),  # Too much is better than not enough
        buffer_chunk_size=-1,  # No dropping frames
        format="fltp",  # Format as float32
    )

    audio_chunks = []
    for ts in timestamps:
        reader.seek(max(0.0, ts - duration))  # Default to closest audio sample. Needs to be non-negative !
        status = reader.fill_buffer()
        if status != 0:
            # Should not happen, but just in case
            logging.warning("Audio stream reached end of recording before decoding desired timestamps.")

        current_audio_chunk = reader.pop_chunks()[0]
        current_audio_chunk_data = current_audio_chunk.t()  # Channel first format

        # Case where the requested audio chunk starts before the beginning of the audio stream
        if ts - duration < 0:
            # No useful audio sample has been recorded
            if ts < 1 / audio_sample_rate:
                current_audio_chunk_data = torch.zeros(
                    (audio_channels, int(ceil(duration * audio_sample_rate)))
                )
            # At least one useful audio sample has been recorded
            else:
                # Remove the superfluous last samples of the audio chunk
                current_audio_chunk_data = current_audio_chunk_data[:, : int(ceil(ts * audio_sample_rate))]
                # Pad the beginning of the audio chunk with zeros
                # TODO(CarolinePascal) : add low level white noise instead of zeros ?
                current_audio_chunk_data = torch.nn.functional.pad(
                    current_audio_chunk_data,
                    (int(ceil((duration - ts) * audio_sample_rate)), 0, 0, 0),  # left, right, top, bottom
                )

        if log_loaded_timestamps:
            logging.info(
                f"audio chunk loaded at timestamp={current_audio_chunk['pts']:.4f} with duration={len(current_audio_chunk) / audio_sample_rate:.4f}"
            )

        audio_chunks.append(current_audio_chunk_data)

    audio_chunks = torch.stack(audio_chunks)

    assert len(timestamps) == len(audio_chunks)
    return audio_chunks


def encode_audio(
    input_path: Path | str,
    output_path: Path | str,
    codec: str = "aac",  # TODO(CarolinePascal) : investigate Fraunhofer FDK AAC (libfdk_aac) codec and and constant (file size control) /variable (quality control) bitrate options
    bit_rate: int | None = None,
    sample_rate: int | None = None,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """Encodes an audio file using ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Open input file
    with av.open(str(input_path), "r") as input:
        input_stream = input.streams.audio[0]  # Assuming the first stream is the audio stream to be encoded

        # Define sub-sampling options
        if sample_rate is None:
            sample_rate = input_stream.rate

        # Create and open output file (overwrite by default)
        with av.open(str(output_path), "w") as output:
            output_stream = output.add_stream(
                codec, rate=sample_rate, layout=CHANNELS_LAYOUTS_MAPPING[input_stream.channels]
            )

            if bit_rate is not None:
                output_stream.bit_rate = bit_rate

            # Loop through input WAV packets and encode them
            for input_frame in input.decode(
                input_stream
            ):  # This step handles both demuxing and decoding under the hood
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)

            # Flush the encoder
            packet = output_stream.encode()
            if packet:
                output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not output_path.exists():
        raise OSError(f"Audio encoding did not work. File not found: {output_path}.")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info
