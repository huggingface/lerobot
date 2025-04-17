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

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from lerobot.common.robot_devices.microphones.configs import MicrophoneConfig, PortAudioMicrophoneConfig
from lerobot.common.robot_devices.microphones.portaudio import find_microphones
from lerobot.common.robot_devices.microphones.utils import (
    async_microphones_read,
    async_microphones_start_recording,
    async_microphones_stop_recording,
    make_microphones_from_configs,
)


def main(
    microphones_configs: dict[str, MicrophoneConfig],
    audio_chunks_number: int,
    audio_chunks_duration: float,
    repetitions: int,
    multiprocessing: bool = False,
):
    recording_dir = Path("outputs/audio_benchmark")
    recording_dir.mkdir(parents=True, exist_ok=True)

    # Create microphones
    microphones = make_microphones_from_configs(microphones_configs)

    # Connect microphones
    for microphone in microphones.values():
        microphone.connect()

    all_audio_chunks = []
    for i in range(repetitions):
        # Create audio chunks
        audio_chunks = {}
        for microphone_key in microphones:
            audio_chunks.update({microphone_key: []})

        # Start recording
        async_microphones_start_recording(
            microphones,
            output_files=[
                recording_dir / f"{microphone_key}_recording_{i}.wav" for microphone_key in microphones
            ],
            multiprocessing=True,
        )

        # Record audio chunks
        for j in range(audio_chunks_number):
            time.sleep(audio_chunks_duration)

            audio_readings = async_microphones_read(microphones)
            for microphone_index, reading in audio_readings.items():
                print(f"{microphone_index} - repetition {i} - chunk {j} - samples {reading.shape[0]}")
                audio_chunks[microphone_index].append(reading)

        # Stop recording
        async_microphones_stop_recording(microphones)

        for microphone_key in microphones:
            audio_chunks[microphone_key] = np.concatenate(audio_chunks[microphone_key], axis=0)

        all_audio_chunks.append(audio_chunks)

    # Disconnect microphones
    for microphone in microphones.values():
        microphone.disconnect()

    # Compute statistics
    cmap = plt.get_cmap("tab10")
    _, ax = plt.subplots(nrows=repetitions, ncols=len(microphones))
    chunk_length = np.zeros((repetitions, len(microphones)))
    record_length = np.zeros((repetitions, len(microphones)))
    for i in range(repetitions):
        for j, (microphone_key, microphone) in enumerate(microphones.items()):
            # Get recorded audio chunks
            recorded_audio_chunks = all_audio_chunks[i][microphone_key]

            # Load recorded file
            recorded_data, _ = sf.read(recording_dir / f"{microphone_key}_recording_{i}.wav")
            if len(recorded_data.shape) == 1:
                recorded_data = np.expand_dims(recorded_data, axis=1)

            record_length[i, j] = recorded_data.shape[0]
            chunk_length[i, j] = recorded_audio_chunks.shape[0]

            for k, (chunk_data, record_data) in enumerate(
                zip(recorded_audio_chunks.T, recorded_data.T, strict=False)
            ):
                # Plot audio chunks and recorded data
                ax[i, j].plot(
                    np.arange(0, len(chunk_data)) / microphone.sample_rate,
                    chunk_data,
                    label=f"audio chunks - channel {k}",
                    color=cmap(2 * k),
                )
                ax[i, j].plot(
                    np.arange(0, len(record_data)) / microphone.sample_rate,
                    record_data,
                    label=f"recorded data - channel {k}",
                    linestyle="dashed",
                    color=cmap(2 * k + 1),
                )

                # Plot absolute difference (errors should be located at the end of the recordings)
                if recorded_data.shape[0] - recorded_audio_chunks.shape[0] > 0:
                    chunk_data = np.append(
                        chunk_data, np.zeros(int(recorded_data.shape[0] - recorded_audio_chunks.shape[0]))
                    )
                else:
                    record_data = np.append(
                        record_data, np.zeros(int(-recorded_data.shape[0] + recorded_audio_chunks.shape[0]))
                    )
                ax[i, j].plot(
                    np.arange(0, len(record_data)) / microphone.sample_rate,
                    np.abs(chunk_data - record_data),
                    label=f"differences - channel {k}",
                    color="red",
                    linestyle="dotted",
                )
                ax[i, j].set_title(f"{microphone_key} - repetition {i}")
            ax[i, j].legend()

    plt.show()

    # Print statistics
    differences = record_length - chunk_length
    for i, microphone in enumerate(microphones.values()):
        print(
            f"Average recorded duration for {microphone_key} : {np.mean(record_length[:, i]) / microphone.sample_rate:.3f} seconds"
        )
        print(
            f"Average chunk duration for {microphone_key} : {np.mean(chunk_length[:, i]) / microphone.sample_rate:.3f} seconds"
        )
        print(f"Average difference for {microphone_key} : {np.mean(differences[:, i]):.3f} samples")
        print(
            f"Average difference for {microphone_key} : {np.mean(differences[:, i]) / microphone.sample_rate:.3f} seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--microphones_indices",
        type=int,
        nargs="+",
        default=[microphone["index"] for microphone in find_microphones()],
    )
    parser.add_argument(
        "--microphones_sample_rate",
        type=float,
        nargs="+",
        default=[None] * len(find_microphones()),
    )
    parser.add_argument(
        "--microphones_channels",
        type=int,
        nargs="+",
        default=[None] * len(find_microphones()),
    )
    parser.add_argument("--audio_chunks_number", type=int, default=2)
    parser.add_argument(
        "--audio_chunks_duration",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
    )

    args = vars(parser.parse_args())

    args["microphones_configs"] = {}
    for index, sample_rate, channels in zip(
        args["microphones_indices"],
        args["microphones_sample_rate"],
        args["microphones_channels"],
        strict=False,
    ):
        microphone_config = PortAudioMicrophoneConfig(
            microphone_index=index,
            sample_rate=sample_rate,
            channels=channels,
        )
        args["microphones_configs"].update({f"microphone_{index}": microphone_config})
    args.pop("microphones_indices")
    args.pop("microphones_sample_rate")
    args.pop("microphones_channels")

    main(**args)
