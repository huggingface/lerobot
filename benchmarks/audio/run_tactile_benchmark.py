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
from pathlib import Path

import numpy as np
import soundfile as sf

from lerobot.microphones.configs import MicrophoneConfig
from lerobot.microphones.touchlab import TouchLabSensorConfig
from lerobot.microphones.utils import (
    async_microphones_start_recording,
    async_microphones_stop_recording,
    make_microphones_from_configs,
)
from lerobot.utils.robot_utils import (
    precise_sleep,
)


def main(
    sensors_configs: dict[str, MicrophoneConfig],
    multiprocessing: bool = False,
):
    recording_dir = Path("outputs/tactile_benchmark")
    recording_dir.mkdir(parents=True, exist_ok=True)

    # Create microphones
    sensors = make_microphones_from_configs(sensors_configs)

    # Connect microphones
    for sensor in sensors.values():
        sensor.connect()

    # Create audio chunks
    data_chunks = {}
    for sensor_key in sensors:
        data_chunks.update({sensor_key: []})

    # Start recording
    async_microphones_start_recording(
        sensors,
        output_files=[recording_dir / f"{sensor_key}_recording.wav" for sensor_key in sensors],
        multiprocessing=multiprocessing,
    )

    # Record audio chunks
    precise_sleep(10.0)

    for sensor_key, sensor in sensors.items():
        data_chunk = sensor.read()
        print(f"{sensor_key} - samples {data_chunk.shape[0]}")
        data_chunks[sensor_key].append(data_chunk)

    # Stop recording
    async_microphones_stop_recording(sensors)

    for sensor_key in sensors:
        data_chunks[sensor_key] = np.concatenate(data_chunks[sensor_key], axis=0)

    # Disconnect microphones
    for sensor in sensors.values():
        sensor.disconnect()

    for sensor_key in sensors:
        data, sample_rate = sf.read(recording_dir / f"{sensor_key}_recording.wav")
        print(f"{sensor_key} - samples {data.shape[0]}")
        print(f"{sensor_key} - sample rate {sample_rate}")
        print(f"{sensor_key} - data {data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensors_ports",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--sensors_baud_rate",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--sensors_sample_rate",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--sensors_channels",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
    )

    args = vars(parser.parse_args())

    args["sensors_configs"] = {}
    for port, baud_rate, sample_rate, channels in zip(
        args["sensors_ports"],
        args["sensors_baud_rate"],
        args["sensors_sample_rate"],
        args["sensors_channels"],
        strict=False,
    ):
        if isinstance(channels, int):
            channels = [channels]
        sensor_config = TouchLabSensorConfig(
            sensor_port=port,
            baud_rate=baud_rate,
            sample_rate=sample_rate,
            channels=channels,
        )
        args["sensors_configs"].update({f"sensor_{port}": sensor_config})
    args.pop("sensors_ports")
    args.pop("sensors_baud_rate")
    args.pop("sensors_sample_rate")
    args.pop("sensors_channels")

    main(**args)
