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

"""
This file contains utilities for recording audio from a microhone.
"""

import argparse
import logging
import shutil
import time
from multiprocessing import Event as process_Event
from multiprocessing import JoinableQueue as process_Queue
from multiprocessing import Process
from os import getcwd
from pathlib import Path
from queue import Empty
from queue import Queue as thread_Queue
from threading import Event, Thread
from threading import Event as thread_Event

import numpy as np
import soundfile as sf

from lerobot.common.robot_devices.microphones.configs import MicrophoneConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceAlreadyRecordingError,
    RobotDeviceNotConnectedError,
    RobotDeviceNotRecordingError,
)
from lerobot.common.utils.utils import capture_timestamp_utc


def find_microphones(raise_when_empty=False, mock=False) -> list[dict]:
    """
    Finds and lists all microphones compatible with sounddevice (and the underlying PortAudio library).
    Most microphones and sound cards are compatible, across all OS (Linux, Mac, Windows).
    """
    microphones = []

    if mock:
        import tests.microphones.mock_sounddevice as sd
    else:
        import sounddevice as sd

    devices = sd.query_devices()
    for device in devices:
        if device["max_input_channels"] > 0:
            microphones.append(
                {
                    "index": device["index"],
                    "name": device["name"],
                }
            )

    if raise_when_empty and len(microphones) == 0:
        raise OSError(
            "Not a single microphone was detected. Try re-plugging the microphone or check the microphone settings."
        )

    return microphones


def record_audio_from_microphones(
    output_dir: Path, microphone_ids: list[int] | None = None, record_time_s: float = 2.0
):
    """
    Records audio from all the channels of the specified microphones for the specified duration.
    If no microphone ids are provided, all available microphones will be used.
    """

    if microphone_ids is None or len(microphone_ids) == 0:
        microphones = find_microphones()
        microphone_ids = [m["index"] for m in microphones]

    microphones = []
    for microphone_id in microphone_ids:
        config = MicrophoneConfig(microphone_index=microphone_id)
        microphone = Microphone(config)
        microphone.connect()
        print(
            f"Recording audio from microphone {microphone_id} for {record_time_s} seconds at {microphone.sample_rate} Hz."
        )
        microphones.append(microphone)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(
            output_dir,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving audio to {output_dir}")

    for microphone in microphones:
        microphone.start_recording(getcwd() / output_dir / f"microphone_{microphone.microphone_index}.wav")

    time.sleep(record_time_s)

    for microphone in microphones:
        microphone.stop_recording()

    # Remark : recording may be resumed here if needed

    for microphone in microphones:
        microphone.disconnect()

    print(f"Images have been saved to {output_dir}")


class Microphone:
    """
    The Microphone class handles all microphones compatible with sounddevice (and the underlying PortAudio library). Most microphones and sound cards are compatible, across all OS (Linux, Mac, Windows).

    A Microphone instance requires the sounddevice index of the microphone, which may be obtained using `python -m sounddevice`. It also requires the recording sample rate as well as the list of recorded channels.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.microphones.configs import MicrophoneConfig

    config = MicrophoneConfig(microphone_index=0, sample_rate=16000, channels=[1])
    microphone = Microphone(config)

    microphone.connect()
    microphone.start_recording("some/output/file.wav")
    ...
    audio_readings = microphone.read()  #Gets all recorded audio data since the last read or since the beginning of the recording
    ...
    microphone.stop_recording()
    microphone.disconnect()
    ```
    """

    def __init__(self, config: MicrophoneConfig):
        self.config = config
        self.microphone_index = config.microphone_index

        # Store the recording sample rate and channels
        self.sample_rate = config.sample_rate
        self.channels = config.channels

        self.mock = config.mock

        # Input audio stream
        self.stream = None

        # Thread/Process-safe concurrent queue to store the recorded/read audio
        self.record_queue = None
        self.read_queue = None

        # Thread/Process to handle data reading and file writing in a separate thread/process (safely)
        self.record_thread = None
        self.record_stop_event = None

        self.logs = {}
        self.is_connected = False
        self.is_recording = False
        self.is_writing = False

    def connect(self) -> None:
        """
        Connects the microphone and checks if the requested acquisition parameters are compatible with the microphone.
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"Microphone {self.microphone_index} is already connected."
            )

        if self.mock:
            import tests.microphones.mock_sounddevice as sd
        else:
            import sounddevice as sd

        # Check if the provided microphone index does match an input device
        is_index_input = sd.query_devices(self.microphone_index)["max_input_channels"] > 0

        if not is_index_input:
            microphones_info = find_microphones()
            available_microphones = [m["index"] for m in microphones_info]
            raise OSError(
                f"Microphone index {self.microphone_index} does not match an input device (microphone). Available input devices : {available_microphones}"
            )

        # Check if provided recording parameters are compatible with the microphone
        actual_microphone = sd.query_devices(self.microphone_index)

        if self.sample_rate is not None:
            if self.sample_rate > actual_microphone["default_samplerate"]:
                raise OSError(
                    f"Provided sample rate {self.sample_rate} is higher than the sample rate of the microphone {actual_microphone['default_samplerate']}."
                )
            elif self.sample_rate < actual_microphone["default_samplerate"]:
                logging.warning(
                    "Provided sample rate is lower than the sample rate of the microphone. Performance may be impacted."
                )
        else:
            self.sample_rate = int(actual_microphone["default_samplerate"])

        if self.channels is not None:
            if any(c > actual_microphone["max_input_channels"] for c in self.channels):
                raise OSError(
                    f"Some of the provided channels {self.channels} are outside the maximum channel range of the microphone {actual_microphone['max_input_channels']}."
                )
        else:
            self.channels = np.arange(1, actual_microphone["max_input_channels"] + 1)

        # Get channels index instead of number for slicing
        self.channels = np.array(self.channels) - 1

        # Create the audio stream
        self.stream = sd.InputStream(
            device=self.microphone_index,
            samplerate=self.sample_rate,
            channels=max(self.channels) + 1,
            dtype="float32",
            callback=self._audio_callback,
        )
        # Remark : the blocksize parameter could be passed to the stream to ensure that audio_callback always receive same length buffers.
        # However, this may lead to additional latency. We thus stick to blocksize=0 which means that audio_callback will receive varying length buffers, but with no additional latency.

        self.is_connected = True

    def _audio_callback(self, indata, frames, time, status) -> None:
        """
        Low-level sounddevice callback.
        """
        if status:
            logging.warning(status)
        # Slicing makes copy unnecessary
        # Two separate queues are necessary because .get() also pops the data from the queue
        if self.is_writing:
            self.record_queue.put(indata[:, self.channels])
        self.read_queue.put(indata[:, self.channels])

    @staticmethod
    def _record_loop(queue, event: Event, sample_rate: int, channels: list[int], output_file: Path) -> None:
        """
        Thread/Process-safe loop to write audio data into a file.
        """
        # Can only be run on a single process/thread for file writing safety
        with sf.SoundFile(
            output_file,
            mode="x",
            samplerate=sample_rate,
            channels=max(channels) + 1,
            subtype=sf.default_subtype(output_file.suffix[1:]),
        ) as file:
            while not event.is_set():
                try:
                    file.write(
                        queue.get(timeout=0.02)
                    )  # Timeout set as twice the usual sounddevice buffer size
                    queue.task_done()
                except Empty:
                    continue

    def _read(self) -> np.ndarray:
        """
        Thread/Process-safe callback to read available audio data
        """
        audio_readings = np.empty((0, len(self.channels)))

        while True:
            try:
                audio_readings = np.concatenate((audio_readings, self.read_queue.get_nowait()), axis=0)
            except Empty:
                break

        self.read_queue = thread_Queue()

        return audio_readings

    def read(self) -> np.ndarray:
        """
        Reads the last audio chunk recorded by the microphone, e.g. all samples recorded since the last read or since the beginning of the recording.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise RobotDeviceNotRecordingError(f"Microphone {self.microphone_index} is not recording.")

        start_time = time.perf_counter()

        audio_readings = self._read()

        # log the number of seconds it took to read the audio chunk
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the audio chunk was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return audio_readings

    def start_recording(self, output_file: str | None = None, multiprocessing: bool | None = False) -> None:
        """
        Starts the recording of the microphone. If output_file is provided, the audio will be written to this file.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if self.is_recording:
            raise RobotDeviceAlreadyRecordingError(
                f"Microphone {self.microphone_index} is already recording."
            )

        # Reset queues
        self.read_queue = thread_Queue()
        if multiprocessing:
            self.record_queue = process_Queue()
        else:
            self.record_queue = thread_Queue()

        # Write recordings into a file if output_file is provided
        if output_file is not None:
            output_file = Path(output_file)
            if output_file.exists():
                output_file.unlink()

            if multiprocessing:
                self.record_stop_event = process_Event()
                self.record_thread = Process(
                    target=Microphone._record_loop,
                    args=(
                        self.record_queue,
                        self.record_stop_event,
                        self.sample_rate,
                        self.channels,
                        output_file,
                    ),
                )
            else:
                self.record_stop_event = thread_Event()
                self.record_thread = Thread(
                    target=Microphone._record_loop,
                    args=(
                        self.record_queue,
                        self.record_stop_event,
                        self.sample_rate,
                        self.channels,
                        output_file,
                    ),
                )
            self.record_thread.daemon = True
            self.record_thread.start()

            self.is_writing = True

        self.is_recording = True
        self.stream.start()

    def stop_recording(self) -> None:
        """
        Stops the recording of the microphones.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise RobotDeviceNotRecordingError(f"Microphone {self.microphone_index} is not recording.")

        if self.stream.active:
            self.stream.stop()  # Wait for all buffers to be processed
            # Remark : stream.abort() flushes the buffers !
            self.is_recording = False

        if self.record_thread is not None:
            self.record_queue.join()
            self.record_stop_event.set()
            self.record_thread.join()
            self.record_thread = None
            self.record_stop_event = None
            self.is_writing = False

    def disconnect(self) -> None:
        """
        Disconnects the microphone and stops the recording.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")

        if self.is_recording:
            self.stop_recording()

        self.stream.close()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Records audio using `Microphone` for all microphones connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--microphone-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of microphones indices used to instantiate the `Microphone`. If not provided, find and use all available microphones indices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/audio_from_microphones",
        help="Set directory to save an audio snippet for each microphone.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the audio. By default, 4 seconds.",
    )
    args = parser.parse_args()
    record_audio_from_microphones(**vars(args))
