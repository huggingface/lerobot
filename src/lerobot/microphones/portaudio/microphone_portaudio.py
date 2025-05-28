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

"""
Provides the PortAudioMicrophone class for capturing audio from microphones using the PortAudio library through the sounddevice Python package.
"""

import logging
import time
from multiprocessing import Event as process_Event, JoinableQueue as process_Queue, Process
from pathlib import Path
from queue import Empty, Queue as thread_Queue
from threading import Event, Event as thread_Event, Thread
from typing import Any

import numpy as np
import sounddevice as sd
from soundfile import SoundFile

from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceAlreadyRecordingError,
    DeviceNotConnectedError,
    DeviceNotRecordingError,
)
from lerobot.utils.utils import capture_timestamp_utc

from ..microphone import Microphone
from .configuration_portaudio import PortAudioMicrophoneConfig


class PortAudioMicrophone(Microphone):
    """
    The PortAudioMicrophone class handles all microphones compatible with sounddevice (and the underlying PortAudio library). Most microphones and sound cards are compatible, across all OS (Linux, Mac, Windows).

    A PortAudioMicrophone instance requires the sounddevice index of the microphone, which may be obtained using `python -m sounddevice`. It also requires the recording sample rate as well as the list of recorded channels.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.microphones.configs import PortAudioMicrophoneConfig

    config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=16000, channels=[1])
    microphone = PortAudioMicrophone(config)

    microphone.connect()
    microphone.start_recording("some/output/file.wav")
    ...
    audio_readings = (
        microphone.read()
    )  # Gets all recorded audio data since the last read or since the beginning of the recording
    ...
    microphone.stop_recording()
    microphone.disconnect()
    ```
    """

    def __init__(self, config: PortAudioMicrophoneConfig):
        """
        Initializes the PortAudioMicrophone instance.

        Args:
            config: The configuration settings for the microphone.
        """
        super().__init__(config)

        # Microphone index
        self.microphone_index = config.microphone_index

        # Input audio stream
        self.stream = None

        # Thread/Process-safe concurrent queue to store the recorded/read audio
        self.record_queue = None
        self.read_queue = None

        # Thread/Process to handle data reading and file writing in a separate thread/process (safely)
        self.record_thread = None
        self.record_stop_event = None

        self.logs = {}
        self._is_connected = False
        self.is_recording = False
        self.is_writing = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_microphones() -> list[dict[str, Any]]:
        """
        Detects available microphones connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected microphone : index, name, sample rate, channels.
        """
        found_microphones_info = []

        devices = sd.query_devices()
        for device in devices:
            if device["max_input_channels"] > 0:
                microphone_info = {
                    "index": device["index"],
                    "name": device["name"],
                    "sample_rate": int(device["default_samplerate"]),
                    "channels": list(range(1, device["max_input_channels"] + 1)),
                }
                found_microphones_info.append(microphone_info)

        return found_microphones_info

    def connect(self) -> None:
        """
        Connects the microphone and checks if the requested acquisition parameters are compatible with the microphone.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"Microphone {self.microphone_index} is already connected.")

        # Check if the provided microphone index does match an input device
        is_index_input = sd.query_devices(self.microphone_index)["max_input_channels"] > 0

        if not is_index_input:
            found_microphones_info = self.find_microphones()
            available_microphones = {m["name"]: m["index"] for m in found_microphones_info}
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

        if self.channels is not None and len(self.channels) > 0:
            if any(c > actual_microphone["max_input_channels"] for c in self.channels):
                raise OSError(
                    f"Some of the provided channels {self.channels} are outside the maximum channel range of the microphone {actual_microphone['max_input_channels']}."
                )
        else:
            self.channels = np.arange(1, actual_microphone["max_input_channels"] + 1)

        # Get channels index instead of number for slicing
        self.channels_index = np.array(self.channels) - 1

        # Create the audio stream
        self.stream = sd.InputStream(
            device=self.microphone_index,
            samplerate=self.sample_rate,
            channels=max(self.channels),
            dtype="float32",
            blocksize=0,  # Varying input buffer length, but no additional latency
            latency="low",  # Low latency mode (not enabled by default !)
            # never_drop_input=True, # Disabled as it generates an error for some devices
            callback=self._audio_callback,
        )

        self._is_connected = True

    def _audio_callback(self, indata, frames, timestamp, status) -> None:
        """
        Low-level sounddevice callback.
        """
        if status:
            logging.warning(status)
        # Slicing makes copy unnecessary
        # Two separate queues are necessary because .get() also pops the data from the queue
        # Remark: this also ensures that file-recorded data and chunk-audio data are the same.
        if self.is_writing:
            self.record_queue.put_nowait(indata[:, self.channels_index])
        self.read_queue.put_nowait(indata[:, self.channels_index])

    @staticmethod
    def _record_loop(queue, event: Event, sample_rate: int, channels: list[int], output_file: Path) -> None:
        """
        Thread/Process-safe loop to write audio data into a file.
        """
        # Can only be run on a single process/thread for file writing safety
        with SoundFile(
            output_file,
            mode="w",
            samplerate=sample_rate,
            channels=max(channels),
            format="WAV",
            subtype="FLOAT",  # By default, a much lower quality WAV file is created !
        ) as file:
            while not event.is_set():
                try:
                    file.write(
                        queue.get(timeout=0.005)
                    )  # Timeout set as the usual sounddevice buffer size. get_nowait is not possible here as it saturates the thread.
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
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.stream.active:
            raise RuntimeError(f"Microphone {self.microphone_index} is not recording.")

        start_time = time.perf_counter()

        audio_readings = self._read()

        # log the number of seconds it took to read the audio chunk
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the audio chunk was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return audio_readings

    def start_recording(
        self,
        output_file: str | None = None,
        multiprocessing: bool | None = False,
        overwrite: bool | None = True,
    ) -> None:
        """
        Starts the recording of the microphone. If output_file is provided, the audio will be written to this file.
        Remark: multiprocessing is implemented, but does not work well with sounddevice (launching delays, tricky memory sharing, sounddevice streams are not picklable (even with dill #pathos), etc.).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if self.is_recording:
            raise DeviceAlreadyRecordingError(f"Microphone {self.microphone_index} is already recording.")

        # Reset queues
        self.read_queue = thread_Queue()
        if multiprocessing:
            self.record_queue = process_Queue()
        else:
            self.record_queue = thread_Queue()

        # Write recordings into a file if output_file is provided
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.exists():
                if overwrite:
                    output_file.unlink()
                else:
                    raise FileExistsError(
                        f"Output file {output_file} already exists. Set overwrite to True to overwrite it."
                    )

            if multiprocessing:
                self.record_stop_event = process_Event()
                self.record_thread = Process(
                    target=PortAudioMicrophone._record_loop,
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
                    target=PortAudioMicrophone._record_loop,
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
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise DeviceNotRecordingError(f"Microphone {self.microphone_index} is not recording.")

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

        self.logs["stop_timestamp"] = capture_timestamp_utc()

    def disconnect(self) -> None:
        """
        Disconnects the microphone and stops the recording.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")

        if self.is_recording:
            self.stop_recording()

        self.stream.close()
        self._is_connected = False
