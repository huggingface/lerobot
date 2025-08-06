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
from queue import Empty
from threading import Barrier, Event, Event as thread_Event, Thread
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
    audio_readings = microphone.read()  # Gets all recorded audio data since the last read or since the beginning of the recording. The longer the period the longer the reading time !
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

        # Input audio stream process and events
        self.stream_process = None
        self.stream_stop_event = process_Event()
        self.stream_start_event = process_Event()
        self.stream_close_event = process_Event()
        self.stream_is_started_event = process_Event()

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

    def _configure_capture_settings(self) -> None:
        """
        Validates the microphone index, sample rate and channels settings specified in the constructor's config to the un-connected microphone.

        This method actually checks the specified settings and fills the sample rate and channels settings if not specified before attempting to start a PortAudio stream.

        Raises:
            RuntimeError: If one of the specified settings is not compatible with the microphone.
            DeviceAlreadyConnectedError: If the microphone is connected when attempting to configure settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"Cannot configure settings for {self} as it is already connected."
            )

        self._validate_microphone_index()
        self._validate_sample_rate()
        self._validate_channels()

    def _validate_microphone_index(self) -> None:
        """ "Validates the microphone index against available devices by checking if it has at least one input channel."""

        is_index_input = (
            self.microphone_index >= 0 and sd.query_devices(self.microphone_index)["max_input_channels"] > 0
        )

        if not is_index_input:
            found_microphones_info = self.find_microphones()
            available_microphones = {m["name"]: m["index"] for m in found_microphones_info}
            raise RuntimeError(
                f"Microphone index {self.microphone_index} does not match an input device (microphone). Available input devices : {available_microphones}"
            )

    def _validate_sample_rate(self) -> None:
        """Validates the sample rate against the actual microphone's default sample rate."""

        actual_sample_rate = sd.query_devices(self.microphone_index)["default_samplerate"]

        if self.sample_rate is not None:
            if self.sample_rate > actual_sample_rate or self.sample_rate < 1000:
                raise RuntimeError(
                    f"Provided sample rate {self.sample_rate} is either too low or too high compared to the sample rate of the microphone {actual_sample_rate}."
                )
            else:
                if self.sample_rate < actual_sample_rate:
                    logging.warning(
                        "Provided sample rate is lower than the sample rate of the microphone. Performance may be impacted."
                    )
                self.sample_rate = int(self.sample_rate)
        else:
            self.sample_rate = int(actual_sample_rate)

    def _validate_channels(self) -> None:
        """Validates the channels against the actual microphone's maximum input channels."""

        actual_max_microphone_channels = sd.query_devices(self.microphone_index)["max_input_channels"]

        if self.channels is not None and len(self.channels) > 0:
            if any(c > actual_max_microphone_channels or c <= 0 for c in self.channels):
                raise RuntimeError(
                    f"Some of the provided channels {self.channels} are outside the maximum channel range of the microphone {actual_max_microphone_channels}."
                )
        else:
            self.channels = np.arange(1, actual_max_microphone_channels + 1)

        # Get channels index instead of number for slicing
        self.channels_index = np.array(self.channels) - 1

    def connect(self) -> None:
        """
        Connects the microphone and checks if the requested acquisition parameters are compatible with the microphone.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"Microphone {self.microphone_index} is already connected.")

        self._configure_capture_settings()
        # Create queues
        self.record_queue = process_Queue()
        self.read_queue = process_Queue()

        # Reset events
        self.stream_start_event.clear()
        self.stream_stop_event.clear()
        self.stream_close_event.clear()
        self.stream_is_started_event.clear()

        # Create and run audio input stream process
        # Remark: this is done in a separate process so that audio recording is not impacted by the main thread CPU usage, especially the busy_wait function.
        self.stream_process = Process(
            target=self._run_audio_input_stream,
            args=(
                self.microphone_index,
                self.sample_rate,
                self.channels,
                self.stream_start_event,
                self.stream_stop_event,
                self.stream_close_event,
                self.stream_is_started_event,
                self.record_queue,
                self.read_queue,
            ),
        )
        self.stream_process.daemon = True
        self.stream_process.start()

        self._is_connected = True

    def disconnect(self) -> None:
        """
        Disconnects the microphone and stops the recording.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")

        if self.is_recording:
            self.stop_recording()

        if self.stream_process is not None:
            self.stream_close_event.set()
            self.read_queue = None
            self.record_queue = None
            self.stream_process.terminate()  # No time to wait
            self.stream_process = None
        self.is_connected = False

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

    @staticmethod
    def _run_audio_input_stream(
        microphone_index,
        sample_rate,
        channels,
        stream_start_event,
        stream_stop_event,
        stream_close_event,
        is_started_event,
        record_queue,
        read_queue,
    ) -> None:
        """
        Process callback used to create an unpickable sounddevice audio input stream and start, stop and close it based on multiprocessing events.
        """

        channels_index = np.array(channels) - 1

        def audio_callback(indata, frames, timestamp, status) -> None:
            """
            Low-level sounddevice callback.
            """
            if status:
                logging.warning(status)
            # Slicing makes copy unnecessary
            # Two separate queues are necessary because .get() also pops the data from the queue
            # Remark: this also ensures that file-recorded data and chunk-audio data are the same.
            record_queue.put_nowait(indata[:, channels_index])
            read_queue.put_nowait(indata[:, channels_index])

        # Create the audio stream
        stream = sd.InputStream(
            device=microphone_index,
            samplerate=sample_rate,
            channels=max(channels),
            dtype="float32",
            blocksize=0,  # Varying input buffer length, but no additional latency
            latency="low",  # Low latency mode (not enabled by default !)
            # never_drop_input=True, # Disabled as it generates an error for some devices
            callback=audio_callback,
        )

        while True:
            start_flag = stream_start_event.wait(timeout=1.0)
            if stream_close_event.is_set():
                break
            elif not start_flag:
                continue
            stream.start()
            is_started_event.set()
            stream_stop_event.wait()
            stream.stop()  # stream.stop() waits for all buffers to be processed
            # Remark : stream.abort() flushes the buffers !
        stream.close()

    def start_recording(
        self,
        output_file: str | None = None,
        multiprocessing: bool | None = False,
        overwrite: bool | None = True,
        barrier: Barrier | None = None,
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
        self._clear_queue(self.read_queue)
        self._clear_queue(self.record_queue)

        # Reset events - stream_start_event is already cleared here
        self.stream_stop_event.clear()
        self.stream_is_started_event.clear()

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
            self.is_writing = True
            if barrier is None:
                self.record_thread.start()

        self.stream_start_event.set()  # Start the input audio stream process
        self.stream_is_started_event.wait()  # Wait for the input audio stream process to be actually started

        if barrier is not None:
            barrier.wait()  # Wait for multiple input audio streams to be started at the same time

            self._clear_queue(self.read_queue)
            self._clear_queue(self.record_queue)
            if output_file is not None:
                self.record_thread.start()

        self.is_recording = True

    def stop_recording(self) -> None:
        """
        Stops the recording of the microphones.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise DeviceNotRecordingError(f"Microphone {self.microphone_index} is not recording.")

        if self.stream_process is not None:
            self.stream_start_event.clear()  # Ensures the stream is not started again !
            self.stream_stop_event.set()
        self.is_recording = False

        if self.record_thread is not None:
            self.record_queue.join()
            self.record_stop_event.set()
            self.record_thread.join()
            self.record_thread = None
            self.record_stop_event = None
        self.is_writing = False

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
            channels=len(channels),
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

    @staticmethod
    def _clear_queue(queue):
        """
        Clears the queue by getting all items until it is empty. The longer the queue, the longer it takes to clear it.
        """
        try:
            while True:
                queue.get_nowait()
                queue.task_done()
        except Empty:
            return
