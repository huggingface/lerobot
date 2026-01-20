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
from multiprocessing import (
    Event as process_Event,
    JoinableQueue as process_Queue,
    Process,
)
from pathlib import Path
from queue import Empty
from threading import Barrier, Event, Event as thread_Event, Thread
from typing import Any

import numpy as np
from soundfile import SoundFile

from lerobot.microphones.portaudio.interface_sounddevice_sdk import ISounddeviceSDK, SounddeviceSDKAdapter
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceAlreadyRecordingError,
    DeviceNotConnectedError,
    DeviceNotRecordingError,
)
from lerobot.utils.shared_array import SharedArray

from ..microphone import Microphone
from .configuration_portaudio import PortAudioMicrophoneConfig

logger = logging.getLogger(__name__)


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

    def __init__(self, config: PortAudioMicrophoneConfig, sounddevice_sdk: ISounddeviceSDK = None):
        """
        Initializes the PortAudioMicrophone instance.

        Args:
            config: The configuration settings for the microphone.
        """
        super().__init__(config)

        if sounddevice_sdk is None:
            self.sounddevice_sdk = SounddeviceSDKAdapter()
        else:
            self.sounddevice_sdk = sounddevice_sdk

        # Microphone index
        self.microphone_index = config.microphone_index

        # Input audio recording process and events
        self.record_process = None
        self.record_stop_event = process_Event()
        self.record_start_event = process_Event()
        self.record_close_event = process_Event()
        self.record_is_started_event = process_Event()
        self.audio_callback_start_event = process_Event()

        # Process-safe concurrent queue to send audio from the recording process to the writing process/thread
        self.write_queue = process_Queue()

        # SharedArray to store audio from the recording process.
        self.read_shared_array = None
        self.local_read_shared_array = None
        # Thread/Process to handle data writing in a separate thread/process (safely)
        self.write_thread = None
        self.write_stop_event = None
        self.write_is_started_event = None

        self.logs = {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.microphone_index})"

    @property
    def is_connected(self) -> bool:
        return self.record_process is not None and self.record_process.is_alive()

    @property
    def is_recording(self) -> bool:
        return self.record_is_started_event.is_set()

    @property
    def is_writing(self) -> bool:
        return self.write_thread is not None and self.write_is_started_event.is_set()

    @staticmethod
    def find_microphones(
        device: int | str | None = None, sounddevice_sdk: ISounddeviceSDK = None
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Detects available microphones connected to the system.

        Args:
            device: The device to find microphones for. If None, all microphones are found.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected microphone : index, name, sample rate, channels.
        """

        if sounddevice_sdk is None:
            sounddevice_sdk = SounddeviceSDKAdapter()

        found_microphones_info = []

        devices = sounddevice_sdk.query_devices()
        for d in devices:
            if d["max_input_channels"] > 0:
                microphone_info = {
                    "index": d["index"],
                    "name": d["name"],
                    "sample_rate": int(d["default_samplerate"]),
                    "channels": np.arange(1, d["max_input_channels"] + 1),
                }

                if device is None or (
                    (isinstance(device, int) and d["index"] == device)
                    or (isinstance(device, str) and d["name"] == device)
                ):
                    found_microphones_info.append(microphone_info)

        if device is not None:
            if len(found_microphones_info) == 0:
                raise RuntimeError(f"No microphone found for device {device}")
            else:
                return found_microphones_info[0]

        if len(found_microphones_info) == 0:
            logger.warning("No microphone found !")

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

        try:
            PortAudioMicrophone.find_microphones(self.microphone_index, self.sounddevice_sdk)
        except RuntimeError as e:
            raise RuntimeError(
                f"{e}. Available microphones: {PortAudioMicrophone.find_microphones(sounddevice_sdk=self.sounddevice_sdk)}"
            ) from e

    def _validate_sample_rate(self) -> None:
        """Validates the sample rate against the actual microphone's default sample rate."""

        actual_sample_rate = PortAudioMicrophone.find_microphones(
            self.microphone_index, self.sounddevice_sdk
        )["sample_rate"]

        if self.sample_rate is not None:
            try:
                self.sample_rate = int(self.sample_rate)
            except ValueError as e:
                raise RuntimeError(
                    f"Cannot convert the provided sample rate ({self.sample_rate} Hz) to an integer."
                ) from e

            if self.sample_rate > actual_sample_rate or self.sample_rate < 1000:
                raise RuntimeError(
                    f"Provided sample rate {self.sample_rate} is either too low or too high compared to the sample rate of the microphone {actual_sample_rate}."
                )
            else:
                if self.sample_rate < actual_sample_rate:
                    logger.warning(
                        "Provided sample rate is lower than the sample rate of the microphone. Performance may be impacted."
                    )
        else:
            self.sample_rate = actual_sample_rate

    def _validate_channels(self) -> None:
        """Validates the channels against the actual microphone's maximum input channels."""

        actual_channels = PortAudioMicrophone.find_microphones(self.microphone_index, self.sounddevice_sdk)[
            "channels"
        ]

        if self.channels is not None and len(self.channels) > 0:
            if not all(channel in actual_channels for channel in self.channels):
                raise RuntimeError(
                    f"Some of the provided channels {self.channels} are outside the possible channel range of the microphone {actual_channels}."
                )
        else:
            self.channels = actual_channels

        # Get channels index instead of number for slicing
        self.channels_index = np.array(self.channels) - 1

    def connect(self) -> None:
        """
        Connects the microphone and checks if the requested acquisition parameters are compatible with the microphone.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"Microphone {self.microphone_index} is already connected.")

        self._configure_capture_settings()

        # Create or reset queue and shared array
        self.read_shared_array = SharedArray(
            shape=(self.sample_rate * 10, len(self.channels)),
            dtype=np.dtype("float32"),
        )
        self.local_read_shared_array = self.read_shared_array.get_local_array()
        self.write_queue = process_Queue()

        # Reset events
        self.record_start_event.clear()
        self.record_stop_event.clear()
        self.record_close_event.clear()
        self.record_is_started_event.clear()
        self.audio_callback_start_event.clear()

        # Create and start an audio input stream with a recording callback
        # Remark: this is done in a separate process so that audio recording is not impacted by the main thread CPU usage, especially the precise_sleep function.
        process_init_event = process_Event()
        self.record_process = Process(
            target=self._record_process,
            args=(
                self.microphone_index,
                self.sample_rate,
                self.channels,
                process_init_event,
                self.record_start_event,
                self.record_stop_event,
                self.record_close_event,
                self.record_is_started_event,
                self.audio_callback_start_event,
                self.write_queue,
                self.read_shared_array,
                self.sounddevice_sdk,
            ),
        )
        self.record_process.daemon = True
        self.record_process.start()

        is_init = process_init_event.wait(
            timeout=5.0
        )  # Wait for the recording process to be started, and to potentially raise an error on failure.
        if not self.is_connected or not is_init:
            raise RuntimeError(f"Error connecting microphone {self.microphone_index}.")

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """
        Disconnects the microphone and stops the recording.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")

        if self.is_recording:
            self.stop_recording()

        self.record_close_event.set()
        self.read_shared_array.delete()
        self.write_queue.close()
        self.record_process.join()

        if self.is_connected:
            raise RuntimeError(f"Error disconnecting microphone {self.microphone_index}.")

        logger.info(f"{self} disconnected.")

    def _read(self) -> np.ndarray:
        """
        Thread/Process-safe callback to read available audio data
        """
        return self.read_shared_array.read(self.local_read_shared_array, flush=True)

    def read(self) -> np.ndarray:
        """
        Reads the last audio chunk recorded by the microphone, e.g. all samples recorded since the last read or since the beginning of the recording.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise RuntimeError(f"Microphone {self.microphone_index} is not recording.")

        start_time = time.perf_counter()

        audio_readings = self._read()

        # log the number of seconds it took to read the audio chunk
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the audio chunk was received
        self.logs["timestamp_utc"] = time.perf_counter()

        return audio_readings

    @staticmethod
    def _record_process(
        microphone_index,
        sample_rate,
        channels,
        process_init_event,
        record_start_event,
        record_stop_event,
        record_close_event,
        record_is_started_event,
        audio_callback_start_event,
        write_queue,
        read_shared_array,
        sounddevice_sdk,
    ) -> None:
        """
        Process callback used to create an unpickable sounddevice audio input stream with a recording callback and start, stop and close it based on multiprocessing events.
        """

        channels_index = np.array(channels) - 1
        local_read_shared_array = read_shared_array.get_local_array()

        def audio_callback(indata, frames, timestamp, status) -> None:
            """
            Low-level sounddevice callback.
            """
            if status:
                logger.warning(status)
            if audio_callback_start_event.is_set():
                write_queue.put_nowait(indata[:, channels_index])
                read_shared_array.write(local_read_shared_array, indata[:, channels_index])

        # Create the audio stream
        # InputStream must be instantiated in the process as it is not pickable.
        stream = sounddevice_sdk.InputStream(
            device=microphone_index,
            samplerate=sample_rate,
            channels=max(channels),
            dtype="float32",
            blocksize=0,  # Varying input buffer length, but no additional latency
            latency="low",  # Low latency mode (not enabled by default !)
            # never_drop_input=True, # Disabled as it generates an error for some devices
            callback=audio_callback,
        )
        process_init_event.set()

        while True:
            start_flag = record_start_event.wait(timeout=0.1)
            if record_close_event.is_set():
                break
            elif not start_flag:
                continue
            stream.start()
            record_is_started_event.set()
            record_stop_event.wait()
            stream.stop()  # stream.stop() waits for all buffers to be processed, stream.abort() flushes the buffers !
            record_is_started_event.clear()
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
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if self.is_recording:
            raise DeviceAlreadyRecordingError(f"Microphone {self.microphone_index} is already recording.")

        # Reset queue and shared memory
        self.read_shared_array.reset()
        self._clear_queue(self.write_queue)

        # Reset stop event
        self.record_stop_event.clear()

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
                self.write_stop_event = process_Event()
                self.write_is_started_event = process_Event()
                self.write_thread = Process(
                    target=PortAudioMicrophone._write_loop,
                    args=(
                        self.write_queue,
                        self.write_stop_event,
                        self.write_is_started_event,
                        self.sample_rate,
                        self.channels,
                        output_file,
                    ),
                )
            else:
                self.write_stop_event = thread_Event()
                self.write_is_started_event = thread_Event()
                self.write_thread = Thread(
                    target=PortAudioMicrophone._write_loop,
                    args=(
                        self.write_queue,
                        self.write_stop_event,
                        self.write_is_started_event,
                        self.sample_rate,
                        self.channels,
                        output_file,
                    ),
                )
            self.write_thread.daemon = True
            self.write_thread.start()
            self.write_is_started_event.wait()  # Wait for the writing thread/process to be started.

        self.record_start_event.set()  # Start the input audio stream process
        self.record_is_started_event.wait()  # Wait for the input audio stream process to be actually started

        if barrier is not None:
            barrier.wait()  # Wait for multiple input audio streams to be started at the same time

        self.audio_callback_start_event.set()

        if not self.is_recording:
            raise RuntimeError(f"Error starting recording for microphone {self.microphone_index}.")
        if output_file is not None and not self.is_writing:
            raise RuntimeError(f"Error starting writing for microphone {self.microphone_index}.")

    def stop_recording(self) -> None:
        """
        Stops the recording of the microphones.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Microphone {self.microphone_index} is not connected.")
        if not self.is_recording:
            raise DeviceNotRecordingError(f"Microphone {self.microphone_index} is not recording.")

        self.audio_callback_start_event.clear()
        self.record_start_event.clear()  # Ensures the audio stream is not started again !
        self.record_stop_event.set()

        # Wait for the stream to be stopped (might lead to race condition if the stream is not properly stopped on array reset and queue clearing)
        timeout = 1.0
        while self.is_recording and timeout > 0:
            time.sleep(0.01)
            timeout -= 0.01

        self.read_shared_array.reset()
        self._clear_queue(self.write_queue, join_queue=True)

        if self.is_writing:
            self.write_stop_event.set()
            self.write_thread.join()

        if self.is_recording:
            raise RuntimeError(f"Error stopping recording for microphone {self.microphone_index}.")
        if self.is_writing:
            raise RuntimeError(f"Error stopping writing for microphone {self.microphone_index}.")

    @staticmethod
    def _write_loop(
        queue,
        write_stop_event: Event,
        write_is_started_event: Event,
        sample_rate: int,
        channels: list[int],
        output_file: Path,
    ) -> None:
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
            write_is_started_event.set()
            while not write_stop_event.is_set():
                try:
                    file.write(
                        queue.get(timeout=0.005)
                    )  # Timeout set as the usual sounddevice buffer size. get_nowait is not possible here as it saturates the thread.
                    queue.task_done()
                except Empty:
                    continue
        write_is_started_event.clear()

    def __del__(self) -> None:
        if self.is_connected:
            self.disconnect()

    @staticmethod
    def _clear_queue(queue, join_queue: bool = False):
        """
        Clears the queue by getting all items until it is empty. The longer the queue, the longer it takes to clear it.
        """
        try:
            while True:
                queue.get_nowait()
                queue.task_done()
        except Empty:
            if join_queue:
                queue.join()
            return
