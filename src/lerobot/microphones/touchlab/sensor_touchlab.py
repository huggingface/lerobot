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
Provides the TouchLabSensor class for capturing tactile data from TouchLab tactile sensors.
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
from serial import Serial
from soundfile import SoundFile

from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceAlreadyRecordingError,
    DeviceNotConnectedError,
    DeviceNotRecordingError,
)
from lerobot.utils.shared_array import SharedArray

from ..microphone import Microphone
from .configuration_touchlab import TouchLabSensorConfig

logger = logging.getLogger(__name__)

MAX_SERIAL_READ_SIZE = 512


class TouchLabSensor(Microphone):
    """
    The TouchLabSensor class handles all TouchLab tactile sensors.

    A TouchLabSensor instance requires the serial port of the tactile sensor, which may be obtained using `python -m lerobot.find_port`. It also requires the recording sample rate as well as the list of recorded channels.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.microphones.configs import TouchLabSensorConfig

    config = TouchLabSensorConfig(sensor_port="/dev/ttyACM0", baud_rate=115200, sample_rate=115, channels=[1])
    microphone = TouchLabSensor(config)

    microphone.connect()
    microphone.start_recording("some/output/file.wav")
    ...
    audio_readings = microphone.read()  # Gets all recorded audio data since the last read or since the beginning of the recording. The longer the period the longer the reading time !
    ...
    microphone.stop_recording()
    microphone.disconnect()
    ```
    """

    def __init__(self, config: TouchLabSensorConfig):
        """ "
        Initializes the TouchLabSensor instance.

        Args:
            config: The configuration settings for the sensor.
        """
        super().__init__(config)

        # Sensor port
        self.sensor_port = config.sensor_port

        # Baud rate
        self.baud_rate = config.baud_rate

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
        return f"{self.__class__.__name__}({self.sensor_port})"

    @property
    def is_connected(self) -> bool:
        """Check if the sensor is currently connected.

        Returns:
            bool: True if the sensor is connected and ready to start recording,
                  False otherwise.
        """
        return self.record_process is not None and self.record_process.is_alive()

    @property
    def is_recording(self) -> bool:
        """Check if the sensor is currently recording.

        Returns:
            bool: True if the sensor is recording, False otherwise.
        """
        return self.record_is_started_event.is_set()

    @property
    def is_writing(self) -> bool:
        """Check if the sensor is currently writing to a file.

        Returns:
            bool: True if the sensor is writing to a file, False otherwise.
        """
        return self.write_thread is not None and self.write_is_started_event.is_set()

    @staticmethod
    def find_microphones() -> list[dict[str, Any]]:
        """Detects available sensors connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected sensor.
        """
        pass

    def connect(self) -> None:
        """
        Establish connection to the sensor.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"Sensor connected to {self.sensor_port} is already connected.")

        # Create or reset queue and shared array
        self.read_shared_array = SharedArray(
            shape=(self.sample_rate * 10, len(self.channels)),
            dtype=np.dtype("int16"),
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
                self.sensor_port,
                self.baud_rate,
                self.channels,
                process_init_event,
                self.record_start_event,
                self.record_stop_event,
                self.record_close_event,
                self.record_is_started_event,
                self.audio_callback_start_event,
                self.write_queue,
                self.read_shared_array,
            ),
        )
        self.record_process.daemon = True
        self.record_process.start()

        is_init = process_init_event.wait(
            timeout=5.0
        )  # Wait for the recording process to be started, and to potentially raise an error on failure.
        if not self.is_connected or not is_init:
            raise RuntimeError(f"Error connecting sensor connected to {self.sensor_port}.")

        logger.info(f"{self} connected.")

    @staticmethod
    def _record_process(
        sensor_port,
        baud_rate,
        channels,
        process_init_event,
        record_start_event,
        record_stop_event,
        record_close_event,
        record_is_started_event,
        audio_callback_start_event,
        write_queue,
        read_shared_array,
    ) -> None:
        channels_index = np.array(channels) - 1
        local_read_shared_array = read_shared_array.get_local_array()

        def tactile_callback(serial_connection):
            """
            Parse the tactile data from the raw input data.
            """
            buffer = serial_connection.readline()

            if audio_callback_start_event.is_set():
                strings = buffer.decode("utf8").split(",")
                num_taxels = len(strings)

                if num_taxels > 0 and num_taxels < MAX_SERIAL_READ_SIZE:  # Make sure we didn't read rubbish
                    indata = np.empty((1, num_taxels))
                    for i in range(num_taxels):
                        indata[0, i] = int(strings[i])

                    write_queue.put_nowait(indata[:, channels_index])
                    read_shared_array.write(local_read_shared_array, indata[:, channels_index])

        process_init_event.set()

        while True:
            start_flag = record_start_event.wait(timeout=0.1)
            if record_close_event.is_set():
                break
            elif not start_flag:
                continue

            with Serial(sensor_port, baud_rate, timeout=0.5) as serial_connection:
                serial_connection.flush()
                record_is_started_event.set()
                while not record_stop_event.is_set():
                    tactile_callback(serial_connection)
                record_is_started_event.clear()
        serial_connection.close()

    def disconnect(self) -> None:
        """
        Disconnect the sensor and release any resources.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Sensor connected to {self.sensor_port} is not connected.")

        if self.is_recording:
            self.stop_recording()

        self.record_close_event.set()
        self.read_shared_array.delete()
        self.write_queue.close()
        self.record_process.join()

        if self.is_connected:
            raise RuntimeError(f"Error disconnecting sensor connected to {self.sensor_port}.")

        logger.info(f"{self} disconnected.")

    def start_recording(
        self,
        output_file: str | Path | None = None,
        multiprocessing: bool | None = False,
        overwrite: bool | None = True,
        barrier: Barrier | None = None,
    ) -> None:
        """
        Start recording tactile data from the sensor.

        Args:
            output_file: Optional path to save the recorded tactile data.
            multiprocessing: If True, enables multiprocessing for recording. Defaults to multithreading otherwise.
            overwrite: If True, overwrites existing files at output_file path.
            barrier: If not None, ensures that multiple sensors start recording at the same time.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Sensor connected to {self.sensor_port} is not connected.")
        if self.is_recording:
            raise DeviceAlreadyRecordingError(f"Sensor connected to {self.sensor_port} is already recording.")

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
                    target=TouchLabSensor._write_loop,
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
                    target=TouchLabSensor._write_loop,
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
            raise RuntimeError(f"Error starting recording for sensor connected to {self.sensor_port}.")
        if output_file is not None and not self.is_writing:
            raise RuntimeError(f"Error starting writing for sensor connected to {self.sensor_port}.")

    def _read(self) -> np.ndarray:
        """
        Thread/Process-safe callback to read available audio data
        """
        return self.read_shared_array.read(self.local_read_shared_array, flush=True)

    def read(self) -> np.ndarray:
        """Capture and return a single audio chunk from the sensor.

        Returns:
            np.ndarray: Captured audio chunk as a numpy array.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Sensor connected to {self.sensor_port} is not connected.")
        if not self.is_recording:
            raise RuntimeError(f"Sensor connected to {self.sensor_port} is not recording.")

        start_time = time.perf_counter()

        tactile_readings = self._read()

        # log the number of seconds it took to read the audio chunk
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the audio chunk was received
        self.logs["timestamp_utc"] = time.perf_counter()

        return tactile_readings

    def _read_loop(self) -> None:
        """Internal loop run by the background thread for asynchronous reading."""

    def stop_recording(self) -> None:
        """Stop recording audio from the sensor."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Sensor connected to {self.sensor_port} is not connected.")
        if not self.is_recording:
            raise DeviceNotRecordingError(f"Sensor connected to {self.sensor_port} is not recording.")

        self.audio_callback_start_event.clear()
        self.record_start_event.clear()  # Ensures the audio stream is not started again !
        self.record_stop_event.set()

        self.read_shared_array.reset()
        self._clear_queue(self.write_queue, join_queue=True)

        if self.is_writing:
            self.write_stop_event.set()
            self.write_thread.join()

        timeout = 1.0
        while self.is_recording and timeout > 0:
            time.sleep(0.01)
            timeout -= 0.01

        if self.is_recording:
            raise RuntimeError(f"Error stopping recording for sensor connected to {self.sensor_port}.")
        if self.is_writing:
            raise RuntimeError(f"Error stopping writing for sensor connected to {self.sensor_port}.")

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
            subtype="PCM_16",  # Subtype for int16 values
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
