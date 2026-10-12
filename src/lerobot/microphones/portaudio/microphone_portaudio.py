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

"""Microphone backend for all devices supported by PortAudio, through the `sounddevice` package."""

import logging
import math
import time
from multiprocessing import (
    Event as process_Event,
    JoinableQueue as process_Queue,
    Process,
    Queue,
    Value,
)
from multiprocessing.queues import JoinableQueue
from pathlib import Path
from queue import Empty
from threading import Barrier, BrokenBarrierError, Event as thread_Event, Thread
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyRecordingError, DeviceNotRecordingError
from lerobot.utils.import_utils import _soundfile_available, require_package
from lerobot.utils.shared_array import SharedArray

if TYPE_CHECKING or _soundfile_available:
    from soundfile import SoundFile
else:
    SoundFile = None

from ..microphone import Microphone
from .configuration_portaudio import PortAudioMicrophoneConfig
from .interface_sounddevice_sdk import ISounddeviceSDK, SounddeviceSDKAdapter

logger = logging.getLogger(__name__)

START_STOP_TIMEOUT_S = 5.0


def _last_sample_time(time_info: Any, frames: int, sample_rate: int) -> float:
    """Capture time of the last sample of a PortAudio callback block, in the `time.perf_counter()` clock.

    PortAudio timestamps are expressed in the stream's own clock. They are converted using the delay
    between the capture of the block's first sample (`inputBufferAdcTime`) and the callback
    invocation (`currentTime`). Some host APIs do not report `inputBufferAdcTime` (it is then 0),
    in which case the last sample is assumed to have been captured when the callback is invoked.
    """
    now = time.perf_counter()
    adc_time = time_info.inputBufferAdcTime
    if adc_time <= 0:
        return now
    last_sample_delay = time_info.currentTime - adc_time - (frames - 1) / sample_rate
    return now - max(0.0, last_sample_delay)


class PortAudioMicrophone(Microphone):
    """Microphone backend for all devices supported by PortAudio, through the `sounddevice` package.

    Most microphones and sound cards are supported, on Linux, macOS and Windows. Devices are identified by
    their `sounddevice` index, listed by `lerobot-find-microphones`.

    Audio is captured by a PortAudio stream living in a separate process, so that recording is not impacted
    by the CPU usage of the main process. Samples are made available to [`~PortAudioMicrophone.read`] and
    [`~PortAudioMicrophone.read_latest`] through a shared-memory ring buffer of `buffer_duration_s` seconds,
    and can optionally be written to a 32-bit float WAV file by a separate thread or process.

    Args:
        config (`PortAudioMicrophoneConfig`):
            The configuration of the microphone.
        sounddevice_sdk (`ISounddeviceSDK`, *optional*):
            The `sounddevice` implementation to use. Defaults to the real library; tests inject a fake one.

    Example:
        ```python
        >>> from lerobot.microphones.portaudio import PortAudioMicrophone, PortAudioMicrophoneConfig
        >>> config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=16000, channels=[1])
        >>> with PortAudioMicrophone(config) as microphone:  # doctest: +SKIP
        ...     microphone.start_recording("episode_0.wav")
        ...     chunk = microphone.read()  # All samples recorded since the previous read
        ...     last_second = microphone.read_latest(1.0)  # The most recent second of audio
        ...     microphone.stop_recording()
        ```
    """

    def __init__(self, config: PortAudioMicrophoneConfig, sounddevice_sdk: ISounddeviceSDK | None = None):
        """Prepare the microphone without connecting to it."""
        super().__init__(config)

        self.config = config
        self.sounddevice_sdk = sounddevice_sdk if sounddevice_sdk is not None else SounddeviceSDKAdapter()

        self.microphone_index = config.microphone_index
        self.buffer_duration_s = config.buffer_duration_s

        # Input audio recording process and events
        self.record_process: Process | None = None
        self.record_stop_event = process_Event()
        self.record_start_event = process_Event()
        self.record_close_event = process_Event()
        self.record_is_started_event = process_Event()
        self.audio_callback_start_event = process_Event()
        self.write_enabled_event = process_Event()

        # Number of input overflows reported by PortAudio, i.e. audio lost before reaching the callback.
        self.input_overflows = Value("q", 0)
        self._reported_input_overflows = 0
        self._reported_dropped_samples = 0

        # Process-safe concurrent queue to send audio from the recording process to the writing process/thread
        self.write_queue: JoinableQueue[NDArray[Any]] = process_Queue()

        # Shared-memory ring buffer to make audio from the recording process available to `read()`.
        self.read_shared_array: SharedArray | None = None
        self.local_read_shared_array: NDArray[Any] | None = None

        # Thread/Process to handle data writing in a separate thread/process (safely)
        self.write_thread: Thread | Process | None = None
        self.write_stop_event: Any = None
        self.write_is_started_event: Any = None

    def __str__(self) -> str:
        """Return a short name identifying the microphone in logs and errors."""
        return f"{self.__class__.__name__}({self.microphone_index})"

    @property
    def is_connected(self) -> bool:
        """Whether the recording process is running."""
        return self.record_process is not None and self.record_process.is_alive()

    @property
    def is_recording(self) -> bool:
        """Whether the PortAudio stream is started."""
        return self.record_is_started_event.is_set()

    @property
    def is_writing(self) -> bool:
        """Whether the recording is being written to a file."""
        return (
            self.write_thread is not None
            and self.write_is_started_event is not None
            and self.write_is_started_event.is_set()
        )

    @staticmethod
    def find_microphones(sounddevice_sdk: ISounddeviceSDK | None = None) -> list[dict[str, Any]]:
        """Detect the PortAudio microphones available on the system.

        Args:
            sounddevice_sdk (`ISounddeviceSDK`, *optional*):
                The `sounddevice` implementation to use. Defaults to the real library.

        Returns:
            `list[dict[str, Any]]`: The `index`, `name`, default `sample_rate` and `channels` (1-indexed) of
            each device that has at least one input channel.
        """
        if sounddevice_sdk is None:
            sounddevice_sdk = SounddeviceSDKAdapter()

        devices = sounddevice_sdk.query_devices()
        if isinstance(devices, dict):
            devices = [devices]

        found_microphones_info = [
            {
                "index": d["index"],
                "name": d["name"],
                "sample_rate": int(d["default_samplerate"]),
                "channels": list(range(1, d["max_input_channels"] + 1)),
            }
            for d in devices
            if d["max_input_channels"] > 0
        ]

        if len(found_microphones_info) == 0:
            logger.warning("No microphone found !")

        return found_microphones_info

    def _get_microphone_info(self) -> dict[str, Any]:
        """Return the `find_microphones` entry of the configured device.

        Raises:
            RuntimeError: If the configured index is not a microphone.
        """
        microphones_info = PortAudioMicrophone.find_microphones(self.sounddevice_sdk)
        for microphone_info in microphones_info:
            if microphone_info["index"] == self.microphone_index:
                return microphone_info
        raise RuntimeError(
            f"No microphone found for device {self.microphone_index}. Available microphones: {microphones_info}"
        )

    def _configure_capture_settings(self) -> None:
        """Validate the configured settings, and fill missing ones with the device's defaults.

        Raises:
            RuntimeError: If one of the settings is not compatible with the microphone.
        """
        microphone_info = self._get_microphone_info()
        self._validate_sample_rate(microphone_info["sample_rate"])
        self._validate_channels(microphone_info["channels"])

    def _validate_sample_rate(self, actual_sample_rate: int) -> None:
        """Validates the sample rate against the actual microphone's default sample rate."""
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
            elif self.sample_rate < actual_sample_rate:
                logger.warning(
                    "Provided sample rate is lower than the sample rate of the microphone. Performance may be impacted."
                )
        else:
            self.sample_rate = actual_sample_rate

    def _validate_channels(self, actual_channels: list[int]) -> None:
        """Validates the channels against the actual microphone's maximum input channels."""
        if self.channels is not None and len(self.channels) > 0:
            self.channels = [int(channel) for channel in self.channels]
            if not all(channel in actual_channels for channel in self.channels):
                raise RuntimeError(
                    f"Some of the provided channels {self.channels} are outside the possible channel range of the microphone {actual_channels}."
                )
        else:
            self.channels = actual_channels

    @check_if_already_connected
    def connect(self) -> None:
        """Validate the settings against the device and open its audio stream in a separate process.

        Missing sample rate and channels are filled with the device's defaults.

        Raises:
            DeviceAlreadyConnectedError: If the microphone is already connected.
            RuntimeError: If the settings are not compatible with the microphone or the stream cannot be opened.
        """
        self._configure_capture_settings()
        assert self.sample_rate is not None and self.channels is not None  # nosec B101

        # Create or reset queue and shared array
        capacity = max(1, round(self.sample_rate * self.buffer_duration_s))
        self.read_shared_array = SharedArray(shape=(capacity, len(self.channels)), dtype=np.dtype("float32"))
        self.local_read_shared_array = self.read_shared_array.get_local_array()
        self.write_queue = process_Queue()

        # Reset events and counters
        self.record_start_event.clear()
        self.record_stop_event.clear()
        self.record_close_event.clear()
        self.record_is_started_event.clear()
        self.audio_callback_start_event.clear()
        self.write_enabled_event.clear()
        self.input_overflows.value = 0
        self._reported_input_overflows = 0

        # Create and start an audio input stream with a recording callback
        # Remark: this is done in a separate process so that audio recording is not impacted by the main thread CPU usage, especially the precise_sleep function.
        process_init_event = process_Event()
        process_error_queue: Queue = Queue()
        self.record_process = Process(
            target=self._record_process,
            args=(
                self.microphone_index,
                self.sample_rate,
                self.channels,
                process_init_event,
                process_error_queue,
                self.record_start_event,
                self.record_stop_event,
                self.record_close_event,
                self.record_is_started_event,
                self.audio_callback_start_event,
                self.write_enabled_event,
                self.input_overflows,
                self.write_queue,
                self.read_shared_array,
                self.sounddevice_sdk,
            ),
            daemon=True,
        )
        self.record_process.start()

        # Wait for the stream to be opened, failing early if the recording process dies.
        deadline = time.perf_counter() + START_STOP_TIMEOUT_S
        while not process_init_event.wait(timeout=0.05):
            if not self.record_process.is_alive() or time.perf_counter() > deadline:
                break

        if not process_init_event.is_set():
            try:
                reason = process_error_queue.get(timeout=0.5)
            except Empty:
                reason = f"the audio stream did not open within {START_STOP_TIMEOUT_S}s"
            self._release_record_process()
            raise RuntimeError(f"Error connecting {self}: {reason}")

        logger.info(f"{self} connected.")

    def _release_record_process(self) -> None:
        """Stops the recording process and releases the shared memory."""
        self.record_close_event.set()
        if self.record_process is not None:
            self.record_process.join(timeout=START_STOP_TIMEOUT_S)
            if self.record_process.is_alive():
                logger.warning(f"{self} recording process did not stop in time, terminating it.")
                self.record_process.terminate()
                self.record_process.join()
            self.record_process = None

        self.write_queue.close()
        if self.read_shared_array is not None:
            self.local_read_shared_array = None
            self.read_shared_array.delete()
            self.read_shared_array = None

    @check_if_not_connected
    def disconnect(self) -> None:
        """Stop the recording if needed, close the audio stream and release the shared memory.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
        """
        if self.is_recording:
            self.stop_recording()

        self._release_record_process()

        logger.info(f"{self} disconnected.")

    def _check_is_recording(self) -> None:
        if not self.is_recording:
            raise DeviceNotRecordingError(f"{self} is not recording. Run `.start_recording()` first.")

    def _warn_on_input_overflows(self) -> None:
        input_overflows = self.input_overflows.value
        if input_overflows > self._reported_input_overflows:
            logger.warning(
                f"{self}: PortAudio reported {input_overflows - self._reported_input_overflows} input "
                "overflow(s), some audio was lost before reaching LeRobot."
            )
            self._reported_input_overflows = input_overflows

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Return all samples recorded since the previous `read()`, or since the recording started.

        If more than `buffer_duration_s` seconds elapse between two calls, the oldest samples are dropped and
        a warning is logged.

        Returns:
            `np.ndarray`: Float32 audio samples of shape `(num_samples, num_channels)`.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
        """
        self._check_is_recording()
        assert self.read_shared_array is not None and self.local_read_shared_array is not None  # nosec B101

        audio_readings, timestamp = self.read_shared_array.read(self.local_read_shared_array, flush=True)
        self.last_read_timestamp = None if math.isnan(timestamp) else timestamp

        dropped_samples = self.read_shared_array.dropped
        if dropped_samples > self._reported_dropped_samples:
            logger.warning(
                f"{self}: {dropped_samples - self._reported_dropped_samples} samples were dropped because "
                f"`read()` was not called for more than {self.buffer_duration_s}s. Call it more often "
                "or increase `buffer_duration_s`."
            )
            self._reported_dropped_samples = dropped_samples
        self._warn_on_input_overflows()

        return audio_readings

    @check_if_not_connected
    def read_latest(self, duration_s: float) -> NDArray[Any]:
        """Return the most recent `duration_s` seconds of audio, without consuming them.

        Args:
            duration_s (`float`):
                Length of the window in seconds. Must be in `(0, buffer_duration_s]`.

        Returns:
            `np.ndarray`: Float32 audio samples of shape `(round(duration_s * sample_rate), num_channels)`, or
            fewer samples if the recording started less than `duration_s` seconds ago.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
            ValueError: If `duration_s` is not in `(0, buffer_duration_s]`.
        """
        self._check_is_recording()
        assert self.read_shared_array is not None and self.local_read_shared_array is not None  # nosec B101
        assert self.sample_rate is not None  # nosec B101

        if not 0 < duration_s <= self.buffer_duration_s:
            raise ValueError(f"`duration_s` must be in (0, {self.buffer_duration_s}], got {duration_s}.")

        num_samples = min(round(duration_s * self.sample_rate), self.read_shared_array.capacity)
        audio_readings, timestamp = self.read_shared_array.read_latest(
            self.local_read_shared_array, num_samples
        )
        self.last_read_timestamp = None if math.isnan(timestamp) else timestamp
        self._warn_on_input_overflows()

        return audio_readings

    @staticmethod
    def _record_process(
        microphone_index,
        sample_rate,
        channels,
        process_init_event,
        process_error_queue,
        record_start_event,
        record_stop_event,
        record_close_event,
        record_is_started_event,
        audio_callback_start_event,
        write_enabled_event,
        input_overflows,
        write_queue,
        read_shared_array,
        sounddevice_sdk,
    ) -> None:
        """Open the audio stream in the recording process, then start and stop it as driven by the events.

        The stream is created here because it cannot be pickled to be sent to another process.
        """
        channels_index = np.array(channels) - 1
        local_read_shared_array = read_shared_array.get_local_array()

        def audio_callback(indata, frames, time_info, status) -> None:
            """Copy each block of samples to the ring buffer and the write queue. Must not block or log."""
            if status and getattr(status, "input_overflow", False):
                with input_overflows.get_lock():
                    input_overflows.value += 1
            if not audio_callback_start_event.is_set():
                return

            # Advanced indexing copies the data, as sounddevice reuses `indata` between callbacks.
            audio_chunk = indata[:, channels_index]
            if write_enabled_event.is_set():
                write_queue.put_nowait(audio_chunk)
            read_shared_array.write(
                local_read_shared_array,
                audio_chunk,
                timestamp=_last_sample_time(time_info, frames, sample_rate),
            )

        # Create the audio stream
        # InputStream must be instantiated in the process as it is not picklable.
        try:
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
        except Exception as e:
            process_error_queue.put(repr(e))
            return

        process_init_event.set()

        try:
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
        finally:
            stream.close()

    @check_if_not_connected
    def start_recording(
        self,
        output_file: str | Path | None = None,
        multiprocessing: bool = False,
        overwrite: bool = True,
        barrier: Barrier | None = None,
    ) -> None:
        """Start capturing audio, and writing it to `output_file` if given.

        Args:
            output_file (`str | Path`, *optional*):
                Path to save the recorded audio to, as a 32-bit float WAV file. Nothing is saved if `None`.
            multiprocessing (`bool`, *optional*, defaults to `False`):
                Write the file from a separate process instead of a thread.
            overwrite (`bool`, *optional*, defaults to `True`):
                Overwrite `output_file` if it already exists.
            barrier (`threading.Barrier`, *optional*):
                Shared with other microphones so that they all start capturing at the same time.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceAlreadyRecordingError: If the microphone is already recording.
            FileExistsError: If `output_file` exists and `overwrite` is `False`.
            RuntimeError: If the recording or the file writer fails to start.
        """
        if self.is_recording:
            raise DeviceAlreadyRecordingError(f"{self} is already recording.")
        assert self.read_shared_array is not None  # nosec B101

        # Reset queue and shared memory
        self.read_shared_array.reset()
        self._reported_dropped_samples = 0
        self.last_read_timestamp = None
        self._clear_queue(self.write_queue)

        # Reset stop event
        self.record_stop_event.clear()

        # Write recordings into a file if output_file is provided
        if output_file is not None:
            self._start_writing(Path(output_file), multiprocessing, overwrite)

        self.record_start_event.set()  # Start the input audio stream process
        if not self.record_is_started_event.wait(timeout=START_STOP_TIMEOUT_S):
            self.record_start_event.clear()
            self._stop_writing()
            raise RuntimeError(f"Error starting recording for {self}.")

        if barrier is not None:
            try:
                barrier.wait()  # Wait for multiple input audio streams to be started at the same time
            except BrokenBarrierError:
                # Another microphone failed to start: do not keep this one recording.
                self.stop_recording()
                raise

        self.audio_callback_start_event.set()

    def _start_writing(self, output_file: Path, multiprocessing: bool, overwrite: bool) -> None:
        """Starts the thread/process writing the recorded audio to `output_file`."""
        require_package("soundfile", extra="audio")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            if overwrite:
                output_file.unlink()
            else:
                raise FileExistsError(
                    f"Output file {output_file} already exists. Set overwrite to True to overwrite it."
                )

        writer_cls: type[Thread] | type[Process]
        if multiprocessing:
            self.write_stop_event, self.write_is_started_event = process_Event(), process_Event()
            writer_cls = Process
        else:
            self.write_stop_event, self.write_is_started_event = thread_Event(), thread_Event()
            writer_cls = Thread
        self.write_thread = writer_cls(
            target=PortAudioMicrophone._write_loop,
            args=(
                self.write_queue,
                self.write_stop_event,
                self.write_is_started_event,
                self.sample_rate,
                self.channels,
                output_file,
            ),
            daemon=True,
        )

        self.write_thread.start()
        if not self.write_is_started_event.wait(timeout=START_STOP_TIMEOUT_S):
            self._stop_writing()
            raise RuntimeError(f"Error starting writing to {output_file} for {self}.")
        self.write_enabled_event.set()

    def _stop_writing(self) -> None:
        """Waits for all queued audio to be written to the file, then stops the writing thread/process."""
        self.write_enabled_event.clear()
        if self.write_thread is None:
            return

        # Wait for the writer to drain the queue, so that the end of the recording is not lost.
        # JoinableQueue.join() has no timeout, hence the helper thread.
        drain_thread = Thread(target=self.write_queue.join, daemon=True)
        drain_thread.start()
        drain_thread.join(timeout=START_STOP_TIMEOUT_S)
        if drain_thread.is_alive():
            logger.warning(f"{self}: timed out writing the end of the recording to file.")

        self.write_stop_event.set()
        self.write_thread.join(timeout=START_STOP_TIMEOUT_S)
        if self.write_thread.is_alive():
            raise RuntimeError(f"Error stopping writing for {self}.")
        self.write_thread = None

    @check_if_not_connected
    def stop_recording(self) -> None:
        """Stop capturing audio, after writing everything recorded so far to the output file if any.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
            RuntimeError: If the recording or the file writer fails to stop.
        """
        self._check_is_recording()

        self.record_start_event.clear()  # Ensures the audio stream is not started again !
        self.record_stop_event.set()

        # Wait for the stream to be stopped. Buffers still pending in PortAudio are processed by the callback.
        deadline = time.perf_counter() + START_STOP_TIMEOUT_S
        while self.is_recording and time.perf_counter() < deadline:
            time.sleep(0.01)
        self.audio_callback_start_event.clear()

        self._stop_writing()

        if self.is_recording:
            raise RuntimeError(f"Error stopping recording for {self}.")

    @staticmethod
    def _write_loop(
        queue,
        write_stop_event,
        write_is_started_event,
        sample_rate: int,
        channels: list[int],
        output_file: Path,
    ) -> None:
        """Write the queued audio to `output_file` until stopped. Runs in a single thread or process."""
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

    @staticmethod
    def _clear_queue(queue) -> None:
        """Discard all items currently in the queue."""
        try:
            while True:
                queue.get_nowait()
                queue.task_done()
        except Empty:
            return
