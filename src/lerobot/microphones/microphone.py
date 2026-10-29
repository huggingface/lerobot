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

"""Abstract base class shared by all microphone backends."""

import abc
from pathlib import Path
from threading import Barrier
from typing import Any

from numpy.typing import NDArray

from .configs import MicrophoneConfig


class Microphone(abc.ABC):
    """Base class for microphone implementations.

    Defines a standard interface for microphone operations across different backends. Subclasses must
    implement all abstract methods.

    Unlike a camera, a microphone produces a continuous stream that must not have gaps, so it has an explicit
    recording phase, usually one per episode: [`~Microphone.connect`], then [`~Microphone.start_recording`],
    then any number of [`~Microphone.read`] / [`~Microphone.read_latest`] calls, then
    [`~Microphone.stop_recording`] and finally [`~Microphone.disconnect`].

    Audio is returned as `(num_samples, num_channels)` arrays. After each read, `last_read_timestamp` holds the
    `time.perf_counter()` time at which the last returned sample was captured, so audio can be aligned with
    other observations.

    **Attributes**:
        - **sample_rate** (`int | None`) -- Sample rate in Hz. Backends may resolve `None` to the device's
          default when connecting.
        - **channels** (`list[int] | None`) -- Channel numbers to record (1-indexed). Backends may resolve
          `None` to all input channels when connecting.
        - **last_read_timestamp** (`float | None`) -- Capture time of the last sample returned by the latest
          read, in the `time.perf_counter()` clock. `None` before the first read of a recording.

    Args:
        config (`MicrophoneConfig`):
            Microphone configuration containing the sample rate and channels.

    Example:
        ```python
        >>> from lerobot.microphones.portaudio import PortAudioMicrophone, PortAudioMicrophoneConfig
        >>> config = PortAudioMicrophoneConfig(microphone_index=0)
        >>> with PortAudioMicrophone(config) as microphone:  # doctest: +SKIP
        ...     microphone.start_recording("episode_0.wav")
        ...     chunk = microphone.read()
        ...     microphone.stop_recording()
        ```
    """

    def __init__(self, config: MicrophoneConfig):
        """Store the settings shared by all backends."""
        self.sample_rate: int | None = config.sample_rate
        self.channels: list[int] | None = config.channels
        self.last_read_timestamp: float | None = None

    def __enter__(self):
        """Connect to the microphone when entering a `with` block."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Disconnect (stopping any ongoing recording) when leaving a `with` block, even on error."""
        self.disconnect()

    def __del__(self) -> None:
        """Disconnect if the object is garbage collected without cleanup."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the microphone is connected and ready to start recording."""
        pass

    @property
    @abc.abstractmethod
    def is_recording(self) -> bool:
        """Whether the microphone is currently recording."""
        pass

    @property
    @abc.abstractmethod
    def is_writing(self) -> bool:
        """Whether the microphone is currently writing the recording to a file."""
        pass

    @staticmethod
    @abc.abstractmethod
    def find_microphones() -> list[dict[str, Any]]:
        """Detect the microphones available on the system.

        Returns:
            `list[dict[str, Any]]`: One dictionary of backend-specific information per detected microphone.
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the connection to the microphone.

        Raises:
            DeviceAlreadyConnectedError: If the microphone is already connected.
        """
        pass

    @abc.abstractmethod
    def start_recording(
        self,
        output_file: str | Path | None = None,
        multiprocessing: bool = False,
        overwrite: bool = True,
        barrier: Barrier | None = None,
    ) -> None:
        """Start recording audio from the microphone.

        Args:
            output_file (`str | Path`, *optional*):
                Path to save the recorded audio to, as a WAV file. Nothing is saved if `None`.
            multiprocessing (`bool`, *optional*, defaults to `False`):
                Write the file from a separate process instead of a thread.
            overwrite (`bool`, *optional*, defaults to `True`):
                Overwrite `output_file` if it already exists.
            barrier (`threading.Barrier`, *optional*):
                Shared with other microphones so that they all start recording at the same time.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceAlreadyRecordingError: If the microphone is already recording.
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[Any]:
        """Return all audio samples recorded since the previous `read()`, or since the recording started.

        This is the method to use when every sample matters, e.g. when recording a dataset. The number of
        returned samples depends on how much time elapsed since the previous call.

        Returns:
            `np.ndarray`: Audio samples of shape `(num_samples, num_channels)`.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
        """
        pass

    @abc.abstractmethod
    def read_latest(self, duration_s: float) -> NDArray[Any]:
        """Return the most recent `duration_s` seconds of audio, without consuming them.

        This is the method to use when a fixed-size window is needed, e.g. as a policy input or for
        visualization. It does not affect what the next `read()` returns. Fewer samples are returned if the
        recording started less than `duration_s` seconds ago.

        Args:
            duration_s (`float`):
                Length of the window in seconds.

        Returns:
            `np.ndarray`: Audio samples of shape `(num_samples, num_channels)`.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
        """
        pass

    @abc.abstractmethod
    def stop_recording(self) -> None:
        """Stop recording audio from the microphone.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
            DeviceNotRecordingError: If the microphone is not recording.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect the microphone and release its resources, stopping any ongoing recording.

        Raises:
            DeviceNotConnectedError: If the microphone is not connected.
        """
        pass
