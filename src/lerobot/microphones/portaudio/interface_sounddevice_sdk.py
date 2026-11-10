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

"""Thin wrapper around `sounddevice`, with a fake implementation to test microphones without hardware."""

import abc
import contextlib
import time
from collections.abc import Callable, Sequence
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package
from lerobot.utils.robot_utils import precise_sleep


# --- Interface definitions for InputStream ---
class IInputStream(abc.ABC):
    """Subset of the `sounddevice.InputStream` interface used by [`PortAudioMicrophone`]."""

    @abc.abstractmethod
    def __init__(
        self,
        samplerate: float | None = None,
        blocksize: int | None = None,
        device: int | str | None = None,
        channels: int | None = None,
        dtype: str | np.dtype | None = None,
        latency: float | str | None = None,
        callback: Callable[[Any, int, Any, Any], None] | None = None,
    ):
        """Open the stream with the given settings, without starting it."""
        pass

    @abc.abstractmethod
    def start(self) -> None:
        """Start calling `callback` with captured audio."""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the stream after processing the pending buffers."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the stream and release the device."""
        pass


class ISounddeviceSDK(abc.ABC):
    """Interface defining the contract for the Sounddevice SDK."""

    InputStream: type[IInputStream]

    @abc.abstractmethod
    def query_devices(
        self, device: int | str | None = None, kind: str | None = None
    ) -> Sequence[dict[str, Any]] | dict[str, Any]:
        """Return the information of one device, or of all devices (see `sounddevice.query_devices`)."""
        pass


# --- Real SDK Adapter ---


class SounddeviceSDKAdapter(ISounddeviceSDK):
    """Adapts the real sounddevice library to the ISounddeviceSDK interface."""

    def __init__(self):
        """Check that `sounddevice` is installed."""
        require_package("sounddevice", extra="audio")

    # --- Inner Class Implementation ---
    class RealInputStream(IInputStream):
        """Wraps `sounddevice.InputStream`."""

        def __init__(
            self,
            samplerate: int | None = None,
            blocksize: int | None = None,
            device: int | None = None,
            channels: int | None = None,
            dtype: str | np.dtype | None = None,
            latency: float | str | None = None,
            callback: Callable[[Any, int, Any, Any], None] | None = None,
        ):
            """Open a `sounddevice.InputStream` with the given settings."""
            import sounddevice

            self._input_stream = sounddevice.InputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                device=device,
                channels=channels,
                dtype=dtype,
                latency=latency,
                callback=callback,
            )

        def start(self) -> None:
            """Start the stream."""
            self._input_stream.start()

        def stop(self) -> None:
            """Stop the stream after processing the pending buffers."""
            self._input_stream.stop()

        def close(self) -> None:
            """Close the stream."""
            self._input_stream.close()

        def __del__(self):
            """Close the stream if it is garbage collected while open."""
            with contextlib.suppress(Exception):
                self._input_stream.close()

        @property
        def active(self) -> bool:
            """Whether the stream is started."""
            return self._input_stream.active

        @property
        def stopped(self) -> bool:
            """Whether the stream is stopped."""
            return self._input_stream.stopped

        @property
        def closed(self) -> bool:
            """Whether the stream is closed."""
            return self._input_stream.closed

    InputStream = RealInputStream

    def query_devices(
        self, device: int | str | None = None, kind: str | None = None
    ) -> Sequence[dict[str, Any]] | dict[str, Any]:
        """Return the information of one device, or of all devices (see `sounddevice.query_devices`)."""
        import sounddevice

        return sounddevice.query_devices(device, kind)


# Emulates a 48kHz stereo microphone
VALID_DTYPE = {
    "float32",
    "int32",
    "int16",
    "int8",
    "uint8",
    np.float32,
    np.int32,
    np.int16,
    np.int8,
    np.uint8,
}
VALID_LATENCY = {"low", "high"}

VALID_DEVICES: list[dict[str, Any]] = [
    {
        "index": 0,
        "name": "Built-in Microphone",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_low_input_latency": 0.01,
        "default_low_output_latency": 0.001,
        "default_high_input_latency": 0.1,
        "default_high_output_latency": 0.01,
        "default_samplerate": 48000.0,
    },
    {
        "index": 1,
        "name": "Built-in Output",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_low_input_latency": 0.04,
        "default_low_output_latency": 0.04,
        "default_high_input_latency": 0.12,
        "default_high_output_latency": 0.12,
        "default_samplerate": 48000.0,
    },
    {
        "index": 2,
        "name": "USB Audio Device",
        "hostapi": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_low_input_latency": 0.03,
        "default_low_output_latency": 0.01,
        "default_high_input_latency": 0.04,
        "default_high_output_latency": 0.03,
        "default_samplerate": 16000.0,
    },
]

# -- Fake SDK Adapter ---


class FakePortAudioError(Exception):
    """Stands in for `sounddevice.PortAudioError`, so that the fake SDK works without sounddevice."""


class FakeSounddeviceSDKAdapter(ISounddeviceSDK):
    """Implements the ISounddeviceSDK interface with fake behaviour for testing."""

    # --- Inner Class Implementation ---
    class FakeInputStream(IInputStream):
        """Emulates an input stream by calling `callback` with random audio from a background thread."""

        def __init__(
            self,
            samplerate: float | None = None,
            blocksize: int | None = None,
            device: int | str | None = None,
            channels: int | None = None,
            dtype: str | None = None,
            latency: str | None = None,
            callback: Callable[[Any, int, Any, Any], None] | None = None,
        ):
            """Validate the settings like PortAudio would, without starting the stream."""
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.device = device
            self.channels = channels
            self.dtype = dtype
            self.latency = latency
            self.callback = callback

            self._validate_settings()

            self._active = False
            self._closed = False
            self._streaming_thread: Thread | None = None
            self._streaming_thread_stop_event = Event()

        @property
        def active(self) -> bool:
            """True when the stream is active, False otherwise."""
            return self._active

        @property
        def stopped(self) -> bool:
            """True when the stream is stopped, False otherwise."""
            return not self._active

        @property
        def closed(self) -> bool:
            """True after a call to close(), False otherwise."""
            return self._closed

        def _get_device_info(self):
            """Returns the device info for the device."""
            for device in VALID_DEVICES:
                if (isinstance(self.device, int) and device["index"] == self.device) or (
                    isinstance(self.device, str) and device["name"] == self.device
                ):
                    return device
            raise FakePortAudioError(f"No input device matching {self.device}")

        def _validate_device(self):
            """Validates the device against the valid devices."""
            valid_device_indices = [device["index"] for device in VALID_DEVICES]
            valid_device_names = [device["name"] for device in VALID_DEVICES]

            if self.device is not None:
                if isinstance(self.device, (int, str)):
                    # Check if device index is valid
                    if isinstance(self.device, int) and self.device not in valid_device_indices:
                        raise FakePortAudioError(f"Error querying device {self.device}")

                    # Check if device name is valid
                    if isinstance(self.device, str) and self.device not in valid_device_names:
                        raise FakePortAudioError(f"No input device matching {self.device}")
                else:
                    raise FakePortAudioError(f"Device must be int or str, got {type(self.device)}")
            else:
                # Default to first input device
                input_devices = [d for d in VALID_DEVICES if d["max_input_channels"] > 0]
                if input_devices:
                    self.device = input_devices[0]["index"]

        def _validate_samplerate(self):
            """Validates the samplerate against the device's maximum samplerate."""
            device_info = self._get_device_info()
            if self.samplerate is None:
                self.samplerate = device_info["default_samplerate"]
            elif self.samplerate > device_info["default_samplerate"] or self.samplerate < 1000:
                raise FakePortAudioError("Error opening InputStream: Invalid sample rate")

        def _validate_channels(self):
            """Validates the channels against the device's maximum channels."""
            device_info = self._get_device_info()
            if self.channels is None:
                self.channels = device_info["max_input_channels"]
            elif self.channels > device_info["max_input_channels"] or self.channels < 1:
                raise FakePortAudioError("Error opening InputStream: Invalid number of channels")

        def _validate_dtype(self):
            """Validates the dtype against the valid dtypes."""
            if self.dtype is not None:
                if self.dtype not in VALID_DTYPE:
                    raise FakePortAudioError("Invalid input sample format")
            else:
                self.dtype = "float32"  # Default dtype

        def _validate_latency(self):
            """Validates the latency against the valid latencies."""
            if self.latency is not None:
                if self.latency not in VALID_LATENCY:
                    raise FakePortAudioError("Invalid latency")
            else:
                self.latency = "low"  # Default latency

            if isinstance(self.latency, str):
                device_info = self._get_device_info()
                if self.latency == "low":
                    self.latency = device_info["default_low_input_latency"]
                elif self.latency == "high":
                    self.latency = device_info["default_high_input_latency"]

        def _validate_settings(self):
            """Validates the input parameters against available devices and valid options."""
            self._validate_device()
            self._validate_samplerate()
            self._validate_channels()
            self._validate_dtype()
            self._validate_latency()

        def _simulated_audio_data(self) -> np.ndarray:
            """Generates a simulated audio signal for testing purposes with proper value ranges."""
            # Settings are resolved to numbers by `_validate_settings`.
            assert isinstance(self.samplerate, (int, float)) and isinstance(self.latency, float)  # nosec B101
            duration_samples = int(self.samplerate * self.latency)

            # Generate output according to dtype
            if self.dtype in {"float32", np.float32}:
                # Generate values between -1 and 1 for float32
                data = np.random.uniform(-1.0, 1.0, (duration_samples, self.channels)).astype(self.dtype)
            else:
                # Use np.iinfo to get proper range for integer types
                info = np.iinfo(self.dtype)
                data = np.random.randint(
                    info.min, info.max + 1, (duration_samples, self.channels), dtype=self.dtype
                )

            return data

        def _streaming_loop(self):
            # Blocks are scheduled on absolute times, like a real device, so that the stream does not drift.
            next_block_time = time.perf_counter()
            while not self._streaming_thread_stop_event.is_set():
                next_block_time += self.latency
                precise_sleep(max(0.0, next_block_time - time.perf_counter()))
                tmp_data = self._simulated_audio_data()
                # The fake stream clock is time.perf_counter(), and the last sample was captured just now.
                now = time.perf_counter()
                time_info = SimpleNamespace(
                    inputBufferAdcTime=now - (len(tmp_data) - 1) / self.samplerate,
                    currentTime=now,
                    outputBufferDacTime=0.0,
                )
                self.callback(tmp_data, len(tmp_data), time_info, None)

        def start(self) -> None:
            """Start the fake input stream."""
            if not self.active and self.callback is not None:
                self._streaming_thread_stop_event.clear()
                self._streaming_thread = Thread(target=self._streaming_loop, daemon=True)
                self._streaming_thread.start()
            self._active = True

        def stop(self) -> None:
            """Stop the fake input stream."""
            if self._streaming_thread is not None:
                self._streaming_thread_stop_event.set()
                self._streaming_thread.join()
                self._streaming_thread = None
            self._active = False

        def close(self) -> None:
            """Close the fake input stream."""
            if self.active and self.callback is not None:
                self.stop()
            self._active = False
            self._closed = True

        def __del__(self):
            """Close the stream if it is garbage collected while open."""
            self.close()

    InputStream = FakeInputStream

    def query_devices(
        self, device: int | str | None = None, kind: str | None = None
    ) -> Sequence[dict[str, Any]] | dict[str, Any]:
        """Returns a realistic list of audio devices including speakers and microphones."""
        if device is not None:
            # Return specific device
            for valid_device in VALID_DEVICES:
                if (isinstance(device, int) and valid_device["index"] == device) or (
                    isinstance(device, str) and valid_device["name"] == device
                ):
                    return valid_device
            raise FakePortAudioError(f"Error querying device {device}")

        elif kind is not None:
            for valid_device in VALID_DEVICES:
                if (
                    valid_device["max_input_channels"] > 0
                    and kind == "input"
                    or valid_device["max_output_channels"] > 0
                    and kind == "output"
                ):
                    return valid_device
            raise FakePortAudioError(f"No {kind} device found")

        return VALID_DEVICES
