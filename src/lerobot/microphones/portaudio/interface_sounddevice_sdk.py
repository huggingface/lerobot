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

import abc
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

import numpy as np
from sounddevice import PortAudioError

from lerobot.utils.robot_utils import precise_sleep


# --- Interface definitions for InputStream ---
class IInputStream(abc.ABC):
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
        pass

    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class ISounddeviceSDK(abc.ABC):
    """Interface defining the contract for the Sounddevice SDK."""

    InputStream: type[IInputStream]

    @abc.abstractmethod
    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        pass


# --- Real SDK Adapter ---


class SounddeviceSDKAdapter(ISounddeviceSDK):
    """Adapts the real sounddevice library to the ISounddeviceSDK interface."""

    _sounddevice = None

    def __init__(self):
        try:
            import sounddevice

            SounddeviceSDKAdapter._sounddevice = sounddevice
        except ImportError as e:
            raise ImportError("sounddevice library not found") from e

    # --- Inner Class Implementation ---
    class RealInputStream(IInputStream):
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
            self._input_stream.start()

        def stop(self) -> None:
            self._input_stream.stop()

        def close(self) -> None:
            self._input_stream.close()

        def __del__(self):
            self._input_stream.stop()
            self._input_stream.close()

        @property
        def active(self) -> bool:
            return self._input_stream.active

        @property
        def stopped(self) -> bool:
            return self._input_stream.stopped

        @property
        def closed(self) -> bool:
            return self._input_stream.closed

    InputStream = RealInputStream

    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        return SounddeviceSDKAdapter._sounddevice.query_devices(device, kind)


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

VALID_DEVICES = [
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


class FakeSounddeviceSDKAdapter(ISounddeviceSDK):
    """Implements the ISounddeviceSDK interface with fake behaviour for testing."""

    # --- Inner Class Implementation ---
    class FakeInputStream(IInputStream):
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

            if self.callback is not None:
                self._streaming_thread = Thread(target=self._streaming_loop, daemon=True)
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
            raise PortAudioError(f"No input device matching {self.device}")

        def _validate_device(self):
            """Validates the device against the valid devices."""
            valid_device_indices = [device["index"] for device in VALID_DEVICES]
            valid_device_names = [device["name"] for device in VALID_DEVICES]

            if self.device is not None:
                if isinstance(self.device, (int, str)):
                    # Check if device index is valid
                    if isinstance(self.device, int) and self.device not in valid_device_indices:
                        raise PortAudioError(f"Error querying device {self.device}")

                    # Check if device name is valid
                    if isinstance(self.device, str) and self.device not in valid_device_names:
                        raise PortAudioError(f"No input device matching {self.device}")
                else:
                    raise PortAudioError(f"Device must be int or str, got {type(self.device)}")
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
                raise PortAudioError("Error opening InputStream: Invalid sample rate")

        def _validate_channels(self):
            """Validates the channels against the device's maximum channels."""
            device_info = self._get_device_info()
            if self.channels is None:
                self.channels = device_info["max_input_channels"]
            elif self.channels > device_info["max_input_channels"] or self.channels < 1:
                raise PortAudioError("Error opening InputStream: Invalid number of channels")

        def _validate_dtype(self):
            """Validates the dtype against the valid dtypes."""
            if self.dtype is not None:
                if self.dtype not in VALID_DTYPE:
                    raise PortAudioError("Invalid input sample format")
            else:
                self.dtype = "float32"  # Default dtype

        def _validate_latency(self):
            """Validates the latency against the valid latencies."""
            if self.latency is not None:
                if self.latency not in VALID_LATENCY:
                    raise PortAudioError("Invalid latency")
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
            if self.callback is not None:
                while not self._streaming_thread_stop_event.is_set():
                    precise_sleep(self.latency)
                    tmp_data = self._simulated_audio_data()
                    self.callback(
                        tmp_data,
                        len(tmp_data),
                        time.perf_counter(),
                        None,
                    )

        def start(self) -> None:
            """Start the fake input stream."""
            if not self.active and self.callback is not None:
                self._streaming_thread.start()
            self._active = True

        def stop(self) -> None:
            """Stop the fake input stream."""
            if self.callback is not None:
                self._streaming_thread_stop_event.set()
                self._streaming_thread.join()
            self._active = False

        def close(self) -> None:
            """Close the fake input stream."""
            if self.active and self.callback is not None:
                self.stop()
            self._active = False
            self._closed = True

        def __del__(self):
            self.close()

    InputStream = FakeInputStream

    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        """Returns a realistic list of audio devices including speakers and microphones."""
        if device is not None:
            # Return specific device
            for valid_device in VALID_DEVICES:
                if (isinstance(device, int) and valid_device["index"] == device) or (
                    isinstance(device, str) and valid_device["name"] == device
                ):
                    return valid_device
            raise PortAudioError(f"Error querying device {device}")

        elif kind is not None:
            for valid_device in VALID_DEVICES:
                if (
                    valid_device["max_input_channels"] > 0
                    and kind == "input"
                    or valid_device["max_output_channels"] > 0
                    and kind == "output"
                ):
                    return valid_device
            raise PortAudioError(f"No {kind} device found")

        return VALID_DEVICES
