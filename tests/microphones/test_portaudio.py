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

import os
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from soundfile import read

from lerobot.microphones.portaudio.configuration_portaudio import PortAudioMicrophoneConfig
from lerobot.microphones.portaudio.interface_sounddevice_sdk import (
    FakeSounddeviceSDKAdapter,
    SounddeviceSDKAdapter,
)
from lerobot.microphones.portaudio.microphone_portaudio import PortAudioMicrophone
from lerobot.microphones.utils import async_microphones_start_recording, async_microphones_stop_recording
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceAlreadyRecordingError,
    DeviceNotConnectedError,
    DeviceNotRecordingError,
)
from lerobot.utils.robot_utils import busy_wait

MODULE_PATH = "lerobot.microphones.portaudio.microphone_portaudio"
RECORDING_DURATION = 1.0

LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS = (
    os.getenv("LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS", "False").lower() == "true"
)


class TestPortAudioMicrophoneConfiguration(unittest.TestCase):
    """Test the PortAudioMicrophone configuration and initialization."""

    def test_config_creation(self):
        """Test creating a valid configuration."""
        config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000, channels=[1, 2])
        self.assertEqual(config.microphone_index, 0)
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.channels, [1, 2])

    def test_config_creation_missing_microphone_index(self):
        """Test creating a configuration with missing microphone index."""
        with self.assertRaises(TypeError):
            PortAudioMicrophoneConfig(sample_rate=48000, channels=[1, 2])

    def test_config_creation_missing_sample_rate(self):
        """Test creating a configuration with missing sample rate."""
        config = PortAudioMicrophoneConfig(microphone_index=0, channels=[1, 2])
        self.assertIsNone(config.sample_rate)

    def test_config_creation_missing_channels(self):
        """Test creating a configuration with missing channels."""
        config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000)
        self.assertIsNone(config.channels)


class TestPortAudioMicrophoneDeviceValidation(unittest.TestCase):
    """Test device validation and configuration."""

    def setUp(self):
        if LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS:
            self.test_sdk = SounddeviceSDKAdapter()
        else:
            self.test_sdk = FakeSounddeviceSDKAdapter()

        def _create_config(
            device: int | str | None = None, kind: str | None = None
        ) -> PortAudioMicrophoneConfig:
            device_info = self.test_sdk.query_devices(device, kind)
            config = PortAudioMicrophoneConfig(
                microphone_index=device_info["index"],
                sample_rate=device_info["default_samplerate"],
                channels=np.arange(device_info["max_input_channels"]) + 1,
            )
            return config

        self._create_config = _create_config

        self.default_config = self._create_config(kind="input")

    def test_find_microphones(self):
        microphones = PortAudioMicrophone.find_microphones(sounddevice_sdk=self.test_sdk)

        for microphone in microphones:
            self.assertIsInstance(microphone["index"], int)
            self.assertIsInstance(microphone["name"], str)
            self.assertIsInstance(microphone["sample_rate"], int)
            self.assertIsInstance(microphone["channels"], np.ndarray)
            self.assertGreater(len(microphone["channels"]), 0)

    def test_init_defaults(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)

        device_info = self.test_sdk.query_devices(kind="input")
        self.assertIsNotNone(microphone)
        self.assertEqual(microphone.microphone_index, device_info["index"])
        self.assertEqual(microphone.sample_rate, device_info["default_samplerate"])
        np.testing.assert_array_equal(microphone.channels, np.arange(device_info["max_input_channels"]) + 1)
        self.assertFalse(microphone.is_connected)
        self.assertFalse(microphone.is_recording)

    def test_connect_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()

        self.assertTrue(microphone.is_connected)
        self.assertFalse(microphone.is_recording)
        self.assertFalse(microphone.is_writing)

    def test_connect_empty_config(self):
        config = deepcopy(self.default_config)
        config.sample_rate = None
        config.channels = None
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)
        microphone.connect()

        device_info = self.test_sdk.query_devices(kind="input")
        self.assertEqual(microphone.sample_rate, device_info["default_samplerate"])
        np.testing.assert_array_equal(microphone.channels, np.arange(device_info["max_input_channels"]) + 1)

    def test_connect_already_connected(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()

        with self.assertRaises(DeviceAlreadyConnectedError):
            microphone.connect()

    def test_connect_invalid_device(self):
        config = self._create_config(kind="output")
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(RuntimeError):
            microphone.connect()

    def test_connect_invalid_index(self):
        config = deepcopy(self.default_config)
        config.microphone_index = -1
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(RuntimeError):
            microphone.connect()

    def test_connect_invalid_sample_rate(self):
        config = deepcopy(self.default_config)
        config.sample_rate = -1
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(RuntimeError):
            microphone.connect()

    def test_connect_float_sample_rate(self):
        config = deepcopy(self.default_config)
        config.sample_rate = int(config.sample_rate) - 0.5
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)
        microphone.connect()

        self.assertIsInstance(microphone.sample_rate, int)
        self.assertEqual(microphone.sample_rate, int(config.sample_rate))

    def test_connect_lower_sample_rate(self):
        config = deepcopy(self.default_config)
        config.sample_rate = 1000  # Lowest possible sample rate
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)

        microphone.connect()
        self.assertEqual(microphone.sample_rate, 1000)

    def test_connect_invalid_channels(self):
        config = deepcopy(self.default_config)
        config.channels = np.append(self.default_config.channels, -1)
        microphone = PortAudioMicrophone(config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(RuntimeError):
            microphone.connect()

    def test_disconnect_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.disconnect()

        self.assertFalse(microphone.is_connected)
        self.assertFalse(microphone.is_recording)
        self.assertFalse(microphone.is_writing)

    def test_disconnect_not_connected(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(DeviceNotConnectedError):
            microphone.disconnect()

    def test_start_recording_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording()

        self.assertTrue(microphone.is_recording)
        self.assertTrue(microphone.is_connected)
        self.assertFalse(microphone.is_writing)

    def test_recoring_not_connected(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(DeviceNotConnectedError):
            microphone.start_recording()

    def test_start_recording_already_recording(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording()

        with self.assertRaises(DeviceAlreadyRecordingError):
            microphone.start_recording()

    def test_start_writing_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording(output_file="test.wav")

        self.assertTrue(microphone.is_recording)
        self.assertTrue(microphone.is_connected)
        self.assertTrue(microphone.is_writing)
        self.assertTrue(Path("test.wav").exists())

    def test_stop_recording_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording()
        busy_wait(RECORDING_DURATION)
        microphone.stop_recording()

        self.assertFalse(microphone.is_recording)
        self.assertTrue(microphone.is_connected)
        self.assertFalse(microphone.is_writing)

    def test_stop_writing_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording(output_file="test.wav")
        busy_wait(RECORDING_DURATION)
        microphone.stop_recording()

        self.assertFalse(microphone.is_recording)
        self.assertTrue(microphone.is_connected)
        self.assertFalse(microphone.is_writing)
        self.assertTrue(Path("test.wav").exists())

    def test_stop_recording_not_connected(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)

        with self.assertRaises(DeviceNotConnectedError):
            microphone.stop_recording()

    def test_stop_recording_not_recording(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()

        with self.assertRaises(DeviceNotRecordingError):
            microphone.stop_recording()

    def test_disconnect_while_recording(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording()
        busy_wait(RECORDING_DURATION)
        microphone.disconnect()

        self.assertFalse(microphone.is_connected)
        self.assertFalse(microphone.is_recording)
        self.assertFalse(microphone.is_writing)

    def test_disconnect_while_writing(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording(output_file="test.wav")
        busy_wait(RECORDING_DURATION)
        microphone.disconnect()

        self.assertFalse(microphone.is_connected)
        self.assertFalse(microphone.is_recording)
        self.assertFalse(microphone.is_writing)
        self.assertTrue(Path("test.wav").exists())

    def test_read_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording()

        busy_wait(RECORDING_DURATION)

        data = microphone.read()

        device_info = self.test_sdk.query_devices(kind="input")
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[1], len(self.default_config.channels))
        self.assertAlmostEqual(
            data.shape[0],
            RECORDING_DURATION * self.default_config.sample_rate,
            delta=2 * self.default_config.sample_rate * device_info["default_low_input_latency"],
        )

    def test_writing_success(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording(output_file="test.wav")

        busy_wait(RECORDING_DURATION)

        microphone.stop_recording()

        data, samplerate = read("test.wav")

        device_info = self.test_sdk.query_devices(kind="input")
        self.assertEqual(samplerate, self.default_config.sample_rate)
        self.assertEqual(data.shape[1], len(self.default_config.channels))
        self.assertAlmostEqual(
            data.shape[0],
            RECORDING_DURATION * self.default_config.sample_rate,
            delta=2 * self.default_config.sample_rate * device_info["default_low_input_latency"],
        )

    def test_read_while_writing(self):
        microphone = PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk)
        microphone.connect()
        microphone.start_recording(output_file="test.wav")

        busy_wait(RECORDING_DURATION)

        read_data = microphone.read()
        microphone.stop_recording()

        writing_data, _ = read("test.wav")

        device_info = self.test_sdk.query_devices(kind="input")
        self.assertAlmostEqual(
            writing_data.shape[0],
            RECORDING_DURATION * self.default_config.sample_rate,
            delta=2 * self.default_config.sample_rate * device_info["default_low_input_latency"],
        )
        self.assertAlmostEqual(
            read_data.shape[0],
            RECORDING_DURATION * self.default_config.sample_rate,
            delta=2 * self.default_config.sample_rate * device_info["default_low_input_latency"],
        )

    def test_async_start_recording(self):
        microphones = {
            "microphone_1": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
            "microphone_2": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
        }
        for microphone in microphones.values():
            microphone.connect()

        async_microphones_start_recording(microphones)

        for microphone in microphones.values():
            self.assertTrue(microphone.is_recording)
            self.assertTrue(microphone.is_connected)
            self.assertFalse(microphone.is_writing)

    def test_async_start_writing(self):
        microphones = {
            "microphone_1": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
            "microphone_2": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
        }
        for microphone in microphones.values():
            microphone.connect()

        async_microphones_start_recording(microphones, output_files=["test_1.wav", "test_2.wav"])

        for microphone in microphones.values():
            self.assertTrue(microphone.is_recording)
            self.assertTrue(microphone.is_connected)
            self.assertTrue(microphone.is_writing)
        self.assertTrue(Path("test_1.wav").exists())
        self.assertTrue(Path("test_2.wav").exists())

    def test_async_stop_recording(self):
        microphones = {
            "microphone_1": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
            "microphone_2": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
        }
        for microphone in microphones.values():
            microphone.connect()

        async_microphones_start_recording(microphones)
        async_microphones_stop_recording(microphones)

        for microphone in microphones.values():
            self.assertFalse(microphone.is_recording)
            self.assertTrue(microphone.is_connected)
            self.assertFalse(microphone.is_writing)

    def test_async_stop_writing(self):
        microphones = {
            "microphone_1": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
            "microphone_2": PortAudioMicrophone(self.default_config, sounddevice_sdk=self.test_sdk),
        }
        for microphone in microphones.values():
            microphone.connect()

        async_microphones_start_recording(microphones, output_files=["test_1.wav", "test_2.wav"])
        async_microphones_stop_recording(microphones)

        for microphone in microphones.values():
            self.assertFalse(microphone.is_recording)
            self.assertTrue(microphone.is_connected)
            self.assertFalse(microphone.is_writing)
        self.assertTrue(Path("test_1.wav").exists())
        self.assertTrue(Path("test_2.wav").exists())


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
