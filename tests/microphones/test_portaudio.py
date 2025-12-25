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
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
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
from lerobot.utils.robot_utils import precise_sleep

MODULE_PATH = "lerobot.microphones.portaudio.microphone_portaudio"
RECORDING_DURATION = 1.0

LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS = (
    os.getenv("LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS", "False").lower() == "true"
)


@pytest.fixture
def test_sdk():
    """Fixture to provide either real or fake SDK based on environment variable."""
    if LEROBOT_USE_REAL_PORTAUDIO_MICROPHONE_TESTS:
        return SounddeviceSDKAdapter()
    else:
        return FakeSounddeviceSDKAdapter()


# Configuration Tests


def test_config_creation():
    """Test creating a valid configuration."""
    config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000, channels=[1, 2])
    assert config.microphone_index == 0
    assert config.sample_rate == 48000
    assert config.channels == [1, 2]


def test_config_creation_missing_microphone_index():
    """Test creating a configuration with missing microphone index."""
    with pytest.raises(TypeError):
        PortAudioMicrophoneConfig(sample_rate=48000, channels=[1, 2])


def test_config_creation_missing_sample_rate():
    """Test creating a configuration with missing sample rate."""
    config = PortAudioMicrophoneConfig(microphone_index=0, channels=[1, 2])
    assert config.sample_rate is None


def test_config_creation_missing_channels():
    """Test creating a configuration with missing channels."""
    config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000)
    assert config.channels is None


@pytest.fixture
def default_config(test_sdk):
    """Fixture to provide a default configuration for input devices."""
    device_info = test_sdk.query_devices(kind="input")
    return PortAudioMicrophoneConfig(
        microphone_index=device_info["index"],
        sample_rate=device_info["default_samplerate"],
        channels=np.arange(device_info["max_input_channels"]) + 1,
    )


# Microphone Tests


def test_find_microphones(test_sdk):
    """Test finding microphones."""
    microphones = PortAudioMicrophone.find_microphones(sounddevice_sdk=test_sdk)

    for microphone in microphones:
        assert isinstance(microphone["index"], int)
        assert isinstance(microphone["name"], str)
        assert isinstance(microphone["sample_rate"], int)
        assert isinstance(microphone["channels"], np.ndarray)
        assert len(microphone["channels"]) > 0


def test_init_defaults(default_config, test_sdk):
    """Test microphone initialization with defaults."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)

    device_info = test_sdk.query_devices(kind="input")
    assert microphone is not None
    assert microphone.microphone_index == device_info["index"]
    assert microphone.sample_rate == device_info["default_samplerate"]
    np.testing.assert_array_equal(microphone.channels, np.arange(device_info["max_input_channels"]) + 1)
    assert not microphone.is_connected
    assert not microphone.is_recording


def test_connect_success(default_config, test_sdk):
    """Test successful connection."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()

    assert microphone.is_connected
    assert not microphone.is_recording
    assert not microphone.is_writing


def test_connect_empty_config(default_config, test_sdk):
    """Test connection with empty config values."""
    config = deepcopy(default_config)
    config.sample_rate = None
    config.channels = None
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)
    microphone.connect()

    device_info = test_sdk.query_devices(kind="input")
    assert microphone.sample_rate == device_info["default_samplerate"]
    np.testing.assert_array_equal(microphone.channels, np.arange(device_info["max_input_channels"]) + 1)


def test_connect_already_connected(default_config, test_sdk):
    """Test connecting when already connected."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()

    with pytest.raises(DeviceAlreadyConnectedError):
        microphone.connect()


def test_connect_invalid_device(test_sdk):
    """Test connecting with invalid device (output device)."""
    device_info = test_sdk.query_devices(kind="output")
    config = PortAudioMicrophoneConfig(
        microphone_index=device_info["index"],
        sample_rate=device_info["default_samplerate"],
        channels=np.arange(device_info["max_input_channels"]) + 1,
    )
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)

    with pytest.raises(RuntimeError):
        microphone.connect()


def test_connect_invalid_index(default_config, test_sdk):
    """Test connecting with invalid device index."""
    config = deepcopy(default_config)
    config.microphone_index = -1
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)

    with pytest.raises(RuntimeError):
        microphone.connect()


def test_connect_invalid_sample_rate(default_config, test_sdk):
    """Test connecting with invalid sample rate."""
    config = deepcopy(default_config)
    config.sample_rate = -1
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)

    with pytest.raises(RuntimeError):
        microphone.connect()


def test_connect_float_sample_rate(default_config, test_sdk):
    """Test connecting with float sample rate."""
    config = deepcopy(default_config)
    config.sample_rate = int(config.sample_rate) - 0.5
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)
    microphone.connect()

    assert isinstance(microphone.sample_rate, int)
    assert microphone.sample_rate == int(config.sample_rate)


def test_connect_lower_sample_rate(default_config, test_sdk):
    """Test connecting with lower sample rate."""
    config = deepcopy(default_config)
    config.sample_rate = 1000  # Lowest possible sample rate
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)

    microphone.connect()
    assert microphone.sample_rate == 1000


def test_connect_invalid_channels(default_config, test_sdk):
    """Test connecting with invalid channels."""
    config = deepcopy(default_config)
    config.channels = np.append(default_config.channels, -1)
    microphone = PortAudioMicrophone(config, sounddevice_sdk=test_sdk)

    with pytest.raises(RuntimeError):
        microphone.connect()


def test_disconnect_success(default_config, test_sdk):
    """Test successful disconnection."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.disconnect()

    assert not microphone.is_connected
    assert not microphone.is_recording
    assert not microphone.is_writing


def test_disconnect_not_connected(default_config, test_sdk):
    """Test disconnecting when not connected."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)

    with pytest.raises(DeviceNotConnectedError):
        microphone.disconnect()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_start_recording_success(default_config, test_sdk, multiprocessing):
    """Test successful recording start."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(multiprocessing=multiprocessing)

    assert microphone.is_recording
    assert microphone.is_connected
    assert not microphone.is_writing


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_recording_not_connected(default_config, test_sdk, multiprocessing):
    """Test starting recording when not connected."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)

    with pytest.raises(DeviceNotConnectedError):
        microphone.start_recording(multiprocessing=multiprocessing)


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_start_recording_already_recording(default_config, test_sdk, multiprocessing):
    """Test starting recording when already recording."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(multiprocessing=multiprocessing)

    with pytest.raises(DeviceAlreadyRecordingError):
        microphone.start_recording(multiprocessing=multiprocessing)


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_start_writing_success(tmp_path, default_config, test_sdk, multiprocessing):
    """Test successful writing start."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(output_file=tmp_path / "test.wav", multiprocessing=multiprocessing)

    assert microphone.is_recording
    assert microphone.is_connected
    assert microphone.is_writing
    assert (tmp_path / "test.wav").exists()

    (tmp_path / "test.wav").unlink()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_start_writing_file_already_exists_no_overwrite(tmp_path, default_config, test_sdk, multiprocessing):
    """Test writing with file that already exists."""
    (tmp_path / "test.wav").touch()
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()

    with pytest.raises(FileExistsError):
        microphone.start_recording(
            output_file=tmp_path / "test.wav", multiprocessing=multiprocessing, overwrite=False
        )

    (tmp_path / "test.wav").unlink()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_stop_recording_success(default_config, test_sdk, multiprocessing):
    """Test successful recording stop."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(multiprocessing=multiprocessing)
    precise_sleep(RECORDING_DURATION)
    microphone.stop_recording()

    assert not microphone.is_recording
    assert microphone.is_connected
    assert not microphone.is_writing


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_stop_writing_success(tmp_path, default_config, test_sdk, multiprocessing):
    """Test successful writing stop."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(output_file=tmp_path / "test.wav", multiprocessing=multiprocessing)
    precise_sleep(RECORDING_DURATION)
    microphone.stop_recording()

    assert not microphone.is_recording
    assert microphone.is_connected
    assert not microphone.is_writing
    assert (tmp_path / "test.wav").exists()

    (tmp_path / "test.wav").unlink()


def test_stop_recording_not_connected(default_config, test_sdk):
    """Test stopping recording when not connected."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)

    with pytest.raises(DeviceNotConnectedError):
        microphone.stop_recording()


def test_stop_recording_not_recording(default_config, test_sdk):
    """Test stopping recording when not recording."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()

    with pytest.raises(DeviceNotRecordingError):
        microphone.stop_recording()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_disconnect_while_recording(default_config, test_sdk, multiprocessing):
    """Test disconnecting while recording."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(multiprocessing=multiprocessing)
    precise_sleep(RECORDING_DURATION)
    microphone.disconnect()

    assert not microphone.is_connected
    assert not microphone.is_recording
    assert not microphone.is_writing


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_disconnect_while_writing(tmp_path, default_config, test_sdk, multiprocessing):
    """Test disconnecting while writing."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(output_file=tmp_path / "test.wav", multiprocessing=multiprocessing)
    precise_sleep(RECORDING_DURATION)
    microphone.disconnect()

    assert not microphone.is_connected
    assert not microphone.is_recording
    assert not microphone.is_writing
    assert Path(tmp_path / "test.wav").exists()

    (tmp_path / "test.wav").unlink()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_read_success(default_config, test_sdk, multiprocessing):
    """Test successful reading of audio data."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(multiprocessing=multiprocessing)

    precise_sleep(RECORDING_DURATION)

    data = microphone.read()

    device_info = test_sdk.query_devices(kind="input")
    assert data is not None
    assert data.shape[1] == len(default_config.channels)
    assert (
        abs(data.shape[0] - RECORDING_DURATION * default_config.sample_rate)
        <= 2 * default_config.sample_rate * device_info["default_low_input_latency"]
    )


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_writing_success(tmp_path, default_config, test_sdk, multiprocessing):
    """Test successful writing to file."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(output_file=tmp_path / "test.wav", multiprocessing=multiprocessing)

    precise_sleep(RECORDING_DURATION)

    microphone.stop_recording()

    data, samplerate = read(tmp_path / "test.wav")

    device_info = test_sdk.query_devices(kind="input")
    assert samplerate == default_config.sample_rate
    assert data.shape[1] == len(default_config.channels)
    assert (
        abs(data.shape[0] - RECORDING_DURATION * default_config.sample_rate)
        <= 2 * default_config.sample_rate * device_info["default_low_input_latency"]
    )

    (tmp_path / "test.wav").unlink()


@pytest.mark.parametrize("multiprocessing", [True, False])
def test_read_while_writing(tmp_path, default_config, test_sdk, multiprocessing):
    """Test reading while writing."""
    microphone = PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk)
    microphone.connect()
    microphone.start_recording(output_file=tmp_path / "test.wav", multiprocessing=multiprocessing)

    precise_sleep(RECORDING_DURATION)

    read_data = microphone.read()
    microphone.stop_recording()

    writing_data, _ = read(tmp_path / "test.wav")

    device_info = test_sdk.query_devices(kind="input")
    assert (
        abs(writing_data.shape[0] - RECORDING_DURATION * default_config.sample_rate)
        <= 2 * default_config.sample_rate * device_info["default_low_input_latency"]
    )
    assert (
        abs(read_data.shape[0] - RECORDING_DURATION * default_config.sample_rate)
        <= 2 * default_config.sample_rate * device_info["default_low_input_latency"]
    )

    (tmp_path / "test.wav").unlink()


def test_async_start_recording(default_config, test_sdk):
    """Test async recording start."""
    microphones = {
        "microphone_1": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
        "microphone_2": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
    }
    for microphone in microphones.values():
        microphone.connect()

    async_microphones_start_recording(microphones)

    for microphone in microphones.values():
        assert microphone.is_recording
        assert microphone.is_connected
        assert not microphone.is_writing


def test_async_start_writing(tmp_path, default_config, test_sdk):
    """Test async writing start."""
    microphones = {
        "microphone_1": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
        "microphone_2": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
    }
    for microphone in microphones.values():
        microphone.connect()

    async_microphones_start_recording(
        microphones, output_files=[tmp_path / "test_1.wav", tmp_path / "test_2.wav"]
    )

    for microphone in microphones.values():
        assert microphone.is_recording
        assert microphone.is_connected
        assert microphone.is_writing
    assert Path(tmp_path / "test_1.wav").exists()
    assert Path(tmp_path / "test_2.wav").exists()

    (tmp_path / "test_1.wav").unlink()
    (tmp_path / "test_2.wav").unlink()


def test_async_stop_recording(default_config, test_sdk):
    """Test async recording stop."""
    microphones = {
        "microphone_1": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
        "microphone_2": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
    }
    for microphone in microphones.values():
        microphone.connect()

    async_microphones_start_recording(microphones)
    async_microphones_stop_recording(microphones)

    for microphone in microphones.values():
        assert not microphone.is_recording
        assert microphone.is_connected
        assert not microphone.is_writing


def test_async_stop_writing(tmp_path, default_config, test_sdk):
    """Test async writing stop."""
    microphones = {
        "microphone_1": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
        "microphone_2": PortAudioMicrophone(default_config, sounddevice_sdk=test_sdk),
    }
    for microphone in microphones.values():
        microphone.connect()

    async_microphones_start_recording(
        microphones, output_files=[tmp_path / "test_1.wav", tmp_path / "test_2.wav"]
    )
    async_microphones_stop_recording(microphones)

    for microphone in microphones.values():
        assert not microphone.is_recording
        assert microphone.is_connected
        assert not microphone.is_writing
    assert Path(tmp_path / "test_1.wav").exists()
    assert Path(tmp_path / "test_2.wav").exists()

    (tmp_path / "test_1.wav").unlink()
    (tmp_path / "test_2.wav").unlink()
