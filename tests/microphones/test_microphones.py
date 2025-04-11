# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Tests for physical microphones and their mocked versions.
If the physical microphone is not connected to the computer, or not working,
the test will be skipped.

Example of running a specific test:
```bash
pytest -sx tests/microphones/test_microphones.py::test_microphone
```

Example of running test on a real microphone connected to the computer:
```bash
pytest -sx 'tests/microphones/test_microphones.py::test_microphone[microphone-False]'
```

Example of running test on a mocked version of the microphone:
```bash
pytest -sx 'tests/microphones/test_microphones.py::test_microphone[microphone-True]'
```
"""

import time

import numpy as np
import pytest
from soundfile import read

from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceAlreadyRecordingError,
    RobotDeviceNotConnectedError,
    RobotDeviceNotRecordingError,
)
from tests.utils import TEST_MICROPHONE_TYPES, make_microphone, require_microphone

# Maximum recording tie difference between two consecutive audio recordings of the same duration.
# Set to 0.02 seconds as twice the default size of sounddvice callback buffer (i.e. we tolerate the loss of one buffer).
MAX_RECORDING_TIME_DIFFERENCE = 0.02

DUMMY_RECORDING = "test_recording.wav"


@pytest.mark.parametrize("microphone_type, mock", TEST_MICROPHONE_TYPES)
@require_microphone
def test_microphone(tmp_path, request, microphone_type, mock):
    """Test assumes that a recroding handled with microphone.start_recording(output_file) and stop_recording() or microphone.read()
    leqds to a sample that does not differ from the requested duration by more than 0.1 seconds.
    """

    microphone_kwargs = {"microphone_type": microphone_type, "mock": mock}

    # Test instantiating
    microphone = make_microphone(**microphone_kwargs)

    # Test start_recording, stop_recording, read and disconnecting before connecting raises an error
    with pytest.raises(RobotDeviceNotConnectedError):
        microphone.start_recording()
    with pytest.raises(RobotDeviceNotConnectedError):
        microphone.stop_recording()
    with pytest.raises(RobotDeviceNotConnectedError):
        microphone.read()
    with pytest.raises(RobotDeviceNotConnectedError):
        microphone.disconnect()

    # Test deleting the object without connecting first
    del microphone

    # Test connecting
    microphone = make_microphone(**microphone_kwargs)
    microphone.connect()
    assert microphone.is_connected
    assert microphone.sample_rate is not None
    assert microphone.channels is not None

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        microphone.connect()

    # Test reading or stop recording before starting recording raises an error
    with pytest.raises(RobotDeviceNotRecordingError):
        microphone.read()
    with pytest.raises(RobotDeviceNotRecordingError):
        microphone.stop_recording()

    # Test start_recording
    fpath = tmp_path / DUMMY_RECORDING
    microphone.start_recording(fpath)
    assert microphone.is_recording

    # Test start_recording twice raises an error
    with pytest.raises(RobotDeviceAlreadyRecordingError):
        microphone.start_recording()

    # Test reading from the microphone
    time.sleep(1.0)
    audio_chunk = microphone.read()
    assert isinstance(audio_chunk, np.ndarray)
    assert audio_chunk.ndim == 2
    _, c = audio_chunk.shape
    assert c == len(microphone.channels)

    # Test stop_recording
    microphone.stop_recording()
    assert fpath.exists()
    assert not microphone.stream.active
    assert microphone.record_thread is None

    # Test stop_recording twice raises an error
    with pytest.raises(RobotDeviceNotRecordingError):
        microphone.stop_recording()

    # Test reading and recording output similar length audio chunks
    microphone.start_recording(tmp_path / DUMMY_RECORDING)
    time.sleep(1.0)
    audio_chunk = microphone.read()
    microphone.stop_recording()

    recorded_audio, recorded_sample_rate = read(fpath)
    assert recorded_sample_rate == microphone.sample_rate

    error_msg = (
        "Recording time difference between read() and stop_recording()",
        (len(audio_chunk) - len(recorded_audio)) / MAX_RECORDING_TIME_DIFFERENCE,
    )
    np.testing.assert_allclose(
        len(audio_chunk),
        len(recorded_audio),
        atol=recorded_sample_rate * MAX_RECORDING_TIME_DIFFERENCE,
        err_msg=error_msg,
    )

    # Test disconnecting
    microphone.disconnect()
    assert not microphone.is_connected

    # Test disconnecting with `__del__`
    microphone = make_microphone(**microphone_kwargs)
    microphone.connect()
    del microphone
