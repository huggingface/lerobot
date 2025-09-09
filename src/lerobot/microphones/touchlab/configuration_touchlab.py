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

from dataclasses import dataclass

from ..configs import MicrophoneConfig


@MicrophoneConfig.register_subclass("touchlab")
@dataclass
class TouchLabSensorConfig(MicrophoneConfig):
    """Configuration class for TouchLab tactile sensors (technically not a microphone, but behaves like one acquisition-wise).

    This class provides configuration options for TouchLab tactile sensors, including serial port, sample rate and channels.

    Example configurations:
    ```python
    # Basic configurations
    TouchLabSensorConfig("/dev/ttyACM0", 16000)  # Serial port /dev/ttyACM0, 16000Hz
    TouchLabSensorConfig("/dev/ttyACM1", 44100)  # Serial port /dev/ttyACM1, 44100Hz
    ```

    Attributes:
        sensor_port: Serial port of the tactile sensor.
        baud_rate: Baud rate of the tactile sensor.
        sample_rate: Sample rate in Hz for the tactile sensor.
        channels: List of channel numbers to use for the tactile sensor.
    """

    sensor_port: str
    baud_rate: int = 115_200
