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

from dataclasses import dataclass

from ..configs import MicrophoneConfig


@MicrophoneConfig.register_subclass("portaudio")
@dataclass
class PortAudioMicrophoneConfig(MicrophoneConfig):
    """Configuration class for PortAudio-based microphone devices.

    This class provides configuration options for microphones accessed through PortAudio with the sounddevice Python package.
    including device index, sample rate and channels.

    Example configurations:
    ```python
    # Basic configurations
    PortAudioMicrophoneConfig(0, 16000, [1])  # Device index 0, 16000Hz, mono
    PortAudioMicrophoneConfig(1, 44100, [1, 2])  # Device index 1, 44100Hz, stereo
    ```

    Attributes:
        microphone_index: Device index for the microphone.
        sample_rate: Sample rate in Hz for the microphone.
        channels: List of channel numbers to use for the microphone.
    """

    microphone_index: int
