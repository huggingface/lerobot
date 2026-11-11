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

"""Configuration of PortAudio microphones."""

from dataclasses import dataclass

from ..configs import MicrophoneConfig


@MicrophoneConfig.register_subclass("portaudio")
@dataclass
class PortAudioMicrophoneConfig(MicrophoneConfig):
    """Configuration for microphones accessed through PortAudio with the `sounddevice` package.

    Use `lerobot-find-microphones` to list the available device indices, sample rates and channels.

    Args:
        microphone_index (`int`):
            `sounddevice` index of the microphone.
        sample_rate (`int`, *optional*):
            Sample rate in Hz. Must not exceed the device's default sample rate, which is used if `None`.
        channels (`list[int]`, *optional*):
            Channel numbers to record (1-indexed). All input channels are recorded if `None`.
        buffer_duration_s (`float`, *optional*, defaults to `10.0`):
            Duration of audio kept in memory for `read()` and `read_latest()`. If `read()` is not called
            for longer than this, the oldest samples are dropped. The file written during the recording is
            not affected.

    Example:
        ```python
        >>> from lerobot.microphones.portaudio import PortAudioMicrophoneConfig
        >>> config = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=16000, channels=[1])
        >>> config.type
        'portaudio'
        ```
    """

    microphone_index: int
    buffer_duration_s: float = 10.0

    def __post_init__(self) -> None:
        """Validate the buffer duration."""
        if self.buffer_duration_s <= 0:
            raise ValueError(f"`buffer_duration_s` must be positive, got {self.buffer_duration_s}.")
