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

"""Base configuration shared by all microphone backends."""

import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class MicrophoneConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration for microphones. Backends register a subclass under their type name.

    Args:
        sample_rate (`int`, *optional*):
            Sample rate in Hz. `None` lets the backend pick the device's default.
        channels (`list[int]`, *optional*):
            Channel numbers to record (1-indexed). `None` lets the backend record all input channels.
    """

    sample_rate: int | None = None
    channels: list[int] | None = None

    @property
    def type(self) -> str:
        """The name under which this configuration class is registered, e.g. `"portaudio"`."""
        return str(self.get_choice_name(self.__class__))
