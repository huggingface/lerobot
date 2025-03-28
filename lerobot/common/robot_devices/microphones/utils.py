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

from typing import Protocol

from lerobot.common.robot_devices.microphones.configs import MicrophoneConfig, MicrophoneConfigBase

# Defines a microphone type
class Microphone(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def start_recording(self, output_file: str | None = None): ...
    def stop_recording(self): ...

def make_microphones_from_configs(microphone_configs: dict[str, MicrophoneConfigBase]) -> list[Microphone]:
    microphones = {}

    for key, cfg in microphone_configs.items():
        if cfg.type == "microphone":
            from lerobot.common.robot_devices.microphones.microphone import Microphone
            microphones[key] = Microphone(cfg)
        else:
            raise ValueError(f"The microphone type '{cfg.type}' is not valid.")

    return microphones

def make_microphone(microphone_type, **kwargs) -> Microphone:
    if microphone_type == "microphone":
        from lerobot.common.robot_devices.microphones.microphone import Microphone
        return Microphone(MicrophoneConfig(**kwargs))
    else:
        raise ValueError(f"The microphone type '{microphone_type}' is not valid.")