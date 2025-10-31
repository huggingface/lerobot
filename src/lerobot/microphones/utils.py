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

from .configs import MicrophoneConfig
from .microphone import Microphone


def make_microphones_from_configs(microphone_configs: dict[str, MicrophoneConfig]) -> dict[str, Microphone]:
    microphones = {}

    for key, cfg in microphone_configs.items():
        if cfg.type == "microphone":
            from .microphone import Microphone

            microphones[key] = Microphone(cfg)
        else:
            raise ValueError(f"The microphone type '{cfg.type}' is not valid.")

    return microphones
