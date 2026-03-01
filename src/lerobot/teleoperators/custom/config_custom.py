#!/usr/bin/env python

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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("custom")
@dataclass
class CustomConfig(TeleoperatorConfig):
    """Custom teleoperator config that dynamically wraps a base teleoperator class.
    
    The base class and its configuration are loaded from a JSON config file at runtime.
    Port and baud_rate are taken from the first device in the config file.
    """
    config_path: str | None = None  # REQUIRED: Path to custom config JSON file
    port: str = "/dev/ttyACM0"  # Default port
    baud_rate: int = 115200  # Default baud rate
