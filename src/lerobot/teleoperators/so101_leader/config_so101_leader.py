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


@TeleoperatorConfig.register_subclass("so101_leader")
@dataclass
class SO101LeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    use_degrees: bool = False
    # Optional filtering to reduce "slow movement jitter" caused by small sensor noise / quantization.
    # If a float: applied to all action keys. If dict: per-key deadband.
    # Deadband operates in the same units returned by `get_action()` (degrees if `use_degrees=true`,
    # otherwise normalized units).
    action_deadband: float | dict[str, float] | None = None
