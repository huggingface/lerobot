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

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SharedBusDeviceConfig:
    """Describes how a logical component attaches to a shared bus."""

    component: str
    motor_id_offset: int = 0


@dataclass
class SharedBusConfig:
    """Configuration for a shared Feetech motor bus."""

    port: str
    components: list[SharedBusDeviceConfig] = field(default_factory=list)
    handshake_on_connect: bool = True
