#!/usr/bin/env python

# Copyright 2026 Gangelia. All rights reserved.
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

"""Evaluation-time fault injection for LeRobot policies.

This package provides a configurable layer that can interrupt postprocessed
actions immediately before ``env.step`` during ``lerobot-eval``. It does not
train policies or demonstrate learned recovery.

Fault injection is **disabled by default**. When ``fault.enabled=false``, no
injector is constructed and evaluation behavior is unchanged.
"""

from lerobot.faults.action_hold import ActionHoldFault, make_fault_injector
from lerobot.faults.config import FaultInjectionConfig, default_fault_config, resolve_fault_log_path
from lerobot.faults.logging import FaultEventLogger

__all__ = [
    "ActionHoldFault",
    "FaultEventLogger",
    "FaultInjectionConfig",
    "default_fault_config",
    "make_fault_injector",
    "resolve_fault_log_path",
]
