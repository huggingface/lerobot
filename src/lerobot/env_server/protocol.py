# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Simulator addressing over the shared version-one envelope.

Operation names live in query keys / CONTROL bodies. No inference message tags change.
"""

from lerobot.transport.wire.protocol import validate_segment


def deployment_prefix(deployment: str) -> str:
    """Return a validated simulator deployment key."""
    return f"lerobot/env/v1/deployments/{validate_segment(deployment)}"


def instance_prefix(deployment: str, instance: str) -> str:
    """Return a simulator boot key under its deployment."""
    return f"{deployment_prefix(deployment)}/instances/{validate_segment(instance)}"


def session_prefix(deployment: str, instance: str, session: str) -> str:
    """Return an admitted world-session key."""
    return f"{instance_prefix(deployment, instance)}/sessions/{validate_segment(session)}"
