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

"""
Transport backends with independently optional dependencies.

Zenoh requires ``lerobot[remote]`` only when used. The RL gRPC helpers are
imported separately from ``lerobot.transport.utils`` and require
``lerobot[grpcio-dep]``. Importing this namespace requires neither backend.
"""

from .zenoh import (
    BoundedQueryable,
    BoundedSubscriber,
    PendingQuery,
    PresenceEvent,
    PresenceToken,
    QueryCancelled,
    TransportError,
    ZenohConfig,
    ZenohTransport,
)

__all__ = [
    "BoundedQueryable",
    "BoundedSubscriber",
    "PendingQuery",
    "PresenceEvent",
    "PresenceToken",
    "QueryCancelled",
    "TransportError",
    "ZenohConfig",
    "ZenohTransport",
]
