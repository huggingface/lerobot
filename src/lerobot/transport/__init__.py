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

Zenoh requires ``lerobot[remote]`` only when used. Importing this namespace
requires neither backend.
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

# NOTE: The RL gRPC modules (transport.utils, services_pb2, services_pb2_grpc) are intentionally NOT
# re-exported here to avoid requiring lerobot[grpcio-dep].
# Import from submodules: ``from lerobot.transport.utils import send_bytes_in_chunks``

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
