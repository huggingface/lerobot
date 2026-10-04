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

"""Exclusive asynchronous policy serving over Zenoh.

Protocol definitions are always available. Client, server and deployment exports
load on demand so importing local inference does not load transport backends.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .chunk_contract import chunk_settings
from .protocol import PROTOCOL_VERSION, AdmissionDeniedError, ErrorCode, ProtocolError

if TYPE_CHECKING:
    from .build_info import SOFTWARE_BUILD
    from .client import RemoteClient, RequestCancelled
    from .configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
    from .server import PolicyServer, SessionWorker

_LAZY_EXPORTS = {
    "SOFTWARE_BUILD": "build_info",
    "RemoteClient": "client",
    "RequestCancelled": "client",
    "ExecutionConfig": "configs",
    "LanguageConfig": "configs",
    "ModelConfig": "configs",
    "ServerConfig": "configs",
    "PolicyServer": "server",
    "SessionWorker": "server",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "PROTOCOL_VERSION",
    "SOFTWARE_BUILD",
    "AdmissionDeniedError",
    "ErrorCode",
    "ExecutionConfig",
    "LanguageConfig",
    "ModelConfig",
    "PolicyServer",
    "ProtocolError",
    "RemoteClient",
    "RequestCancelled",
    "ServerConfig",
    "SessionWorker",
    "chunk_settings",
]
