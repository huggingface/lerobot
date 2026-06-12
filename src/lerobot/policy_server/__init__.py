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

"""Multi-client GPU policy serving over Zenoh (``lerobot-policy-server``).

The wire schema (:mod:`.schema`) and codecs (:mod:`.codec`) are shared
with the edge-side :class:`~lerobot.rollout.inference.remote.RemoteInferenceEngine`.
Heavy/optional imports (msgpack, zenoh, torch server) are deferred so the
schema stays importable without the ``async`` extra.
"""

from typing import Any

from .manifest import (
    DebugSpec,
    ModelSpec,
    PolicyServerManifest,
    ZenohSpec,
)
from .schema import SCHEMA_VERSION, MsgHeader, service_prefix

__all__ = [
    "SCHEMA_VERSION",
    "DebugSpec",
    "ModelSpec",
    "MsgHeader",
    "PolicyServer",
    "PolicyServerManifest",
    "ZenohSpec",
    "codec",
    "service_prefix",
]


def __getattr__(name: str) -> Any:
    import importlib

    if name == "PolicyServer":
        return importlib.import_module(".server", __name__).PolicyServer
    if name == "codec":
        return importlib.import_module(".codec", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
