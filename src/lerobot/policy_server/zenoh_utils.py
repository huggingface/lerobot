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

"""Zenoh session construction shared by the policy server and the remote engine.

Verified against eclipse-zenoh 1.9 (thread-based; no asyncio API).
Multicast scouting is always disabled — fleet "discovery" is static
endpoint configuration plus liveliness tokens, never protocol magic.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_ZENOH_IMPORT_HINT = (
    "Remote inference requires the 'async' extra: pip install 'lerobot[async]' (eclipse-zenoh + msgpack)"
)


def import_zenoh():
    """Import zenoh lazily with an actionable error message."""
    try:
        import zenoh
    except ImportError as e:
        raise ImportError(_ZENOH_IMPORT_HINT) from e
    return zenoh


def build_zenoh_config(
    *,
    mode: str = "client",
    connect_endpoints: list[str] | None = None,
    listen_endpoints: list[str] | None = None,
    tls_root_ca_certificate: str | None = None,
    tls_connect_certificate: str | None = None,
    tls_connect_private_key: str | None = None,
    extra_config_json5: str | None = None,
):
    """Build a zenoh.Config (values are JSON5 strings — note the inner quoting)."""
    zenoh = import_zenoh()
    cfg = zenoh.Config()
    cfg.insert_json5("mode", json.dumps(mode))
    cfg.insert_json5("scouting/multicast/enabled", "false")
    if connect_endpoints:
        cfg.insert_json5("connect/endpoints", json.dumps(list(connect_endpoints)))
    if listen_endpoints:
        cfg.insert_json5("listen/endpoints", json.dumps(list(listen_endpoints)))
    if tls_root_ca_certificate:
        cfg.insert_json5("transport/link/tls/root_ca_certificate", json.dumps(tls_root_ca_certificate))
    if tls_connect_certificate:
        cfg.insert_json5("transport/link/tls/connect_certificate", json.dumps(tls_connect_certificate))
    if tls_connect_private_key:
        cfg.insert_json5("transport/link/tls/connect_private_key", json.dumps(tls_connect_private_key))
    if extra_config_json5:
        merged = json.loads(extra_config_json5)
        for key, value in merged.items():
            cfg.insert_json5(key, json.dumps(value))
    return cfg


def action_publisher_qos(zenoh) -> dict:
    """QoS for the action topic: RELIABLE + congestion DROP (never BLOCK) + express.

    DROP so one dead robot uplink can never stall the server's publish
    path; a dropped chunk is recoverable by design — the client's action
    buffer keeps the robot moving and the next chunk replaces it.
    """
    return {
        "reliability": zenoh.Reliability.RELIABLE,
        "congestion_control": zenoh.CongestionControl.DROP,
        "express": True,
        "priority": zenoh.Priority.INTERACTIVE_HIGH,
    }


def obs_publisher_qos(zenoh) -> dict:
    """QoS for the observation topic: best-effort drop, default priority.

    Intentional drop already happened at the client's one-slot holder;
    if the uplink stalls, dropping a frame protects the control loop.
    """
    return {
        "reliability": zenoh.Reliability.BEST_EFFORT,
        "congestion_control": zenoh.CongestionControl.DROP,
        "express": False,
        "priority": zenoh.Priority.DATA,
    }
