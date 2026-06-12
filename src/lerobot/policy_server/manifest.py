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

"""Policy-server manifest: one process = one (model, revision, dtype, device) on one GPU.

Loaded from YAML via ``lerobot-policy-server --manifest server.yaml`` (or
individual ``--model.repo_or_path=...`` CLI overrides through draccus).
Dynamic model loading is deliberately unsupported: pre-warmed processes
keep capacity planning honest and keep code-carrying payloads off the wire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)

SERVING_MODE_AUTO = "auto"
SERVING_MODE_SHARED = "shared"
SERVING_MODE_EXCLUSIVE = "exclusive"


@dataclass
class ModelSpec:
    """Which policy this process serves, and where it runs."""

    repo_or_path: str = ""
    revision: str = "main"
    # Optional torch dtype cast applied after load (e.g. "bfloat16").
    dtype: str | None = None
    device: str = "cuda"


@dataclass
class ZenohSpec:
    """Transport endpoints and security.

    Robots and servers both *dial out* to a ``zenohd`` router in
    production (``mode=client``).  ``mode=peer`` + ``listen_endpoints``
    supports router-less LAN and loopback test deployments.  Multicast
    scouting is always disabled: fleet discovery is configuration, not
    protocol magic.
    """

    mode: str = "client"  # "client" (via router) | "peer" (direct)
    connect_endpoints: list[str] = field(default_factory=list)
    listen_endpoints: list[str] = field(default_factory=list)
    # mTLS material (PEM paths). All three are required for TLS endpoints.
    tls_root_ca_certificate: str | None = None
    tls_connect_certificate: str | None = None
    tls_connect_private_key: str | None = None
    # Escape hatch: raw JSON5 merged into the zenoh config last.
    extra_config_json5: str | None = None


@dataclass
class DebugSpec:
    """Optional bounded request/response capture for offline replay."""

    capture_dir: str | None = None
    capture_max: int = 256


@dataclass
class PolicyServerManifest:
    """Top-level config for ``lerobot-policy-server``."""

    model: ModelSpec = field(default_factory=ModelSpec)
    zenoh: ZenohSpec = field(default_factory=ZenohSpec)

    # The task namespace this service is published under.  When
    # ``pin_task`` is true, session opens with a different task string
    # are rejected; otherwise VLA clients may set the task per session.
    default_task: str = ""
    pin_task: bool = False
    # Optional override for the <task_slug> key segment (defaults to a
    # slug of ``default_task``).
    service_name: str = ""

    # "auto" resolves from the policy classification (shared for
    # chunk-stateless policies, exclusive otherwise).  "exclusive" can be
    # forced; "shared" cannot override a chunk-stateful classification.
    serving_mode: str = SERVING_MODE_AUTO
    max_sessions: int = 5
    warmup_inferences: int = 2

    # FPS contract: warn on mismatch unless strict.
    trained_fps: float = 30.0
    strict_fps: bool = False

    # RTC behaviour for this server process (global to the shared policy:
    # ``init_rtc_processor`` mutates the policy instance, so it is a
    # per-process decision, not per-session).
    rtc: RTCConfig = field(default_factory=RTCConfig)

    # Sessions with no liveliness token and no traffic for this long are
    # garbage-collected (belt-and-braces behind liveliness GC).
    session_idle_timeout_s: float = 300.0

    # HTTP health + Prometheus metrics port; 0 disables the endpoint.
    health_port: int = 9100

    debug: DebugSpec = field(default_factory=DebugSpec)

    def __post_init__(self) -> None:
        if not self.model.repo_or_path:
            raise ValueError("--model.repo_or_path is required (the policy this server serves)")
        if self.serving_mode not in (SERVING_MODE_AUTO, SERVING_MODE_SHARED, SERVING_MODE_EXCLUSIVE):
            raise ValueError(f"serving_mode must be one of auto|shared|exclusive, got {self.serving_mode!r}")
        if self.max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {self.max_sessions}")
        if self.zenoh.mode not in ("client", "peer"):
            raise ValueError(f"zenoh.mode must be 'client' or 'peer', got {self.zenoh.mode!r}")
        if self.zenoh.mode == "client" and not self.zenoh.connect_endpoints:
            raise ValueError("zenoh.connect_endpoints is required in client mode (router address)")
        tls_fields = (
            self.zenoh.tls_root_ca_certificate,
            self.zenoh.tls_connect_certificate,
            self.zenoh.tls_connect_private_key,
        )
        if any(tls_fields) and not all(tls_fields):
            raise ValueError(
                "TLS requires all of tls_root_ca_certificate, tls_connect_certificate, "
                "tls_connect_private_key"
            )
