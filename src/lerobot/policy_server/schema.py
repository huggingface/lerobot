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

"""Wire schema for remote policy inference.

Message dataclasses, the packed attachment header, and the Zenoh
key-expression layout shared by the policy server and the remote
inference engine.  This module is transport-free (no zenoh import) so
codecs and validation can be unit-tested without the optional extra.

Schema discipline: bodies are MessagePack maps decoded tolerantly
(unknown keys ignored, missing optional keys defaulted) so evolution is
additive-only.  Any change to the attachment layout requires a
``SCHEMA_VERSION`` bump; versions are negotiated at session open.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
# Oldest schema version this build can still serve.
MIN_SUPPORTED_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Attachment header (fixed layout, packed little-endian)
#
# Parsed without touching the msgpack body so routing, correlation and
# supersession decisions never pay deserialization costs.  The
# ``client_mono_ns`` field is a client-monotonic timestamp that is
# OPAQUE to the server: it is echoed back verbatim so the client can
# compute round-trip times on its own clock.  Wall-clock instants never
# cross machines (the clock iron rule).
# ---------------------------------------------------------------------------

_HEADER_STRUCT = struct.Struct("<HBQIqI")  # schema_version, msg_type, seq_id, episode_id, mono_ns, epoch

MSG_TYPE_OBS = 1
MSG_TYPE_CHUNK = 2
MSG_TYPE_EVENT = 3


@dataclass
class MsgHeader:
    """Packed per-message header carried in the Zenoh attachment."""

    schema_version: int = SCHEMA_VERSION
    msg_type: int = MSG_TYPE_OBS
    seq_id: int = 0
    episode_id: int = 0
    client_mono_ns: int = 0
    session_epoch: int = 0

    def pack(self) -> bytes:
        return _HEADER_STRUCT.pack(
            self.schema_version,
            self.msg_type,
            self.seq_id,
            self.episode_id,
            self.client_mono_ns,
            self.session_epoch,
        )

    @classmethod
    def unpack(cls, data: bytes) -> MsgHeader:
        if len(data) != _HEADER_STRUCT.size:
            raise ValueError(f"Bad header length: {len(data)} (expected {_HEADER_STRUCT.size})")
        version, msg_type, seq_id, episode_id, mono_ns, epoch = _HEADER_STRUCT.unpack(data)
        return cls(
            schema_version=version,
            msg_type=msg_type,
            seq_id=seq_id,
            episode_id=episode_id,
            client_mono_ns=mono_ns,
            session_epoch=epoch,
        )


HEADER_SIZE = _HEADER_STRUCT.size

# ---------------------------------------------------------------------------
# Message bodies
#
# ``np.ndarray`` fields travel as raw little-endian bytes + dtype + shape
# (see codec.py).  Images travel JPEG-compressed by default.
# ---------------------------------------------------------------------------

IMAGE_CODEC_JPEG = "jpeg"
IMAGE_CODEC_RAW = "raw"


@dataclass
class ObservationMsg:
    """Client → server: one inference request (data plane)."""

    state: np.ndarray | None = None  # float32 [state_dim]
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> uint8 HWC RGB
    task: str = ""
    inference_delay_steps: int = 0
    # RTC prefixes: the unexecuted tail of the previous chunk, in model
    # space (original) and robot space (postprocessed).  Both are needed
    # because the server re-anchors relative-action prefixes against the
    # current state and the client's ActionQueue.merge needs both chunks.
    prefix_model: np.ndarray | None = None  # float32 [T, action_dim]
    prefix_robot: np.ndarray | None = None  # float32 [T, action_dim]
    episode_start: bool = False
    # JPEG quality the images were encoded with; 0 means raw.
    jpeg_quality: int = 90


@dataclass
class ActionChunkMsg:
    """Server → client: one action chunk (data plane)."""

    seq_id_echo: int = 0
    client_mono_ns_echo: int = 0
    episode_id_echo: int = 0
    chunk_model: np.ndarray | None = None  # float32 [H, action_dim] (pre-postprocessor)
    chunk_robot: np.ndarray | None = None  # float32 [H, action_dim] (postprocessed)
    # Durations only — measured on the server's monotonic clock, never
    # compared against client time (the clock iron rule).
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    # Observations from this client that were superseded (overwritten in
    # the latest-only mailbox) since the previous reply — makes drops visible.
    superseded_seqs: int = 0
    server_load: float = 0.0  # active_sessions / max_sessions


@dataclass
class SessionOpenMsg:
    """Client → server (control plane): open and validate a session."""

    op: str = "open"
    client_uuid: str = ""
    robot_type: str = ""
    policy_type: str = ""
    fps: float = 0.0
    # Hard sync-safety contract: must equal the server's action feature
    # names *and order* — this maps chunk columns to motors.
    action_names: list[str] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=list)  # canonical keys (post-rename)
    state_dim: int = 0
    schema_version: int = SCHEMA_VERSION
    rtc_enabled: bool = False
    task: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionAckMsg:
    """Server → client (control plane): session accept/reject + capabilities."""

    accepted: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    session_id: str = ""
    model_repo: str = ""
    model_revision: str = ""
    policy_type: str = ""
    action_names: list[str] = field(default_factory=list)
    expected_cameras: list[str] = field(default_factory=list)
    state_dim: int = 0
    chunk_size: int = 0
    trained_fps: float = 0.0
    supports_rtc: bool = False
    rtc_execution_horizon: int = 0
    serving_mode: str = ""
    warmed_up: bool = False
    schema_version: int = SCHEMA_VERSION
    server_load: float = 0.0


@dataclass
class StatusMsg:
    """Server → client (control plane): pre-flight capability snapshot."""

    model_repo: str = ""
    model_revision: str = ""
    policy_type: str = ""
    action_names: list[str] = field(default_factory=list)
    expected_cameras: list[str] = field(default_factory=list)
    state_dim: int = 0
    chunk_size: int = 0
    trained_fps: float = 0.0
    supports_rtc: bool = False
    rtc_execution_horizon: int = 0
    serving_mode: str = ""
    warmed_up: bool = False
    min_schema_version: int = MIN_SUPPORTED_SCHEMA_VERSION
    max_schema_version: int = SCHEMA_VERSION
    active_sessions: int = 0
    max_sessions: int = 0


@dataclass
class ResetMsg:
    """Client → server (control plane): episode boundary (acknowledged)."""

    client_uuid: str = ""
    episode_id: int = 0


@dataclass
class ResetAckMsg:
    """Server → client: reset acknowledgement."""

    ok: bool = True
    error: str = ""


@dataclass
class SessionCloseMsg:
    """Client → server (control plane): graceful session close."""

    op: str = "close"
    client_uuid: str = ""
    session_id: str = ""


# ---------------------------------------------------------------------------
# Key-expression schema
#
#   @lerobot/<service>/<client_uuid>/obs       client → server   (pub/sub)
#   @lerobot/<service>/<client_uuid>/action    server → client   (pub/sub)
#   @lerobot/<service>/status                  queryable (capabilities)
#   @lerobot/<service>/session                 queryable (open/close)
#   @lerobot/<service>/<client_uuid>/reset     queryable (episode boundary)
#   @lerobot/<service>/<client_uuid>/alive     liveliness token (client)
#   @lerobot/<service>/server/alive            liveliness token (server)
#
# where <service> = <model_slug>/<revision_slug>/<task_slug>.  The task
# segment is a *namespace label* derived from the server's default task
# (or an explicit service name) — the actual inference task string
# travels in the session/observation messages.
#
# ``@lerobot`` is a verbatim chunk: it is only matched by an identical
# chunk, so third-party ``**`` subscribers on a shared router can never
# scrape this tree.
# ---------------------------------------------------------------------------

KEY_ROOT = "@lerobot"

# Conservative allowlist for user-supplied key segments.  Everything
# else (including '/', '*', '$', '?', '#', whitespace) is folded to '-'.
_SEGMENT_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Reserved final chunks of the key tree; a client UUID must never
# collide with them.
RESERVED_SEGMENTS = frozenset({"status", "session", "server", "obs", "action", "reset", "alive"})


def sanitize_key_segment(segment: str) -> str:
    """Fold an arbitrary string into a single safe Zenoh key chunk."""
    slug = _SEGMENT_SANITIZE_RE.sub("-", segment.strip()).strip("-.")
    if not slug:
        raise ValueError(f"Key segment {segment!r} sanitizes to an empty chunk")
    if slug in RESERVED_SEGMENTS:
        raise ValueError(f"Key segment {segment!r} collides with reserved chunk {slug!r}")
    return slug


def service_prefix(model_id: str, revision: str, task: str) -> str:
    """Build the shared key prefix for one served (model, revision, task) triple."""
    return "/".join(
        (
            KEY_ROOT,
            sanitize_key_segment(model_id),
            sanitize_key_segment(revision or "main"),
            sanitize_key_segment(task or "default"),
        )
    )


def obs_key(prefix: str, client_uuid: str) -> str:
    return f"{prefix}/{sanitize_key_segment(client_uuid)}/obs"


def action_key(prefix: str, client_uuid: str) -> str:
    return f"{prefix}/{sanitize_key_segment(client_uuid)}/action"


def reset_key(prefix: str, client_uuid: str) -> str:
    return f"{prefix}/{sanitize_key_segment(client_uuid)}/reset"


def client_alive_key(prefix: str, client_uuid: str) -> str:
    return f"{prefix}/{sanitize_key_segment(client_uuid)}/alive"


def status_key(prefix: str) -> str:
    return f"{prefix}/status"


def session_key(prefix: str) -> str:
    return f"{prefix}/session"


def server_alive_key(prefix: str) -> str:
    return f"{prefix}/server/alive"


# Single-depth wildcards only — '**' would also match status/session/alive.
def obs_wildcard(prefix: str) -> str:
    return f"{prefix}/*/obs"


def reset_wildcard(prefix: str) -> str:
    return f"{prefix}/*/reset"


def client_alive_wildcard(prefix: str) -> str:
    return f"{prefix}/*/alive"


def client_uuid_from_key(key: str) -> str:
    """Extract the client UUID chunk from an obs/reset/alive key."""
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError(f"Key {key!r} has no client chunk")
    return parts[-2]
