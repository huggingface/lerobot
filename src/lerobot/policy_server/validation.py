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

"""Serving-mode classification and session capability validation.

Multi-tenancy is engineered, not assumed: sharing one policy instance
across sessions is only safe when ``predict_action_chunk`` touches no
instance state.  That property has been verified per policy family and
is encoded here as an explicit registry — never inferred.

- ``act``/``pi0``/``pi05``: chunk-stateless (verified in-tree).
- ``smolvla``: populates its ``_queues`` *inside* ``predict_action_chunk``;
  with ``n_obs_steps == 1`` the queue is overwritten with the request's
  own observation before being read, so sharing is safe.  With history
  (``n_obs_steps > 1``) requests would read other sessions' frames →
  exclusive.
- ``diffusion``: ``predict_action_chunk`` reads ``_queues`` that only
  ``select_action`` populates → exclusive, with the server populating
  the observation queues per request (mirroring ``select_action``).
- Policies without a ``predict_action_chunk`` override are refused.
- Unverified chunk-API policies default to exclusive; ``shared`` cannot
  be forced for them (the roadmap upstreams a
  ``supports_stateless_chunking`` attribute to policy classes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from lerobot.policies.pretrained import PreTrainedPolicy

from .manifest import (
    SERVING_MODE_EXCLUSIVE,
    SERVING_MODE_SHARED,
    PolicyServerManifest,
)
from .schema import SessionOpenMsg, StatusMsg

logger = logging.getLogger(__name__)


class ServingClass(Enum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    REFUSED = "refused"


# Verified chunk-stateless families (predict_action_chunk touches no
# cross-request instance state).
VERIFIED_CHUNK_STATELESS: frozenset[str] = frozenset({"act", "pi0", "pi05"})

# Families whose predict_action_chunk reads select_action-fed queues:
# the server must populate the observation queues per request.
QUEUE_POPULATED_IN_SELECT: frozenset[str] = frozenset({"diffusion"})

# Families whose predict_action_chunk accepts the RTC kwargs
# (inference_delay / prev_chunk_left_over) — see each family's
# ActionSelectKwargs TypedDict.
RTC_CAPABLE: frozenset[str] = frozenset({"pi0", "pi05", "smolvla"})


@dataclass
class PolicyClassification:
    serving_class: ServingClass
    supports_rtc: bool
    needs_queue_population: bool
    reason: str


def _has_chunk_api(policy: PreTrainedPolicy) -> bool:
    method = getattr(type(policy), "predict_action_chunk", None)
    return method is not None and method is not PreTrainedPolicy.predict_action_chunk


def classify_policy(policy: PreTrainedPolicy) -> PolicyClassification:
    """Classify a loaded policy into a serving class. Registry-driven, never inferred."""
    name = getattr(policy, "name", type(policy).__name__)

    if not _has_chunk_api(policy):
        return PolicyClassification(
            ServingClass.REFUSED,
            supports_rtc=False,
            needs_queue_population=False,
            reason=f"policy '{name}' does not implement predict_action_chunk",
        )

    supports_rtc = name in RTC_CAPABLE

    if name in VERIFIED_CHUNK_STATELESS:
        return PolicyClassification(
            ServingClass.SHARED, supports_rtc, False, f"'{name}' is verified chunk-stateless"
        )

    if name == "smolvla":
        n_obs_steps = getattr(policy.config, "n_obs_steps", 1)
        if n_obs_steps == 1:
            return PolicyClassification(
                ServingClass.SHARED,
                supports_rtc,
                False,
                "'smolvla' with n_obs_steps=1 overwrites its queues per request",
            )
        return PolicyClassification(
            ServingClass.EXCLUSIVE,
            supports_rtc,
            False,
            f"'smolvla' with n_obs_steps={n_obs_steps} keeps observation history across requests",
        )

    if name in QUEUE_POPULATED_IN_SELECT:
        return PolicyClassification(
            ServingClass.EXCLUSIVE,
            supports_rtc,
            True,
            f"'{name}' predict_action_chunk reads select_action-fed queues",
        )

    return PolicyClassification(
        ServingClass.EXCLUSIVE,
        supports_rtc,
        False,
        f"'{name}' has a chunk API but is not in the verified chunk-stateless registry",
    )


def resolve_serving_mode(
    classification: PolicyClassification, manifest: PolicyServerManifest
) -> tuple[str, int]:
    """Resolve the final (serving_mode, max_sessions) from classification + manifest.

    The manifest may force ``exclusive`` but can never force ``shared``
    for a policy that is not verified chunk-stateless.
    """
    if classification.serving_class is ServingClass.REFUSED:
        raise ValueError(f"Refusing to serve this policy: {classification.reason}")

    if manifest.serving_mode == SERVING_MODE_SHARED:
        if classification.serving_class is not ServingClass.SHARED:
            raise ValueError(
                f"serving_mode=shared is unsafe for this policy: {classification.reason}. "
                "Use serving_mode=exclusive (or auto)."
            )
        mode = SERVING_MODE_SHARED
    elif manifest.serving_mode == SERVING_MODE_EXCLUSIVE:
        mode = SERVING_MODE_EXCLUSIVE
    else:  # auto
        mode = (
            SERVING_MODE_SHARED
            if classification.serving_class is ServingClass.SHARED
            else SERVING_MODE_EXCLUSIVE
        )

    max_sessions = manifest.max_sessions
    if mode == SERVING_MODE_EXCLUSIVE and max_sessions != 1:
        logger.warning(
            "serving_mode=exclusive forces max_sessions=1 (manifest had %d)", manifest.max_sessions
        )
        max_sessions = 1
    return mode, max_sessions


# ---------------------------------------------------------------------------
# Session-open validation (fail fast, fail loud)
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    error: str = ""  # non-empty → hard reject
    warnings: list[str] = field(default_factory=list)
    # RTC requested but unsupported → downgrade to plain chunk-append.
    rtc_downgraded: bool = False

    @property
    def ok(self) -> bool:
        return not self.error


def validate_session_open(
    msg: SessionOpenMsg,
    capabilities: StatusMsg,
    manifest: PolicyServerManifest,
    active_sessions: int,
) -> ValidationResult:
    """Apply the capability matrix from the design doc (§8.4)."""
    result = ValidationResult()

    # Schema version: client must be within the server's supported range.
    if not (capabilities.min_schema_version <= msg.schema_version <= capabilities.max_schema_version):
        result.error = (
            f"schema_version {msg.schema_version} outside supported range "
            f"[{capabilities.min_schema_version}, {capabilities.max_schema_version}]"
        )
        return result

    # Capacity: reject with current load so the client can retry another replica.
    if active_sessions >= capabilities.max_sessions:
        result.error = f"server full: {active_sessions}/{capabilities.max_sessions} sessions active"
        return result

    # Action names AND order: the hard sync-safety contract mapping
    # chunk columns to motors.
    if capabilities.action_names and msg.action_names != capabilities.action_names:
        result.error = (
            "action feature names/order mismatch — refusing to map chunk columns to motors.\n"
            f"  server: {capabilities.action_names}\n"
            f"  client: {msg.action_names}"
        )
        return result

    # State dim.
    if capabilities.state_dim and msg.state_dim and msg.state_dim != capabilities.state_dim:
        result.error = f"state dim mismatch: server={capabilities.state_dim}, client={msg.state_dim}"
        return result

    # Camera names: the client set must cover the policy's visual features.
    missing = set(capabilities.expected_cameras) - set(msg.camera_names)
    if missing:
        result.error = (
            f"missing camera features {sorted(missing)} "
            f"(client provides {sorted(msg.camera_names)}; resolution may differ — names may not)"
        )
        return result

    # Task pinning.
    if manifest.pin_task and msg.task and msg.task != manifest.default_task:
        result.error = f"task is pinned to {manifest.default_task!r} on this server, got {msg.task!r}"
        return result

    # fps: warn unless strict.
    if capabilities.trained_fps and msg.fps and abs(msg.fps - capabilities.trained_fps) > 1e-6:
        fps_msg = f"client fps={msg.fps:g} != trained fps={capabilities.trained_fps:g}"
        if manifest.strict_fps:
            result.error = fps_msg + " (strict_fps=true)"
            return result
        result.warnings.append(fps_msg)

    # Policy type sanity (informational mismatch is a warning, not fatal:
    # the action/state/camera contracts above are the binding ones).
    if msg.policy_type and capabilities.policy_type and msg.policy_type != capabilities.policy_type:
        result.warnings.append(
            f"client expected policy_type={msg.policy_type!r}, server runs {capabilities.policy_type!r}"
        )

    # RTC: requested but unsupported → serve plain chunks, client appends.
    if msg.rtc_enabled and not capabilities.supports_rtc:
        result.rtc_downgraded = True
        result.warnings.append(
            "RTC requested but this server/policy does not support it — downgrading to chunk-append"
        )

    return result
