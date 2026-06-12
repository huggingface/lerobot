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

"""Unit tests for serving-mode classification and session-open validation.

Uses tiny fake policy classes (deliberately NOT subclassing
``PreTrainedPolicy``): classification keys off the ``name`` attribute and
the presence of a ``predict_action_chunk`` override, never off the class
hierarchy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.policy_server.schema import (
    MIN_SUPPORTED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SessionOpenMsg,
    StatusMsg,
)
from lerobot.policy_server.validation import (
    PolicyClassification,
    ServingClass,
    classify_policy,
    resolve_serving_mode,
    validate_session_open,
)
from tests.policy_server.conftest import ACTION_NAMES, STATE_DIM, TASK, make_manifest

# ---------------------------------------------------------------------------
# Fake policy classes (classification only needs `name`, an optional
# `predict_action_chunk` method, and `.config` for smolvla)
# ---------------------------------------------------------------------------


def _fake_policy(name: str, *, chunk_api: bool = True, n_obs_steps: int | None = None):
    """Build a minimal fake policy instance with a class-level chunk method."""

    namespace = {"name": name}
    if chunk_api:
        namespace["predict_action_chunk"] = lambda self, batch, **kwargs: None
    cls = type(f"Fake_{name}_Policy", (), namespace)
    policy = cls()
    if n_obs_steps is not None:
        policy.config = SimpleNamespace(n_obs_steps=n_obs_steps)
    return policy


# ---------------------------------------------------------------------------
# classify_policy
# ---------------------------------------------------------------------------


def test_classify_act_is_shared_without_rtc():
    classification = classify_policy(_fake_policy("act"))
    assert classification.serving_class is ServingClass.SHARED
    assert classification.supports_rtc is False
    assert classification.needs_queue_population is False


@pytest.mark.parametrize("name", ["pi0", "pi05"])
def test_classify_pi_families_are_shared_with_rtc(name):
    classification = classify_policy(_fake_policy(name))
    assert classification.serving_class is ServingClass.SHARED
    assert classification.supports_rtc is True
    assert classification.needs_queue_population is False


def test_classify_smolvla_single_obs_step_is_shared():
    classification = classify_policy(_fake_policy("smolvla", n_obs_steps=1))
    assert classification.serving_class is ServingClass.SHARED
    assert classification.supports_rtc is True


def test_classify_smolvla_with_history_is_exclusive():
    classification = classify_policy(_fake_policy("smolvla", n_obs_steps=2))
    assert classification.serving_class is ServingClass.EXCLUSIVE
    assert classification.supports_rtc is True
    assert classification.needs_queue_population is False


def test_classify_diffusion_is_exclusive_with_queue_population():
    classification = classify_policy(_fake_policy("diffusion"))
    assert classification.serving_class is ServingClass.EXCLUSIVE
    assert classification.supports_rtc is False
    assert classification.needs_queue_population is True


def test_classify_without_chunk_api_is_refused():
    classification = classify_policy(_fake_policy("act", chunk_api=False))
    assert classification.serving_class is ServingClass.REFUSED
    assert classification.supports_rtc is False
    assert "predict_action_chunk" in classification.reason


def test_classify_unknown_name_with_chunk_api_is_exclusive():
    classification = classify_policy(_fake_policy("totally_new_policy"))
    assert classification.serving_class is ServingClass.EXCLUSIVE
    assert classification.supports_rtc is False
    assert classification.needs_queue_population is False
    assert "verified" in classification.reason


# ---------------------------------------------------------------------------
# resolve_serving_mode
# ---------------------------------------------------------------------------


def _classification(serving_class: ServingClass, reason: str = "test") -> PolicyClassification:
    return PolicyClassification(
        serving_class, supports_rtc=False, needs_queue_population=False, reason=reason
    )


def test_resolve_auto_maps_shared_to_shared():
    mode, max_sessions = resolve_serving_mode(
        _classification(ServingClass.SHARED), make_manifest(serving_mode="auto", max_sessions=4)
    )
    assert mode == "shared"
    assert max_sessions == 4


def test_resolve_auto_maps_exclusive_to_exclusive():
    mode, max_sessions = resolve_serving_mode(
        _classification(ServingClass.EXCLUSIVE), make_manifest(serving_mode="auto", max_sessions=4)
    )
    assert mode == "exclusive"
    assert max_sessions == 1


def test_resolve_forced_shared_rejected_for_non_verified_policy():
    with pytest.raises(ValueError, match="unsafe"):
        resolve_serving_mode(_classification(ServingClass.EXCLUSIVE), make_manifest(serving_mode="shared"))


def test_resolve_forced_shared_allowed_for_verified_policy():
    mode, max_sessions = resolve_serving_mode(
        _classification(ServingClass.SHARED), make_manifest(serving_mode="shared", max_sessions=4)
    )
    assert mode == "shared"
    assert max_sessions == 4


def test_resolve_forced_exclusive_allowed_for_shared_policy():
    mode, _ = resolve_serving_mode(
        _classification(ServingClass.SHARED), make_manifest(serving_mode="exclusive")
    )
    assert mode == "exclusive"


def test_resolve_exclusive_forces_single_session():
    mode, max_sessions = resolve_serving_mode(
        _classification(ServingClass.SHARED), make_manifest(serving_mode="exclusive", max_sessions=4)
    )
    assert mode == "exclusive"
    assert max_sessions == 1


def test_resolve_refused_raises_with_reason():
    with pytest.raises(ValueError, match="no chunk API here"):
        resolve_serving_mode(
            _classification(ServingClass.REFUSED, reason="no chunk API here"), make_manifest()
        )


# ---------------------------------------------------------------------------
# validate_session_open
# ---------------------------------------------------------------------------

EXPECTED_CAMERAS = ["observation.images.front"]


@pytest.fixture
def capabilities() -> StatusMsg:
    return StatusMsg(
        model_repo="mock/model",
        policy_type="mockchunk",
        action_names=list(ACTION_NAMES),
        expected_cameras=list(EXPECTED_CAMERAS),
        state_dim=STATE_DIM,
        chunk_size=20,
        trained_fps=30.0,
        supports_rtc=True,
        min_schema_version=MIN_SUPPORTED_SCHEMA_VERSION,
        max_schema_version=SCHEMA_VERSION,
        max_sessions=4,
    )


def _open_msg(**overrides) -> SessionOpenMsg:
    kwargs: dict = {
        "client_uuid": "client-1",
        "policy_type": "mockchunk",
        "fps": 30.0,
        "action_names": list(ACTION_NAMES),
        "camera_names": list(EXPECTED_CAMERAS),
        "state_dim": STATE_DIM,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
    }
    kwargs.update(overrides)
    return SessionOpenMsg(**kwargs)


def test_validate_happy_path(capabilities):
    result = validate_session_open(_open_msg(), capabilities, make_manifest(), active_sessions=0)
    assert result.ok
    assert result.error == ""
    assert result.warnings == []
    assert result.rtc_downgraded is False


def test_validate_action_name_order_is_a_hard_reject(capabilities):
    # Same set of names, different order: chunk columns would map to the
    # wrong motors, so this must be a hard reject.
    result = validate_session_open(
        _open_msg(action_names=list(reversed(ACTION_NAMES))),
        capabilities,
        make_manifest(),
        active_sessions=0,
    )
    assert not result.ok
    assert "action" in result.error
    assert "mismatch" in result.error


def test_validate_missing_camera_rejected(capabilities):
    result = validate_session_open(
        _open_msg(camera_names=[]), capabilities, make_manifest(), active_sessions=0
    )
    assert not result.ok
    assert "observation.images.front" in result.error


def test_validate_wrong_state_dim_rejected(capabilities):
    result = validate_session_open(
        _open_msg(state_dim=STATE_DIM + 1), capabilities, make_manifest(), active_sessions=0
    )
    assert not result.ok
    assert "state dim" in result.error


def test_validate_schema_version_out_of_range_rejected(capabilities):
    result = validate_session_open(
        _open_msg(schema_version=SCHEMA_VERSION + 99),
        capabilities,
        make_manifest(),
        active_sessions=0,
    )
    assert not result.ok
    assert "schema_version" in result.error


def test_validate_at_capacity_rejected_with_load(capabilities):
    result = validate_session_open(
        _open_msg(), capabilities, make_manifest(), active_sessions=capabilities.max_sessions
    )
    assert not result.ok
    assert "full" in result.error
    assert f"{capabilities.max_sessions}/{capabilities.max_sessions}" in result.error


def test_validate_pinned_task_rejects_other_task(capabilities):
    result = validate_session_open(
        _open_msg(task="another task"),
        capabilities,
        make_manifest(pin_task=True),
        active_sessions=0,
    )
    assert not result.ok
    assert "pinned" in result.error


def test_validate_fps_mismatch_strict_rejects(capabilities):
    result = validate_session_open(
        _open_msg(fps=15.0), capabilities, make_manifest(strict_fps=True), active_sessions=0
    )
    assert not result.ok
    assert "fps" in result.error


def test_validate_fps_mismatch_lenient_warns_only(capabilities):
    result = validate_session_open(
        _open_msg(fps=15.0), capabilities, make_manifest(strict_fps=False), active_sessions=0
    )
    assert result.ok
    assert len(result.warnings) == 1
    assert "fps" in result.warnings[0]


def test_validate_rtc_downgraded_when_unsupported(capabilities):
    capabilities.supports_rtc = False
    result = validate_session_open(
        _open_msg(rtc_enabled=True), capabilities, make_manifest(), active_sessions=0
    )
    assert result.ok
    assert result.rtc_downgraded is True
    assert any("RTC" in warning for warning in result.warnings)


def test_validate_empty_capability_action_names_skips_action_check(capabilities):
    capabilities.action_names = []
    result = validate_session_open(
        _open_msg(action_names=["whatever.pos"]),
        capabilities,
        make_manifest(),
        active_sessions=0,
    )
    assert result.ok
