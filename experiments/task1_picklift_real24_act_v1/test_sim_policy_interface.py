from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deployment_safety import (
    CALIBRATION_BOUNDS_DEG,
    SIM_STATE_CALIBRATION_TOLERANCE,
    apply_action_safety,
    project_sim_state_to_calibration,
)
from sim_policy_inference import validate_observation
from summarize_sim_results import load_jsonl, summarize


def test_validate_observation_accepts_contract() -> None:
    state, image = validate_observation(
        np.zeros(6, dtype=np.float32),
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    ("state", "image", "message"),
    [
        (np.zeros(5, dtype=np.float32), np.zeros((480, 640, 3), dtype=np.uint8), "shape"),
        (np.zeros(6, dtype=np.int64), np.zeros((480, 640, 3), dtype=np.uint8), "floating"),
        (np.zeros(6, dtype=np.float32), np.zeros((3, 480, 640), dtype=np.uint8), "RGB shape"),
        (np.zeros(6, dtype=np.float32), np.zeros((480, 640, 3), dtype=np.float32), "uint8"),
    ],
)
def test_validate_observation_fails_closed(
    state: np.ndarray, image: np.ndarray, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_observation(state, image)


def test_action_safety_applies_calibration_then_relative_clamp() -> None:
    current = np.zeros(6, dtype=np.float32)
    raw = np.asarray([200, -200, 4, -4, 6, 200], dtype=np.float32)
    stages = apply_action_safety(raw, current)
    np.testing.assert_allclose(stages["sent_action"], [5, -5, 4, -4, 5, 5])
    assert stages["calibration_clip_mask"].tolist() == [True, True, False, False, False, True]
    assert stages["relative_clip_mask"].tolist() == [True, True, False, False, True, True]


def test_action_safety_rejects_state_outside_calibration() -> None:
    state = np.zeros(6)
    state[0] = CALIBRATION_BOUNDS_DEG[0, 1] + 0.1
    with pytest.raises(RuntimeError, match="outside"):
        apply_action_safety(np.zeros(6), state)


def test_sim_state_projection_accepts_only_tiny_boundary_noise() -> None:
    state = np.zeros(6, dtype=np.float32)
    state[-1] = -0.0017
    projected = project_sim_state_to_calibration(state)
    np.testing.assert_allclose(projected["state"], np.zeros(6))
    assert projected["projection_mask"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert projected["projection_delta"][-1] == pytest.approx(0.0017)


def test_sim_state_projection_rejects_coordinate_mismatch() -> None:
    state = np.zeros(6, dtype=np.float32)
    state[-1] = -(SIM_STATE_CALIBRATION_TOLERANCE + 0.001)
    with pytest.raises(RuntimeError, match="coordinate contract mismatch"):
        project_sim_state_to_calibration(state)


def test_sim_state_projection_accepts_measured_endpoint_tracking_overshoot() -> None:
    state = np.zeros(6, dtype=np.float32)
    state[1] = -98.9798355102539
    projected = project_sim_state_to_calibration(state)
    assert projected["projection_mask"][1]
    assert projected["projection_delta"][1] == pytest.approx(
        0.03478056519895745
    )
    assert projected["state"][1] == pytest.approx(
        CALIBRATION_BOUNDS_DEG[1, 0]
    )


def test_summary_reports_overall_and_per_cell(tmp_path: Path) -> None:
    base = {
        "phase_id": "gate12",
        "seed": 9000,
        "initial_pose": [0, 0, 0, 0, 0, 0],
        "env_steps": 10,
        "first_success_step": None,
        "confirmed_success_step": None,
        "max_lift_m": 0.01,
        "is_grasped": False,
        "terminated": False,
        "truncated": True,
        "timeout": True,
        "termination_reason": "timeout",
        "raw_action_count": 10,
        "calibration_clipped_action_count": 1,
        "relative_clipped_action_count": 5,
        "sent_action_count": 10,
    }
    records = [
        {**base, "cell": "r1c1", "success": True, "failure_type": None},
        {**base, "cell": "r1c1", "success": False, "failure_type": "timeout"},
        {**base, "cell": "r1c2", "success": False, "failure_type": "missed_grasp"},
    ]
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    payload = summarize(load_jsonl(path))
    assert payload["overall"]["episodes"] == 3
    assert payload["overall"]["success_rate"] == pytest.approx(1 / 3)
    assert payload["by_cell"]["r1c1"]["success_rate"] == 0.5
    assert payload["by_cell"]["r1c2"]["failure_types"] == {"missed_grasp": 1}
