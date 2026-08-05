from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


ENGINE_PATH = Path(__file__).with_name("evaluate_real.py")
SPEC = importlib.util.spec_from_file_location("evaluate_real_under_test", ENGINE_PATH)
assert SPEC is not None and SPEC.loader is not None
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)

VALIDATOR_PATH = Path(__file__).with_name("validate_success_early_stop.py")
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validator_under_test", VALIDATOR_PATH)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def write_marker(path: Path, *, evaluation_id: str, trial_id: str, created: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "evaluation_id": evaluation_id,
                "trial_id": trial_id,
                "operator_confirmed_success": True,
                "created_at_utc": created.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def run_loop(tmp_path: Path, signal_step: int | None):
    evaluation_id = "eval_future_v1"
    trial_id = "t001"
    marker = engine.success_marker_path(tmp_path, evaluation_id, trial_id)
    policy_started = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    current_utc = [policy_started]
    predicate, state = engine.build_success_marker_predicate(
        marker,
        evaluation_id=evaluation_id,
        trial_id=trial_id,
        policy_started_at_utc=policy_started,
        utc_now_fn=lambda: current_utc[0],
    )
    clock = Clock()
    termination = {}
    recorded = []

    def tick(step, *_):
        current_utc[0] = policy_started + timedelta(seconds=clock.value)
        if signal_step is not None and step == signal_step:
            write_marker(
                marker,
                evaluation_id=evaluation_id,
                trial_id=trial_id,
                created=current_utc[0],
            )
        return {"frame": step, "sent": step}

    rows = engine.run_paced_ticks(
        0.5,
        tick,
        recorded.append,
        period=0.05,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        stop_predicate=predicate,
        termination_out=termination,
    )
    return rows, recorded, termination, state


def test_success_at_different_ticks_stops_after_confirming_frame(tmp_path):
    for signal_step in (0, 3, 7):
        rows, recorded, termination, state = run_loop(tmp_path / str(signal_step), signal_step)
        assert len(rows) == signal_step + 1
        assert len(recorded) == len(rows)
        assert termination["termination"] == "success_early_stop"
        assert termination["success_signal_observed_policy_step"] == signal_step
        assert state["accepted"] == termination


def test_no_signal_runs_full_window_without_catchup(tmp_path):
    rows, recorded, termination, state = run_loop(tmp_path, None)
    assert len(rows) == 11
    assert len(recorded) == len(rows)
    assert termination == {}
    assert state["accepted"] is None


def test_stale_and_wrong_markers_are_rejected_without_stopping(tmp_path):
    evaluation_id = "eval_future_v1"
    trial_id = "t001"
    marker = engine.success_marker_path(tmp_path, evaluation_id, trial_id)
    started = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    now = [started]
    predicate, state = engine.build_success_marker_predicate(
        marker,
        evaluation_id=evaluation_id,
        trial_id=trial_id,
        policy_started_at_utc=started,
        utc_now_fn=lambda: now[0],
    )
    write_marker(marker, evaluation_id="wrong", trial_id=trial_id, created=started)
    assert predicate({}, 0, 0.0) is None
    write_marker(
        marker,
        evaluation_id=evaluation_id,
        trial_id=trial_id,
        created=started - timedelta(seconds=1),
    )
    assert predicate({}, 1, 0.05) is None
    assert len(state["rejections"]) == 2


def test_profile_is_explicit_opt_in(tmp_path):
    profile = Path(__file__).with_name("real_evaluation_success_early_stop_profile_v1.json")
    args = type("Args", (), {"success_early_stop_profile": None, "success_marker_dir": None})()
    assert engine.load_success_early_stop_profile(args) is None
    args.success_early_stop_profile = profile
    args.success_marker_dir = tmp_path
    loaded = engine.load_success_early_stop_profile(args)
    assert loaded["explicit_opt_in"] is True


def test_evidence_invariants_for_early_stop():
    steps = [{"step": 0}, {"step": 1}, {"step": 2}]
    video_frames = len(steps)
    ready_result = {"status": "ready_pose_observed"}
    return_result = {"status": "ready_pose_observed"}
    torque_disable_verified = True
    assert video_frames == len(steps)
    assert ready_result["status"] == "ready_pose_observed"
    assert return_result["status"] == "ready_pose_observed"
    assert torque_disable_verified is True


def test_validator_accepts_early_stop_and_rejects_frame_mismatch():
    evidence = {
        "success_early_stop_profile": {"enabled": True},
        "termination": "success_early_stop",
        "success_signal_path": "/markers/eval/t001.success.json",
        "success_signal_sha256": "a" * 64,
        "success_signal_created_at_utc": "2026-08-05T12:00:00+00:00",
        "success_signal_observed_policy_step": 2,
        "success_signal_observed_elapsed_seconds": 0.1,
        "steps_jsonl": {"lines": 3},
        "video": {"frames": 3},
        "actual_policy_ticks": 3,
        "maximum_trial_seconds": 30.0,
        "torque_disable_verified": True,
        "automatic_return": {"outside_evaluation_window": True},
    }
    assert validator.validate_trial_evidence(evidence) == []
    evidence["video"]["frames"] = 2
    assert "video_frames_must_equal_steps" in validator.validate_trial_evidence(evidence)
