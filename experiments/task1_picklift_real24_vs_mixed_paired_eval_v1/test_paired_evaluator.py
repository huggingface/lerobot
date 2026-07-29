from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import paired_evaluator as evaluator


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_frozen_plan_has_exact_24_trial_paired_order() -> None:
    plan = evaluator.load_frozen_plan()
    assert len(plan["trials"]) == 24
    assert [(row["cell_id"], row["model_id"]) for row in plan["trials"]] == list(
        evaluator.EXPECTED_SCHEDULE
    )
    assert {
        model_id: sum(row["model_id"] == model_id for row in plan["trials"])
        for model_id in evaluator.MODEL_IDS
    } == {"real24_only": 12, "real24_questsim24": 12}


def test_plan_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    altered = tmp_path / "altered_plan.json"
    altered.write_text(
        evaluator.DEFAULT_PLAN.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="plan hash"):
        evaluator.load_frozen_plan(altered)


def test_checkpoint_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    plan = copy.deepcopy(evaluator.load_frozen_plan())
    bad_checkpoint = tmp_path / "bad_checkpoint"
    bad_checkpoint.mkdir()
    (bad_checkpoint / "model.safetensors").write_bytes(b"not the frozen model")
    (bad_checkpoint / "config.json").write_text(
        json.dumps({"type": "act", "chunk_size": 67, "n_action_steps": 67}),
        encoding="utf-8",
    )
    plan["models"]["real24_only"]["checkpoint"] = str(bad_checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        evaluator.verify_static_files(plan)


def test_fake_robot_camera_bus_exercises_both_models_and_same_ready_pose() -> None:
    plan = evaluator.load_frozen_plan()
    result = evaluator.run_fake_protocol(plan)
    assert result["fake_hardware_only"] is True
    assert result["real_device_accessed"] is False
    assert result["trials_exercised"] == 24
    assert result["models_in_frozen_order"] == [
        trial["model_id"] for trial in plan["trials"]
    ]
    assert result["policy_reset_calls"] == {
        "real24_only": 12,
        "real24_questsim24": 12,
    }
    assert result["all_ready_before_policy"] is True
    assert result["all_ready_after_trial"] is True
    assert result["all_canonical_rgb_640x480"] is True
    assert result["all_official_sent_equals_requested"] is True
    assert result["all_torque_disabled"] is True
    ready = plan["setup"]["ready_pose_state"]
    assert all(row["tick0_state"] == ready for row in result["records"])


def test_success_inside_window_does_not_require_held_at_end() -> None:
    plan = evaluator.load_frozen_plan()
    result = evaluator.run_fake_protocol(plan)["success_contract_probe"]
    assert result == {
        "valid_success_seen_inside_window": True,
        "held_at_window_end": False,
        "scored_success": True,
        "policy_window_unchanged": True,
    }


def test_slow_tick_does_not_create_catch_up_burst() -> None:
    clock = FakeClock()
    tick_compute = [0.20, 0.01, 0.01]
    starts: list[float] = []
    records: list[dict] = []

    def tick(step: int, tick_started: float, loop_started: float) -> dict:
        del loop_started
        starts.append(tick_started)
        clock.advance(tick_compute[step])
        return {}

    evaluator.run_paced_ticks(
        0.26,
        tick,
        lambda row: records.append(dict(row)),
        period=0.05,
        now_fn=clock.now,
        sleep_fn=clock.advance,
    )
    np.testing.assert_allclose(starts, [0.0, 0.20, 0.25])
    assert [row["step"] for row in records] == [0, 1, 2]
    assert records[0]["scheduled_sleep_seconds"] == 0.0
    assert records[1]["scheduled_sleep_seconds"] == pytest.approx(0.04)


def test_software_dry_run_never_loads_hardware_engine(monkeypatch) -> None:
    plan = evaluator.load_frozen_plan()
    monkeypatch.setattr(
        evaluator,
        "verify_static_files",
        lambda unused_plan: {
            "plan_sha256": evaluator.EXPECTED_PLAN_SHA256,
            "models": {},
        },
    )
    monkeypatch.setattr(
        evaluator,
        "load_official_engine",
        lambda unused_plan: (_ for _ in ()).throw(
            AssertionError("hardware engine must not load in software dry-run")
        ),
    )
    result = evaluator.software_dry_run(plan)
    assert result["status"] == "software_dry_run_passed_hardware_not_accessed"
    assert all(value is False for value in result["hardware_access"].values())


def test_official_send_engine_is_exact_pinned_source() -> None:
    plan = evaluator.load_frozen_plan()
    path = evaluator.resolve_repo_path(plan["execution_engine"]["path"])
    assert evaluator.sha256_file(path) == evaluator.EXPECTED_ENGINE_SHA256
    source = path.read_text(encoding="utf-8")
    assert "max_relative_target=None" in source
    assert "requested_action = raw_action.copy()" in source
    assert "robot.send_action(action_dict(requested_action))" in source
    assert "steps = run_paced_ticks" in source
    assert "return_result = move_to_frozen_ready_pose" in source


def test_frozen_order_rejects_skips_and_allows_one_linked_infrastructure_replacement(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(evaluator.load_frozen_plan())
    plan["evidence_root"] = str(tmp_path)
    first = plan["trials"][0]
    second = plan["trials"][1]
    with pytest.raises(RuntimeError, match="next missing trial"):
        evaluator.validate_execution_order(plan, second, replacement=False)
    stem, link = evaluator.validate_execution_order(
        plan,
        first,
        replacement=False,
    )
    assert stem == first["spawn_region"]
    assert link is None
    original = evaluator.original_evidence_path(plan, first)
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(
        json.dumps(
            {
                "status": "aborted_with_error",
                "termination": "hardware_or_runtime_error",
            }
        ),
        encoding="utf-8",
    )
    replacement_stem, link = evaluator.validate_execution_order(
        plan,
        first,
        replacement=True,
    )
    assert replacement_stem.endswith("__replacement1")
    assert link == first["spawn_region"]
    (original.parent / f"{replacement_stem}.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="already exists"):
        evaluator.validate_execution_order(plan, first, replacement=True)
