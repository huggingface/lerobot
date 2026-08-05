from __future__ import annotations

import copy
import json
from pathlib import Path

import generate_evaluation_plan as generator
import paired_evaluator as evaluator
import pytest


def load_plan() -> dict:
    return evaluator.load_frozen_plan()


def test_frozen_plan_and_research_source_hashes() -> None:
    plan = load_plan()
    assert evaluator.sha256_file(evaluator.DEFAULT_PLAN) == evaluator.EXPECTED_PLAN_SHA256
    assert (
        evaluator.sha256_file(evaluator.SOURCE_RESEARCH_CONTRACT)
        == evaluator.EXPECTED_RESEARCH_HASHES["experiment_design"]
    )
    assert (
        evaluator.sha256_file(evaluator.SOURCE_TRAINING_RESULT)
        == evaluator.EXPECTED_RESEARCH_HASHES["training_result"]
    )
    assert (
        evaluator.sha256_file(evaluator.SOURCE_POSE_MANIFEST)
        == evaluator.EXPECTED_RESEARCH_HASHES["pose_manifest"]
    )
    assert plan["research_contract"]["research_repo_commit"] == evaluator.EXPECTED_RESEARCH_COMMIT
    assert plan["authorization"]["hardware_authorized"] is False


def test_plan_is_reproducible_from_frozen_pose_manifest() -> None:
    assert generator.build_plan() == load_plan()


def test_exact_96_order_matches_frozen_pose_manifest() -> None:
    plan = load_plan()
    trials = plan["trials"]
    poses = json.loads(evaluator.SOURCE_POSE_MANIFEST.read_text(encoding="utf-8"))["ordered_eval_poses"]
    assert len(trials) == 96
    assert [trial["order"] for trial in trials] == list(range(1, 97))
    assert [trial["eval_pose_id"] for trial in trials[::2]] == [pose["eval_pose_id"] for pose in poses]
    assert [trial["model_key"] for trial in trials[:6]] == [
        "real24_localsim24_gap",
        "real24_localsim48_full",
        "real24_localsim48_full",
        "real24_localsim24_gap",
        "real24_localsim24_gap",
        "real24_localsim48_full",
    ]
    assert {key: sum(trial["model_key"] == key for trial in trials) for key in evaluator.MODEL_IDS} == {
        "real24_localsim24_gap": 48,
        "real24_localsim48_full": 48,
    }


def test_each_pair_preserves_identical_frozen_pose() -> None:
    trials = load_plan()["trials"]
    fields = (
        "pose_order",
        "eval_pose_id",
        "source_order_sha256",
        "cell",
        "coverage_tier",
        "quadrant",
        "nominal_x_forward_m",
        "nominal_y_lateral_m",
        "nominal_yaw_degrees_modulo_90",
    )
    for left, right in zip(trials[::2], trials[1::2], strict=True):
        assert all(left[field] == right[field] for field in fields)
        assert {left["model_key"], right["model_key"]} == set(evaluator.MODEL_IDS)
    source = trials[::2]
    assert {
        tier: sum(row["coverage_tier"] == tier for row in source)
        for tier in ("seen_by_real48", "added_by_real96", "unseen_by_both")
    } == {"seen_by_real48": 24, "added_by_real96": 18, "unseen_by_both": 6}
    assert {yaw: sum(row["nominal_yaw_degrees_modulo_90"] == yaw for row in source) for yaw in (0, 45)} == {
        0: 24,
        45: 24,
    }


def test_fake_protocol_exercises_all_trials_without_devices() -> None:
    result = evaluator.run_fake_protocol(load_plan())
    assert result["fake_hardware_only"] is True
    assert result["real_device_accessed"] is False
    assert result["trials_exercised"] == 96
    assert result["policy_reset_calls"] == {
        "real24_localsim24_gap": 48,
        "real24_localsim48_full": 48,
    }
    assert result["all_ready_before_policy"] is True
    assert result["all_ready_after_trial"] is True
    assert result["all_canonical_rgb_640x480"] is True
    assert result["all_pre_action_frames_before_policy_send"] is True
    assert result["all_official_sent_equals_requested"] is True
    assert result["all_torque_disabled"] is True


def test_success_contract_is_strictly_over_5cm_and_never_early_stops() -> None:
    plan = load_plan()
    assert plan["success_contract"]["unsupported_lift_strictly_greater_than_m"] == 0.05
    assert plan["success_contract"]["continuous_hold_seconds_minimum"] == 0.5
    assert plan["success_contract"]["changes_policy_action_window"] is False
    assert plan["setup"]["stop_on_success"] is False


def test_ready_return_uses_60_official_send_interpolation_steps() -> None:
    result = evaluator.run_fake_interpolated_ready_probe(load_plan())
    assert result["commands_sent"] == 60
    assert result["trajectory_rows"] == 60
    assert result["all_official_sent_equals_requested"] is True
    assert result["requested_final_state"] == load_plan()["setup"]["ready_pose_state"]


def test_slow_tick_never_generates_catch_up_burst() -> None:
    class Clock:
        value = 0.0

        @classmethod
        def now(cls) -> float:
            return cls.value

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.value += seconds

    def tick(step: int, tick_started: float, loop_started: float) -> dict:
        del step, tick_started, loop_started
        Clock.value += 0.08
        return {}

    records = evaluator.run_paced_ticks(
        0.22, tick, lambda row: None, period=0.05, now_fn=Clock.now, sleep_fn=Clock.sleep
    )
    assert len(records) == 3
    assert [row["scheduled_sleep_seconds"] for row in records] == [0.0, 0.0, 0.0]
    assert [row["tick_started_elapsed_seconds"] for row in records] == pytest.approx([0.0, 0.08, 0.16])


def test_plan_hash_tamper_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "evaluation_plan.json"
    tampered.write_bytes(evaluator.DEFAULT_PLAN.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="plan hash"):
        evaluator.load_frozen_plan(tampered)


def test_checkpoint_hash_tamper_fails_closed() -> None:
    plan = copy.deepcopy(load_plan())
    plan["models"]["real24_localsim48_full"]["model_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        evaluator.verify_static_files(plan)


def test_static_contract_verifies_models_processor_calibration_and_no_devices() -> None:
    static = evaluator.verify_static_files(load_plan())
    assert set(static["models"]) == set(evaluator.MODEL_IDS)
    assert static["hardware_identity_contract"]["devices_opened"] is False
    assert (
        static["hardware_identity_contract"]["calibration_sha256"]
        == load_plan()["setup"]["follower_calibration_sha256"]
    )
    for model in static["models"].values():
        assert model["chunk_size"] == 67
        assert model["n_action_steps"] == 67
        assert model["input_features"] == {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
        }
        assert model["output_features"] == {"action": {"type": "ACTION", "shape": [6]}}


def test_fresh_order_and_one_linked_replacement(tmp_path: Path) -> None:
    plan = copy.deepcopy(load_plan())
    plan["evidence_root"] = str(tmp_path)
    first, second = plan["trials"][:2]
    stem, linked = evaluator.validate_execution_order(plan, first, replacement=False)
    assert stem == first["artifact_stem"] and linked is None
    with pytest.raises(RuntimeError, match="next missing trial"):
        evaluator.validate_execution_order(plan, second, replacement=False)
    original = evaluator.original_evidence_path(plan, first)
    original.write_text(
        json.dumps({"status": "aborted_with_error", "termination": "hardware_or_runtime_error"}),
        encoding="utf-8",
    )
    replacement, linked = evaluator.validate_execution_order(plan, first, replacement=True)
    assert replacement == f"{first['artifact_stem']}__replacement1"
    assert linked == first["artifact_stem"]


def test_evidence_schema_covers_execution_review_and_reporting_contracts() -> None:
    contract = load_plan()["evidence_contract"]
    per_trial = " ".join(contract["per_trial"])
    reporting = set(contract["reporting"])
    for fragment in (
        "pre-action canonical frame",
        "requested/observed ready",
        "policy reset and tick0",
        "canonical video",
        "raw/requested/official sent",
        "operator label",
        "canonical-video review label",
        "adjudication",
        "return trajectory",
        "torque disable",
    ):
        assert fragment in per_trial
    assert reporting == {
        "overall by model",
        "coverage tiers 24/18/6",
        "cell",
        "yaw",
        "failure categories",
        "paired LocalSim48-full minus LocalSim24-gap success difference",
    }


def test_full_software_dry_run_has_no_hardware_access() -> None:
    result = evaluator.software_dry_run(load_plan())
    assert result["status"] == "software_dry_run_passed_hardware_not_accessed"
    assert result["hardware_access"] == {
        "serial": False,
        "camera": False,
        "robot": False,
        "torque": False,
        "rollout": False,
    }
    assert result["fake_protocol"]["trials_exercised"] == 96
