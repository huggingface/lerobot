from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import paired_evaluator as evaluator


def load_plan() -> dict:
    return evaluator.load_frozen_plan()


def test_frozen_plan_hash_and_research_identity() -> None:
    plan = load_plan()
    assert evaluator.sha256_file(evaluator.DEFAULT_PLAN) == evaluator.EXPECTED_PLAN_SHA256
    assert (
        evaluator.sha256_file(evaluator.RESEARCH_IDENTITY_VERIFICATION)
        == evaluator.EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256
    )
    assert plan["research_contract"]["research_repo_commit"] == evaluator.EXPECTED_RESEARCH_COMMIT
    assert plan["authorization"]["hardware_authorized"] is False


def test_exact_24_order_and_alternating_first_model() -> None:
    trials = load_plan()["trials"]
    assert len(trials) == 24
    assert [trial["order"] for trial in trials] == list(range(1, 25))
    assert [trial["cell_id"] for trial in trials[::2]] == [
        "r3c3",
        "r3c4",
        "r2c4",
        "r2c1",
        "r2c2",
        "r2c3",
        "r1c2",
        "r1c4",
        "r1c1",
        "r1c3",
        "r3c2",
        "r3c1",
    ]
    expected_models = []
    for source_order in range(1, 13):
        expected_models.extend(
            ("real24_only", "questsim48_v7")
            if source_order % 2
            else ("questsim48_v7", "real24_only")
        )
    assert [trial["model_id"] for trial in trials] == expected_models


def test_each_pair_is_same_frozen_15mm_pose() -> None:
    trials = load_plan()["trials"]
    paired_fields = (
        "source_pose_order",
        "source_pose_id",
        "source_order_sha256",
        "cell_id",
        "quadrant",
        "nominal_x_forward_m",
        "nominal_y_lateral_m",
        "nominal_yaw_degrees_modulo_90",
        "operator_placement_prompt_zh",
    )
    for left, right in zip(trials[::2], trials[1::2], strict=True):
        assert all(left[field] == right[field] for field in paired_fields)
    assert {
        quadrant: sum(trial["quadrant"] == quadrant for trial in trials[::2])
        for quadrant in (
            "x_minus_y_minus",
            "x_minus_y_plus",
            "x_plus_y_minus",
            "x_plus_y_plus",
        )
    } == {
        "x_minus_y_minus": 3,
        "x_minus_y_plus": 3,
        "x_plus_y_minus": 3,
        "x_plus_y_plus": 3,
    }
    assert {
        yaw: sum(trial["nominal_yaw_degrees_modulo_90"] == yaw for trial in trials[::2])
        for yaw in (0, 45)
    } == {0: 6, 45: 6}


def test_fake_protocol_exercises_both_models_without_devices() -> None:
    result = evaluator.run_fake_protocol(load_plan())
    assert result["fake_hardware_only"] is True
    assert result["real_device_accessed"] is False
    assert result["trials_exercised"] == 24
    assert result["policy_reset_calls"] == {"real24_only": 12, "questsim48_v7": 12}
    assert result["all_ready_before_policy"] is True
    assert result["all_ready_after_trial"] is True
    assert result["all_canonical_rgb_640x480"] is True
    assert result["all_pre_action_frames_before_policy_send"] is True
    assert result["all_official_sent_equals_requested"] is True
    assert result["all_torque_disabled"] is True
    assert result["success_contract_probe"]["held_at_window_end"] is False
    assert result["success_contract_probe"]["scored_success"] is True
    assert result["success_contract_probe"]["policy_window_unchanged"] is True


def test_ready_return_uses_60_official_send_interpolation_steps() -> None:
    result = evaluator.run_fake_interpolated_ready_probe(load_plan())
    assert result["real_device_accessed"] is False
    assert result["commands_sent"] == 60
    assert result["trajectory_rows"] == 60
    assert result["all_interpolation_phase"] is True
    assert result["first_alpha"] == pytest.approx(1 / 60)
    assert result["last_alpha"] == 1.0
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
        0.22,
        tick,
        lambda row: None,
        period=0.05,
        now_fn=Clock.now,
        sleep_fn=Clock.sleep,
    )
    assert len(records) == 3
    assert [row["scheduled_sleep_seconds"] for row in records] == [0.0, 0.0, 0.0]
    assert [row["tick_started_elapsed_seconds"] for row in records] == pytest.approx(
        [0.0, 0.08, 0.16]
    )


def test_plan_hash_tamper_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "evaluation_plan.json"
    tampered.write_bytes(evaluator.DEFAULT_PLAN.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="plan hash"):
        evaluator.load_frozen_plan(tampered)


def test_checkpoint_hash_reference_tamper_fails_closed() -> None:
    plan = copy.deepcopy(load_plan())
    plan["models"]["questsim48_v7"]["model_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        evaluator.verify_static_files(plan)


def test_static_contract_verifies_both_checkpoint_processors() -> None:
    static = evaluator.verify_static_files(load_plan())
    assert set(static["models"]) == {"real24_only", "questsim48_v7"}
    for model in static["models"].values():
        assert model["chunk_size"] == 67
        assert model["n_action_steps"] == 67
        assert model["input_features"] == {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
        }
        assert model["output_features"] == {
            "action": {"type": "ACTION", "shape": [6]}
        }
        assert model["use_imagenet_stats"] is True


def test_fresh_execution_order_and_one_linked_infrastructure_replacement(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(load_plan())
    plan["evidence_root"] = str(tmp_path)
    first, second = plan["trials"][:2]
    stem, replacement_for = evaluator.validate_execution_order(
        plan,
        first,
        replacement=False,
    )
    assert stem == first["spawn_region"]
    assert replacement_for is None
    with pytest.raises(RuntimeError, match="next missing trial"):
        evaluator.validate_execution_order(plan, second, replacement=False)

    original_path = evaluator.original_evidence_path(plan, first)
    original_path.write_text(
        json.dumps(
            {
                "status": "aborted_with_error",
                "termination": "hardware_or_runtime_error",
            }
        ),
        encoding="utf-8",
    )
    replacement_stem, replacement_for = evaluator.validate_execution_order(
        plan,
        first,
        replacement=True,
    )
    assert replacement_stem == f"{first['spawn_region']}__replacement1"
    assert replacement_for == first["spawn_region"]


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
    assert result["fake_protocol"]["trials_exercised"] == 24
