from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import evalv2_pilot as evaluator
import numpy as np
import pytest


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_frozen_plan_has_exact_research_pose_order_and_balance() -> None:
    plan = evaluator.load_frozen_plan()
    assert plan["research_contract"]["research_repo_commit"] == evaluator.EXPECTED_RESEARCH_COMMIT
    assert {
        key: plan["research_contract"][key]["sha256"]
        for key in evaluator.EXPECTED_RESEARCH_HASHES
    } == evaluator.EXPECTED_RESEARCH_HASHES
    assert plan["protocol_revision"]["immutable_completed_orders"] == [1, 2, 3, 4]
    assert plan["protocol_revision"]["continuation_orders"] == list(range(5, 13))
    assert plan["protocol_revision"]["scored_window_changed"] is False
    assert len(plan["trials"]) == 12
    assert [
        (
            row["trial_id"],
            row["cell_id"],
            row["quadrant"],
            row["nominal_yaw_degrees_modulo_90"],
        )
        for row in plan["trials"]
    ] == list(evaluator.EXPECTED_POSE_SCHEDULE)
    assert {row["cell_id"] for row in plan["trials"]} == {
        f"r{row}c{column}" for row in range(1, 4) for column in range(1, 5)
    }
    assert {
        quadrant: sum(row["quadrant"] == quadrant for row in plan["trials"])
        for quadrant in ("x_minus_y_minus", "x_minus_y_plus", "x_plus_y_minus", "x_plus_y_plus")
    } == {
        "x_minus_y_minus": 3,
        "x_minus_y_plus": 3,
        "x_plus_y_minus": 3,
        "x_plus_y_plus": 3,
    }
    assert {
        yaw: sum(row["nominal_yaw_degrees_modulo_90"] == yaw for row in plan["trials"])
        for yaw in (0, 45)
    } == {0: 6, 45: 6}


def test_v2_coordinates_use_15mm_offsets_and_10mm_lattice() -> None:
    plan = evaluator.load_frozen_plan()
    centers = {
        "r1": 0.225,
        "r2": 0.275,
        "r3": 0.325,
        "c1": -0.075,
        "c2": -0.025,
        "c3": 0.025,
        "c4": 0.075,
    }
    assert [
        (row["nominal_x_forward_m"], row["nominal_y_lateral_m"])
        for row in plan["trials"]
    ] == list(evaluator.EXPECTED_NOMINAL_XY)
    for row in plan["trials"]:
        assert abs(row["nominal_x_forward_m"] - centers[row["cell_id"][:2]]) == pytest.approx(0.015)
        assert abs(row["nominal_y_lateral_m"] - centers[row["cell_id"][2:]]) == pytest.approx(0.015)
        assert row["nominal_x_forward_m"] / 0.01 == pytest.approx(
            round(row["nominal_x_forward_m"] / 0.01)
        )
        assert row["nominal_y_lateral_m"] / 0.01 == pytest.approx(
            round(row["nominal_y_lateral_m"] / 0.01)
        )
        predecessor_pilot_id = "task1_picklift_offcenter_yaw_eval_v2_difficulty_pilot_v1"
        expected_order_hash = hashlib.sha256(
            f"{predecessor_pilot_id}|{row['cell_id']}".encode()
        ).hexdigest()
        assert row["source_order_sha256"] == expected_order_hash


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


def test_fake_robot_camera_bus_exercises_all_12_poses() -> None:
    plan = evaluator.load_frozen_plan()
    result = evaluator.run_fake_protocol(plan)
    assert result["fake_hardware_only"] is True
    assert result["real_device_accessed"] is False
    assert result["trials_exercised"] == 12
    assert result["pose_trial_ids_in_frozen_order"] == [row["trial_id"] for row in plan["trials"]]
    assert result["models_in_frozen_order"] == ["real24_only"] * 12
    assert result["policy_reset_calls"] == {"real24_only": 12}
    assert result["all_ready_before_policy"] is True
    assert result["all_ready_after_trial"] is True
    assert result["all_canonical_rgb_640x480"] is True
    assert result["all_pre_action_frames_before_policy_send"] is True
    assert result["all_official_sent_equals_requested"] is True
    assert result["all_torque_disabled"] is True
    ready = plan["setup"]["ready_pose_state"]
    assert all(row["tick0_state"] == ready for row in result["records"])


def test_operator_prompts_use_repeatable_grid_coordinates() -> None:
    plan = evaluator.load_frozen_plan()
    prompts = [row["operator_placement_prompt_zh"] for row in plan["trials"]]
    assert prompts[0] == (
        "请把红色方块中心放在任务网格坐标 x=31cm、y=+4cm 的交点；"
        "方块边与网格线平行（0°）。"
    )
    assert all("任务网格坐标" in prompt and "交点" in prompt for prompt in prompts)
    assert all(("网格线平行" in prompt) or ("转45°" in prompt) for prompt in prompts)
    assert all("0.3125" not in prompt and "0.0125" not in prompt for prompt in prompts)
    assert plan["placement_contract"]["operator_numeric_coordinate_entry_required"] is True
    assert plan["placement_contract"]["nominal_center_offset_from_cell_center_m"] == 0.015
    assert plan["placement_contract"]["nominal_center_lattice_m"] == 0.01
    assert plan["placement_contract"]["manual_pose_is_measurement_truth"] is False


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


def test_fake_ready_return_uses_exact_three_second_interpolation() -> None:
    plan = evaluator.load_frozen_plan()
    result = evaluator.run_fake_interpolated_ready_probe(plan)
    assert result["fake_hardware_only"] is True
    assert result["real_device_accessed"] is False
    assert result["commands_sent"] == 60
    assert result["trajectory_rows"] == 60
    assert result["all_interpolation_phase"] is True
    assert result["first_alpha"] == pytest.approx(1.0 / 60.0)
    assert result["last_alpha"] == 1.0
    assert result["requested_final_state"] == plan["setup"]["ready_pose_state"]
    assert result["maximum_requested_step_degrees"] == pytest.approx(1.5)
    assert result["expected_maximum_requested_step_degrees"] == pytest.approx(1.5)
    assert result["all_official_sent_equals_requested"] is True
    assert result["elapsed_seconds"] == pytest.approx(3.0)
    assert result["result"]["settle_steps"] == 0


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
    assert result["fake_interpolated_ready_return"]["commands_sent"] == 60


def test_official_send_and_pre_action_order_are_exact_pinned_source() -> None:
    plan = evaluator.load_frozen_plan()
    result = evaluator.verify_static_files(plan)
    assert result["official_send_contract"] == {
        "max_relative_target_none": True,
        "runner_absolute_clamp": False,
        "runner_step_limiter": None,
        "no_catch_up_pacing": True,
        "canonical_frame_written_before_inference_and_send": True,
        "ready_return_trajectory": "linear_3s_20hz_then_hold_target",
        "ready_return_official_send": True,
    }


def test_checkpoint_owned_processor_uses_saved_imagenet_stats() -> None:
    result = evaluator.verify_static_files(evaluator.load_frozen_plan())
    assert set(result["models"]) == {"real24_only"}
    model = result["models"]["real24_only"]
    assert model["model_sha256"] == evaluator.EXPECTED_MODEL_SHA256
    assert model["use_imagenet_stats"] is True
    assert model["visual_mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert model["visual_std"] == pytest.approx([0.229, 0.224, 0.225])


def test_pre_action_sidecar_is_nominal_not_measurement_truth(tmp_path: Path, monkeypatch) -> None:
    plan = evaluator.load_frozen_plan()
    trial = plan["trials"][0]
    video_path = tmp_path / "canonical.mp4"
    video_path.write_bytes(b"fake canonical bytes")
    evidence = {
        "video": {
            "path": str(video_path),
            "sha256": evaluator.sha256_file(video_path),
            "frames": 1,
            "source": "canonical_rgb_act_input",
        }
    }
    monkeypatch.setattr(
        evaluator,
        "extract_pre_action_frame",
        lambda video, output: {
            "status": "frozen",
            "path": str(output),
            "sha256": "fake",
            "frame_index": 0,
            "width": 640,
            "height": 480,
        },
    )
    pre_action = evaluator.build_pre_action_evidence(
        evidence,
        trial,
        tmp_path,
        trial["spawn_region"],
    )
    assert pre_action["frame_index"] == 0
    assert pre_action["width"] == 640
    assert pre_action["height"] == 480
    assert pre_action["nominal_requested_pose"] == {
        "cell_id": "r3c3",
        "quadrant": "x_minus_y_plus",
        "x_forward_m": 0.31,
        "y_lateral_m": 0.04,
        "yaw_degrees_modulo_90": 0,
    }
    assert pre_action["manual_pose_is_measurement_truth"] is False
    assert pre_action["placement_claim"] == "nominal_manual_placement_only_not_instrumented_ground_truth"


def test_frozen_order_rejects_skips_and_allows_one_linked_infrastructure_replacement(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(evaluator.load_frozen_plan())
    plan["evidence_root"] = str(tmp_path)
    first = plan["trials"][0]
    second = plan["trials"][1]
    with pytest.raises(RuntimeError, match="next missing trial"):
        evaluator.validate_execution_order(plan, second, replacement=False)
    stem, link = evaluator.validate_execution_order(plan, first, replacement=False)
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
    replacement_stem, link = evaluator.validate_execution_order(plan, first, replacement=True)
    assert replacement_stem.endswith("__replacement1")
    assert link == first["spawn_region"]
    (original.parent / f"{replacement_stem}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        evaluator.validate_execution_order(plan, first, replacement=True)


def test_continuation_order_accepts_trial5_after_four_immutable_predecessors(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(evaluator.load_frozen_plan())
    plan["evidence_root"] = str(tmp_path)
    for row in plan["trials"][:4]:
        path = evaluator.original_evidence_path(plan, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"status": "completed_pending_operator_annotation"}),
            encoding="utf-8",
        )
    fifth = plan["trials"][4]
    stem, link = evaluator.validate_execution_order(plan, fifth, replacement=False)
    assert stem == fifth["spawn_region"]
    assert link is None


def test_operator_placement_mismatch_allows_one_linked_replacement(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(evaluator.load_frozen_plan())
    plan["evidence_root"] = str(tmp_path)
    trial = plan["trials"][7]
    original = evaluator.original_evidence_path(plan, trial)
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(
        json.dumps(
            {
                "status": "completed_pending_operator_annotation",
                "termination": "maximum_duration",
            }
        ),
        encoding="utf-8",
    )
    marker_path = evaluator.infrastructure_invalid_marker_path(plan, trial)
    marker_path.write_text(
        json.dumps(
            {
                "status": "infrastructure_invalid",
                "reason": "operator_placement_mismatch",
                "scored_trial": False,
                "replacement_allowed": True,
                "trial_id": trial["trial_id"],
                "artifact_stem": trial["spawn_region"],
                "original_evidence": {
                    "sha256": evaluator.sha256_file(original),
                },
            }
        ),
        encoding="utf-8",
    )
    replacement_stem, link = evaluator.validate_execution_order(
        plan,
        trial,
        replacement=True,
    )
    assert replacement_stem == f"{trial['spawn_region']}__replacement1"
    assert link == trial["spawn_region"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["original_evidence"]["sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker is malformed"):
        evaluator.validate_execution_order(plan, trial, replacement=True)


def test_gripper_nuisance_does_not_modify_action_contract() -> None:
    plan = evaluator.load_frozen_plan()
    assert plan["gripper_nuisance"] == {
        "status": "recorded_versioned_non_blocking",
        "safe_open_added": False,
        "real_default_opening_changed": False,
        "allowed_range": [0, 100],
        "extra_action_restriction_added": False,
    }
    assert plan["setup"]["max_relative_target"] is None
    assert plan["setup"]["custom_absolute_action_clamp"] is False
    assert plan["setup"]["custom_relative_step_limit_degrees"] is None
    assert plan["setup"]["ready_pose_interpolation_duration_seconds"] == 3.0
    assert plan["setup"]["ready_pose_interpolation_steps"] == 60
