from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import run_remote_sim as runner
from sim_policy_inference import (
    FIXED_CHECKPOINT,
    FIXED_MODEL_SHA256,
    requested_action_from_raw,
    sha256_file,
    validate_observation,
    validate_saved_imagenet_processor,
)


def test_owner_plan_hash_and_unrestricted_action_contract() -> None:
    plan = runner.load_owner_plan()
    contract = plan["policy_environment_contract"]
    assert contract["follower_calibration_state_gate"] is False
    assert contract["sim_state_projection"] is False
    assert contract["max_relative_target"] is None
    assert contract["custom_absolute_calibration_clamp"] is False
    assert contract["custom_relative_clamp"] is False
    assert contract["additional_action_limit"] is False


def test_plan_hash_mismatch_fails_closed(tmp_path: Path, monkeypatch) -> None:
    altered = tmp_path / "plan.json"
    altered.write_text(
        runner.PLAN_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PLAN_PATH", altered)
    with pytest.raises(RuntimeError, match="plan hash"):
        runner.load_owner_plan()


def test_fixed_mixed_v2_checkpoint_hash() -> None:
    assert (
        sha256_file(FIXED_CHECKPOINT / "model.safetensors")
        == FIXED_MODEL_SHA256
    )


def test_checkpoint_owned_processor_is_imagenet_normalized() -> None:
    result = validate_saved_imagenet_processor(FIXED_CHECKPOINT)
    assert result["status"] == "pass_checkpoint_owned_imagenet_stats"
    assert result["use_imagenet_stats"] is True
    assert result["visual_mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert result["visual_std"] == pytest.approx([0.229, 0.224, 0.225])


def test_state_validation_accepts_values_outside_follower_calibration() -> None:
    state = np.asarray(
        [-500.0, 500.0, -300.0, 300.0, -200.0, 150.0],
        dtype=np.float32,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    accepted_state, accepted_image = validate_observation(state, image)
    np.testing.assert_array_equal(accepted_state, state)
    assert accepted_image.shape == (480, 640, 3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_state_validation_rejects_only_nonfinite_values(bad: float) -> None:
    state = np.zeros(6, dtype=np.float32)
    state[2] = bad
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        validate_observation(
            state,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )


def test_raw_action_is_requested_without_any_clamp() -> None:
    raw = np.asarray(
        [-999.0, 999.0, -321.5, 456.25, -777.0, 222.0],
        dtype=np.float32,
    )
    requested = requested_action_from_raw(raw)
    np.testing.assert_array_equal(requested, raw)
    assert requested is not raw


def test_policy_interface_has_no_follower_safety_import() -> None:
    source = (
        Path(__file__).resolve().parent / "sim_policy_inference.py"
    ).read_text(encoding="utf-8")
    assert "deployment_safety" not in source
    assert "project_sim_state_to_calibration" not in source
    assert "clamp_action_fail_closed" not in source
    assert "apply_action_safety" not in source


def fake_episode(cell: str, success: bool) -> dict:
    return {
        "cell": cell,
        "success": success,
        "interface_valid": True,
        "ready_pose_tick0_valid": True,
        "object_spawn_plan_valid": True,
        "first_success_step": 100 if success else None,
        "confirmed_success_step": 124 if success else None,
        "max_lift_m": 0.06 if success else 0.01,
        "is_grasped": success,
        "env_steps": 1500,
        "raw_action_count": 600,
        "requested_action_count": 600,
        "environment_clipped_action_count": 3,
        "environment_clipped_joint_value_count": 4,
        "failure_type": "none" if success else "policy_task_failure",
        "termination_reason": "max_steps_reached",
    }


def test_gate_success_is_not_interface_gate_condition() -> None:
    episodes = [
        fake_episode(f"r{row}c{column}", False)
        for row in range(1, 4)
        for column in range(1, 5)
    ]
    summary = runner.summarize_episodes("gate12", episodes)
    assert summary["interface_pass"] is True
    assert summary["overall"]["successes"] == 0
    assert summary["task_success_is_gate_condition"] is False


def test_episode_interface_requires_full_600_ticks_and_1500_steps() -> None:
    runtime = {
        "success": False,
        "env_steps": 1500,
        "timeout": True,
        "termination_reason": "max_steps_reached",
        "failure_type": "policy_task_failure",
    }
    record = runner.build_episode_record(
        runtime_summary=runtime,
        raw_action_count=600,
        requested_action_count=600,
        valid_observation_count=600,
        environment_clipped_action_count=0,
        environment_clipped_joint_value_count=0,
        ready_pose_tick0_valid=True,
        object_spawn_plan_valid=True,
        interface_error=None,
    )
    assert record["interface_valid"] is True
    short = dict(runtime, env_steps=1499)
    record = runner.build_episode_record(
        runtime_summary=short,
        raw_action_count=599,
        requested_action_count=599,
        valid_observation_count=599,
        environment_clipped_action_count=0,
        environment_clipped_joint_value_count=0,
        ready_pose_tick0_valid=True,
        object_spawn_plan_valid=True,
        interface_error=None,
    )
    assert record["interface_valid"] is False


def test_frozen_phase_paths_are_unique_and_do_not_overlap_real_only() -> None:
    plan = runner.load_owner_plan()
    gate = Path(plan["phases"]["gate12"]["output_dir"])
    frozen = Path(plan["phases"]["frozen120"]["output_dir"])
    assert gate != frozen
    assert "mixed_v2_act_remote_sim_eval_v1" in str(gate)
    assert "mixed_v2_act_remote_sim_eval_v1" in str(frozen)
    assert "real24_act_v1_real_to_remote_sim_v1" not in str(gate)
