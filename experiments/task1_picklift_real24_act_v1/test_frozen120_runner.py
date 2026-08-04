"""Pure summary checks for the frozen120 hardware-free runner."""

from __future__ import annotations

from copy import deepcopy

from run_remote_sim_frozen120 import summarize_frozen120


def make_episode(cell: str) -> dict[str, object]:
    return {
        "cell": cell,
        "success": False,
        "interface_valid": True,
        "ready_pose_tick0_valid": True,
        "object_spawn_plan_valid": True,
        "env_steps": 1500,
        "raw_action_count": 600,
        "calibration_clipped_action_count": 1,
        "relative_clipped_action_count": 2,
        "calibration_clipped_joint_value_count": 3,
        "relative_clipped_joint_value_count": 4,
        "sent_action_count": 600,
        "environment_clipped_action_count": 5,
        "sim_state_projected_tick_count": 6,
        "maximum_absolute_sim_state_projection_delta": 0.02,
        "ready_pose_validation": {
            "maximum_absolute_tick0_delta": 2.0e-6,
        },
        "failure_type": "policy_task_failure",
        "termination_reason": "max_steps_reached",
    }


def make_complete_plan() -> list[dict[str, object]]:
    return [
        make_episode(f"r{row}c{column}")
        for _repeat in range(10)
        for row in range(1, 4)
        for column in range(1, 5)
    ]


def test_complete_fixed_plan_passes_and_aggregates() -> None:
    summary = summarize_frozen120(make_complete_plan())

    assert summary["frozen120_interface_pass"] is True
    assert summary["overall"]["episodes"] == 120
    assert summary["overall"]["env_steps"] == 180000
    assert summary["overall"]["raw_action_count"] == 72000
    assert summary["overall"]["sim_state_projected_tick_count"] == 720
    assert summary["overall"][
        "maximum_absolute_sim_state_projection_delta"
    ] == 0.02
    assert set(summary["by_cell"]) == {
        f"r{row}c{column}"
        for row in range(1, 4)
        for column in range(1, 5)
    }
    assert all(
        cell_summary["episodes"] == 10
        for cell_summary in summary["by_cell"].values()
    )


def test_any_interface_failure_fails_gate_without_dropping_episode() -> None:
    episodes = make_complete_plan()
    failed = deepcopy(episodes[37])
    failed["interface_valid"] = False
    failed["failure_type"] = "interface_error"
    episodes[37] = failed

    summary = summarize_frozen120(episodes)

    assert summary["frozen120_interface_pass"] is False
    assert summary["overall"]["episodes"] == 120
    assert summary["overall"]["interface_valid_episodes"] == 119
    assert summary["overall"]["failure_types"]["interface_error"] == 1
