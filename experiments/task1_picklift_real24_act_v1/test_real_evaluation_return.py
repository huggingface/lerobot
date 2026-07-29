from __future__ import annotations

import hashlib
import inspect
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import evaluate_real
import numpy as np
from deployment_safety import JOINT_ORDER, sha256_file
from evaluate_real import (
    ACTION_PATH_PROFILE_ID,
    EXPECTED_PLAN_SHA256,
    EXPECTED_PROFILE_SHA256,
    POLICY_TIMING_PROFILE_ID,
    READY_POSE,
    READY_POSE_STATE_SHA256,
    action_dict,
    load_frozen_contract,
    move_to_frozen_ready_pose,
    ready_pose_state_sha256,
    reset_policy_after_ready_pose,
    run_paced_ticks,
)

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower.so_follower import SOFollower


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeBus:
    def __init__(self, state: np.ndarray):
        self.state = state.astype(np.float32)

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Position"
        return {
            joint: float(self.state[index])
            for index, joint in enumerate(JOINT_ORDER)
        }


class FakeRobot:
    def __init__(self, state: np.ndarray):
        self.bus = FakeBus(state)
        self.requests: list[np.ndarray] = []

    def send_action(self, requested: dict[str, float]) -> dict[str, float]:
        vector = np.asarray(
            [requested[f"{joint}.pos"] for joint in JOINT_ORDER],
            dtype=np.float32,
        )
        self.requests.append(vector)
        self.bus.state = vector.copy()
        return requested


class FakeModel:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_ready_pose_hash_matches_frozen_remote_contract() -> None:
    assert ready_pose_state_sha256() == READY_POSE_STATE_SHA256
    payload = json.dumps(
        [float(value) for value in READY_POSE],
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == READY_POSE_STATE_SHA256


def test_ready_move_requests_full_frozen_pose_without_relative_step_limit() -> None:
    clock = FakeClock()
    initial = np.asarray([0, -80, 70, 70, 20, 50], dtype=np.float32)
    robot = FakeRobot(initial)
    evidence = StringIO()

    result = move_to_frozen_ready_pose(
        robot,
        evidence,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert result["status"] == "ready_pose_observed"
    assert result["requested_state"] == READY_POSE.tolist()
    np.testing.assert_allclose(result["observed_state"], READY_POSE)
    assert len(robot.requests) == 1
    np.testing.assert_allclose(robot.requests[0], READY_POSE)
    assert float(np.max(np.abs(READY_POSE - initial))) > 5.0
    row = json.loads(evidence.getvalue())
    np.testing.assert_allclose(row["requested_action"], READY_POSE)
    np.testing.assert_allclose(row["sent_action"], READY_POSE)
    assert not any(row["upstream_action_modified_mask"])


def test_policy_reset_occurs_only_for_observed_ready_pose() -> None:
    model = FakeModel()
    reset_at = reset_policy_after_ready_pose(
        model,
        {"status": "ready_pose_observed"},
    )

    assert model.reset_calls == 1
    assert reset_at.endswith("+00:00")


def test_action_request_preserves_out_of_calibration_values() -> None:
    raw = np.asarray([-200, 200, -150, 150, -180, 120], dtype=np.float32)

    requested = action_dict(raw)
    reconstructed = np.asarray(
        [requested[f"{joint}.pos"] for joint in JOINT_ORDER],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(reconstructed, raw)


def test_per_tick_pacing_does_not_catch_up_after_slow_first_tick() -> None:
    clock = FakeClock()
    tick_compute = [0.20, 0.01, 0.01]
    tick_starts: list[float] = []
    recorded: list[dict] = []

    def tick(step: int, tick_started: float, loop_started: float) -> dict:
        assert loop_started == 0.0
        tick_starts.append(tick_started)
        clock.advance(tick_compute[step])
        return {"payload": step}

    run_paced_ticks(
        0.30,
        tick,
        lambda record: recorded.append(dict(record)),
        period=0.05,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    np.testing.assert_allclose(tick_starts, [0.0, 0.20, 0.25])
    np.testing.assert_allclose(clock.sleeps, [0.0, 0.04, 0.04])
    assert [row["step"] for row in recorded] == [0, 1, 2]
    assert recorded[0]["scheduled_sleep_seconds"] == 0.0
    assert recorded[1]["tick_started_elapsed_seconds"] == 0.20


def test_runner_order_is_ready_then_reset_then_tick0_loop() -> None:
    source = inspect.getsource(evaluate_real.run_hardware_trial)

    ready_index = source.index(
        "ready_result = move_to_frozen_ready_pose(robot, ready_handle)"
    )
    reset_index = source.index(
        "policy_reset_at_utc = reset_policy_after_ready_pose"
    )
    loop_index = source.index("steps = run_paced_ticks")

    assert ready_index < reset_index < loop_index


def test_runner_has_no_custom_policy_action_clamp() -> None:
    source = inspect.getsource(evaluate_real.run_hardware_trial)

    assert "clamp_action_fail_closed" not in source
    assert "max_relative_target=None" in source
    assert '"custom_action_transform": "none"' in source
    validator_source = (
        Path(__file__).resolve().parent / "validate_checkpoint.py"
    ).read_text(encoding="utf-8")
    assert "clamp_action_fail_closed" not in validator_source
    assert '"custom_absolute_action_clamp": False' in validator_source


def test_upstream_action_behavior_is_documented_from_actual_source() -> None:
    send_source = inspect.getsource(SOFollower.send_action)
    unnormalize_source = inspect.getsource(FeetechMotorsBus._unnormalize)
    degrees_branch = unnormalize_source.split(
        "elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:"
    )[1].split("else:")[0]

    assert "if self.config.max_relative_target is not None:" in send_source
    assert 'self.bus.sync_write("Goal_Position", goal_pos)' in send_source
    assert "bounded_val" not in degrees_branch
    assert "min(100.0, max(0.0, val))" in unnormalize_source


def test_versioned_plan_and_profile_hashes_and_contract() -> None:
    experiment = Path(__file__).resolve().parent
    plan_path = experiment / "evaluation_plan_ready_pose_official_send_v1.json"
    profile_path = (
        experiment / "real_evaluation_profile_ready_pose_official_send_v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert sha256_file(plan_path) == EXPECTED_PLAN_SHA256
    assert sha256_file(profile_path) == EXPECTED_PROFILE_SHA256
    assert plan["setup"]["max_relative_target"] is None
    assert plan["setup"]["custom_absolute_action_clamp"] is False
    assert plan["setup"]["act_chunk_size"] == 67
    assert plan["setup"]["act_n_action_steps"] == 67
    assert profile["action_path"]["profile_id"] == ACTION_PATH_PROFILE_ID
    assert profile["policy_timing"]["profile_id"] == POLICY_TIMING_PROFILE_ID
    assert profile["ready_pose"]["state"] == READY_POSE.tolist()


def test_load_frozen_contract_is_software_only() -> None:
    experiment = Path(__file__).resolve().parent
    args = SimpleNamespace(
        plan=experiment / "evaluation_plan_ready_pose_official_send_v1.json",
        profile=(
            experiment
            / "real_evaluation_profile_ready_pose_official_send_v1.json"
        ),
        spawn_region="r1c1",
        maximum_trial_seconds=30.0,
    )

    plan, profile, trial = load_frozen_contract(args)

    assert plan["evaluation_id"] == evaluate_real.EXPECTED_EVALUATION_ID
    assert profile["ready_pose"]["state_sha256"] == READY_POSE_STATE_SHA256
    assert trial["order"] == 1
