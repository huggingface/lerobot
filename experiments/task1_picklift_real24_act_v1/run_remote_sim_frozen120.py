"""Run the authorized hardware-free Task1 ACT aligned frozen120 diagnostic.

This runner is intentionally frozen120-only. It imports the frozen ACT
inference interface and the deployed Remote Nexus adapter, but starts no
service, accesses no camera/serial/robot hardware, and writes no LeRobot
Dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


EXPERIMENT_DIR = Path(
    "/home/ubuntu24/Teleop/lerobot/experiments/"
    "task1_picklift_real24_act_v1"
)
REMOTE_ROOT = Path("/home/ubuntu24/SO101QuestRemote")
REMOTE_ADAPTER_DIR = REMOTE_ROOT / "robot-host"
REMOTE_DEPLOYED_COMMIT_FILE = REMOTE_ROOT / ".deployed-commit"
EXPECTED_REMOTE_DEPLOYED_COMMIT = (
    "1dfac5c108e830a55b114b1a75bd00bdb5d877b7"
)
EXPECTED_PLAN_SHA256 = (
    "c26bf3caa788708cf8ccb332e2f84771d271b05795634a9b1022ed97d06b8c86"
)
EXPECTED_MODEL_SHA256 = (
    "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
)
EXPECTED_EPISODES = 120
EXPECTED_RESET_PROFILE_ID = "picklift_real24_aligned_reset_v2"
EXPECTED_READY_POSE_PROFILE_ID = "task1_real24_ready_pose_reset_v1"
EXPECTED_READY_POSE_STATE_SHA256 = (
    "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
)
EXPECTED_READY_POSE_SOURCE_EPISODE_INDEX = 13
EXPECTED_READY_POSE_SOURCE_FRAME_INDEX = 0
EXPECTED_READY_POSE_TOLERANCE = 1.0e-5
LEGAL_TERMINATION_REASONS = {
    "environment_terminated",
    "environment_truncated",
    "max_steps_reached",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(handle: Any, payload: Any) -> None:
    handle.write(canonical_json(payload) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_contract() -> tuple[Any, dict[str, Any], tuple[Any, ...]]:
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    if str(REMOTE_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_ADAPTER_DIR))

    from nexus_picklift_policy_adapter import (
        EVALUATION_PHASE_ID,
        build_real24_act100k_plan,
        experiment_contract,
    )

    if EVALUATION_PHASE_ID != "frozen120":
        raise RuntimeError("Remote evaluation phase id drifted.")
    contract = experiment_contract()
    plan_sha = contract["plan"]["plan_sha256"]
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RuntimeError(
            f"Remote plan SHA mismatch: {plan_sha} != {EXPECTED_PLAN_SHA256}"
        )
    trials = build_real24_act100k_plan()[EVALUATION_PHASE_ID]
    if len(trials) != EXPECTED_EPISODES:
        raise RuntimeError(
            "Remote frozen plan must contain exactly 120 episodes."
        )
    if [trial.phase_episode_index for trial in trials] != list(
        range(EXPECTED_EPISODES)
    ):
        raise RuntimeError("Remote frozen120 trial ordering drifted.")
    if [trial.cell_name for trial in trials] != [
        f"r{row}c{column}"
        for _seed in range(1000, 1010)
        for row in range(1, 4)
        for column in range(1, 5)
    ]:
        raise RuntimeError(
            "Remote frozen120 order is not seed-major then row-major cell."
        )
    if [trial.seed for trial in trials] != [
        seed
        for seed in range(1000, 1010)
        for _cell in range(12)
    ]:
        raise RuntimeError("Remote frozen120 seed ordering drifted.")
    if [trial.repeat_index for trial in trials] != [
        repeat_index
        for repeat_index in range(10)
        for _cell in range(12)
    ]:
        raise RuntimeError("Remote frozen120 repeat ordering drifted.")
    return EVALUATION_PHASE_ID, contract, trials


def validate_deployment() -> dict[str, Any]:
    deployed_commit = REMOTE_DEPLOYED_COMMIT_FILE.read_text(
        encoding="utf-8"
    ).strip()
    if deployed_commit != EXPECTED_REMOTE_DEPLOYED_COMMIT:
        raise RuntimeError(
            "Remote deployed commit mismatch; refusing frozen120."
        )
    phase_id, contract, trials = load_contract()
    if contract["upstream_act_binding"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Remote contract checkpoint binding drifted.")
    if contract["plan"]["evaluation_phase_id"] != "frozen120":
        raise RuntimeError("Remote evaluation phase contract drifted.")
    if contract["clock"]["maximum_policy_ticks"] != 600:
        raise RuntimeError("Remote 30-second policy-tick limit drifted.")
    if contract["clock"]["maximum_env_steps"] != 1500:
        raise RuntimeError("Remote 30-second environment-step limit drifted.")
    from picklift_ready_pose import (
        ACTIVE_PICKLIFT_RESET_PROFILE,
        REAL24_READY_POSE,
        READY_POSE_TOLERANCE_DATASET_UNITS,
    )

    reset_contract = ACTIVE_PICKLIFT_RESET_PROFILE.contract()
    ready_contract = REAL24_READY_POSE.contract()
    if reset_contract["profile_id"] != EXPECTED_RESET_PROFILE_ID:
        raise RuntimeError("Remote active reset profile drifted.")
    if reset_contract["ready_pose"] != ready_contract:
        raise RuntimeError("Remote active reset/ready-pose binding drifted.")
    if ready_contract["profile_id"] != EXPECTED_READY_POSE_PROFILE_ID:
        raise RuntimeError("Remote ready-pose profile drifted.")
    if ready_contract["state_sha256"] != EXPECTED_READY_POSE_STATE_SHA256:
        raise RuntimeError("Remote ready-pose state SHA drifted.")
    if (
        ready_contract["source_episode_index"]
        != EXPECTED_READY_POSE_SOURCE_EPISODE_INDEX
        or ready_contract["source_frame_index"]
        != EXPECTED_READY_POSE_SOURCE_FRAME_INDEX
    ):
        raise RuntimeError("Remote ready-pose source frame drifted.")
    if READY_POSE_TOLERANCE_DATASET_UNITS != EXPECTED_READY_POSE_TOLERANCE:
        raise RuntimeError("Remote ready-pose tolerance drifted.")
    return {
        "phase_id": phase_id,
        "episode_count": len(trials),
        "plan_sha256": contract["plan"]["plan_sha256"],
        "remote_deployed_commit": deployed_commit,
        "model_sha256": contract["upstream_act_binding"]["model_sha256"],
        "reset_profile": reset_contract,
        "ready_pose": ready_contract,
        "ready_pose_tolerance": READY_POSE_TOLERANCE_DATASET_UNITS,
        "clock": contract["clock"],
        "status": "pass",
    }


def validate_policy_inputs(inputs: dict[str, Any]) -> None:
    if set(inputs) != {
        "observation.state",
        "observation.images.front",
    }:
        raise RuntimeError("Remote observation keys drifted.")
    state = inputs["observation.state"]
    image = inputs["observation.images.front"]
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (6,)
        or state.dtype != np.float32
        or not bool(np.isfinite(state).all())
    ):
        raise RuntimeError("Remote state is not finite float32[6].")
    if (
        not isinstance(image, np.ndarray)
        or image.shape != (480, 640, 3)
        or image.dtype != np.uint8
    ):
        raise RuntimeError("Remote front image is not uint8[480,640,3] RGB.")


def validate_ready_pose_evidence(
    evidence: dict[str, Any] | None,
    tick0_state: np.ndarray,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise RuntimeError("Remote ready-pose reset evidence is unavailable.")
    if evidence.get("contract") != expected_contract:
        raise RuntimeError("Remote ready-pose evidence contract drifted.")
    application = evidence.get("application")
    if not isinstance(application, dict):
        raise RuntimeError("Remote ready-pose application evidence is missing.")
    requested = np.asarray(
        application.get("requested_state_dataset_units"),
        dtype=np.float64,
    )
    observed = np.asarray(
        application.get("observed_tick0_state_dataset_units"),
        dtype=np.float64,
    )
    delta = np.asarray(
        application.get("per_joint_delta_dataset_units"),
        dtype=np.float64,
    )
    state = np.asarray(tick0_state, dtype=np.float64)
    if any(values.shape != (6,) for values in (requested, observed, delta, state)):
        raise RuntimeError("Ready-pose evidence arrays must all have shape [6].")
    if not all(
        bool(np.isfinite(values).all())
        for values in (requested, observed, delta, state)
    ):
        raise RuntimeError("Ready-pose evidence contains non-finite values.")
    expected = np.asarray(
        expected_contract["state_dataset_units"],
        dtype=np.float64,
    )
    tolerance = float(application.get("absolute_tolerance_dataset_units"))
    if tolerance != EXPECTED_READY_POSE_TOLERANCE:
        raise RuntimeError("Ready-pose application tolerance drifted.")
    if not np.array_equal(requested, expected):
        raise RuntimeError("Ready-pose requested state drifted.")
    if not np.allclose(delta, observed - requested, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Ready-pose recorded delta is inconsistent.")
    maximum_absolute_delta = float(np.max(np.abs(delta)))
    if maximum_absolute_delta > tolerance:
        raise RuntimeError("Ready-pose observed state exceeds tolerance.")
    if not np.allclose(state, observed, rtol=0.0, atol=tolerance):
        raise RuntimeError("Returned tick0 state differs from ready-pose evidence.")
    required_flags = {
        "within_tolerance": True,
        "robot_qvel_zero": True,
        "object_pose_unchanged": True,
        "simulation_time_advanced": False,
        "nexus_init_noise_overwritten": True,
        "env_step_after_override": 0,
        "last_step_result_cleared": True,
    }
    for field, expected_value in required_flags.items():
        if application.get(field) != expected_value:
            raise RuntimeError(
                f"Ready-pose application flag drifted: {field}."
            )
    return {
        "ready_pose_tick0_valid": True,
        "maximum_absolute_tick0_delta": maximum_absolute_delta,
        "requested_tick0_state": requested.tolist(),
        "observed_tick0_state": observed.tolist(),
        "per_joint_tick0_delta": delta.tolist(),
        "absolute_tolerance": tolerance,
    }


def validate_object_spawn(
    trial_manifest: dict[str, Any],
    task_manifest: dict[str, Any],
) -> dict[str, Any]:
    planned = trial_manifest["spawn"]
    observed = task_manifest["spawn"]
    mismatches = {
        key: {"planned": value, "observed": observed.get(key)}
        for key, value in planned.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Remote object spawn differs from frozen plan: {mismatches}"
        )
    return {
        "object_spawn_plan_valid": True,
        "planned_spawn": planned,
        "actual_initial_pose": observed["actual_initial_pose"],
        "placement_method": observed["placement_method"],
    }


def episode_record(
    *,
    runtime_summary: dict[str, Any],
    raw_action_count: int,
    calibration_clipped_action_count: int,
    relative_clipped_action_count: int,
    calibration_clipped_joint_value_count: int,
    relative_clipped_joint_value_count: int,
    sent_action_count: int,
    valid_observation_count: int,
    sim_state_projected_tick_count: int,
    maximum_absolute_sim_state_projection_delta: float,
    ready_pose_tick0_valid: bool,
    object_spawn_plan_valid: bool,
    expected_policy_ticks: int,
    interface_error: str | None,
) -> dict[str, Any]:
    record = dict(runtime_summary)
    record.update(
        {
            "raw_action_count": raw_action_count,
            "calibration_clipped_action_count": (
                calibration_clipped_action_count
            ),
            "relative_clipped_action_count": relative_clipped_action_count,
            "calibration_clipped_joint_value_count": (
                calibration_clipped_joint_value_count
            ),
            "relative_clipped_joint_value_count": (
                relative_clipped_joint_value_count
            ),
            "sent_action_count": sent_action_count,
            "valid_observation_count": valid_observation_count,
            "sim_state_projected_tick_count": (
                sim_state_projected_tick_count
            ),
            "maximum_absolute_sim_state_projection_delta": (
                maximum_absolute_sim_state_projection_delta
            ),
            "ready_pose_tick0_valid": ready_pose_tick0_valid,
            "object_spawn_plan_valid": object_spawn_plan_valid,
            "interface_error": interface_error,
        }
    )
    legal_end = (
        record["termination_reason"] in LEGAL_TERMINATION_REASONS
        and (
            bool(record["terminated"])
            or bool(record["truncated"])
            or bool(record["timeout"])
        )
    )
    record["interface_valid"] = bool(
        interface_error is None
        and valid_observation_count == raw_action_count
        and raw_action_count == sent_action_count
        and 0 < raw_action_count <= expected_policy_ticks
        and (
            not bool(record["timeout"])
            or raw_action_count == expected_policy_ticks
        )
        and int(record["env_steps"]) > 0
        and legal_end
        and ready_pose_tick0_valid
        and object_spawn_plan_valid
    )
    if interface_error is not None:
        record["failure_type"] = "interface_error"
    return record


def summarize_frozen120(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_cell[str(episode["cell"])].append(episode)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(item["success"]) for item in group)
        return {
            "episodes": len(group),
            "interface_valid_episodes": sum(
                bool(item["interface_valid"]) for item in group
            ),
            "ready_pose_tick0_valid_episodes": sum(
                bool(item["ready_pose_tick0_valid"]) for item in group
            ),
            "object_spawn_plan_valid_episodes": sum(
                bool(item["object_spawn_plan_valid"]) for item in group
            ),
            "successes": successes,
            "success_rate": successes / len(group),
            "env_steps": sum(int(item["env_steps"]) for item in group),
            "raw_action_count": sum(
                int(item["raw_action_count"]) for item in group
            ),
            "calibration_clipped_action_count": sum(
                int(item["calibration_clipped_action_count"])
                for item in group
            ),
            "relative_clipped_action_count": sum(
                int(item["relative_clipped_action_count"]) for item in group
            ),
            "calibration_clipped_joint_value_count": sum(
                int(item["calibration_clipped_joint_value_count"])
                for item in group
            ),
            "relative_clipped_joint_value_count": sum(
                int(item["relative_clipped_joint_value_count"])
                for item in group
            ),
            "sent_action_count": sum(
                int(item["sent_action_count"]) for item in group
            ),
            "environment_clipped_action_count": sum(
                int(item["environment_clipped_action_count"])
                for item in group
            ),
            "sim_state_projected_tick_count": sum(
                int(item["sim_state_projected_tick_count"])
                for item in group
            ),
            "maximum_absolute_sim_state_projection_delta": max(
                float(item["maximum_absolute_sim_state_projection_delta"])
                for item in group
            ),
            "maximum_absolute_ready_pose_tick0_delta": max(
                float(
                    item["ready_pose_validation"][
                        "maximum_absolute_tick0_delta"
                    ]
                )
                if item["ready_pose_validation"] is not None
                else 0.0
                for item in group
            ),
            "failure_types": dict(
                Counter(
                    str(item["failure_type"])
                    for item in group
                    if not item["success"]
                )
            ),
            "termination_reasons": dict(
                Counter(str(item["termination_reason"]) for item in group)
            ),
        }

    overall = summarize_group(episodes)
    expected_cells = {
        f"r{row}c{column}"
        for row in range(1, 4)
        for column in range(1, 5)
    }
    interface_pass = bool(
        len(episodes) == EXPECTED_EPISODES
        and set(by_cell) == expected_cells
        and all(len(group) == 10 for group in by_cell.values())
        and overall["interface_valid_episodes"] == EXPECTED_EPISODES
        and (
            overall["ready_pose_tick0_valid_episodes"]
            == EXPECTED_EPISODES
        )
        and (
            overall["object_spawn_plan_valid_episodes"]
            == EXPECTED_EPISODES
        )
    )
    return {
        "schema_version": 1,
        "status": "diagnostic_only_not_a_real_robot_or_paper_result",
        "phase_id": "frozen120",
        "plan_definition": (
            "all 120 fixed seed-major trials verify the frozen ready pose at "
            "tick0, preserve the frozen object spawn plan, produce exact "
            "state/RGB, execute all 600 policy ticks without adaptive stopping, "
            "and produce legal termination/timeout evidence"
        ),
        "frozen120_interface_pass": interface_pass,
        "overall": overall,
        "by_cell": {
            cell: summarize_group(by_cell[cell]) for cell in sorted(by_cell)
        },
    }


def write_hashes(output_dir: Path) -> dict[str, str]:
    names = (
        "episodes.jsonl",
        "ticks.jsonl",
        "run_manifest.json",
        "summary.json",
    )
    hashes = {name: sha256_file(output_dir / name) for name in names}
    (output_dir / "hashes.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    return hashes


def fallback_runtime_summary(trial: Any) -> dict[str, Any]:
    return {
        "phase_id": trial.phase_id,
        "cell": trial.cell_name,
        "seed": trial.seed,
        "initial_pose": trial.manifest()["initial_pose"],
        "success": False,
        "env_steps": 0,
        "first_success_step": None,
        "confirmed_success_step": None,
        "max_lift_m": 0.0,
        "is_grasped": False,
        "terminated": False,
        "truncated": False,
        "timeout": False,
        "termination_reason": "interface_error_before_episode_manifest",
        "environment_clipped_action_count": 0,
        "failure_type": "incomplete",
    }


def run_frozen120(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(
            f"Evidence directory already exists; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    validation = validate_deployment()
    _, contract, trials = load_contract()

    from nexus_picklift_policy_adapter import (
        create_nexus_picklift_policy_adapter,
    )
    from picklift_ready_pose import REAL24_READY_POSE
    from sim_policy_inference import Task1ActSimInference

    source_commit = git_head(EXPERIMENT_DIR.parents[1])
    run_manifest = {
        "schema_version": 1,
        "experiment_run_id": output_dir.name,
        "phase_id": "frozen120",
        "status": "running",
        "research_status": (
            "diagnostic_only_not_real_robot_evaluation_not_a_paper_result"
        ),
        "started_at_utc": started_at.isoformat(),
        "source_commit": source_commit,
        "remote_deployed_commit": validation["remote_deployed_commit"],
        "remote_adapter_path": str(
            REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
        ),
        "remote_adapter_sha256": sha256_file(
            REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
        ),
        "remote_reset_profile": validation["reset_profile"],
        "ready_pose": validation["ready_pose"],
        "ready_pose_tolerance": validation["ready_pose_tolerance"],
        "ready_pose_explicitly_passed_to_every_reset": True,
        "plan_sha256": validation["plan_sha256"],
        "checkpoint_path": contract["upstream_act_binding"]["checkpoint_path"],
        "model_sha256": validation["model_sha256"],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hardware_accessed": False,
        "gateway_or_quest_started": False,
        "lerobot_dataset_written": False,
        "phase_scope": "authorized_frozen120_only_no_other_phase",
        "contract": contract,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    policy = Task1ActSimInference()
    if policy.model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Loaded policy model SHA mismatch.")
    adapter = create_nexus_picklift_policy_adapter()
    episodes: list[dict[str, Any]] = []
    try:
        with (
            (output_dir / "ticks.jsonl").open(
                "x", encoding="utf-8", buffering=1
            ) as ticks_handle,
            (output_dir / "episodes.jsonl").open(
                "x", encoding="utf-8", buffering=1
            ) as episodes_handle,
        ):
            for trial in trials:
                policy.reset_episode()
                raw_action_count = 0
                calibration_clipped_action_count = 0
                relative_clipped_action_count = 0
                calibration_clipped_joint_value_count = 0
                relative_clipped_joint_value_count = 0
                sent_action_count = 0
                valid_observation_count = 0
                sim_state_projected_tick_count = 0
                maximum_absolute_sim_state_projection_delta = 0.0
                ready_pose_tick0_valid = False
                object_spawn_plan_valid = False
                ready_pose_validation: dict[str, Any] | None = None
                object_spawn_validation: dict[str, Any] | None = None
                interface_error: str | None = None
                try:
                    observation = adapter.reset(
                        trial,
                        ready_pose=REAL24_READY_POSE,
                    )
                    reset_manifest = adapter.episode_manifest()
                    ready_pose_validation = validate_ready_pose_evidence(
                        reset_manifest["ready_pose_reset"],
                        observation.policy_inputs()["observation.state"],
                        validation["ready_pose"],
                    )
                    ready_pose_tick0_valid = True
                    object_spawn_validation = validate_object_spawn(
                        trial.manifest(),
                        reset_manifest["task_manifest"],
                    )
                    object_spawn_plan_valid = True
                    while adapter.phase == "active":
                        inputs = observation.policy_inputs()
                        validate_policy_inputs(inputs)
                        valid_observation_count += 1
                        policy_step = policy.infer(
                            inputs["observation.state"],
                            inputs["observation.images.front"],
                        )
                        raw_action_count += 1
                        calibration_clipped_action_count += int(
                            bool(policy_step.calibration_clip_mask.any())
                        )
                        relative_clipped_action_count += int(
                            bool(policy_step.relative_clip_mask.any())
                        )
                        calibration_clipped_joint_value_count += int(
                            policy_step.calibration_clip_mask.sum()
                        )
                        relative_clipped_joint_value_count += int(
                            policy_step.relative_clip_mask.sum()
                        )
                        if bool(policy_step.sim_state_projection_mask.any()):
                            sim_state_projected_tick_count += 1
                            maximum_absolute_sim_state_projection_delta = max(
                                maximum_absolute_sim_state_projection_delta,
                                float(
                                    np.max(
                                        np.abs(
                                            policy_step.sim_state_projection_delta
                                        )
                                    )
                                ),
                            )
                        tick_result = adapter.apply_action(
                            policy_step.sent_action
                        )
                        sent_action_count += 1
                        append_jsonl(
                            ticks_handle,
                            {
                                "phase_id": trial.phase_id,
                                "cell": trial.cell_name,
                                "seed": trial.seed,
                                "observation": observation.evidence(),
                                "policy": policy_step.to_jsonable(),
                                "remote": tick_result.evidence(),
                            },
                        )
                        if adapter.phase == "active":
                            observation = adapter.observe()
                except Exception as exc:
                    interface_error = f"{type(exc).__name__}: {exc}"

                manifest: dict[str, Any] | None = None
                if adapter.trial is trial:
                    try:
                        manifest = adapter.episode_manifest()
                    except Exception as exc:
                        manifest_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        interface_error = (
                            f"{interface_error}; manifest={manifest_error}"
                            if interface_error is not None
                            else f"manifest={manifest_error}"
                        )
                runtime_summary = (
                    manifest["runtime_summary"]
                    if manifest is not None
                    else fallback_runtime_summary(trial)
                )
                record = episode_record(
                    runtime_summary=runtime_summary,
                    raw_action_count=raw_action_count,
                    calibration_clipped_action_count=(
                        calibration_clipped_action_count
                    ),
                    relative_clipped_action_count=(
                        relative_clipped_action_count
                    ),
                    calibration_clipped_joint_value_count=(
                        calibration_clipped_joint_value_count
                    ),
                    relative_clipped_joint_value_count=(
                        relative_clipped_joint_value_count
                    ),
                    sent_action_count=sent_action_count,
                    valid_observation_count=valid_observation_count,
                    sim_state_projected_tick_count=(
                        sim_state_projected_tick_count
                    ),
                    maximum_absolute_sim_state_projection_delta=(
                        maximum_absolute_sim_state_projection_delta
                    ),
                    ready_pose_tick0_valid=ready_pose_tick0_valid,
                    object_spawn_plan_valid=object_spawn_plan_valid,
                    expected_policy_ticks=contract["clock"][
                        "maximum_policy_ticks"
                    ],
                    interface_error=interface_error,
                )
                record.update(
                    {
                        "phase_episode_index": trial.phase_episode_index,
                        "repeat_index": trial.repeat_index,
                        "trial_manifest": trial.manifest(),
                        "object_spawn": (
                            manifest["task_manifest"]["spawn"]
                            if manifest is not None
                            else None
                        ),
                        "object_spawn_validation": object_spawn_validation,
                        "ready_pose_reset": (
                            manifest["ready_pose_reset"]
                            if manifest is not None
                            else None
                        ),
                        "ready_pose_validation": ready_pose_validation,
                        "reset_profile_id": EXPECTED_RESET_PROFILE_ID,
                        "ready_pose_profile_id": (
                            EXPECTED_READY_POSE_PROFILE_ID
                        ),
                        "experiment_run_id": output_dir.name,
                        "source_commit": source_commit,
                        "remote_deployed_commit": validation[
                            "remote_deployed_commit"
                        ],
                        "model_sha256": policy.model_hash,
                    }
                )
                append_jsonl(episodes_handle, record)
                episodes.append(record)
                print(
                    canonical_json(
                        {
                            "cell": record["cell"],
                            "episode": len(episodes),
                            "interface_valid": record["interface_valid"],
                            "ready_pose_tick0_valid": record[
                                "ready_pose_tick0_valid"
                            ],
                            "object_spawn_plan_valid": record[
                                "object_spawn_plan_valid"
                            ],
                            "success": record["success"],
                            "env_steps": record["env_steps"],
                            "termination_reason": record[
                                "termination_reason"
                            ],
                        }
                    ),
                    flush=True,
                )
    finally:
        adapter.close()

    summary = summarize_frozen120(episodes)
    summary["experiment_run_id"] = output_dir.name
    summary["plan_sha256"] = validation["plan_sha256"]
    summary["model_sha256"] = policy.model_hash
    summary["remote_deployed_commit"] = validation["remote_deployed_commit"]
    summary["reset_profile"] = validation["reset_profile"]
    summary["ready_pose"] = validation["ready_pose"]
    summary["ready_pose_tolerance"] = validation["ready_pose_tolerance"]
    summary["runtime_seconds"] = time.monotonic() - started_monotonic
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "summary.json", summary)

    run_manifest.update(
        {
            "status": (
                "frozen120_complete"
                if summary["frozen120_interface_pass"]
                else "frozen120_complete_with_interface_failures"
            ),
            "completed_at_utc": summary["completed_at_utc"],
            "runtime_seconds": summary["runtime_seconds"],
            "episode_count": len(episodes),
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    hashes = write_hashes(output_dir)
    print(
        canonical_json(
            {
                "frozen120_interface_pass": summary[
                    "frozen120_interface_pass"
                ],
                "output_dir": str(output_dir),
                "hashes": hashes,
            }
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hardware-free Task1 ACT Remote frozen120 only."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="Validate frozen identities without creating a Nexus environment.",
    )
    args = parser.parse_args()
    if args.validate_contract_only == (args.output_dir is not None):
        parser.error(
            "choose exactly one of --validate-contract-only or --output-dir"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.validate_contract_only:
        print(json.dumps(validate_deployment(), indent=2, sort_keys=True))
        return
    summary = run_frozen120(args.output_dir.resolve())
    if not summary["frozen120_interface_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
