"""Run the frozen mixed-v2 ACT Task1 diagnostic through the formal Nexus adapter.

This module starts no gateway and imports no serial, camera, robot, torque, or
real-worker path. The policy observation is the adapter's canonical RGB plus
dataset-unit state. ACT raw output is passed unchanged to adapter.apply_action;
only the formal Nexus sink may apply its published environment action-space
clip.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan.json"
EXPECTED_OWNER_PLAN_SHA256 = (
    "f06a201d7261f75b8dbb24a8510a9c4ab52f48c475cb2744ee2547b6f51a4aa3"
)
REMOTE_ROOT = Path("/home/ubuntu24/SO101QuestRemote")
REMOTE_ADAPTER_DIR = REMOTE_ROOT / "robot-host"
REMOTE_DEPLOYED_COMMIT_FILE = REMOTE_ROOT / ".deployed-commit"
EXPECTED_REMOTE_DEPLOYED_COMMIT = (
    "1dfac5c108e830a55b114b1a75bd00bdb5d877b7"
)
EXPECTED_REMOTE_ADAPTER_SHA256 = (
    "8cd784ee8f36c3b5b36204ce4fe1b2c017af715dab120514113d33b00d7366ff"
)
EXPECTED_REMOTE_PLAN_SHA256 = (
    "c26bf3caa788708cf8ccb332e2f84771d271b05795634a9b1022ed97d06b8c86"
)
EXPECTED_MODEL_SHA256 = (
    "b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68"
)
EXPECTED_RESET_PROFILE_ID = "picklift_real24_aligned_reset_v2"
EXPECTED_READY_POSE_PROFILE_ID = "task1_real24_ready_pose_reset_v1"
EXPECTED_READY_POSE_STATE_SHA256 = (
    "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
)
EXPECTED_READY_POSE_TOLERANCE = 1.0e-5
EXPECTED_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXPECTED_CAMERA_PROFILE_ID = (
    "picklift_manual_viewer_center_crop_18pct_camera_v5"
)
EXPECTED_SUCCESS_SOURCE = "official MuJoCoPickLift-v1 info.success only"
LEGAL_TERMINATION_REASONS = {"max_steps_reached"}
COMPARISON_BOUNDARY = (
    "Mixed v2 versus Mixed v1 is a same-Remote-contract, single-seed, "
    "small-sample descriptive diagnostic. It does not isolate the bundled "
    "engineering changes, establish a causal effect, or constitute a "
    "real-robot or paper performance conclusion."
)


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


def load_owner_plan() -> dict[str, Any]:
    if sha256_file(PLAN_PATH) != EXPECTED_OWNER_PLAN_SHA256:
        raise RuntimeError("Mixed-v2-sim owner plan hash mismatch.")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["model"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Mixed v2 checkpoint binding drifted.")
    if plan["remote"]["deployed_commit"] != EXPECTED_REMOTE_DEPLOYED_COMMIT:
        raise RuntimeError("Remote deployed-commit binding drifted.")
    action = plan["policy_environment_contract"]
    if action["joint_order"] != list(EXPECTED_JOINT_ORDER):
        raise RuntimeError("Owner plan joint order drifted.")
    forbidden = (
        action["follower_calibration_state_gate"],
        action["sim_state_projection"],
        action["custom_absolute_calibration_clamp"],
        action["custom_relative_clamp"],
        action["additional_action_limit"],
    )
    if any(forbidden):
        raise RuntimeError("Owner plan added forbidden state/action processing.")
    if action["max_relative_target"] is not None:
        raise RuntimeError("Owner plan max_relative_target must be null.")
    return plan


def import_remote_contract(
    phase: str,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if str(REMOTE_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_ADAPTER_DIR))
    from nexus_picklift_policy_adapter import (
        POLICY_FEATURE_NAMES,
        build_real24_act100k_plan,
        experiment_contract,
    )

    contract = experiment_contract()
    if contract["plan"]["plan_sha256"] != EXPECTED_REMOTE_PLAN_SHA256:
        raise RuntimeError("Remote frozen plan SHA mismatch.")
    if tuple(POLICY_FEATURE_NAMES) != EXPECTED_JOINT_ORDER:
        raise RuntimeError("Remote adapter joint order drifted.")
    trials = build_real24_act100k_plan()[phase]
    expected_count = 12 if phase == "gate12" else 120
    if len(trials) != expected_count:
        raise RuntimeError(f"Remote {phase} episode count drifted.")
    if [trial.phase_episode_index for trial in trials] != list(
        range(expected_count)
    ):
        raise RuntimeError(f"Remote {phase} phase indices drifted.")
    if phase == "gate12":
        expected_cells = [
            f"r{row}c{column}"
            for row in range(1, 4)
            for column in range(1, 5)
        ]
        expected_seeds = list(range(9000, 9012))
    else:
        expected_cells = [
            f"r{row}c{column}"
            for _seed in range(1000, 1010)
            for row in range(1, 4)
            for column in range(1, 5)
        ]
        expected_seeds = [
            seed for seed in range(1000, 1010) for _cell in range(12)
        ]
    if [trial.cell_name for trial in trials] != expected_cells:
        raise RuntimeError(f"Remote {phase} cell order drifted.")
    if [trial.seed for trial in trials] != expected_seeds:
        raise RuntimeError(f"Remote {phase} seed order drifted.")
    return contract, trials


def validate_dataset_unit_conversion() -> dict[str, Any]:
    from so101_nexus import lerobot_dataset as conversion

    probe = np.asarray(
        [12.5, -88.0, 42.0, 91.0, -3.0, 37.5],
        dtype=np.float64,
    )
    radians = conversion.dataset_row_to_sim_qpos(probe)
    recovered = conversion.sim_qpos_to_dataset_row(radians)
    if not np.allclose(probe, recovered, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("Nexus dataset-unit conversion roundtrip failed.")
    source_path = Path(inspect.getsourcefile(conversion) or "")
    return {
        "joint_order": list(EXPECTED_JOINT_ORDER),
        "body_joint_units": "degrees",
        "gripper_units": "RANGE_0_100 percent",
        "roundtrip_probe": probe.tolist(),
        "roundtrip_valid": True,
        "conversion_source": str(source_path),
        "conversion_source_sha256": sha256_file(source_path),
    }


def validate_deployment(phase: str) -> dict[str, Any]:
    plan = load_owner_plan()
    deployed = REMOTE_DEPLOYED_COMMIT_FILE.read_text(encoding="utf-8").strip()
    if deployed != EXPECTED_REMOTE_DEPLOYED_COMMIT:
        raise RuntimeError("Remote deployed commit mismatch.")
    adapter_path = REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
    if sha256_file(adapter_path) != EXPECTED_REMOTE_ADAPTER_SHA256:
        raise RuntimeError("Remote formal adapter hash mismatch.")
    contract, trials = import_remote_contract(phase)
    environment = contract["environment"]
    clock = contract["clock"]
    if environment["camera_profile_id"] != EXPECTED_CAMERA_PROFILE_ID:
        raise RuntimeError("Remote v5 camera profile drifted.")
    if environment["production_spawn_profile_version"] != 5:
        raise RuntimeError("Remote v5 spawn profile drifted.")
    if environment["success_source"] != EXPECTED_SUCCESS_SOURCE:
        raise RuntimeError("Remote official success source drifted.")
    if environment["success_hold_env_steps"] != 25:
        raise RuntimeError("Remote official success hold drifted.")
    expected_clock = {
        "policy_hz": 20,
        "environment_hz": 50,
        "maximum_policy_ticks": 600,
        "maximum_env_steps": 1500,
        "maximum_episode_seconds": 30,
        "adaptive_stopping": False,
        "continue_after_confirmed_success": True,
    }
    for key, value in expected_clock.items():
        if clock.get(key) != value:
            raise RuntimeError(f"Remote clock field drifted: {key}.")
    from picklift_ready_pose import (
        ACTIVE_PICKLIFT_RESET_PROFILE,
        READY_POSE_TOLERANCE_DATASET_UNITS,
        REAL24_READY_POSE,
    )

    reset_contract = ACTIVE_PICKLIFT_RESET_PROFILE.contract()
    ready_contract = REAL24_READY_POSE.contract()
    if reset_contract["profile_id"] != EXPECTED_RESET_PROFILE_ID:
        raise RuntimeError("Remote aligned reset profile drifted.")
    if reset_contract["ready_pose"] != ready_contract:
        raise RuntimeError("Remote reset/ready-pose binding drifted.")
    if ready_contract["profile_id"] != EXPECTED_READY_POSE_PROFILE_ID:
        raise RuntimeError("Remote ready-pose profile drifted.")
    if ready_contract["state_sha256"] != EXPECTED_READY_POSE_STATE_SHA256:
        raise RuntimeError("Remote ready-pose state SHA drifted.")
    if READY_POSE_TOLERANCE_DATASET_UNITS != EXPECTED_READY_POSE_TOLERANCE:
        raise RuntimeError("Remote ready-pose tolerance drifted.")
    checkpoint = Path(plan["model"]["checkpoint"])
    model_hash = sha256_file(checkpoint / "model.safetensors")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Mixed v2 checkpoint file hash mismatch.")
    return {
        "status": "pass",
        "phase": phase,
        "owner_plan_sha256": EXPECTED_OWNER_PLAN_SHA256,
        "remote_plan_sha256": EXPECTED_REMOTE_PLAN_SHA256,
        "remote_deployed_commit": deployed,
        "remote_adapter_sha256": EXPECTED_REMOTE_ADAPTER_SHA256,
        "episode_count": len(trials),
        "model_sha256": model_hash,
        "checkpoint_path": str(checkpoint),
        "reset_profile": reset_contract,
        "ready_pose": ready_contract,
        "ready_pose_tolerance": READY_POSE_TOLERANCE_DATASET_UNITS,
        "clock": clock,
        "environment": environment,
        "dataset_unit_contract": validate_dataset_unit_conversion(),
        "remote_historical_upstream_binding": contract[
            "upstream_act_binding"
        ],
        "owner_policy_binding_overrides_only_external_policy": True,
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
        raise RuntimeError("Remote state is not finite float32[6] dataset units.")
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
        raise RuntimeError("Remote ready-pose evidence is unavailable.")
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
    tick0 = np.asarray(tick0_state, dtype=np.float64)
    if any(row.shape != (6,) for row in (requested, observed, delta, tick0)):
        raise RuntimeError("Ready-pose evidence arrays must have shape [6].")
    if not all(
        bool(np.isfinite(row).all())
        for row in (requested, observed, delta, tick0)
    ):
        raise RuntimeError("Ready-pose evidence contains non-finite values.")
    expected = np.asarray(
        expected_contract["state_dataset_units"],
        dtype=np.float64,
    )
    tolerance = float(application["absolute_tolerance_dataset_units"])
    if tolerance != EXPECTED_READY_POSE_TOLERANCE:
        raise RuntimeError("Ready-pose evidence tolerance drifted.")
    if not np.array_equal(requested, expected):
        raise RuntimeError("Ready-pose request drifted.")
    if not np.allclose(delta, observed - requested, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Ready-pose delta evidence is inconsistent.")
    maximum_delta = float(np.max(np.abs(delta)))
    if maximum_delta > tolerance:
        raise RuntimeError("Ready-pose tick0 exceeds tolerance.")
    if not np.allclose(tick0, observed, rtol=0.0, atol=tolerance):
        raise RuntimeError("Tick0 state differs from ready-pose evidence.")
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
            raise RuntimeError(f"Ready-pose flag drifted: {field}.")
    return {
        "ready_pose_tick0_valid": True,
        "maximum_absolute_tick0_delta": maximum_delta,
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
        raise RuntimeError(f"Remote object spawn drifted: {mismatches}")
    return {
        "object_spawn_plan_valid": True,
        "planned_spawn": planned,
        "actual_initial_pose": observed["actual_initial_pose"],
        "placement_method": observed["placement_method"],
    }


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


def build_episode_record(
    *,
    runtime_summary: dict[str, Any],
    raw_action_count: int,
    requested_action_count: int,
    valid_observation_count: int,
    environment_clipped_action_count: int,
    environment_clipped_joint_value_count: int,
    ready_pose_tick0_valid: bool,
    object_spawn_plan_valid: bool,
    interface_error: str | None,
) -> dict[str, Any]:
    record = dict(runtime_summary)
    record.update(
        {
            "raw_action_count": raw_action_count,
            "requested_action_count": requested_action_count,
            "valid_observation_count": valid_observation_count,
            "environment_clipped_action_count": environment_clipped_action_count,
            "environment_clipped_joint_value_count": (
                environment_clipped_joint_value_count
            ),
            "ready_pose_tick0_valid": ready_pose_tick0_valid,
            "object_spawn_plan_valid": object_spawn_plan_valid,
            "runner_state_projection": "none",
            "runner_absolute_calibration_clamp": "none",
            "runner_relative_clamp": "none",
            "max_relative_target": None,
            "interface_error": interface_error,
        }
    )
    record["interface_valid"] = bool(
        interface_error is None
        and valid_observation_count == 600
        and raw_action_count == 600
        and requested_action_count == 600
        and int(record["env_steps"]) == 1500
        and bool(record["timeout"])
        and record["termination_reason"] in LEGAL_TERMINATION_REASONS
        and ready_pose_tick0_valid
        and object_spawn_plan_valid
    )
    if interface_error is not None:
        record["failure_type"] = "interface_error"
    return record


def summarize_episodes(
    phase: str,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_cell[str(episode["cell"])].append(episode)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(item["success"]) for item in group)
        confirmed = [
            int(item["confirmed_success_step"])
            for item in group
            if item["confirmed_success_step"] is not None
        ]
        first = [
            int(item["first_success_step"])
            for item in group
            if item["first_success_step"] is not None
        ]
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
            "success_rate": successes / len(group) if group else 0.0,
            "first_success_steps": first,
            "confirmed_success_steps": confirmed,
            "maximum_lift_m": max(
                (float(item["max_lift_m"]) for item in group),
                default=0.0,
            ),
            "final_is_grasped_episodes": sum(
                bool(item["is_grasped"]) for item in group
            ),
            "env_steps": sum(int(item["env_steps"]) for item in group),
            "raw_action_count": sum(
                int(item["raw_action_count"]) for item in group
            ),
            "requested_action_count": sum(
                int(item["requested_action_count"]) for item in group
            ),
            "environment_clipped_action_count": sum(
                int(item["environment_clipped_action_count"]) for item in group
            ),
            "environment_clipped_joint_value_count": sum(
                int(item["environment_clipped_joint_value_count"])
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

    expected_count = 12 if phase == "gate12" else 120
    expected_per_cell = 1 if phase == "gate12" else 10
    expected_cells = {
        f"r{row}c{column}"
        for row in range(1, 4)
        for column in range(1, 5)
    }
    overall = summarize_group(episodes)
    interface_pass = bool(
        len(episodes) == expected_count
        and set(by_cell) == expected_cells
        and all(len(group) == expected_per_cell for group in by_cell.values())
        and overall["interface_valid_episodes"] == expected_count
        and overall["ready_pose_tick0_valid_episodes"] == expected_count
        and overall["object_spawn_plan_valid_episodes"] == expected_count
    )
    return {
        "schema_version": 1,
        "status": "diagnostic_only_not_real_robot_or_paper_result",
        "phase_id": phase,
        "interface_pass": interface_pass,
        "task_success_is_gate_condition": False,
        "full_600_ticks_1500_env_steps_required": True,
        "overall": overall,
        "by_cell": {
            cell: summarize_group(by_cell[cell]) for cell in sorted(by_cell)
        },
        "comparison_boundary": COMPARISON_BOUNDARY,
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


def run_phase(phase: str, output_dir: Path) -> dict[str, Any]:
    plan = load_owner_plan()
    expected_output = Path(plan["phases"][phase]["output_dir"])
    if output_dir != expected_output:
        raise RuntimeError(
            f"Output path differs from frozen {phase} path: {expected_output}"
        )
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite evidence: {output_dir}")
    validation = validate_deployment(phase)
    contract, trials = import_remote_contract(phase)

    from sim_policy_inference import Task1MixedV2ActSimInference

    policy = Task1MixedV2ActSimInference()
    if policy.model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Loaded mixed v2 ACT model SHA mismatch.")

    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    run_manifest = {
        "schema_version": 1,
        "experiment_run_id": output_dir.name,
        "phase_id": phase,
        "status": "running",
        "research_status": plan["research_status"],
        "started_at_utc": started_at.isoformat(),
        "source_commit": git_head(EXPERIMENT_DIR.parents[1]),
        "owner_plan_path": str(PLAN_PATH),
        "owner_plan_sha256": EXPECTED_OWNER_PLAN_SHA256,
        "remote_deployed_commit": validation["remote_deployed_commit"],
        "remote_adapter_path": str(
            REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
        ),
        "remote_adapter_sha256": validation["remote_adapter_sha256"],
        "remote_plan_sha256": validation["remote_plan_sha256"],
        "remote_contract": contract,
        "remote_reset_profile": validation["reset_profile"],
        "ready_pose": validation["ready_pose"],
        "ready_pose_tolerance": validation["ready_pose_tolerance"],
        "ready_pose_explicitly_passed_to_every_reset": True,
        "checkpoint_path": str(policy.checkpoint),
        "model_sha256": policy.model_hash,
        "checkpoint_owned_processor": policy.processor_contract,
        "policy_interface_path": str(
            EXPERIMENT_DIR / "sim_policy_inference.py"
        ),
        "policy_interface_sha256": sha256_file(
            EXPERIMENT_DIR / "sim_policy_inference.py"
        ),
        "action_processing": plan["policy_environment_contract"],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hardware_accessed": False,
        "gateway_or_quest_started": False,
        "lerobot_dataset_written": False,
        "comparison_boundary": COMPARISON_BOUNDARY,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    from nexus_picklift_policy_adapter import (
        create_nexus_picklift_policy_adapter,
    )
    from picklift_ready_pose import REAL24_READY_POSE

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
                requested_action_count = 0
                valid_observation_count = 0
                environment_clipped_action_count = 0
                environment_clipped_joint_value_count = 0
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
                    inputs = observation.policy_inputs()
                    validate_policy_inputs(inputs)
                    ready_pose_validation = validate_ready_pose_evidence(
                        reset_manifest["ready_pose_reset"],
                        inputs["observation.state"],
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
                        requested_action_count += 1
                        tick_result = adapter.apply_action(
                            policy_step.requested_action
                        )
                        remote_evidence = tick_result.evidence()
                        adapter_received = np.asarray(
                            remote_evidence["sent_action"],
                            dtype=np.float32,
                        )
                        if not np.array_equal(
                            adapter_received,
                            policy_step.requested_action,
                        ):
                            raise RuntimeError(
                                "Adapter input differs from unchanged requested action."
                            )
                        environment_mask = np.asarray(
                            remote_evidence["environment_clipped_mask"],
                            dtype=bool,
                        )
                        environment_clipped_action_count += int(
                            bool(environment_mask.any())
                        )
                        environment_clipped_joint_value_count += int(
                            environment_mask.sum()
                        )
                        append_jsonl(
                            ticks_handle,
                            {
                                "phase_id": trial.phase_id,
                                "phase_episode_index": (
                                    trial.phase_episode_index
                                ),
                                "cell": trial.cell_name,
                                "seed": trial.seed,
                                "observation": observation.evidence(),
                                "policy": policy_step.to_jsonable(),
                                "remote": remote_evidence,
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
                        manifest_error = f"{type(exc).__name__}: {exc}"
                        interface_error = (
                            f"{interface_error}; manifest={manifest_error}"
                            if interface_error is not None
                            else f"manifest={manifest_error}"
                        )
                runtime = (
                    manifest["runtime_summary"]
                    if manifest is not None
                    else fallback_runtime_summary(trial)
                )
                record = build_episode_record(
                    runtime_summary=runtime,
                    raw_action_count=raw_action_count,
                    requested_action_count=requested_action_count,
                    valid_observation_count=valid_observation_count,
                    environment_clipped_action_count=(
                        environment_clipped_action_count
                    ),
                    environment_clipped_joint_value_count=(
                        environment_clipped_joint_value_count
                    ),
                    ready_pose_tick0_valid=ready_pose_tick0_valid,
                    object_spawn_plan_valid=object_spawn_plan_valid,
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
                            "episode": len(episodes),
                            "cell": record["cell"],
                            "seed": record["seed"],
                            "interface_valid": record["interface_valid"],
                            "success": record["success"],
                            "env_steps": record["env_steps"],
                            "environment_clip_ticks": record[
                                "environment_clipped_action_count"
                            ],
                            "termination_reason": record[
                                "termination_reason"
                            ],
                        }
                    ),
                    flush=True,
                )
    finally:
        adapter.close()

    summary = summarize_episodes(phase, episodes)
    summary.update(
        {
            "experiment_run_id": output_dir.name,
            "owner_plan_sha256": EXPECTED_OWNER_PLAN_SHA256,
            "remote_plan_sha256": EXPECTED_REMOTE_PLAN_SHA256,
            "model_sha256": policy.model_hash,
            "checkpoint_owned_processor": policy.processor_contract,
            "remote_deployed_commit": validation["remote_deployed_commit"],
            "reset_profile": validation["reset_profile"],
            "ready_pose": validation["ready_pose"],
            "runtime_seconds": time.monotonic() - started_monotonic,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(output_dir / "summary.json", summary)
    run_manifest.update(
        {
            "status": (
                f"{phase}_complete"
                if summary["interface_pass"]
                else f"{phase}_complete_with_interface_failures"
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
                "phase": phase,
                "interface_pass": summary["interface_pass"],
                "output_dir": str(output_dir),
                "hashes": hashes,
            }
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen mixed v2 ACT formal Remote Nexus diagnostic."
    )
    parser.add_argument("--phase", choices=("gate12", "frozen120"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-contract-only", action="store_true")
    args = parser.parse_args()
    if args.validate_contract_only:
        if args.phase is None or args.output_dir is not None:
            parser.error(
                "--validate-contract-only requires --phase and no --output-dir"
            )
    elif args.phase is None or args.output_dir is None:
        parser.error("execution requires --phase and --output-dir")
    return args


def main() -> None:
    args = parse_args()
    if args.validate_contract_only:
        print(json.dumps(validate_deployment(args.phase), indent=2, sort_keys=True))
        return
    summary = run_phase(args.phase, args.output_dir.resolve())
    if not summary["interface_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
