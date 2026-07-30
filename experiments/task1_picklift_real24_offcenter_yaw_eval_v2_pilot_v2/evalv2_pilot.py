from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_PLAN = EXPERIMENT_DIR / "evaluation_plan.json"
RESEARCH_IDENTITY_VERIFICATION = EXPERIMENT_DIR / "research_identity_verification.json"
EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256 = (
    "39491b33ab46e7b8ca957b2ed8bbf68e74182898b070829f68418e697056b24c"
)
EXPECTED_PLAN_SHA256 = "48a159bf74ea61b9c444903f048f4c6315de0417248b37b500e9a1cb04b27cb5"
EXPECTED_ENGINE_SHA256 = "380b8c1c13f0f38a59e129b78d845a1cbd8916411af1f61a56b9267e83205f96"
EXPECTED_PROFILE_SHA256 = "60025f6478a63bcf9b301a75cd27124b19f1cf4f6b142f8576e6b10abf4f95a5"
EXPECTED_READY_MOVE_TOLERANCE = 3.0
EXPECTED_EVALUATION_ID = "task1_picklift_real24_offcenter_yaw_eval_v2_difficulty_pilot_v2_grid15mm"
EXPECTED_RESEARCH_COMMIT = "9d220248f5cff7c9eb78837f2636bf979185f01d"
EXPECTED_RESEARCH_HASHES = {
    "protocol": "1a5605878778506502b9d7d06e2445ce9aa509aa82f6523d646bf4a0b195eb97",
    "pose_manifest": "9acfcb7c49e6ff03f9de7b03008c204342e7e846a741a7676c795c8f14043200",
    "validation_manifest": "609d6934c1afca811622821821265762a1d483224ab9ffa8dc1a988eb844b5c3",
    "alignment_manifest": "14b01f42045865a477d764037f512559461d5c8e364d6ab75fe1d1509d0b92a1",
}
EXPECTED_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
EXPECTED_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00"
EXPECTED_CAMERA_DEVICE = "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0"
DEFAULT_CALIBRATION = Path(
    "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json"
)
READY_POSE_STATE_SHA256 = "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
MODEL_IDS = ("real24_only",)
EXPECTED_POSE_SCHEDULE = (
    ("evalv2_pilot_v2_r3c3", "r3c3", "x_minus_y_plus", 0),
    ("evalv2_pilot_v2_r3c4", "r3c4", "x_minus_y_minus", 45),
    ("evalv2_pilot_v2_r2c4", "r2c4", "x_minus_y_plus", 0),
    ("evalv2_pilot_v2_r2c1", "r2c1", "x_plus_y_minus", 45),
    ("evalv2_pilot_v2_r2c2", "r2c2", "x_plus_y_plus", 0),
    ("evalv2_pilot_v2_r2c3", "r2c3", "x_minus_y_minus", 45),
    ("evalv2_pilot_v2_r1c2", "r1c2", "x_minus_y_plus", 45),
    ("evalv2_pilot_v2_r1c4", "r1c4", "x_plus_y_plus", 45),
    ("evalv2_pilot_v2_r1c1", "r1c1", "x_minus_y_minus", 0),
    ("evalv2_pilot_v2_r1c3", "r1c3", "x_plus_y_minus", 0),
    ("evalv2_pilot_v2_r3c2", "r3c2", "x_plus_y_minus", 45),
    ("evalv2_pilot_v2_r3c1", "r3c1", "x_plus_y_plus", 0),
)
EXPECTED_NOMINAL_XY = (
    (0.31, 0.04),
    (0.31, 0.06),
    (0.26, 0.09),
    (0.29, -0.09),
    (0.29, -0.01),
    (0.26, 0.01),
    (0.21, -0.01),
    (0.24, 0.09),
    (0.21, -0.09),
    (0.24, 0.01),
    (0.34, -0.04),
    (0.34, -0.06),
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_joint_vector(values: Any, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (6,):
        raise RuntimeError(f"{label} must have shape (6,), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"{label} contains NaN or infinity.")
    return vector


def ready_pose_state_sha256(values: Any) -> str:
    vector = finite_joint_vector(values, "ready pose")
    payload = json.dumps(
        [float(value) for value in vector],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_frozen_plan(path: Path = DEFAULT_PLAN) -> dict:
    if sha256_file(path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Eval-v2 pilot plan hash differs from the frozen plan.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["evaluation_id"] != EXPECTED_EVALUATION_ID:
        raise RuntimeError("Unexpected Eval-v2 pilot identity.")
    research = plan["research_contract"]
    if research["research_repo_commit"] != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("Research-control commit reference changed.")
    for identity, expected_hash in EXPECTED_RESEARCH_HASHES.items():
        if research[identity]["sha256"] != expected_hash:
            raise RuntimeError(f"Research {identity} hash reference changed.")
    alignment = research["alignment_manifest"]
    if alignment["alignment_id"] != "task1_picklift_sim_real_geometry_alignment_v1":
        raise RuntimeError("Accepted alignment identity changed.")
    if alignment["joint_geometry_profile_id"] != "picklift_real_to_mujoco_joint_geometry_v1":
        raise RuntimeError("Accepted joint-geometry identity changed.")
    if alignment["sim_camera_profile_id"] != "picklift_real_canonical_12cell_camera_v6":
        raise RuntimeError("Accepted Sim camera identity changed.")
    if alignment["real_camera_profile_id"] != "icspring_front_crop_1280x960_to_640x480_v1":
        raise RuntimeError("Frozen Real camera identity changed.")
    if plan["execution_engine"]["source_sha256"] != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Official-send engine hash reference changed.")
    if plan["evaluation_profile"]["sha256"] != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Evaluation-profile hash reference changed.")
    if tuple(plan["models"]) != MODEL_IDS:
        raise RuntimeError("The single frozen model identity changed.")
    if plan["models"]["real24_only"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("The fixed Real24-only model hash reference changed.")
    setup = plan["setup"]
    if setup["max_relative_target"] is not None:
        raise RuntimeError("Eval-v2 pilot requires max_relative_target=None.")
    if setup["custom_absolute_action_clamp"] is not False:
        raise RuntimeError("Runner-side absolute action clamp must remain disabled.")
    if setup["custom_relative_step_limit_degrees"] is not None:
        raise RuntimeError("Runner-side relative step limiter must remain disabled.")
    if setup["control_fps"] != 20 or setup["maximum_trial_seconds"] != 30:
        raise RuntimeError("Frozen policy timing changed.")
    if setup["act_chunk_size"] != 67 or setup["act_n_action_steps"] != 67:
        raise RuntimeError("Frozen ACT action queue configuration changed.")
    if setup["ready_pose_before_every_trial"] is not True:
        raise RuntimeError("Ready pose is required before every trial.")
    if setup["ready_pose_after_every_trial"] is not True:
        raise RuntimeError("Ready pose is required after every trial.")
    if setup["policy_reset_after_ready_pose"] is not True:
        raise RuntimeError("Policy reset must happen after ready pose.")
    if setup["stop_on_success"] is not False:
        raise RuntimeError("Success must not shorten the 30-second policy window.")
    if setup["camera_profile_id"] != "icspring_front_crop_1280x960_to_640x480_v1":
        raise RuntimeError("Real camera profile changed.")
    if setup["canonical_rgb_width"] != 640 or setup["canonical_rgb_height"] != 480:
        raise RuntimeError("Canonical RGB geometry changed.")
    if setup["ready_pose_arrival_tolerance_degrees"] != EXPECTED_READY_MOVE_TOLERANCE:
        raise RuntimeError("Ready-pose arrival tolerance differs from revision v2.")
    if ready_pose_state_sha256(setup["ready_pose_state"]) != READY_POSE_STATE_SHA256:
        raise RuntimeError("Frozen ready-pose vector hash mismatch.")
    success = plan["success_contract"]
    if success["must_remain_held_until_timeout"] is not False:
        raise RuntimeError("Success must not require a hold through timeout.")
    if success["changes_policy_action_window"] is not False:
        raise RuntimeError("A success label must not shorten the action window.")
    trials = plan["trials"]
    if len(trials) != 12:
        raise RuntimeError("Eval-v2 pilot must contain exactly 12 scored trials.")
    actual_schedule = tuple(
        (
            trial["trial_id"],
            trial["cell_id"],
            trial["quadrant"],
            trial["nominal_yaw_degrees_modulo_90"],
        )
        for trial in trials
    )
    if actual_schedule != EXPECTED_POSE_SCHEDULE:
        raise RuntimeError("Pilot pose order differs from the frozen research pose manifest.")
    actual_xy = tuple(
        (trial["nominal_x_forward_m"], trial["nominal_y_lateral_m"])
        for trial in trials
    )
    if actual_xy != EXPECTED_NOMINAL_XY:
        raise RuntimeError("Pilot nominal coordinates differ from the frozen 15 mm grid revision.")
    if [trial["order"] for trial in trials] != list(range(1, 13)):
        raise RuntimeError("Pilot order indices must be contiguous 1..12.")
    if len({trial["cell_id"] for trial in trials}) != 12:
        raise RuntimeError("Every 5 cm cell must appear exactly once.")
    if len({trial["spawn_region"] for trial in trials}) != 12:
        raise RuntimeError("Each trial needs a unique immutable artifact stem.")
    if any(trial["model_id"] != "real24_only" for trial in trials):
        raise RuntimeError("Pilot must use only the fixed Real24-only checkpoint.")
    quadrant_counts = {
        quadrant: sum(trial["quadrant"] == quadrant for trial in trials)
        for quadrant in ("x_minus_y_minus", "x_minus_y_plus", "x_plus_y_minus", "x_plus_y_plus")
    }
    if set(quadrant_counts.values()) != {3}:
        raise RuntimeError(f"Quadrant balance changed: {quadrant_counts}.")
    yaw_counts = {
        yaw: sum(trial["nominal_yaw_degrees_modulo_90"] == yaw for trial in trials)
        for yaw in (0, 45)
    }
    if yaw_counts != {0: 6, 45: 6}:
        raise RuntimeError(f"Yaw balance changed: {yaw_counts}.")
    placement = plan["placement_contract"]
    if placement["manual_pose_is_measurement_truth"] is not False:
        raise RuntimeError("Manual placement must remain nominal, not measurement truth.")
    if placement["operator_numeric_coordinate_entry_required"] is not True:
        raise RuntimeError("The revised pilot must use explicit grid coordinates.")
    if placement["nominal_center_offset_from_cell_center_m"] != 0.015:
        raise RuntimeError("The revised pilot must use the frozen 15 mm offset.")
    if placement["nominal_center_lattice_m"] != 0.01:
        raise RuntimeError("The revised pilot nominal centers must use the 10 mm lattice.")
    if plan["gripper_nuisance"]["safe_open_added"] is not False:
        raise RuntimeError("Pilot must not add a safe-open action.")
    if plan["gripper_nuisance"]["allowed_range"] != [0, 100]:
        raise RuntimeError("Frozen gripper range changed.")
    return plan


def verify_static_files(plan: dict) -> dict:
    engine_path = resolve_repo_path(plan["execution_engine"]["path"])
    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    if sha256_file(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Current official-send engine differs from commit 34cc7ac.")
    if sha256_file(profile_path) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Current evaluation profile differs from the frozen profile.")
    if sha256_file(RESEARCH_IDENTITY_VERIFICATION) != EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256:
        raise RuntimeError("Research identity verification record changed.")
    research_verification = json.loads(
        RESEARCH_IDENTITY_VERIFICATION.read_text(encoding="utf-8")
    )
    if research_verification["head_commit"] != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("Verified research commit does not match the frozen plan.")
    if research_verification["working_tree_status"] != "clean":
        raise RuntimeError("Research identity was not verified from a clean checkout.")
    if any(
        research_verification["files"][name]["sha256"] != expected
        or research_verification["files"][name]["match"] is not True
        for name, expected in EXPECTED_RESEARCH_HASHES.items()
    ):
        raise RuntimeError("Research identity verification hashes changed.")
    engine_source = engine_path.read_text(encoding="utf-8")
    required_engine_fragments = (
        "max_relative_target=None",
        "requested_action = raw_action.copy()",
        "robot.send_action(action_dict(requested_action))",
        "steps = run_paced_ticks",
        "return_result = move_to_frozen_ready_pose",
    )
    if any(fragment not in engine_source for fragment in required_engine_fragments):
        raise RuntimeError("Official-send engine contract fragments changed.")
    frame_write = engine_source.index("video_process.stdin.write(frame_bytes)")
    tick0_inference = engine_source.index("select_action(")
    action_send = engine_source.index("robot.send_action(action_dict(requested_action))")
    if not frame_write < tick0_inference < action_send:
        raise RuntimeError("Canonical pre-action frame ordering changed.")
    models: dict[str, dict] = {}
    for model_id, model in plan["models"].items():
        checkpoint = Path(model["checkpoint"])
        weights = checkpoint / "model.safetensors"
        config_path = checkpoint / "config.json"
        train_config_path = checkpoint / "train_config.json"
        preprocessor_path = checkpoint / "policy_preprocessor.json"
        normalizer_stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        actual_hash = sha256_file(weights)
        if actual_hash != model["model_sha256"]:
            raise RuntimeError(f"{model_id} checkpoint hash mismatch.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("type") != "act":
            raise RuntimeError(f"{model_id} is not an ACT checkpoint.")
        if config.get("chunk_size") != 67 or config.get("n_action_steps") != 67:
            raise RuntimeError(f"{model_id} ACT queue configuration changed.")
        expected_hashes = {
            config_path: model["config_sha256"],
            train_config_path: model["train_config_sha256"],
            preprocessor_path: model["policy_preprocessor_sha256"],
            normalizer_stats_path: model["normalizer_stats_sha256"],
        }
        for path, expected_hash in expected_hashes.items():
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"{model_id} frozen file hash mismatch: {path.name}.")
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        if train_config["dataset"].get("use_imagenet_stats") is not True:
            raise RuntimeError(f"{model_id} does not use ImageNet visual stats.")
        stats = load_file(normalizer_stats_path)
        image_mean = stats["observation.images.front.mean"].cpu().numpy().reshape(-1)
        image_std = stats["observation.images.front.std"].cpu().numpy().reshape(-1)
        if not np.allclose(image_mean, IMAGENET_MEAN, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(f"{model_id} saved visual mean is not ImageNet mean.")
        if not np.allclose(image_std, IMAGENET_STD, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(f"{model_id} saved visual std is not ImageNet std.")
        models[model_id] = {
            "checkpoint": str(checkpoint),
            "model_sha256": actual_hash,
            "config_sha256": sha256_file(config_path),
            "train_config_sha256": sha256_file(train_config_path),
            "policy_preprocessor_sha256": sha256_file(preprocessor_path),
            "normalizer_stats_sha256": sha256_file(normalizer_stats_path),
            "use_imagenet_stats": True,
            "visual_mean": image_mean.tolist(),
            "visual_std": image_std.tolist(),
            "chunk_size": config["chunk_size"],
            "n_action_steps": config["n_action_steps"],
        }
    return {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "engine_path": str(engine_path),
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "profile_path": str(profile_path),
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "research_identity_verification": {
            "path": str(RESEARCH_IDENTITY_VERIFICATION),
            "sha256": EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256,
            "research_repo_commit": EXPECTED_RESEARCH_COMMIT,
        },
        "official_send_contract": {
            "max_relative_target_none": True,
            "runner_absolute_clamp": False,
            "runner_step_limiter": None,
            "no_catch_up_pacing": True,
            "canonical_frame_written_before_inference_and_send": True,
        },
        "models": models,
    }


class FakeBus:
    def __init__(self, ready_pose: np.ndarray) -> None:
        self.state = ready_pose.copy()
        self.torque_enabled = False
        self.sent: list[np.ndarray] = []

    def move_to_ready(self, requested: np.ndarray) -> np.ndarray:
        self.state = finite_joint_vector(requested, "fake ready request").copy()
        return self.state.copy()

    def send(self, requested: np.ndarray) -> np.ndarray:
        action = finite_joint_vector(requested, "fake policy action").copy()
        self.sent.append(action)
        self.state = action.copy()
        return action


class FakeCamera:
    def capture(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeRobot:
    def __init__(self, ready_pose: np.ndarray) -> None:
        self.bus = FakeBus(ready_pose)
        self.camera = FakeCamera()

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bus.state.copy(), self.camera.capture()


class FakePolicy:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def action(self, state: np.ndarray) -> np.ndarray:
        return finite_joint_vector(state, "fake observation").copy()


def run_fake_protocol(plan: dict) -> dict:
    ready_pose = finite_joint_vector(
        plan["setup"]["ready_pose_state"],
        "ready pose",
    )
    robot = FakeRobot(ready_pose)
    policies = {model_id: FakePolicy(model_id) for model_id in MODEL_IDS}
    records = []
    for trial in plan["trials"]:
        robot.bus.torque_enabled = True
        observed_ready = robot.bus.move_to_ready(ready_pose)
        policy = policies[trial["model_id"]]
        policy.reset()
        tick0_state, canonical_rgb = robot.observation()
        raw_requested = policy.action(tick0_state)
        official_sent = robot.bus.send(raw_requested)
        observed_return = robot.bus.move_to_ready(ready_pose)
        robot.bus.torque_enabled = False
        records.append(
            {
                "order": trial["order"],
                "trial_id": trial["trial_id"],
                "cell_id": trial["cell_id"],
                "quadrant": trial["quadrant"],
                "nominal_yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
                "operator_placement_prompt_zh": trial["operator_placement_prompt_zh"],
                "model_id": trial["model_id"],
                "ready_before_policy_matches": bool(np.array_equal(observed_ready, ready_pose)),
                "policy_reset_before_tick0": True,
                "tick0_state": tick0_state.tolist(),
                "canonical_rgb_shape": list(canonical_rgb.shape),
                "pre_action_frame": {
                    "frame_index": 0,
                    "canonical_rgb_shape": list(canonical_rgb.shape),
                    "captured_before_tick0_inference": True,
                    "captured_before_tick0_action_send": True,
                    "nominal_requested_pose": {
                        "x_forward_m": trial["nominal_x_forward_m"],
                        "y_lateral_m": trial["nominal_y_lateral_m"],
                        "yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
                    },
                    "manual_pose_is_measurement_truth": False,
                },
                "raw_requested_action": raw_requested.tolist(),
                "official_sent_action": official_sent.tolist(),
                "ready_after_trial_matches": bool(np.array_equal(observed_return, ready_pose)),
                "torque_disabled": not robot.bus.torque_enabled,
            }
        )
    return {
        "fake_hardware_only": True,
        "real_device_accessed": False,
        "trials_exercised": len(records),
        "pose_trial_ids_in_frozen_order": [row["trial_id"] for row in records],
        "models_in_frozen_order": [row["model_id"] for row in records],
        "policy_reset_calls": {model_id: policy.reset_calls for model_id, policy in policies.items()},
        "all_ready_before_policy": all(row["ready_before_policy_matches"] for row in records),
        "all_ready_after_trial": all(row["ready_after_trial_matches"] for row in records),
        "all_canonical_rgb_640x480": all(row["canonical_rgb_shape"] == [480, 640, 3] for row in records),
        "all_pre_action_frames_before_policy_send": all(
            row["pre_action_frame"]["captured_before_tick0_inference"]
            and row["pre_action_frame"]["captured_before_tick0_action_send"]
            and row["pre_action_frame"]["manual_pose_is_measurement_truth"] is False
            for row in records
        ),
        "all_official_sent_equals_requested": all(
            row["official_sent_action"] == row["raw_requested_action"] for row in records
        ),
        "all_torque_disabled": all(row["torque_disabled"] for row in records),
        "success_contract_probe": {
            "valid_success_seen_inside_window": True,
            "held_at_window_end": False,
            "scored_success": (plan["success_contract"]["must_remain_held_until_timeout"] is False),
            "policy_window_unchanged": (plan["success_contract"]["changes_policy_action_window"] is False),
        },
        "records": records,
    }


def run_paced_ticks(
    duration_seconds: float,
    tick_fn: Callable[[int, float, float], dict],
    record_fn: Callable[[dict], None],
    *,
    period: float,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict]:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite.")
    records = []
    loop_started = now_fn()
    step = 0
    while now_fn() - loop_started < duration_seconds:
        tick_started = now_fn()
        record = tick_fn(step, tick_started, loop_started)
        compute_seconds = now_fn() - tick_started
        scheduled_sleep_seconds = max(0.0, period - compute_seconds)
        record.update(
            {
                "step": step,
                "tick_started_elapsed_seconds": tick_started - loop_started,
                "loop_compute_seconds": compute_seconds,
                "scheduled_sleep_seconds": scheduled_sleep_seconds,
            }
        )
        record_fn(record)
        records.append(record)
        sleep_fn(scheduled_sleep_seconds)
        record["tick_completed_elapsed_seconds"] = now_fn() - loop_started
        step += 1
    return records


def software_dry_run(plan: dict) -> dict:
    static = verify_static_files(plan)
    fake = run_fake_protocol(plan)
    if fake["trials_exercised"] != 12:
        raise RuntimeError("Fake protocol did not exercise all 12 pilot trials.")
    if fake["policy_reset_calls"] != {"real24_only": 12}:
        raise RuntimeError("The fixed policy must be reset once for each pilot trial.")
    checks = (
        fake["all_ready_before_policy"],
        fake["all_ready_after_trial"],
        fake["all_canonical_rgb_640x480"],
        fake["all_pre_action_frames_before_policy_send"],
        fake["all_official_sent_equals_requested"],
        fake["all_torque_disabled"],
        fake["success_contract_probe"]["scored_success"],
        fake["success_contract_probe"]["policy_window_unchanged"],
    )
    if not all(checks):
        raise RuntimeError("Fake Eval-v2 pilot protocol verification failed.")
    return {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "software_dry_run_passed_hardware_not_accessed",
        "hardware_access": {
            "serial": False,
            "camera": False,
            "robot": False,
            "torque": False,
            "rollout": False,
        },
        "static_verification": static,
        "fake_protocol": fake,
        "next_gate": "Stop before the user turns on Follower 12 V.",
    }


def write_software_evidence(plan: dict, dry_run: dict) -> dict:
    evidence_root = Path(plan["evidence_root"])
    software_root = evidence_root / "software_preparation_v1"
    software_root.mkdir(parents=True, exist_ok=True)
    plan_copy = software_root / "evaluation_plan.json"
    research_copy = software_root / "research_identity_verification.json"
    dry_run_path = software_root / "dry_run.json"
    manifest_path = software_root / "manifest.json"
    hashes_path = software_root / "hashes.sha256"
    for path in (plan_copy, research_copy, dry_run_path, manifest_path, hashes_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite frozen evidence: {path}")
    shutil.copyfile(DEFAULT_PLAN, plan_copy)
    shutil.copyfile(RESEARCH_IDENTITY_VERIFICATION, research_copy)
    dry_run_path.write_text(
        json.dumps(dry_run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "status": dry_run["status"],
        "evidence_root": str(evidence_root),
        "research_contract": plan["research_contract"],
        "plan": {
            "repo_path": str(DEFAULT_PLAN),
            "evidence_copy": str(plan_copy),
            "sha256": sha256_file(plan_copy),
        },
        "research_identity_verification": {
            "repo_path": str(RESEARCH_IDENTITY_VERIFICATION),
            "evidence_copy": str(research_copy),
            "sha256": sha256_file(research_copy),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "static_verification": dry_run["static_verification"],
        "dry_run": {
            "path": str(dry_run_path),
            "sha256": sha256_file(dry_run_path),
        },
        "hardware_access": dry_run["hardware_access"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hash_rows = [
        (sha256_file(plan_copy), plan_copy),
        (sha256_file(research_copy), research_copy),
        (sha256_file(dry_run_path), dry_run_path),
        (sha256_file(manifest_path), manifest_path),
    ]
    hashes_path.write_text(
        "".join(f"{digest}  {path.name}\n" for digest, path in hash_rows),
        encoding="utf-8",
    )
    return {
        "evidence_root": str(evidence_root),
        "software_root": str(software_root),
        "plan_sha256": sha256_file(plan_copy),
        "research_identity_verification_sha256": sha256_file(research_copy),
        "dry_run_sha256": sha256_file(dry_run_path),
        "manifest_sha256": sha256_file(manifest_path),
        "hashes_sha256": sha256_file(hashes_path),
    }


def find_trial(plan: dict, trial_id: str) -> dict:
    matches = [trial for trial in plan["trials"] if trial["trial_id"] == trial_id]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown frozen trial id: {trial_id}")
    return matches[0]


def original_evidence_path(plan: dict, trial: dict) -> Path:
    return Path(plan["evidence_root"]) / "trials" / f"{trial['spawn_region']}.json"


def validate_execution_order(
    plan: dict,
    trial: dict,
    *,
    replacement: bool,
) -> tuple[str, str | None]:
    trials_root = Path(plan["evidence_root"]) / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    original_path = original_evidence_path(plan, trial)
    first_missing = next(
        (candidate for candidate in plan["trials"] if not original_evidence_path(plan, candidate).exists()),
        None,
    )
    if not replacement:
        if first_missing is None:
            raise RuntimeError("All 12 frozen original pilot trials already have evidence.")
        if first_missing["trial_id"] != trial["trial_id"]:
            raise RuntimeError(
                "Requested trial is not the next missing trial in frozen order: "
                f"expected {first_missing['trial_id']}."
            )
        return trial["spawn_region"], None
    if not original_path.exists():
        raise RuntimeError("Replacement requires preserved original evidence.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    infrastructure_invalid = (
        original.get("status") == "aborted_with_error"
        or original.get("termination") == "hardware_or_runtime_error"
    )
    if not infrastructure_invalid:
        raise RuntimeError("Replacement is allowed only for infrastructure-invalid trials.")
    replacement_stem = f"{trial['spawn_region']}__replacement1"
    if (trials_root / f"{replacement_stem}.json").exists():
        raise RuntimeError("The one allowed linked replacement already exists.")
    return replacement_stem, trial["spawn_region"]


def load_official_engine(plan: dict):
    engine_path = resolve_repo_path(plan["execution_engine"]["path"])
    if sha256_file(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Official-send engine source hash changed.")
    if str(engine_path.parent) not in sys.path:
        sys.path.insert(0, str(engine_path.parent))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "task1_official_send_engine",
        engine_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the frozen official-send engine.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_pre_action_frame(video_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite canonical pre-action frame: {output_path}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required to freeze the pre-action frame.")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("Pre-action frame does not contain exactly one image stream.")
    width = streams[0].get("width")
    height = streams[0].get("height")
    if (width, height) != (640, 480):
        raise RuntimeError(f"Pre-action frame must be 640x480, got {width}x{height}.")
    return {
        "status": "frozen",
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "frame_index": 0,
        "width": width,
        "height": height,
        "source": "canonical_rgb_act_input_video_frame_0",
        "capture_semantics": (
            "The official engine writes canonical frame 0 before tick-0 inference "
            "and before the first policy action send."
        ),
    }


def build_pre_action_evidence(evidence: dict, trial: dict, trials_root: Path, artifact_stem: str) -> dict:
    nominal_pose = {
        "cell_id": trial["cell_id"],
        "quadrant": trial["quadrant"],
        "x_forward_m": trial["nominal_x_forward_m"],
        "y_lateral_m": trial["nominal_y_lateral_m"],
        "yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
    }
    video = evidence.get("video", {})
    video_path = Path(video["path"]) if video.get("path") else None
    if (
        video.get("source") != "canonical_rgb_act_input"
        or video.get("frames", 0) < 1
        or video_path is None
        or not video_path.exists()
    ):
        return {
            "status": "unavailable_in_infrastructure_invalid_trial",
            "nominal_requested_pose": nominal_pose,
            "manual_pose_is_measurement_truth": False,
        }
    if sha256_file(video_path) != video["sha256"]:
        raise RuntimeError("Canonical video hash changed before pre-action extraction.")
    frame_path = trials_root / f"{artifact_stem}.pre_action.png"
    frozen = extract_pre_action_frame(video_path, frame_path)
    frozen.update(
        {
            "nominal_requested_pose": nominal_pose,
            "manual_pose_is_measurement_truth": False,
            "placement_claim": "nominal_manual_placement_only_not_instrumented_ground_truth",
        }
    )
    return frozen


def write_evalv2_sidecar(
    plan: dict,
    trial: dict,
    artifact_stem: str,
    replacement_for: str | None,
) -> None:
    trials_root = Path(plan["evidence_root"]) / "trials"
    engine_evidence_path = trials_root / f"{artifact_stem}.json"
    if not engine_evidence_path.exists():
        return
    sidecar_path = trials_root / f"{artifact_stem}.evalv2_pilot.json"
    if sidecar_path.exists():
        raise RuntimeError(f"Refusing to overwrite Eval-v2 pilot sidecar: {sidecar_path}")
    evidence = json.loads(engine_evidence_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(evidence["started_at_utc"])
    ended = datetime.fromisoformat(evidence["ended_at_utc"])
    pre_action = build_pre_action_evidence(evidence, trial, trials_root, artifact_stem)
    sidecar = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "trial": trial,
        "artifact_stem": artifact_stem,
        "replacement_for": replacement_for,
        "engine_evidence": {
            "path": str(engine_evidence_path),
            "sha256": sha256_file(engine_evidence_path),
        },
        "actual_policy_ticks": evidence["steps_jsonl"]["lines"],
        "wall_duration_seconds": (ended - started).total_seconds(),
        "pre_action_frame": pre_action,
        "placement_contract": plan["placement_contract"],
        "operator_label": {"status": "pending"},
        "canonical_video_review_label": {"status": "pending"},
        "success_contract": plan["success_contract"],
        "return": evidence["automatic_return"],
        "torque_disable_verified": evidence["torque_disable_verified"],
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_hardware(args: argparse.Namespace, plan: dict) -> None:
    if not args.operator_confirmed_ready:
        raise RuntimeError("--execute-hardware requires --operator-confirmed-ready.")
    if args.trial_id is None:
        raise RuntimeError("--execute-hardware requires --trial-id.")
    verify_static_files(plan)
    trial = find_trial(plan, args.trial_id)
    artifact_stem, replacement_for = validate_execution_order(
        plan,
        trial,
        replacement=args.replacement,
    )
    engine = load_official_engine(plan)
    model = plan["models"][trial["model_id"]]
    engine.EXPECTED_MODEL_SHA256 = model["model_sha256"]
    engine.EXPECTED_PLAN_SHA256 = EXPECTED_PLAN_SHA256
    engine.EXPECTED_PROFILE_SHA256 = EXPECTED_PROFILE_SHA256
    engine.EXPECTED_EVALUATION_ID = EXPECTED_EVALUATION_ID
    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    ready_movement = profile["ready_pose_movement"]
    if ready_movement["arrival_tolerance_degrees"] != EXPECTED_READY_MOVE_TOLERANCE:
        raise RuntimeError("Loaded profile has the wrong ready-pose tolerance.")
    engine.READY_MOVE_TOLERANCE = EXPECTED_READY_MOVE_TOLERANCE
    engine.READY_MOVE_PROFILE_ID = ready_movement["profile_id"]
    engine_args = argparse.Namespace(
        execute_hardware=True,
        operator_confirmed_ready=True,
        spawn_region=trial["spawn_region"],
        follower_port=args.follower_port,
        camera_device=args.camera_device,
        checkpoint=Path(model["checkpoint"]),
        calibration=args.calibration,
        plan=args.plan,
        profile=profile_path,
        evidence_dir=Path(plan["evidence_root"]) / "trials",
        maximum_trial_seconds=30.0,
    )
    preflight = engine.preflight(engine_args)
    preflight.update(
        {
            "evalv2_pilot_evaluation_id": plan["evaluation_id"],
            "evalv2_pilot_plan_sha256": EXPECTED_PLAN_SHA256,
            "evalv2_pilot_trial": trial,
            "replacement_for": replacement_for,
            "success_contract": plan["success_contract"],
            "operator_label": {"status": "pending"},
            "canonical_video_review_label": {"status": "pending"},
        }
    )
    engine_args.spawn_region = artifact_stem
    try:
        engine.run_hardware_trial(engine_args, preflight)
    finally:
        write_evalv2_sidecar(
            plan,
            trial,
            artifact_stem,
            replacement_for,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Task1 Real24-only off-center/yaw Eval-v2 pilot. "
            "Software dry-run never inspects camera, serial, robot, or torque."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--software-dry-run", action="store_true")
    mode.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--freeze-software-evidence", action="store_true")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--trial-id")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--operator-confirmed-ready", action="store_true")
    parser.add_argument("--follower-port", default=EXPECTED_FOLLOWER_PORT)
    parser.add_argument("--camera-device", default=EXPECTED_CAMERA_DEVICE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_frozen_plan(args.plan)
    if args.software_dry_run:
        if args.trial_id or args.replacement or args.operator_confirmed_ready:
            raise RuntimeError("Hardware-only arguments are invalid in dry-run mode.")
        dry_run = software_dry_run(plan)
        if args.freeze_software_evidence:
            dry_run["frozen_evidence"] = write_software_evidence(plan, dry_run)
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        print("DRY RUN ONLY: no serial, camera, robot, torque, 12 V, or rollout was accessed.")
        return
    if args.freeze_software_evidence:
        raise RuntimeError("--freeze-software-evidence is dry-run only.")
    execute_hardware(args, plan)


if __name__ == "__main__":
    main()
