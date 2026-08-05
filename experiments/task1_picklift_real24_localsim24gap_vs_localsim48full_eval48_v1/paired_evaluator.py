from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
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
SOURCE_RESEARCH_CONTRACT = EXPERIMENT_DIR / "source_authorization.json"
SOURCE_TRAINING_RESULT = EXPERIMENT_DIR / "source_training_result.json"
SOURCE_POSE_MANIFEST = EXPERIMENT_DIR / "source_pose_manifest.json"
SOURCE_EVAL48_PLAN = REPO_ROOT / "experiments/task1_picklift_real48_vs_real96_eval48_v1/evaluation_plan.json"
EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256 = (
    "9b3d3c4b3ed65156a300c1731737b6dc789c86547eb848e208545e86a646ecb5"
)
EXPECTED_PLAN_SHA256 = "61efcb5e4298c86a103d41a963643962801375a4a78f0919b50c037c783ce176"
EXPECTED_ENGINE_SHA256 = "380b8c1c13f0f38a59e129b78d845a1cbd8916411af1f61a56b9267e83205f96"
EXPECTED_PROFILE_SHA256 = "6b031bb4c980467addb3e69d68a16032ceae7e45fb3f8e2288d8a4989ff3cbf3"
EXPECTED_READY_MOVE_TOLERANCE = 3.0
READY_INTERPOLATION_DURATION_SECONDS = 3.0
READY_INTERPOLATION_CONTROL_FPS = 20
READY_INTERPOLATION_STEPS = 60
EXPECTED_EVALUATION_ID = "task1_picklift_real24_localsim24gap_vs_localsim48full_eval48_v1"
EXPECTED_RESEARCH_COMMIT = "340facbbcf7b8eb60a062e8ec54d64b96ce0ba86"
EXPECTED_RESEARCH_HASHES = {
    "experiment_design": "696980b1a78d5f2d2ee71c96a72e7ede23a34b51fddde93caabfa04767394342",
    "training_result": "a2a320bcfbc3ff6bfcbf2000a17ca15804e1a7b00e9477e9f16a1577b82b2477",
    "pose_manifest": "f6bc79e9b99818f12f0e6a374688850374ea6f5cb971ba5da7ef3f32ae8322e7",
}
EXPECTED_MODEL_SHA256 = {
    "real24_localsim24_gap": "e9bbbc96e3104435d670e450090ab143610e2cdba8d38485beec339d5230577c",
    "real24_localsim48_full": "735af1dc914c1ea5b82fada65a3c72439cb5603ac8731425f43819f849972c0e",
}
EXPECTED_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00"
EXPECTED_CAMERA_DEVICE = "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0"
DEFAULT_CALIBRATION = Path(
    "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json"
)
READY_POSE_STATE_SHA256 = "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
MODEL_IDS = ("real24_localsim24_gap", "real24_localsim48_full")
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
        raise RuntimeError("Eval48 plan hash differs from the frozen plan.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["evaluation_id"] != EXPECTED_EVALUATION_ID:
        raise RuntimeError("Unexpected Eval48 identity.")
    research = plan["research_contract"]
    if research["research_repo_commit"] != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("Research-control commit reference changed.")
    for identity, expected_hash in EXPECTED_RESEARCH_HASHES.items():
        if research[identity]["sha256"] != expected_hash:
            raise RuntimeError(f"Research {identity} hash reference changed.")
    if plan["execution_engine"]["source_sha256"] != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Official-send engine hash reference changed.")
    if plan["evaluation_profile"]["sha256"] != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Evaluation-profile hash reference changed.")
    if tuple(plan["models"].keys()) != MODEL_IDS:
        raise RuntimeError("The two frozen model identities or their order changed.")
    for model_id, expected_hash in EXPECTED_MODEL_SHA256.items():
        if plan["models"][model_id]["model_sha256"] != expected_hash:
            raise RuntimeError(f"The fixed {model_id} model hash reference changed.")
    setup = plan["setup"]
    if setup["max_relative_target"] is not None:
        raise RuntimeError("Eval48 requires max_relative_target=None.")
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
    if setup["ready_pose_movement_profile_id"] != (
        "task1_real24_ready_pose_interpolated3s_official_send_tolerance3_v1"
    ):
        raise RuntimeError("Ready/return interpolation profile identity changed.")
    if setup["ready_pose_interpolation_duration_seconds"] != READY_INTERPOLATION_DURATION_SECONDS:
        raise RuntimeError("Ready/return interpolation duration changed.")
    if setup["ready_pose_interpolation_steps"] != READY_INTERPOLATION_STEPS:
        raise RuntimeError("Ready/return interpolation step count changed.")
    if setup["ready_pose_post_interpolation_hold_until_tolerance"] is not True:
        raise RuntimeError("Ready/return must hold the exact target until tolerance is observed.")
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
    if success["unsupported_lift_strictly_greater_than_m"] != 0.05:
        raise RuntimeError("Success lift threshold must remain strictly greater than 5 cm.")
    if success["continuous_hold_seconds_minimum"] != 0.5:
        raise RuntimeError("Success hold duration changed.")
    if success["must_remain_held_until_timeout"] is not False:
        raise RuntimeError("Success must not require a hold through timeout.")
    if success["changes_policy_action_window"] is not False:
        raise RuntimeError("A success label must not shorten the action window.")
    trials = plan["trials"]
    if len(trials) != 96:
        raise RuntimeError("Paired Eval48 plan must contain exactly 96 scored trials.")
    if [trial["order"] for trial in trials] != list(range(1, 97)):
        raise RuntimeError("Paired order indices must be contiguous 1..96.")
    if len({trial["artifact_stem"] for trial in trials}) != 96:
        raise RuntimeError("Each trial needs a unique immutable artifact stem.")
    source = json.loads(SOURCE_POSE_MANIFEST.read_text(encoding="utf-8"))
    source_poses = source["ordered_eval_poses"]
    source_trials = trials[::2]
    if [trial["eval_pose_id"] for trial in source_trials] != [pose["eval_pose_id"] for pose in source_poses]:
        raise RuntimeError("Pose order differs from the frozen research manifest.")
    if any(
        left["pose_order"] != pair_index
        or right["pose_order"] != pair_index
        or left["eval_pose_id"] != right["eval_pose_id"]
        or left["cell"] != right["cell"]
        or left["quadrant"] != right["quadrant"]
        or left["nominal_x_forward_m"] != right["nominal_x_forward_m"]
        or left["nominal_y_lateral_m"] != right["nominal_y_lateral_m"]
        or left["nominal_yaw_degrees_modulo_90"] != right["nominal_yaw_degrees_modulo_90"]
        or left["source_order_sha256"] != right["source_order_sha256"]
        for pair_index, (left, right) in enumerate(
            zip(trials[::2], trials[1::2], strict=True),
            start=1,
        )
    ):
        raise RuntimeError("A paired pose does not preserve identical frozen pose fields.")
    expected_pair_models = [
        (MODEL_IDS[0], MODEL_IDS[1]) if pose["order"] % 2 == 1 else (MODEL_IDS[1], MODEL_IDS[0])
        for pose in source_poses
    ]
    actual_pair_models = [
        (left["model_key"], right["model_key"]) for left, right in zip(trials[::2], trials[1::2], strict=True)
    ]
    if actual_pair_models != expected_pair_models:
        raise RuntimeError("First/second model order differs from the frozen pose manifest.")
    if {model_id: sum(trial["model_key"] == model_id for trial in trials) for model_id in MODEL_IDS} != {
        MODEL_IDS[0]: 48,
        MODEL_IDS[1]: 48,
    }:
        raise RuntimeError("Each fixed checkpoint must appear exactly once per frozen pose.")
    if {
        tier: sum(trial["coverage_tier"] == tier for trial in source_trials)
        for tier in ("seen_by_real48", "added_by_real96", "unseen_by_both")
    } != {"seen_by_real48": 24, "added_by_real96": 18, "unseen_by_both": 6}:
        raise RuntimeError("Coverage-tier balance changed.")
    if any(
        sum(trial["cell"] == cell for trial in source_trials) != 4
        for cell in {trial["cell"] for trial in source_trials}
    ):
        raise RuntimeError("Each cell must contain exactly four frozen poses.")
    yaw_counts = {
        yaw: sum(trial["nominal_yaw_degrees_modulo_90"] == yaw for trial in source_trials) for yaw in (0, 45)
    }
    if yaw_counts != {0: 24, 45: 24}:
        raise RuntimeError(f"Yaw balance changed: {yaw_counts}.")
    placement = plan["placement_contract"]
    if placement["manual_pose_is_measurement_truth"] is not False:
        raise RuntimeError("Manual placement must remain nominal, not measurement truth.")
    replacement = plan["replacement_contract"]
    if replacement["model_or_task_failure_retry_allowed"] is not False:
        raise RuntimeError("Model/task failure retries must remain prohibited.")
    if replacement["maximum_linked_replacements_per_original"] != 1:
        raise RuntimeError("At most one linked infrastructure replacement is allowed.")
    if set(replacement["allowed_only_for"]) != {
        "policy_window_never_started",
        "confirmed_operator_placement_error",
        "infrastructure_error",
    }:
        raise RuntimeError("Replacement reason contract changed.")
    authorization = plan["authorization"]
    if authorization["hardware_authorized"] is not False:
        raise RuntimeError("The frozen software-preparation plan must stop before hardware authorization.")
    if any(
        authorization[key] is not False
        for key in (
            "serial_accessed_during_preparation",
            "camera_accessed_during_preparation",
            "robot_accessed_during_preparation",
            "torque_accessed_during_preparation",
            "rollout_executed_during_preparation",
        )
    ):
        raise RuntimeError("Software preparation may not claim any hardware access.")
    return plan


def verify_static_files(plan: dict) -> dict:
    engine_path = resolve_repo_path(plan["execution_engine"]["path"])
    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    calibration_path = Path(plan["setup"]["follower_calibration_path"])
    if sha256_file(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Current official-send engine differs from commit 34cc7ac.")
    if sha256_file(profile_path) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Current evaluation profile differs from the frozen profile.")
    if sha256_file(calibration_path) != plan["setup"]["follower_calibration_sha256"]:
        raise RuntimeError("Frozen Follower calibration hash changed.")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    movement = profile["ready_pose_movement"]
    if movement["profile_id"] != "task1_real24_ready_pose_interpolated3s_official_send_tolerance3_v1":
        raise RuntimeError("Interpolated ready/return profile id changed.")
    if movement["trajectory"] != "time_parameterized_linear_interpolation":
        raise RuntimeError("Ready/return trajectory type changed.")
    if movement["interpolation_duration_seconds"] != READY_INTERPOLATION_DURATION_SECONDS:
        raise RuntimeError("Ready/return duration changed.")
    if movement["interpolation_steps"] != READY_INTERPOLATION_STEPS:
        raise RuntimeError("Ready/return step count changed.")
    if movement["custom_relative_step_limit"] is not None:
        raise RuntimeError("Ready/return revision must not add a current-relative limiter.")
    if sha256_file(RESEARCH_IDENTITY_VERIFICATION) != EXPECTED_RESEARCH_IDENTITY_VERIFICATION_SHA256:
        raise RuntimeError("Research identity verification record changed.")
    research_verification = json.loads(RESEARCH_IDENTITY_VERIFICATION.read_text(encoding="utf-8"))
    if research_verification["head_commit"] != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("Verified research commit does not match the frozen plan.")
    if research_verification["tracked_working_tree_status"] != "clean":
        raise RuntimeError("Research identity was not verified from a clean tracked checkout.")
    if any(
        research_verification["files"][name]["sha256"] != expected
        or research_verification["files"][name]["match"] is not True
        for name, expected in EXPECTED_RESEARCH_HASHES.items()
    ):
        raise RuntimeError("Research identity verification hashes changed.")
    if sha256_file(SOURCE_RESEARCH_CONTRACT) != EXPECTED_RESEARCH_HASHES["experiment_design"]:
        raise RuntimeError("Research contract snapshot changed.")
    if sha256_file(SOURCE_TRAINING_RESULT) != EXPECTED_RESEARCH_HASHES["training_result"]:
        raise RuntimeError("Research training-result snapshot changed.")
    if sha256_file(SOURCE_POSE_MANIFEST) != EXPECTED_RESEARCH_HASHES["pose_manifest"]:
        raise RuntimeError("Research pose snapshot changed.")
    if sha256_file(SOURCE_EVAL48_PLAN) != (
        "7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1"
    ):
        raise RuntimeError("Frozen source Eval48 plan changed.")
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
        expected_inputs = {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
        }
        expected_outputs = {
            "action": {"type": "ACTION", "shape": [6]},
        }
        if config.get("input_features") != expected_inputs:
            raise RuntimeError(f"{model_id} frozen front+state input contract changed.")
        if config.get("output_features") != expected_outputs:
            raise RuntimeError(f"{model_id} frozen action[6] output contract changed.")
        expected_hashes = {
            config_path: model["config_sha256"],
            train_config_path: model["train_config_sha256"],
            preprocessor_path: model["policy_preprocessor_sha256"],
            normalizer_stats_path: model["processor_stats_sha256"],
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
            "input_features": config["input_features"],
            "output_features": config["output_features"],
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
        "hardware_identity_contract": {
            "follower_port_by_id": plan["setup"]["follower_port_by_id"],
            "camera_device_by_id": plan["setup"]["camera_device_by_id"],
            "camera_profile_id": plan["setup"]["camera_profile_id"],
            "calibration_path": plan["setup"]["follower_calibration_path"],
            "calibration_sha256": sha256_file(Path(plan["setup"]["follower_calibration_path"])),
            "devices_opened": False,
        },
        "official_send_contract": {
            "max_relative_target_none": True,
            "runner_absolute_clamp": False,
            "runner_step_limiter": None,
            "no_catch_up_pacing": True,
            "canonical_frame_written_before_inference_and_send": True,
            "ready_return_trajectory": "linear_3s_20hz_then_hold_target",
            "ready_return_official_send": True,
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
        policy = policies[trial["model_key"]]
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
                "cell": trial["cell"],
                "coverage_tier": trial["coverage_tier"],
                "quadrant": trial["quadrant"],
                "nominal_yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
                "operator_placement_prompt_zh": trial["operator_placement_prompt_zh"],
                "model_key": trial["model_key"],
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
        "models_in_frozen_order": [row["model_key"] for row in records],
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


def run_fake_interpolated_ready_probe(plan: dict) -> dict:
    ready_pose = finite_joint_vector(plan["setup"]["ready_pose_state"], "ready pose")
    start_offset = np.asarray([30.0, 90.0, -45.0, 24.0, 6.0, -12.0], dtype=np.float32)

    class ProbeClock:
        value = 0.0

        @classmethod
        def now(cls) -> float:
            return cls.value

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.value += seconds

    class ProbeRobot:
        def __init__(self) -> None:
            self.state = ready_pose + start_offset
            self.sent: list[np.ndarray] = []

        def send_action(self, requested: np.ndarray) -> np.ndarray:
            vector = finite_joint_vector(requested, "fake interpolated ready action")
            self.sent.append(vector.copy())
            self.state = vector.copy()
            return vector.copy()

    class ProbeEngine:
        READY_POSE = ready_pose
        READY_MOVE_TOLERANCE = EXPECTED_READY_MOVE_TOLERANCE
        READY_MOVE_TIMEOUT_SECONDS = 60.0
        precise_sleep = ProbeClock.sleep

        @staticmethod
        def read_present_position(robot: ProbeRobot) -> np.ndarray:
            return robot.state.copy()

        @staticmethod
        def action_dict(requested: np.ndarray) -> np.ndarray:
            return requested.copy()

        @staticmethod
        def sent_action_vector(sent: np.ndarray) -> np.ndarray:
            return finite_joint_vector(sent, "fake official sent action")

    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    robot = ProbeRobot()
    trajectory = io.StringIO()
    move = build_interpolated_ready_move(ProbeEngine, profile)
    result = move(
        robot,
        trajectory,
        now_fn=ProbeClock.now,
        sleep_fn=ProbeClock.sleep,
    )
    records = [json.loads(line) for line in trajectory.getvalue().splitlines()]
    requested = np.asarray([row["requested_action"] for row in records], dtype=np.float64)
    requested_step_delta = np.asarray(
        [row["requested_step_delta"] for row in records],
        dtype=np.float64,
    )
    return {
        "fake_hardware_only": True,
        "real_device_accessed": False,
        "result": result,
        "commands_sent": len(robot.sent),
        "trajectory_rows": len(records),
        "all_interpolation_phase": all(row["trajectory_phase"] == "linear_interpolation" for row in records),
        "first_alpha": records[0]["interpolation_alpha"],
        "last_alpha": records[-1]["interpolation_alpha"],
        "requested_start_state": (ready_pose + start_offset).tolist(),
        "requested_final_state": requested[-1].tolist(),
        "maximum_requested_step_degrees": float(np.max(np.abs(requested_step_delta))),
        "expected_maximum_requested_step_degrees": float(
            np.max(np.abs(start_offset.astype(np.float64))) / READY_INTERPOLATION_STEPS
        ),
        "all_official_sent_equals_requested": all(
            not any(row["upstream_action_modified_mask"]) for row in records
        ),
        "elapsed_seconds": result["elapsed_seconds"],
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
    ready_return = run_fake_interpolated_ready_probe(plan)
    if fake["trials_exercised"] != 96:
        raise RuntimeError("Fake protocol did not exercise all 96 paired trials.")
    if fake["policy_reset_calls"] != {MODEL_IDS[0]: 48, MODEL_IDS[1]: 48}:
        raise RuntimeError("Each fixed policy must be reset once per paired pose.")
    checks = (
        fake["all_ready_before_policy"],
        fake["all_ready_after_trial"],
        fake["all_canonical_rgb_640x480"],
        fake["all_pre_action_frames_before_policy_send"],
        fake["all_official_sent_equals_requested"],
        fake["all_torque_disabled"],
        fake["success_contract_probe"]["scored_success"],
        fake["success_contract_probe"]["policy_window_unchanged"],
        ready_return["commands_sent"] == READY_INTERPOLATION_STEPS,
        ready_return["trajectory_rows"] == READY_INTERPOLATION_STEPS,
        ready_return["all_interpolation_phase"],
        ready_return["first_alpha"] == 1.0 / READY_INTERPOLATION_STEPS,
        ready_return["last_alpha"] == 1.0,
        ready_return["requested_final_state"] == plan["setup"]["ready_pose_state"],
        ready_return["all_official_sent_equals_requested"],
        abs(
            ready_return["maximum_requested_step_degrees"]
            - ready_return["expected_maximum_requested_step_degrees"]
        )
        <= 1.0e-5,
    )
    if not all(checks):
        raise RuntimeError("Fake paired Eval48 protocol verification failed.")
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
        "fake_interpolated_ready_return": ready_return,
        "next_gate": "Stop before the user turns on Follower 12 V.",
    }


def write_software_evidence(plan: dict, dry_run: dict) -> dict:
    evidence_root = Path(plan["evidence_root"])
    software_root = evidence_root / "software_preparation_v1"
    software_root.mkdir(parents=True, exist_ok=True)
    plan_copy = software_root / "evaluation_plan.json"
    research_copy = software_root / "research_identity_verification.json"
    contract_copy = software_root / "source_research_contract.json"
    training_copy = software_root / "source_training_result.json"
    poses_copy = software_root / "source_pose_manifest.json"
    source_plan_copy = software_root / "source_eval48_plan.json"
    profile_copy = software_root / "real_evaluation_profile.json"
    dry_run_path = software_root / "dry_run.json"
    manifest_path = software_root / "manifest.json"
    hashes_path = software_root / "hashes.sha256"
    offline_smokes = {
        model_id: software_root / f"offline_inference_{model_id}.json" for model_id in MODEL_IDS
    }
    for model_id, path in offline_smokes.items():
        if not path.exists():
            raise RuntimeError(f"Missing required CUDA offline inference smoke: {model_id}")
        smoke = json.loads(path.read_text(encoding="utf-8"))
        if (
            smoke.get("status") != "pass"
            or smoke.get("output_shape") != [1, 6]
            or smoke.get("output_finite") is not True
        ):
            raise RuntimeError(f"Invalid CUDA offline inference smoke: {model_id}")
        if smoke.get("model_sha256") != plan["models"][model_id]["model_sha256"]:
            raise RuntimeError(f"Offline inference checkpoint mismatch: {model_id}")
    for path in (
        plan_copy,
        research_copy,
        contract_copy,
        training_copy,
        poses_copy,
        source_plan_copy,
        profile_copy,
        dry_run_path,
        manifest_path,
        hashes_path,
    ):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite frozen evidence: {path}")
    shutil.copyfile(DEFAULT_PLAN, plan_copy)
    shutil.copyfile(RESEARCH_IDENTITY_VERIFICATION, research_copy)
    shutil.copyfile(SOURCE_RESEARCH_CONTRACT, contract_copy)
    shutil.copyfile(SOURCE_TRAINING_RESULT, training_copy)
    shutil.copyfile(SOURCE_POSE_MANIFEST, poses_copy)
    shutil.copyfile(SOURCE_EVAL48_PLAN, source_plan_copy)
    shutil.copyfile(resolve_repo_path(plan["evaluation_profile"]["path"]), profile_copy)
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
        "source_snapshots": {
            "experiment_design": {"path": str(contract_copy), "sha256": sha256_file(contract_copy)},
            "training_result": {"path": str(training_copy), "sha256": sha256_file(training_copy)},
            "pose_manifest": {"path": str(poses_copy), "sha256": sha256_file(poses_copy)},
            "source_eval48_plan": {"path": str(source_plan_copy), "sha256": sha256_file(source_plan_copy)},
            "evaluation_profile": {"path": str(profile_copy), "sha256": sha256_file(profile_copy)},
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
        "offline_inference_smokes": {
            model_id: {"path": str(path), "sha256": sha256_file(path)}
            for model_id, path in offline_smokes.items()
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
        (sha256_file(contract_copy), contract_copy),
        (sha256_file(training_copy), training_copy),
        (sha256_file(poses_copy), poses_copy),
        (sha256_file(source_plan_copy), source_plan_copy),
        (sha256_file(profile_copy), profile_copy),
        (sha256_file(dry_run_path), dry_run_path),
        (sha256_file(manifest_path), manifest_path),
    ]
    hash_rows.extend((sha256_file(path), path) for path in offline_smokes.values())
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
    return Path(plan["evidence_root"]) / "trials" / f"{trial['artifact_stem']}.json"


def infrastructure_invalid_marker_path(plan: dict, trial: dict) -> Path:
    return Path(plan["evidence_root"]) / "trials" / f"{trial['artifact_stem']}.infrastructure_invalid.json"


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
            raise RuntimeError("All 96 frozen original paired trials already have evidence.")
        if first_missing["trial_id"] != trial["trial_id"]:
            raise RuntimeError(
                "Requested trial is not the next missing trial in frozen order: "
                f"expected {first_missing['trial_id']}."
            )
        return trial["artifact_stem"], None
    if not original_path.exists():
        raise RuntimeError("Replacement requires preserved original evidence.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    engine_infrastructure_invalid = (
        original.get("status") == "aborted_with_error"
        or original.get("termination") == "hardware_or_runtime_error"
    )
    marker_path = infrastructure_invalid_marker_path(plan, trial)
    marker_invalid = False
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_invalid = (
            marker.get("status") == "infrastructure_invalid"
            and marker.get("reason") in plan["replacement_contract"]["allowed_only_for"]
            and marker.get("scored_trial") is False
            and marker.get("replacement_allowed") is True
            and marker.get("trial_id") == trial["trial_id"]
            and marker.get("artifact_stem") == trial["artifact_stem"]
            and marker.get("original_evidence", {}).get("sha256") == sha256_file(original_path)
        )
        if not marker_invalid:
            raise RuntimeError(
                "Infrastructure-invalid marker is malformed or does not bind original evidence."
            )
    infrastructure_invalid = engine_infrastructure_invalid or marker_invalid
    if not infrastructure_invalid:
        raise RuntimeError("Replacement is allowed only for infrastructure-invalid trials.")
    replacement_stem = f"{trial['artifact_stem']}__replacement1"
    if (trials_root / f"{replacement_stem}.json").exists():
        raise RuntimeError("The one allowed linked replacement already exists.")
    return replacement_stem, trial["artifact_stem"]


def build_interpolated_ready_move(engine, profile: dict):
    movement = profile["ready_pose_movement"]
    duration = float(movement["interpolation_duration_seconds"])
    control_fps = int(movement["control_fps"])
    interpolation_steps = int(movement["interpolation_steps"])
    if duration != READY_INTERPOLATION_DURATION_SECONDS:
        raise RuntimeError("Interpolated ready/return duration differs from the frozen profile.")
    if control_fps != READY_INTERPOLATION_CONTROL_FPS:
        raise RuntimeError("Interpolated ready/return control rate differs from the frozen profile.")
    if interpolation_steps != READY_INTERPOLATION_STEPS:
        raise RuntimeError("Interpolated ready/return step count differs from the frozen profile.")
    if interpolation_steps != round(duration * control_fps):
        raise RuntimeError("Interpolated ready/return duration, rate, and step count are inconsistent.")

    def move_to_frozen_ready_pose_interpolated(
        robot,
        trajectory_handle,
        *,
        now_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> dict:
        if sleep_fn is None:
            sleep_fn = engine.precise_sleep
        started = now_fn()
        period = 1.0 / control_fps
        start_state = finite_joint_vector(
            engine.read_present_position(robot),
            "ready movement start state",
        )
        target = finite_joint_vector(engine.READY_POSE, "frozen ready target")
        initial_delta = target.astype(np.float64) - start_state.astype(np.float64)
        initial_maximum_error = float(np.max(np.abs(initial_delta)))
        if initial_maximum_error <= engine.READY_MOVE_TOLERANCE:
            return {
                "status": "ready_pose_observed",
                "profile_id": movement["profile_id"],
                "requested_state": target.tolist(),
                "observed_state": start_state.tolist(),
                "delta_requested_minus_observed": initial_delta.tolist(),
                "maximum_absolute_error": initial_maximum_error,
                "steps": 0,
                "elapsed_seconds": now_fn() - started,
                "trajectory": "time_parameterized_linear_interpolation",
                "planned_interpolation_duration_seconds": duration,
                "planned_interpolation_steps": interpolation_steps,
                "interpolation_steps_executed": 0,
                "settle_steps": 0,
            }

        steps = 0
        previous_requested = start_state.copy()
        for interpolation_index in range(interpolation_steps):
            tick_started = now_fn()
            if tick_started - started >= engine.READY_MOVE_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "Frozen ready pose was not reached within the existing "
                    f"{engine.READY_MOVE_TIMEOUT_SECONDS}-second movement timeout."
                )
            observed = finite_joint_vector(
                engine.read_present_position(robot),
                "ready movement observed state",
            )
            alpha = float(interpolation_index + 1) / float(interpolation_steps)
            requested = (
                start_state.astype(np.float64)
                + alpha * (target.astype(np.float64) - start_state.astype(np.float64))
            ).astype(np.float32)
            sent = engine.sent_action_vector(robot.send_action(engine.action_dict(requested)))
            delta = target.astype(np.float64) - observed.astype(np.float64)
            upstream_modified_mask = ~np.isclose(
                sent,
                requested,
                rtol=0.0,
                atol=1.0e-6,
            )
            compute_seconds = now_fn() - tick_started
            scheduled_sleep_seconds = max(0.0, period - compute_seconds)
            record = {
                "movement_step": steps,
                "trajectory_phase": "linear_interpolation",
                "interpolation_alpha": alpha,
                "planned_interpolation_steps": interpolation_steps,
                "start_state": start_state.tolist(),
                "observed_state": observed.tolist(),
                "requested_action": requested.tolist(),
                "requested_step_delta": (
                    requested.astype(np.float64) - previous_requested.astype(np.float64)
                ).tolist(),
                "sent_action": sent.tolist(),
                "upstream_action_modified_mask": upstream_modified_mask.tolist(),
                "delta_ready_target_minus_observed": delta.tolist(),
                "maximum_absolute_error_to_ready_before_step": float(np.max(np.abs(delta))),
                "loop_compute_seconds": compute_seconds,
                "scheduled_sleep_seconds": scheduled_sleep_seconds,
            }
            trajectory_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            previous_requested = requested.copy()
            steps += 1
            sleep_fn(scheduled_sleep_seconds)

        settle_steps = 0
        while True:
            tick_started = now_fn()
            observed = finite_joint_vector(
                engine.read_present_position(robot),
                "ready movement observed state",
            )
            delta = target.astype(np.float64) - observed.astype(np.float64)
            maximum_error = float(np.max(np.abs(delta)))
            if maximum_error <= engine.READY_MOVE_TOLERANCE:
                return {
                    "status": "ready_pose_observed",
                    "profile_id": movement["profile_id"],
                    "requested_state": target.tolist(),
                    "observed_state": observed.tolist(),
                    "delta_requested_minus_observed": delta.tolist(),
                    "maximum_absolute_error": maximum_error,
                    "steps": steps,
                    "elapsed_seconds": now_fn() - started,
                    "trajectory": "time_parameterized_linear_interpolation",
                    "planned_interpolation_duration_seconds": duration,
                    "planned_interpolation_steps": interpolation_steps,
                    "interpolation_steps_executed": interpolation_steps,
                    "settle_steps": settle_steps,
                }
            if now_fn() - started >= engine.READY_MOVE_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "Frozen ready pose was not reached within the existing "
                    f"{engine.READY_MOVE_TIMEOUT_SECONDS}-second movement timeout."
                )
            sent = engine.sent_action_vector(robot.send_action(engine.action_dict(target)))
            upstream_modified_mask = ~np.isclose(
                sent,
                target,
                rtol=0.0,
                atol=1.0e-6,
            )
            compute_seconds = now_fn() - tick_started
            scheduled_sleep_seconds = max(0.0, period - compute_seconds)
            record = {
                "movement_step": steps,
                "trajectory_phase": "hold_target_until_tolerance",
                "interpolation_alpha": 1.0,
                "planned_interpolation_steps": interpolation_steps,
                "start_state": start_state.tolist(),
                "observed_state": observed.tolist(),
                "requested_action": target.tolist(),
                "requested_step_delta": (
                    target.astype(np.float64) - previous_requested.astype(np.float64)
                ).tolist(),
                "sent_action": sent.tolist(),
                "upstream_action_modified_mask": upstream_modified_mask.tolist(),
                "delta_ready_target_minus_observed": delta.tolist(),
                "maximum_absolute_error_to_ready_before_step": maximum_error,
                "loop_compute_seconds": compute_seconds,
                "scheduled_sleep_seconds": scheduled_sleep_seconds,
            }
            trajectory_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            previous_requested = target.copy()
            steps += 1
            settle_steps += 1
            sleep_fn(scheduled_sleep_seconds)

    return move_to_frozen_ready_pose_interpolated


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
        "cell": trial["cell"],
        "coverage_tier": trial["coverage_tier"],
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


def write_eval48_sidecar(
    plan: dict,
    trial: dict,
    artifact_stem: str,
    replacement_for: str | None,
) -> None:
    trials_root = Path(plan["evidence_root"]) / "trials"
    engine_evidence_path = trials_root / f"{artifact_stem}.json"
    if not engine_evidence_path.exists():
        return
    sidecar_path = trials_root / f"{artifact_stem}.paired_eval48.json"
    if sidecar_path.exists():
        raise RuntimeError(f"Refusing to overwrite paired Eval48 sidecar: {sidecar_path}")
    evidence = json.loads(engine_evidence_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(evidence["started_at_utc"])
    ended = datetime.fromisoformat(evidence["ended_at_utc"])
    pre_action = build_pre_action_evidence(evidence, trial, trials_root, artifact_stem)
    sidecar = {
        "schema": "task1_picklift_real24_localsim_gap_full_eval48_trial_sidecar_v1",
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
        "adjudication": {"status": "not_required_unless_labels_disagree"},
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
    model = plan["models"][trial["model_key"]]
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
    engine.move_to_frozen_ready_pose = build_interpolated_ready_move(
        engine,
        profile,
    )
    engine_args = argparse.Namespace(
        execute_hardware=True,
        operator_confirmed_ready=True,
        spawn_region=trial["artifact_stem"],
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
            "paired_eval48_evaluation_id": plan["evaluation_id"],
            "paired_eval48_plan_sha256": EXPECTED_PLAN_SHA256,
            "paired_eval48_trial": trial,
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
        write_eval48_sidecar(
            plan,
            trial,
            artifact_stem,
            replacement_for,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Task1 Real24+LocalSim gap24 versus full48 paired Eval48. "
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
