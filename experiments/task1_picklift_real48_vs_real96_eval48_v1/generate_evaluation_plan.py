from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_CONTRACT = ROOT / "source_research_contract.json"
SOURCE_POSES = ROOT / "source_pose_manifest.json"
PROFILE = ROOT / "real_evaluation_profile.json"
OUTPUT = ROOT / "evaluation_plan.json"

EVALUATION_ID = "task1_picklift_real48_vs_real96_eval48_v1"
RESEARCH_COMMIT = "73908355df1add52cd04753216c13f8b1c0b400a"
SOURCE_CONTRACT_SHA256 = "46b2ec8335d9b4415967efdd70dbe854d113d2966307a0eaaf38278744650af1"
SOURCE_POSES_SHA256 = "f6bc79e9b99818f12f0e6a374688850374ea6f5cb971ba5da7ef3f32ae8322e7"
ENGINE_SHA256 = "380b8c1c13f0f38a59e129b78d845a1cbd8916411af1f61a56b9267e83205f96"
PROFILE_SHA256 = "6b031bb4c980467addb3e69d68a16032ceae7e45fb3f8e2288d8a4989ff3cbf3"

MODEL_KEYS = {
    "ACT_Real48_seed1000_step100000": "real48",
    "ACT_Real96_seed1000_step100000": "real96",
}
MODELS = {
    "real48": {
        "model_id": "ACT_Real48_seed1000_step100000",
        "checkpoint": "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real48_vs_real96_act_v1/real48/full_100k/checkpoints/100000/pretrained_model",
        "model_sha256": "73f61996c0ebba444c1bce070ec36735425f3307e420eab47cf29a3ab7ffa14c",
        "config_sha256": "d2b1f0c4eb93dbb567150ebe7a4fc9636c40b368bdff8366e3079dc48277fef4",
        "train_config_sha256": "c91e79a19ef9ce4bd2153352d56fad359f6a9f44e60465bc0412a946b73383c7",
        "policy_preprocessor_sha256": "adc8a12dd079a93b4e6fd4e7f15e93126c9927463593a4de3174643e59fca28a",
        "processor_stats_sha256": "062d15518fd1e1c898179c024c9043d19d07da7a7ccf95ddb55e4e572674a906",
        "dataset_root": "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real48_accepted_v1/accepted",
        "dataset_repo_id": "local/task1_picklift_real96_accepted_v1_accepted",
        "offline_smoke_sample_index": 0,
    },
    "real96": {
        "model_id": "ACT_Real96_seed1000_step100000",
        "checkpoint": "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real48_vs_real96_act_v1/real96/full_100k/checkpoints/100000/pretrained_model",
        "model_sha256": "2d80bbddff5c3e5862a6e9f0b639619628fb637b9beea4826a6469e95f851e44",
        "config_sha256": "d2b1f0c4eb93dbb567150ebe7a4fc9636c40b368bdff8366e3079dc48277fef4",
        "train_config_sha256": "547ce0e3d6e29f8f8343714914fa35870ac3069ea0340bcd23a4678f4fef23e8",
        "policy_preprocessor_sha256": "adc8a12dd079a93b4e6fd4e7f15e93126c9927463593a4de3174643e59fca28a",
        "processor_stats_sha256": "7006945a9ba1961e592398d09359e201fbd2cdf13ef863baad1d82ef1eac690a",
        "dataset_root": "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real96_accepted_v1",
        "dataset_repo_id": "local/task1_picklift_real96_accepted_v1",
        "offline_smoke_sample_index": 0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def placement_prompt(pose: dict, *, keep_existing: bool) -> str:
    prefix = (
        f"保持第{pose['order']}个冻结姿态不动"
        if keep_existing
        else f"摆放第{pose['order']}个冻结姿态"
    )
    yaw = (
        "方块边与任务网格线平行（0°）"
        if pose["yaw_degrees_modulo_90"] == 0
        else "方块绕中心旋转45°（边与任务网格线成45°）"
    )
    return (
        f"{prefix}：{pose['cell']}（第{pose['row']}行第{pose['column']}列），"
        f"红块中心放在任务网格 X={pose['x_forward_m'] * 100:g} cm、"
        f"Y={pose['y_lateral_m'] * 100:+g} cm 的交点；{yaw}。"
    )


def build_trial(pose: dict, model_key: str, order: int, within_pair: int) -> dict:
    short_model = "real48" if model_key == "real48" else "real96"
    trial_id = f"t{order:03d}_p{pose['order']:02d}_{short_model}"
    return {
        "order": order,
        "trial_id": trial_id,
        "artifact_stem": trial_id,
        "spawn_region": trial_id,
        "pose_order": pose["order"],
        "within_pair_order": within_pair,
        "eval_pose_id": pose["eval_pose_id"],
        "source_order_sha256": pose["order_sha256"],
        "cell": pose["cell"],
        "row": pose["row"],
        "column": pose["column"],
        "coverage_tier": pose["coverage_tier"],
        "seen_by_real48": pose["seen_by_real48"],
        "seen_by_real96": pose["seen_by_real96"],
        "position_kind": pose["position_kind"],
        "quadrant": pose["quadrant"],
        "nominal_x_forward_m": pose["x_forward_m"],
        "nominal_y_lateral_m": pose["y_lateral_m"],
        "nominal_yaw_degrees_modulo_90": pose["yaw_degrees_modulo_90"],
        "source_collection_plan_item_id": pose["source_collection_plan_item_id"],
        "model_key": model_key,
        "model_id": MODELS[model_key]["model_id"],
        "operator_placement_prompt_zh": placement_prompt(
            pose,
            keep_existing=within_pair == 2,
        ),
        "manual_pose_is_measurement_truth": False,
        "policy_failure_retry_allowed": False,
    }


def build_plan() -> dict:
    if sha256_file(SOURCE_CONTRACT) != SOURCE_CONTRACT_SHA256:
        raise RuntimeError("Research contract source hash mismatch")
    if sha256_file(SOURCE_POSES) != SOURCE_POSES_SHA256:
        raise RuntimeError("Research pose-manifest source hash mismatch")
    if sha256_file(PROFILE) != PROFILE_SHA256:
        raise RuntimeError("Evaluation profile hash mismatch")
    contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_POSES.read_text(encoding="utf-8"))
    if contract["evaluation_id"] != EVALUATION_ID or source["evaluation_id"] != EVALUATION_ID:
        raise RuntimeError("Evaluation identity mismatch")
    poses = source["ordered_eval_poses"]
    if len(poses) != 48 or [pose["order"] for pose in poses] != list(range(1, 49)):
        raise RuntimeError("Frozen source must contain ordered poses 1..48")
    trials: list[dict] = []
    for pose in poses:
        first = MODEL_KEYS[pose["first_model"]]
        second = MODEL_KEYS[pose["second_model"]]
        trials.append(build_trial(pose, first, len(trials) + 1, 1))
        trials.append(build_trial(pose, second, len(trials) + 1, 2))
    return {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "status": "software_gate_frozen_hardware_not_authorized",
        "purpose": contract["purpose"],
        "comparison_role": "Task1 matched Real-data budget paired real evaluation",
        "research_contract": {
            "research_repo_commit": RESEARCH_COMMIT,
            "experiment_design": {
                "path": "experiment-design/task1-picklift-real48-vs-real96-eval48-v1.json",
                "repo_snapshot": "source_research_contract.json",
                "sha256": SOURCE_CONTRACT_SHA256,
            },
            "pose_manifest": {
                "path": "manifests/task1-picklift-real48-vs-real96-eval48-poses-v1.json",
                "repo_snapshot": "source_pose_manifest.json",
                "sha256": SOURCE_POSES_SHA256,
            },
        },
        "execution_engine": {
            "owning_commit": "34cc7ac5e43d029a81f5089d76a811e5a59014b1",
            "path": "experiments/task1_picklift_real24_act_v1/evaluate_real.py",
            "source_sha256": ENGINE_SHA256,
        },
        "evaluation_profile": {
            "profile_id": "task1_real48_vs_real96_eval48_official_send_interpolated3s_tolerance3_v1",
            "path": "experiments/task1_picklift_real48_vs_real96_eval48_v1/real_evaluation_profile.json",
            "sha256": PROFILE_SHA256,
        },
        "evidence_root": "/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real48_vs_real96_eval48_v1",
        "models": MODELS,
        "setup": {
            "task_id": "PickLift-Nexus-v1",
            "task_frame_id": "picklift_task_grid",
            "camera_profile_id": "icspring_front_crop_1280x960_to_640x480_v1",
            "camera_device_by_id": "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
            "canonical_rgb_width": 640,
            "canonical_rgb_height": 480,
            "follower_port_by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00",
            "follower_calibration_path": "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json",
            "follower_calibration_sha256": "c78e4f7e1383571c6aa496f62996f518b3e4122f78244d2bbc094658bc0cb8a0",
            "control_fps": 20,
            "maximum_trial_seconds": 30,
            "nominal_policy_ticks_at_target_rate": 600,
            "stop_on_success": False,
            "max_relative_target": None,
            "custom_absolute_action_clamp": False,
            "custom_relative_step_limit_degrees": None,
            "action_path_profile_id": "lerobot_so101_official_send_no_custom_clamp_v1",
            "policy_timing_profile_id": "lerobot_per_tick_pacing_no_catchup_v1",
            "ready_pose_profile_id": "task1_real24_ready_pose_reset_v1",
            "ready_pose_state": [7.4285712242126465, -98.32967376708984, 45.010990142822266, 92.21977996826172, 1.8461538553237915, 19.765840530395508],
            "ready_pose_state_sha256": "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56",
            "ready_pose_arrival_tolerance_degrees": 3.0,
            "ready_pose_before_every_trial": True,
            "ready_pose_after_every_trial": True,
            "policy_reset_after_ready_pose": True,
            "ready_pose_movement_profile_id": "task1_real24_ready_pose_interpolated3s_official_send_tolerance3_v1",
            "ready_pose_interpolation_duration_seconds": 3.0,
            "ready_pose_interpolation_steps": 60,
            "ready_pose_post_interpolation_hold_until_tolerance": True,
            "act_chunk_size": 67,
            "act_n_action_steps": 67,
        },
        "placement_contract": {
            "task_grid": "existing 10 mm visual grid over the frozen 5 cm cells",
            "operator_numeric_coordinate_instruction": True,
            "manual_pose_is_measurement_truth": False,
            "pre_action_evidence": "canonical 640x480 frame 0 before tick0 inference and action send",
            "yaw_0": "cube edges parallel to task-grid lines",
            "yaw_45": "cube rotated nominally 45 degrees about its center",
        },
        "success_contract": {
            "within_scored_window": True,
            "bilateral_finger_grasp": True,
            "unsupported_lift_strictly_greater_than_m": 0.05,
            "continuous_hold_seconds_minimum": 0.5,
            "must_remain_held_until_timeout": False,
            "changes_policy_action_window": False,
            "operator_label_required": True,
            "canonical_video_review_required": True,
            "immutable_adjudication_only_on_disagreement": True,
            "failure_categories": ["missed_grasp", "spatial_offset", "post_grasp_drop", "other"],
        },
        "replacement_contract": {
            "model_or_task_failure_retry_allowed": False,
            "maximum_linked_replacements_per_original": 1,
            "allowed_only_for": ["policy_window_never_started", "confirmed_operator_placement_error", "infrastructure_error"],
            "original_evidence_preserved": True,
        },
        "evidence_contract": {
            "per_trial": ["trial/model/pose/profile identities", "pre-action canonical frame and nominal pose", "requested/observed ready and delta", "policy reset and tick0", "canonical video", "per-tick observation/raw/requested/official sent/upstream modification", "wall duration and actual ticks", "operator label", "canonical-video review label", "immutable adjudication if needed", "return trajectory", "torque disable verification"],
            "reporting": ["overall by model", "coverage tiers 24/18/6", "cell", "yaw", "failure categories", "paired Real96 minus Real48 success difference"],
        },
        "authorization": {
            "hardware_authorized": False,
            "first_hardware_action_requires_later_explicit_go": True,
            "serial_accessed_during_preparation": False,
            "camera_accessed_during_preparation": False,
            "robot_accessed_during_preparation": False,
            "torque_accessed_during_preparation": False,
            "rollout_executed_during_preparation": False,
        },
        "balance_invariants": source["balance_invariants"],
        "trials": trials,
        "next_hardware_gate": "Stop before Follower 12 V; wait for separate research-control hardware GO.",
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen plan: {OUTPUT}")
    plan = build_plan()
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256_file(OUTPUT), "trials": len(plan["trials"])}, indent=2))


if __name__ == "__main__":
    main()
