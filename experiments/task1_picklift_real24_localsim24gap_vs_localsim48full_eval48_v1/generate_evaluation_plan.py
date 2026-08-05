from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SOURCE_AUTHORIZATION = ROOT / "source_authorization.json"
SOURCE_TRAINING_RESULT = ROOT / "source_training_result.json"
SOURCE_POSES = ROOT / "source_pose_manifest.json"
SOURCE_EVAL48_PLAN = REPO_ROOT / "experiments/task1_picklift_real48_vs_real96_eval48_v1/evaluation_plan.json"
PROFILE = REPO_ROOT / "experiments/task1_picklift_real48_vs_real96_eval48_v1/real_evaluation_profile.json"
OUTPUT = ROOT / "evaluation_plan.json"

EVALUATION_ID = "task1_picklift_real24_localsim24gap_vs_localsim48full_eval48_v1"
RESEARCH_COMMIT = "340facbbcf7b8eb60a062e8ec54d64b96ce0ba86"
AUTHORIZATION_SHA256 = "696980b1a78d5f2d2ee71c96a72e7ede23a34b51fddde93caabfa04767394342"
TRAINING_RESULT_SHA256 = "a2a320bcfbc3ff6bfcbf2000a17ca15804e1a7b00e9477e9f16a1577b82b2477"
SOURCE_POSES_SHA256 = "f6bc79e9b99818f12f0e6a374688850374ea6f5cb971ba5da7ef3f32ae8322e7"
SOURCE_EVAL48_PLAN_SHA256 = "7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1"
ENGINE_SHA256 = "380b8c1c13f0f38a59e129b78d845a1cbd8916411af1f61a56b9267e83205f96"
PROFILE_SHA256 = "6b031bb4c980467addb3e69d68a16032ceae7e45fb3f8e2288d8a4989ff3cbf3"

MODEL_C = "real24_localsim24_gap"
MODEL_D = "real24_localsim48_full"
MODELS = {
    MODEL_C: {
        "model_id": "ACT_Real24_LocalSim24Gap_seed1000_step100000",
        "checkpoint": (
            "/home/ubuntu24/Teleop/artifacts/training/"
            "task1_picklift_real24_localsim48_gap_recovery_act_v1/"
            "real24_sim24_gap/full_100k/checkpoints/100000/pretrained_model"
        ),
        "model_sha256": ("e9bbbc96e3104435d670e450090ab143610e2cdba8d38485beec339d5230577c"),
        "config_sha256": ("58b1d128ac6ac3b81729ed2ed1a8f10f93f74203fae443cff9dcef5ac641ea8e"),
        "train_config_sha256": ("c5b257c118f8604307fc6d459e0eef9ff5a22e6846c66f3a366bd423cfd20849"),
        "policy_preprocessor_sha256": ("efa0df0e288722aa67eb13ae6198c80f897032524026b06aef162d3bbd007219"),
        "processor_stats_sha256": ("6f8cfd553d02e759515f92c0d4e40046f0391198c28f31f6dfa1c438e8a3c44c"),
        "training_dataset": "Real24 + LocalSim gap24",
    },
    MODEL_D: {
        "model_id": "ACT_Real24_LocalSim48Full_seed1000_step100000",
        "checkpoint": (
            "/home/ubuntu24/Teleop/artifacts/training/"
            "task1_picklift_real24_localsim48_gap_recovery_act_v1/"
            "real24_sim48_full/full_100k/checkpoints/100000/pretrained_model"
        ),
        "model_sha256": ("735af1dc914c1ea5b82fada65a3c72439cb5603ac8731425f43819f849972c0e"),
        "config_sha256": ("58b1d128ac6ac3b81729ed2ed1a8f10f93f74203fae443cff9dcef5ac641ea8e"),
        "train_config_sha256": ("f6b93d898d48351c7908ada4938b82113224483fb92e2476c1ccaab56fa38547"),
        "policy_preprocessor_sha256": ("efa0df0e288722aa67eb13ae6198c80f897032524026b06aef162d3bbd007219"),
        "processor_stats_sha256": ("52611ce7bf838f15ae4737a4bc422321510898055382b8f0beb71690a78e1975"),
        "training_dataset": "Real24 + LocalSim full48",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def placement_prompt(pose: dict, *, keep_existing: bool) -> str:
    prefix = f"保持第{pose['order']}个冻结姿态不动" if keep_existing else f"摆放第{pose['order']}个冻结姿态"
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
    short_model = "localsim24gap" if model_key == MODEL_C else "localsim48full"
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
        "operator_placement_prompt_zh": placement_prompt(pose, keep_existing=within_pair == 2),
        "manual_pose_is_measurement_truth": False,
        "policy_failure_retry_allowed": False,
    }


def verify_source_pose_order(poses: list[dict]) -> None:
    source_plan = json.loads(SOURCE_EVAL48_PLAN.read_text(encoding="utf-8"))
    source_trials = source_plan["trials"][::2]
    fields = {
        "eval_pose_id": "eval_pose_id",
        "order_sha256": "source_order_sha256",
        "cell": "cell",
        "x_forward_m": "nominal_x_forward_m",
        "y_lateral_m": "nominal_y_lateral_m",
        "yaw_degrees_modulo_90": "nominal_yaw_degrees_modulo_90",
    }
    if len(source_trials) != len(poses):
        raise RuntimeError("Source Eval48 plan does not contain 48 paired poses")
    for pose, trial in zip(poses, source_trials, strict=True):
        if pose["order"] != trial["pose_order"] or any(
            pose[pose_field] != trial[trial_field] for pose_field, trial_field in fields.items()
        ):
            raise RuntimeError(f"Source Eval48 pose mismatch at order {pose['order']}")


def build_plan() -> dict:
    expected_files = {
        SOURCE_AUTHORIZATION: AUTHORIZATION_SHA256,
        SOURCE_TRAINING_RESULT: TRAINING_RESULT_SHA256,
        SOURCE_POSES: SOURCE_POSES_SHA256,
        SOURCE_EVAL48_PLAN: SOURCE_EVAL48_PLAN_SHA256,
        PROFILE: PROFILE_SHA256,
    }
    for path, expected in expected_files.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
    authorization = json.loads(SOURCE_AUTHORIZATION.read_text(encoding="utf-8"))
    training_result = json.loads(SOURCE_TRAINING_RESULT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_POSES.read_text(encoding="utf-8"))
    if authorization["authorization_id"] != (
        "GO_TASK1_REAL24_LOCALSIM24GAP_VS_LOCALSIM48FULL_EVAL48_SOFTWARE_GATE_V1"
    ):
        raise RuntimeError("Unexpected research authorization")
    if training_result["status"] != ("offline_training_complete_ready_for_same_eval48_software_gate"):
        raise RuntimeError("Training result is not ready for this software gate")
    poses = source["ordered_eval_poses"]
    if len(poses) != 48 or [pose["order"] for pose in poses] != list(range(1, 49)):
        raise RuntimeError("Frozen source must contain ordered poses 1..48")
    verify_source_pose_order(poses)
    trials: list[dict] = []
    for pose in poses:
        first, second = (MODEL_C, MODEL_D) if pose["order"] % 2 == 1 else (MODEL_D, MODEL_C)
        trials.append(build_trial(pose, first, len(trials) + 1, 1))
        trials.append(build_trial(pose, second, len(trials) + 1, 2))
    return {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "status": "software_gate_frozen_hardware_not_authorized",
        "purpose": (
            "Matched real Eval48 comparison of Real24+LocalSim gap24 versus "
            "Real24+LocalSim full48 ACT checkpoints."
        ),
        "comparison_role": "Task1 LocalSim simulation-data composition gate",
        "research_contract": {
            "research_repo_commit": RESEARCH_COMMIT,
            "experiment_design": {
                "path": (
                    "manifests/task1-picklift-real24-localsim24gap-vs-"
                    "localsim48full-eval48-software-authorization-v1.json"
                ),
                "repo_snapshot": "source_authorization.json",
                "sha256": AUTHORIZATION_SHA256,
            },
            "training_result": {
                "path": (
                    "manifests/task1-picklift-real24-localsim24gap-localsim48full-act-training-result-v1.json"
                ),
                "repo_snapshot": "source_training_result.json",
                "sha256": TRAINING_RESULT_SHA256,
            },
            "pose_manifest": {
                "path": "manifests/task1-picklift-real48-vs-real96-eval48-poses-v1.json",
                "repo_snapshot": "source_pose_manifest.json",
                "sha256": SOURCE_POSES_SHA256,
            },
            "source_eval48_plan": {
                "path": str(SOURCE_EVAL48_PLAN),
                "sha256": SOURCE_EVAL48_PLAN_SHA256,
                "pose_order_reselected": False,
            },
        },
        "execution_engine": {
            "owning_commit": "34cc7ac5e43d029a81f5089d76a811e5a59014b1",
            "path": "experiments/task1_picklift_real24_act_v1/evaluate_real.py",
            "source_sha256": ENGINE_SHA256,
        },
        "evaluation_profile": {
            "profile_id": ("task1_real48_vs_real96_eval48_official_send_interpolated3s_tolerance3_v1"),
            "path": ("experiments/task1_picklift_real48_vs_real96_eval48_v1/real_evaluation_profile.json"),
            "sha256": PROFILE_SHA256,
        },
        "evidence_root": (f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}"),
        "models": MODELS,
        "setup": {
            "task_id": "PickLift-Nexus-v1",
            "task_frame_id": "picklift_task_grid",
            "camera_profile_id": "icspring_front_crop_1280x960_to_640x480_v1",
            "camera_device_by_id": ("/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0"),
            "canonical_rgb_width": 640,
            "canonical_rgb_height": 480,
            "follower_port_by_id": ("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00"),
            "follower_calibration_path": (
                "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/"
                "so_follower/so101_follower_main.json"
            ),
            "follower_calibration_sha256": (
                "c78e4f7e1383571c6aa496f62996f518b3e4122f78244d2bbc094658bc0cb8a0"
            ),
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
            "ready_pose_state": [
                7.4285712242126465,
                -98.32967376708984,
                45.010990142822266,
                92.21977996826172,
                1.8461538553237915,
                19.765840530395508,
            ],
            "ready_pose_state_sha256": ("ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"),
            "ready_pose_arrival_tolerance_degrees": 3.0,
            "ready_pose_before_every_trial": True,
            "ready_pose_after_every_trial": True,
            "policy_reset_after_ready_pose": True,
            "ready_pose_movement_profile_id": (
                "task1_real24_ready_pose_interpolated3s_official_send_tolerance3_v1"
            ),
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
            "pre_action_evidence": ("canonical 640x480 frame 0 before tick0 inference and action send"),
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
            "failure_categories": [
                "missed_grasp",
                "spatial_offset",
                "post_grasp_drop",
                "other",
            ],
        },
        "replacement_contract": {
            "model_or_task_failure_retry_allowed": False,
            "maximum_linked_replacements_per_original": 1,
            "allowed_only_for": [
                "policy_window_never_started",
                "confirmed_operator_placement_error",
                "infrastructure_error",
            ],
            "original_evidence_preserved": True,
        },
        "evidence_contract": {
            "per_trial": [
                "trial/model/pose/profile identities",
                "pre-action canonical frame and nominal pose",
                "requested/observed ready and delta",
                "policy reset and tick0",
                "canonical video",
                "per-tick observation/raw/requested/official sent/upstream modification",
                "wall duration and actual ticks",
                "operator label",
                "canonical-video review label",
                "immutable adjudication if needed",
                "return trajectory",
                "torque disable verification",
            ],
            "reporting": [
                "overall by model",
                "coverage tiers 24/18/6",
                "cell",
                "yaw",
                "failure categories",
                "paired LocalSim48-full minus LocalSim24-gap success difference",
            ],
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
        "balance_invariants": {
            **source["balance_invariants"],
            "first_model_counts": {MODEL_C: 24, MODEL_D: 24},
        },
        "trials": trials,
        "next_hardware_gate": ("Stop before Follower 12 V; wait for separate research-control hardware GO."),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen plan: {OUTPUT}")
    plan = build_plan()
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT), "trials": 96},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
