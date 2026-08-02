from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from examples.picklift_v3.batch_record import BATCH_WORKFLOW_VERSION, validate_batch_config
from examples.picklift_v3.real96_plan import (
    COLLECTION_PLAN_ID,
    COLLECTION_PLAN_SHA256,
    POSE_MANIFEST_ID,
    POSE_MANIFEST_SHA256,
    RESEARCH_CONTRACT_COMMIT,
    RESEARCH_CONTRACT_PARENT,
    SESSION_SEQUENCE_SHA256,
    SUBSET_MANIFEST_ID,
    SUBSET_MANIFEST_SHA256,
    batch_spawns,
)
from examples.picklift_v3.record import (
    FPS,
    REAL96_COLLECTION_PROTOCOL_VERSION,
    REAL96_SPAWN_PROTOCOL_VERSION,
    REAL_ACK,
)

READY_POSE_PROFILE = "task1_real24_ready_pose_reset_v1"
READY_POSE_STATE_SHA256 = "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
CAMERA_PROFILE = "icspring_front_crop_1280x960_to_640x480_v1"
COMPLETED_SESSION_FREEZES = {
    1: {
        "session_id": "task1_real96_s01",
        "freeze_id": "task1_picklift_real96_s01_raw_attempts_freeze_v1",
        "raw_tree_sha256": "f628ab551aadea2c40402a48b7bda9625342d2b7921e960035e9620f2970fd2a",
        "accepted_success_episodes": 24,
    },
    2: {
        "session_id": "task1_real96_s02",
        "freeze_id": "task1_picklift_real96_s02_raw_attempts_freeze_v1",
        "raw_tree_sha256": "ed02dda9b55400b3a4be6a837c98fff3db83ff4c7ffd5de2466c64eb39773f4b",
        "accepted_success_episodes": 24,
    },
}


def make_config(
    *,
    session_index: int,
    operator_id: str,
    dataset_root: Path,
    device_config: dict,
    powered_ack: str = "",
) -> dict:
    session_id = f"task1_real96_s{session_index:02d}"
    required_device_fields = (
        "camera_device",
        "robot_id",
        "follower_port",
        "leader_id",
        "leader_port",
    )
    missing = [key for key in required_device_fields if not device_config.get(key)]
    if missing:
        raise ValueError(f"device config missing: {', '.join(missing)}")
    if session_index not in SESSION_SEQUENCE_SHA256:
        raise ValueError("only independently transferred sessions are deployable in this checkout")
    prior_sessions = [COMPLETED_SESSION_FREEZES[index] for index in range(1, session_index)]
    immediate_predecessor = prior_sessions[-1] if prior_sessions else None
    return {
        "collection_workflow_version": BATCH_WORKFLOW_VERSION,
        "successes_per_spawn": 1,
        "max_attempts": 72,
        "base_config": {
            "mode": "real",
            "dataset_root": str(dataset_root.resolve()),
            "repo_id": f"local/{session_id}_raw_attempts",
            "operator_id": operator_id,
            "session_id": session_id,
            "task_id": "PickLift-Nexus-v1",
            "task_version": "1",
            "task": "Pick up the red cube.",
            "task_spec_revision": "task1_picklift_final_v2",
            "task_frame_id": "picklift_task_grid_v2",
            "alignment_reference_id": "picklift_red_cube_alignment_v2",
            "real_world_setup_version": (
                "picklift_real_setup_spawn_v4_5cm_grid_camera_aligned_task_grid_v2_reference_v2"
            ),
            "camera_config_version": CAMERA_PROFILE,
            "camera_profile_id": CAMERA_PROFILE,
            "camera_device": device_config["camera_device"],
            "camera_intrinsics_version": "uncalibrated_front_v1",
            "camera_extrinsics_version": "uncalibrated_front_mount_v1",
            "robot_id": device_config["robot_id"],
            "robot_calibration_id": device_config["robot_id"],
            "follower_serial_id": "1a86_USB_Single_Serial_5C82110904",
            "follower_port": device_config["follower_port"],
            "leader_id": device_config["leader_id"],
            "leader_calibration_id": device_config["leader_id"],
            "leader_serial_id": "1a86_USB_Single_Serial_5C82107516",
            "leader_port": device_config["leader_port"],
            "collection_protocol_version": REAL96_COLLECTION_PROTOCOL_VERSION,
            "spawn_protocol_version": REAL96_SPAWN_PROTOCOL_VERSION,
            "yaw_annotation_mode": "predeclared_nominal_0_or_45",
            "yaw_intended_range_deg": [0, 45],
            "yaw_sampling_method": "frozen_real96_pose_manifest_v1",
            "yaw_distribution_claim": "balanced_predeclared_nominal",
            "yaw_randomization_confirmed": False,
            "success_annotation_source": "operator_visual_v1",
            "success_detection_mode": "manual_proxy_for_nexus_v1",
            "lift_height_m": None,
            "is_grasped": None,
            "result": "pending",
            "formal_data": True,
            "control_hz": 50,
            "camera_acquisition_fps": 30,
            "alignment_mode": "direct_absolute",
            "startup_hold_s": 0,
            "operator_cue_wait": False,
            "operator_ui": True,
            "record_fps": FPS,
            "episode_seconds": 30,
            "success": False,
            "termination_reason": "operator_end_or_max_duration",
            "use_videos": True,
            "max_relative_target": None,
            "powered_real_run_ack": powered_ack,
            "research_contract_commit": RESEARCH_CONTRACT_COMMIT,
            "research_contract_parent": RESEARCH_CONTRACT_PARENT,
            "task_contract_id": "task1_picklift_final_v2",
            "task_contract_version": 2,
            "collection_plan_id": COLLECTION_PLAN_ID,
            "collection_plan_sha256": COLLECTION_PLAN_SHA256,
            "pose_manifest_id": POSE_MANIFEST_ID,
            "pose_manifest_sha256": POSE_MANIFEST_SHA256,
            "subset_manifest_id": SUBSET_MANIFEST_ID,
            "subset_manifest_sha256": SUBSET_MANIFEST_SHA256,
            "session_sequence_sha256": SESSION_SEQUENCE_SHA256[session_index],
            "predecessor_session_id": (
                immediate_predecessor["session_id"] if immediate_predecessor else None
            ),
            "predecessor_freeze_id": (immediate_predecessor["freeze_id"] if immediate_predecessor else None),
            "predecessor_raw_tree_sha256": (
                immediate_predecessor["raw_tree_sha256"] if immediate_predecessor else None
            ),
            "completed_session_freezes": prior_sessions,
            "cumulative_accepted_before_session": sum(
                item["accepted_success_episodes"] for item in prior_sessions
            ),
            "ready_pose_profile": READY_POSE_PROFILE,
            "ready_pose_state_sha256": READY_POSE_STATE_SHA256,
            "follower_calibration_path": device_config["follower_calibration_path"],
            "follower_calibration_sha256": device_config["follower_calibration_sha256"],
            "leader_calibration_path": device_config["leader_calibration_path"],
            "leader_calibration_sha256": device_config["leader_calibration_sha256"],
        },
        "spawns": batch_spawns(session_index),
    }


def validate_without_hardware(cfg: dict) -> None:
    validation_copy = json.loads(json.dumps(cfg))
    validation_copy["base_config"]["powered_real_run_ack"] = REAL_ACK
    validate_batch_config(validation_copy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--operator-id", default="operator_01")
    parser.add_argument("--device-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device_config = json.loads(args.device_config.read_text())
    calibration_root = Path.home() / ".cache/huggingface/lerobot/calibration"
    calibration_paths = {
        "follower_calibration_path": (
            calibration_root / f"robots/so_follower/{device_config['robot_id']}.json"
        ),
        "leader_calibration_path": (
            calibration_root / f"teleoperators/so_leader/{device_config['leader_id']}.json"
        ),
    }
    for key, path in calibration_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing calibration profile: {path}")
        device_config[key] = str(path.resolve())
        device_config[key.replace("_path", "_sha256")] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = make_config(
        session_index=args.session,
        operator_id=args.operator_id,
        dataset_root=args.dataset_root,
        device_config=device_config,
    )
    validate_without_hardware(cfg)
    if args.dataset_root.exists():
        raise FileExistsError(f"new raw-attempt dataset root already exists: {args.dataset_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"prepared {args.output}; powered acknowledgement remains empty; no devices opened")


if __name__ == "__main__":
    main()
