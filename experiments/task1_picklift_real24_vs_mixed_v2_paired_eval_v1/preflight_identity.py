from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from paired_evaluator import (
    DEFAULT_CALIBRATION,
    EXPECTED_CAMERA_DEVICE,
    EXPECTED_FOLLOWER_PORT,
    REPO_ROOT,
    load_frozen_plan,
    sha256_file,
    verify_static_files,
)

EXPECTED_CALIBRATION_SHA256 = "c78e4f7e1383571c6aa496f62996f518b3e4122f78244d2bbc094658bc0cb8a0"
CAMERA_PROFILE_SOURCE = REPO_ROOT / "examples/picklift_v3/camera_profile.py"
ENGINE_DEPENDENCY = REPO_ROOT / "experiments/task1_picklift_real24_act_v1/deployment_safety.py"


def device_identity(path_value: str) -> dict:
    path = Path(path_value)
    if not path.is_symlink():
        raise RuntimeError(f"Expected by-id symlink is unavailable: {path}")
    target = path.resolve(strict=True)
    target_stat = target.stat()
    if not stat.S_ISCHR(target_stat.st_mode):
        raise RuntimeError(f"By-id target is not a character device: {target}")
    busy = subprocess.run(
        ["fuser", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if busy.returncode not in (0, 1):
        raise RuntimeError(f"Unable to inspect device ownership for {target}.")
    pids = sorted({int(token) for token in (busy.stdout + " " + busy.stderr).split() if token.isdigit()})
    return {
        "by_id_path": str(path),
        "symlink_target": os.readlink(path),
        "resolved_device": str(target),
        "character_device": True,
        "device_major": os.major(target_stat.st_rdev),
        "device_minor": os.minor(target_stat.st_rdev),
        "busy_process_ids": pids,
    }


def snapshot() -> dict:
    plan = load_frozen_plan()
    static = verify_static_files(plan)
    calibration_hash = sha256_file(DEFAULT_CALIBRATION)
    if calibration_hash != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("Follower calibration hash differs from the frozen setup.")
    return {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "identity_snapshot_passed_devices_not_opened",
        "repo_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "plan_sha256": static["plan_sha256"],
        "models": static["models"],
        "official_send_engine": {
            "path": static["engine_path"],
            "sha256": static["engine_sha256"],
            "current_dependency_path": str(ENGINE_DEPENDENCY),
            "current_dependency_sha256": sha256_file(ENGINE_DEPENDENCY),
        },
        "profile": {
            "path": static["profile_path"],
            "sha256": static["profile_sha256"],
            "ready_pose_arrival_tolerance_degrees": 3.0,
        },
        "calibration": {
            "path": str(DEFAULT_CALIBRATION),
            "sha256": calibration_hash,
            "recalibrated": False,
        },
        "follower": device_identity(EXPECTED_FOLLOWER_PORT),
        "camera": {
            **device_identity(EXPECTED_CAMERA_DEVICE),
            "profile_id": plan["setup"]["camera_profile_id"],
            "profile_source": str(CAMERA_PROFILE_SOURCE),
            "profile_source_sha256": sha256_file(CAMERA_PROFILE_SOURCE),
            "canonical_policy_input_hwc": [480, 640, 3],
            "canonical_policy_input_dtype": "uint8 RGB",
        },
        "physical_grid": {
            "setup_version": plan["setup"]["real_world_setup_version"],
            "task_frame_id": plan["setup"]["task_frame_id"],
            "requires_onsite_unchanged_confirmation": True,
        },
        "power_and_torque": {
            "status": "not_queried_before_onsite_confirmation",
        },
        "access": {
            "serial_opened": False,
            "camera_opened": False,
            "robot_connected": False,
            "torque_commanded": False,
            "action_sent": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    result = snapshot()
    if args.freeze:
        plan = load_frozen_plan()
        root = Path(plan["evidence_root"]) / "software_preparation_v1"
        root.mkdir(parents=True, exist_ok=True)
        output = root / "device_identity.json"
        hashes = root / "device_identity.sha256"
        if output.exists() or hashes.exists():
            raise RuntimeError("Refusing to overwrite frozen device identity evidence.")
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes.write_text(
            f"{sha256_file(output)}  {output.name}\n",
            encoding="utf-8",
        )
        result["frozen_evidence"] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "hashes_path": str(hashes),
            "hashes_sha256": sha256_file(hashes),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
