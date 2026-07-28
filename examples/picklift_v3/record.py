from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from examples.picklift_v3.alignment_reference import (
    ALIGNMENT_REFERENCE_V1_ID,
    alignment_reference,
    validate_alignment_reference_config,
)
from examples.picklift_v3.backend import (
    JOINTS,
    RealSO101Backend,
    SyntheticBackend,
)
from examples.picklift_v3.camera_profile import (
    camera_profile,
    validate_camera_profile_config,
)
from examples.picklift_v3.task_frame import (
    TASK_GRID_FRAME_V1_ID,
    task_frame,
    validate_task_frame_config,
)
from lerobot import __version__ as lerobot_version
from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

FPS = 20
REAL_ACK = "I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK"
SPAWN_PROTOCOL_VERSION = "picklift_spawn_v5"
SPAWN_PROTOCOLS = {
    "picklift_spawn_v1": {
        "x": (20.0, 40.0),
        "y": (15.0, 25.0),
        "row_axis": "y",
        "column_axis": "x",
        "x_description": "mat horizontal",
        "y_description": "forward depth",
    },
    "picklift_spawn_v2": {
        "x": (10.0, 25.0),
        "y": (-10.0, 10.0),
        "row_axis": "x",
        "column_axis": "y",
        "x_description": "task-grid +X forward",
        "y_description": "task-grid +Y lateral",
    },
    "picklift_spawn_v3": {
        "x": (20.0, 35.0),
        "y": (-10.0, 10.0),
        "row_axis": "x",
        "column_axis": "y",
        "x_description": "task-grid +X forward",
        "y_description": "task-grid +Y lateral",
    },
    "picklift_spawn_v4": {
        "x": (20.0, 35.0),
        "y": (-10.0, 10.0),
        "row_axis": "x",
        "column_axis": "y",
        "x_description": "task-grid +X forward",
        "y_description": "task-grid +Y lateral",
        "cell_edges": {
            "x": (20.0, 25.0, 30.0, 35.0),
            "y": (-10.0, -5.0, 0.0, 5.0, 10.0),
        },
        "cell_size_cm": 5.0,
    },
    "picklift_spawn_v5": {
        "x": (20.0, 35.0),
        "y": (-10.0, 10.0),
        "row_axis": "x",
        "column_axis": "y",
        "x_description": "task-grid +X forward",
        "y_description": "task-grid +Y lateral",
        "cell_edges": {
            "x": (20.0, 25.0, 30.0, 35.0),
            "y": (-10.0, -5.0, 0.0, 5.0, 10.0),
        },
        "cell_size_cm": 5.0,
    },
}
RESULTS = {"pending", "success", "failure", "discard"}
REQUIRED = (
    "dataset_root",
    "repo_id",
    "operator_id",
    "session_id",
    "task_id",
    "task_version",
    "task",
    "task_frame_id",
    "alignment_reference_id",
    "real_world_setup_version",
    "camera_config_version",
    "camera_profile_id",
    "camera_device",
    "camera_intrinsics_version",
    "camera_extrinsics_version",
    "robot_id",
    "robot_calibration_id",
    "follower_serial_id",
    "leader_id",
    "leader_calibration_id",
    "leader_serial_id",
    "spawn_id",
    "spawn_protocol_version",
    "spawn_region",
    "spawn_x_cm",
    "spawn_y_cm",
    "spawn_yaw_deg",
    "result",
    "formal_data",
    "termination_reason",
    "success",
)
V5_REQUIRED = (
    "collection_protocol_version",
    "task_spec_revision",
    "yaw_annotation_mode",
    "yaw_intended_range_deg",
    "yaw_sampling_method",
    "yaw_distribution_claim",
    "yaw_randomization_confirmed",
    "success_annotation_source",
    "success_detection_mode",
    "lift_height_m",
    "is_grasped",
)
SUCCESS_CONTRACT = {
    "criteria_version": "picklift_manual_success_v1",
    "automatic_detection_available": False,
    "minimum_visual_lift_height_m": 0.05,
    "recommended_target_lift_range_m": [0.06, 0.08],
    "requires_visible_bilateral_finger_grasp": True,
    "minimum_stable_hold_s": 0.5,
    "requires_success_pose_held_at_end": True,
    "annotation_timing": "operator_selects_result_after_manual_end",
    "failure_definition": "task_success_criteria_not_met",
    "discard_definition": "recording_configuration_or_safety_anomaly",
}


class Backend(Protocol):
    def connect(self) -> None: ...
    def read_pre_action(self) -> tuple[np.ndarray, np.ndarray]: ...
    def requested_action(self) -> np.ndarray: ...
    def send_action(self, action: np.ndarray) -> np.ndarray: ...
    def preview_frame(self) -> np.ndarray: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ControlSample:
    sequence: int
    captured_at: float
    state: np.ndarray
    front: np.ndarray
    sent: np.ndarray


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def spawn_contract(protocol_version: str) -> dict:
    try:
        protocol = SPAWN_PROTOCOLS[protocol_version]
    except KeyError as exc:
        raise ValueError(f"unsupported spawn_protocol_version: {protocol_version}") from exc
    contract = {
        "protocol_version": protocol_version,
        "x_cm": {
            "min": protocol["x"][0],
            "max": protocol["x"][1],
            "description": protocol["x_description"],
        },
        "y_cm": {
            "min": protocol["y"][0],
            "max": protocol["y"][1],
            "description": protocol["y_description"],
        },
        "region_rows_increase_along": protocol["row_axis"],
        "region_columns_increase_along": protocol["column_axis"],
    }
    if "cell_edges" in protocol:
        row_edges = protocol["cell_edges"][protocol["row_axis"]]
        column_edges = protocol["cell_edges"][protocol["column_axis"]]
        contract.update(
            {
                "cell_size_cm": protocol["cell_size_cm"],
                "cell_edges_cm": {axis: list(edges) for axis, edges in protocol["cell_edges"].items()},
                "grid_shape": {
                    "rows": len(row_edges) - 1,
                    "columns": len(column_edges) - 1,
                    "cells": (len(row_edges) - 1) * (len(column_edges) - 1),
                },
            }
        )
    return contract


def spawn_region_for(
    x_cm: float,
    y_cm: float,
    protocol_version: str = SPAWN_PROTOCOL_VERSION,
) -> str:
    try:
        protocol = SPAWN_PROTOCOLS[protocol_version]
    except KeyError as exc:
        raise ValueError(f"unsupported spawn_protocol_version: {protocol_version}") from exc
    x_min, x_max = protocol["x"]
    y_min, y_max = protocol["y"]
    if not x_min <= x_cm <= x_max:
        raise ValueError(f"spawn_x_cm must be within {protocol['x_description']} {x_min:g}..{x_max:g} cm")
    if not y_min <= y_cm <= y_max:
        raise ValueError(f"spawn_y_cm must be within {protocol['y_description']} {y_min:g}..{y_max:g} cm")
    row_axis = protocol["row_axis"]
    column_axis = protocol["column_axis"]
    values = {"x": x_cm, "y": y_cm}
    if "cell_edges" in protocol:
        row_edges = protocol["cell_edges"][row_axis]
        column_edges = protocol["cell_edges"][column_axis]
        row = min(bisect_right(row_edges, values[row_axis]), len(row_edges) - 1)
        column = min(
            bisect_right(column_edges, values[column_axis]),
            len(column_edges) - 1,
        )
    else:
        bounds = {"x": (x_min, x_max), "y": (y_min, y_max)}
        row_min, row_max = bounds[row_axis]
        column_min, column_max = bounds[column_axis]
        row = min(int((values[row_axis] - row_min) / ((row_max - row_min) / 3)), 2) + 1
        column = (
            min(
                int((values[column_axis] - column_min) / ((column_max - column_min) / 3)),
                2,
            )
            + 1
        )
    return f"r{row}c{column}"


def spawn_ui_summary(cfg: dict) -> str:
    cell_count = spawn_contract(cfg["spawn_protocol_version"]).get("grid_shape", {}).get("cells", 9)
    if cfg["spawn_protocol_version"] == "picklift_spawn_v5":
        return (
            f"picklift_spawn_v5 | {cell_count} cells\n"
            f"{cfg['spawn_id']} {cfg['spawn_region']} | "
            f"X={cfg['spawn_x_cm']} Y={cfg['spawn_y_cm']}\n"
            "Yaw arbitrary 0..90 | no measure"
        )
    return (
        f"{cfg['spawn_protocol_version']} | {cfg['spawn_id']} | {cfg['spawn_region']}\n"
        f"Xfwd={cfg['spawn_x_cm']}cm Ylat={cfg['spawn_y_cm']}cm "
        f"yaw={cfg['spawn_yaw_deg']}\n"
        f"{cell_count} cells | {cfg['alignment_reference_id']}"
    )


def validate_config(cfg: dict) -> None:
    if (
        not cfg.get("alignment_reference_id")
        and cfg.get("task_frame_id") == TASK_GRID_FRAME_V1_ID
        and cfg.get("spawn_protocol_version") in {"picklift_spawn_v1", "picklift_spawn_v2"}
    ):
        # Compatibility for configs frozen before alignment references became a
        # separate provenance object. New protocols must always provide one.
        cfg["alignment_reference_id"] = ALIGNMENT_REFERENCE_V1_ID
    protocol_version = cfg.get("spawn_protocol_version")
    if protocol_version not in SPAWN_PROTOCOLS:
        raise ValueError(f"unsupported spawn_protocol_version: {protocol_version}")
    nullable = {"spawn_yaw_deg"} if protocol_version == "picklift_spawn_v5" else set()
    missing = [key for key in REQUIRED if key not in cfg or (cfg[key] in ("", None) and key not in nullable)]
    if protocol_version == "picklift_spawn_v5":
        nullable_v5 = {"lift_height_m", "is_grasped"}
        missing.extend(
            key
            for key in V5_REQUIRED
            if key not in cfg or (cfg[key] in ("", None) and key not in nullable_v5)
        )
    if missing:
        raise ValueError(f"missing explicit configuration values: {', '.join(missing)}")
    if cfg.get("record_fps") != FPS:
        raise ValueError("record_fps must be exactly 20")
    if float(cfg.get("control_hz", 0)) < FPS:
        raise ValueError("control_hz must be >= 20")
    if int(cfg.get("camera_acquisition_fps", 0)) < FPS:
        raise ValueError("camera_acquisition_fps must be >= 20")
    validate_camera_profile_config(cfg)
    validate_task_frame_config(cfg)
    validate_alignment_reference_config(cfg)
    if cfg["mode"] == "real" and cfg["camera_config_version"] != cfg["camera_profile_id"]:
        raise ValueError("camera_config_version must equal the immutable real camera_profile_id")
    if cfg.get("alignment_mode") not in {"relative_rebase", "direct_absolute"}:
        raise ValueError("alignment_mode must be relative_rebase or direct_absolute")
    if float(cfg.get("startup_hold_s", -1)) < 0:
        raise ValueError("startup_hold_s must be >= 0")
    if not isinstance(cfg["success"], bool):
        raise ValueError("success must be boolean")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(cfg["spawn_id"])):
        raise ValueError("spawn_id must be a stable token")
    x_cm = float(cfg["spawn_x_cm"])
    y_cm = float(cfg["spawn_y_cm"])
    expected_region = spawn_region_for(x_cm, y_cm, cfg["spawn_protocol_version"])
    if cfg["spawn_region"] != expected_region:
        raise ValueError(f"spawn_region does not match actual coordinates: expected {expected_region}")
    if (
        cfg["spawn_protocol_version"] in {"picklift_spawn_v4", "picklift_spawn_v5"}
        and cfg["formal_data"]
        and x_cm == 25.0
        and y_cm == 0.0
    ):
        raise ValueError("(25, 0) is alignment-only and cannot be a formal randomized episode pose")
    if cfg["spawn_protocol_version"] == "picklift_spawn_v5":
        if cfg["collection_protocol_version"] != "picklift_collection_v5":
            raise ValueError("v5 requires collection_protocol_version=picklift_collection_v5")
        if cfg["task_spec_revision"] != "picklift_taskspec_v2_unmeasured_yaw":
            raise ValueError("v5 requires task_spec_revision=picklift_taskspec_v2_unmeasured_yaw")
        if cfg["yaw_annotation_mode"] != "unmeasured_random" or cfg["spawn_yaw_deg"] is not None:
            raise ValueError("v5 requires yaw_annotation_mode=unmeasured_random and spawn_yaw_deg=null")
        if cfg["yaw_intended_range_deg"] != [0, 90]:
            raise ValueError("v5 requires yaw_intended_range_deg=[0, 90]")
        if cfg["yaw_sampling_method"] != "operator_unmeasured_arbitrary":
            raise ValueError("v5 requires yaw_sampling_method=operator_unmeasured_arbitrary")
        if cfg["yaw_distribution_claim"] != "unknown":
            raise ValueError("v5 requires yaw_distribution_claim=unknown")
        if not isinstance(cfg["yaw_randomization_confirmed"], bool):
            raise ValueError("yaw_randomization_confirmed must be boolean")
        if cfg["success_annotation_source"] != "operator_visual_v1":
            raise ValueError("v5 requires success_annotation_source=operator_visual_v1")
        if cfg["success_detection_mode"] != "manual_proxy_for_nexus_v1":
            raise ValueError("v5 requires success_detection_mode=manual_proxy_for_nexus_v1")
        if cfg["lift_height_m"] is not None or cfg["is_grasped"] is not None:
            raise ValueError("v5 unavailable lift_height_m and is_grasped measurements must remain null")
    else:
        yaw_deg = float(cfg["spawn_yaw_deg"])
        if not 0 <= yaw_deg <= 90:
            raise ValueError("spawn_yaw_deg must be within 0..90 degrees")
    if cfg["result"] not in RESULTS:
        raise ValueError(f"result must be one of {sorted(RESULTS)}")
    if cfg["result"] == "pending" and not cfg.get("operator_ui", False):
        raise ValueError("pending result requires operator_ui result review")
    if cfg["result"] != "pending" and cfg["success"] != (cfg["result"] == "success"):
        raise ValueError("success must be true exactly when result is success")
    if cfg["mode"] not in {"synthetic", "real"}:
        raise ValueError("mode must be synthetic or real")
    if re.search(r"@|\\s", cfg["operator_id"]):
        raise ValueError("operator_id must be a pseudonymous token, not direct identity information")
    root = Path(cfg["dataset_root"]).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dataset_root must be new and empty: {root}")
    if cfg["mode"] == "real":
        placeholders = [k for k, v in cfg.items() if isinstance(v, str) and "REPLACE" in v]
        if placeholders:
            raise ValueError(f"real config still contains placeholders: {', '.join(placeholders)}")
        if cfg.get("powered_real_run_ack") != REAL_ACK:
            raise PermissionError(
                "real mode is locked until the powered safety check; do not enable 12 V for validation"
            )
        for key in ("follower_port", "leader_port"):
            if not str(cfg.get(key, "")).startswith("/dev/serial/by-id/"):
                raise ValueError(f"{key} must use a stable /dev/serial/by-id path")


def features(use_videos: bool) -> dict:
    names = list(JOINTS)
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.front": {
            "dtype": "video" if use_videos else "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record(cfg: dict, backend: Backend | None = None) -> Path:
    validate_config(cfg)
    root = Path(cfg["dataset_root"]).resolve()
    start: str | None = None
    backend = backend or (SyntheticBackend(cfg) if cfg["mode"] == "synthetic" else RealSO101Backend(cfg))
    dataset = LeRobotDataset.create(
        cfg["repo_id"],
        fps=FPS,
        root=root,
        robot_type="so101_follower" if cfg["mode"] == "real" else "synthetic_so101",
        features=features(bool(cfg["use_videos"])),
        use_videos=bool(cfg["use_videos"]),
    )
    dropped = 0
    sync_anomalies = 0
    actual_termination_reason = cfg["termination_reason"]
    actual_result = cfg["result"]
    actual_success = cfg["success"]
    sample_count = round(float(cfg["episode_seconds"]) * FPS)
    period = 1 / FPS
    control_period = 1 / float(cfg["control_hz"])
    condition = threading.Condition()
    stop = threading.Event()
    latest: ControlSample | None = None
    control_error: BaseException | None = None

    def control_loop() -> None:
        nonlocal latest, control_error
        sequence = 0
        next_tick = time.perf_counter()
        try:
            while not stop.is_set():
                state, front = backend.read_pre_action()
                requested = backend.requested_action()
                sent = backend.send_action(requested)
                sample = ControlSample(sequence, time.perf_counter(), state, front, sent)
                with condition:
                    latest = sample
                    condition.notify_all()
                sequence += 1
                next_tick += control_period
                stop.wait(max(0.0, next_tick - time.perf_counter()))
        except BaseException as exc:
            control_error = exc
            with condition:
                condition.notify_all()

    backend.connect()
    ui = None
    if cfg.get("operator_ui", False):
        from examples.picklift_v3.operator_ui import OperatorUI

        ui = OperatorUI(target_frames=sample_count)
        ui.open()
        ui.wait_for_start(backend.preview_frame, message=spawn_ui_summary(cfg))
    elif cfg.get("operator_cue_wait", False):
        input("CONTROL_READY: waiting for operator cue (press ENTER to start)")
    start = utc_now()
    worker = threading.Thread(target=control_loop, name="picklift-control", daemon=True)
    worker.start()
    try:
        next_sample = time.perf_counter()
        last_sequence = -1
        for sample_index in range(sample_count):
            delay = next_sample - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with condition:
                condition.wait_for(
                    lambda minimum_sequence=last_sequence: (
                        control_error is not None
                        or (latest is not None and latest.sequence > minimum_sequence)
                    ),
                    timeout=period,
                )
                if control_error is not None:
                    raise RuntimeError("control loop failed") from control_error
                if latest is None or latest.sequence <= last_sequence:
                    dropped += 1
                    raise RuntimeError("no fresh synchronized control sample; fail-closed")
                sample = latest
            state, front, sent = sample.state, sample.front, sample.sent
            if time.perf_counter() - sample.captured_at > period:
                sync_anomalies += 1
                raise RuntimeError("stale synchronized sample; fail-closed")
            if state.shape != (6,) or state.dtype != np.float32:
                raise RuntimeError("pre-action follower state contract violation")
            if sent.shape != (6,) or sent.dtype != np.float32:
                raise RuntimeError("actual-sent action contract violation")
            if front.shape != (480, 640, 3) or front.dtype != np.uint8:
                raise RuntimeError("front synchronization/canonical image contract violation")
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": sent,
                    "observation.images.front": front,
                    "task": cfg["task"],
                }
            )
            if ui is not None:
                command = ui.show(
                    front,
                    status="RECORDING",
                    elapsed_s=((sample_index + 1) / FPS),
                    frames=sample_index + 1,
                    message=(
                        "SUCCESS: lift >=5cm (aim 6-8)\n"
                        "Cube visibly between both fingers\n"
                        "Hold 0.5-1s through END"
                    ),
                )
                if command == "stop":
                    actual_termination_reason = "operator_end"
                    break
                if command == "quit":
                    actual_termination_reason = "operator_quit"
                    break
            last_sequence = sample.sequence
            next_sample += period
        stop.set()
        worker.join(timeout=2)
        if worker.is_alive():
            raise RuntimeError("control loop did not stop cleanly")
        print("ACTION_WINDOW_COMPLETE", flush=True)
        if ui is not None:
            actual_result = ui.review_result(front)
            actual_success = actual_result == "success"
        dataset.save_episode()
        dataset.finalize()
        if ui is not None:
            ui.show_complete(front, root)
    finally:
        stop.set()
        worker.join(timeout=2)
        backend.close()
        if ui is not None:
            ui.close()
    end = utc_now()
    common = {
        **{k: cfg[k] for k in REQUIRED},
        **{k: cfg[k] for k in V5_REQUIRED if k in cfg},
        "backend": "real" if cfg["mode"] == "real" else "synthetic",
        "control_mode": "leader_follower",
        "spawn_contract": spawn_contract(cfg["spawn_protocol_version"]),
        "collection_commit": git_commit(),
        "lerobot_version": lerobot_version,
        "lerobot_dataset_version": CODEBASE_VERSION,
        "control_hz": cfg["control_hz"],
        "camera_acquisition_fps": cfg["camera_acquisition_fps"],
        "record_fps": FPS,
        "joint_order": list(JOINTS),
        "task_frame": task_frame(cfg["task_frame_id"]),
        "alignment_reference": alignment_reference(cfg["alignment_reference_id"]),
        "camera_profile": camera_profile(cfg["camera_profile_id"]),
        "canonical_front": camera_profile(cfg["camera_profile_id"])["output"],
        "alignment_mode": cfg["alignment_mode"],
        "startup_hold_s": cfg["startup_hold_s"],
        "initial_rebase_offset": (
            backend.rebaser.offset.tolist()
            if isinstance(backend, RealSO101Backend) and backend.rebaser.offset is not None
            else [0.0] * 6
        ),
        "raw_evidence": "not_recorded",
        "start_time": start,
        "end_time": end,
        "dropped_frames": dropped,
        "sync_anomalies": sync_anomalies,
        "formal_data": cfg["formal_data"],
    }
    if cfg["spawn_protocol_version"] == "picklift_spawn_v5":
        common["success_contract"] = SUCCESS_CONTRACT
    common["configured_termination_reason"] = cfg["termination_reason"]
    common["termination_reason"] = actual_termination_reason
    common["configured_result"] = cfg["result"]
    common["result"] = actual_result
    common["success"] = actual_success
    write_json(root / "provenance/dataset.json", common)
    write_json(root / "provenance/session.json", common)
    write_json(root / "provenance/episodes/episode_000000.json", {**common, "episode_index": 0})
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    if args.validate_only:
        print("configuration valid; no devices opened")
        return
    print(record(cfg))


if __name__ == "__main__":
    main()
