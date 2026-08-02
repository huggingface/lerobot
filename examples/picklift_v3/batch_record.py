from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from examples.picklift_v3.backend import RealSO101Backend, SyntheticBackend
from examples.picklift_v3.operator_ui import OperatorUI
from examples.picklift_v3.record import (
    FPS,
    REAL96_COLLECTION_PROTOCOL_VERSION,
    REAL96_PLAN_FIELDS,
    REAL96_SESSION_FIELDS,
    REAL96_SPAWN_PROTOCOL_VERSION,
    Backend,
    capture_episode,
    create_dataset,
    episode_provenance,
    spawn_ui_summary,
    validate_config,
    write_json,
)

BATCH_WORKFLOW_VERSION = "picklift_continuous_batch_v4_real96_attempt_ledger"
SPAWN_FIELDS = (
    "spawn_id",
    "spawn_region",
    "spawn_x_cm",
    "spawn_y_cm",
    "spawn_yaw_deg",
)
REAL96_BATCH_FIELDS = REAL96_SESSION_FIELDS


class LiveResetFollower:
    def __init__(self, backend: Backend, control_hz: float, initial_frame: np.ndarray):
        self.backend = backend
        self.period = 1 / control_hz
        self._frame = initial_frame
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="picklift-live-reset",
            daemon=True,
        )

    def _run(self) -> None:
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                _, front = self.backend.read_pre_action()
                requested = self.backend.requested_action()
                self.backend.send_action(requested)
                with self._lock:
                    self._frame = front
                next_tick += self.period
                self._stop.wait(max(0.0, next_tick - time.perf_counter()))
        except BaseException as exc:
            self._error = exc

    def start(self) -> None:
        self._worker.start()

    def frame(self) -> np.ndarray:
        if self._error is not None:
            raise RuntimeError("live reset follower failed") from self._error
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2)
        if self._worker.is_alive():
            raise RuntimeError("live reset follower did not stop cleanly")
        if self._error is not None:
            raise RuntimeError("live reset follower failed") from self._error


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _attempt_config(base: dict, spawn: dict, repetition: int, successes_per_spawn: int) -> dict:
    cfg = {**deepcopy(base), **deepcopy(spawn)}
    if successes_per_spawn > 1:
        cfg["spawn_id"] = f"{spawn['spawn_id']}_rep{repetition + 1:02d}"
    cfg["result"] = "pending"
    cfg["success"] = False
    cfg["yaw_randomization_confirmed"] = False
    return cfg


def validate_batch_config(batch_cfg: dict) -> list[dict]:
    if batch_cfg.get("collection_workflow_version") != BATCH_WORKFLOW_VERSION:
        raise ValueError(f"collection_workflow_version must be {BATCH_WORKFLOW_VERSION}")
    base = batch_cfg.get("base_config")
    spawns = batch_cfg.get("spawns")
    if not isinstance(base, dict):
        raise ValueError("base_config must be an object")
    if not isinstance(spawns, list) or not spawns:
        raise ValueError("spawns must be a non-empty list")
    successes_per_spawn = batch_cfg.get("successes_per_spawn")
    if not isinstance(successes_per_spawn, int) or successes_per_spawn <= 0:
        raise ValueError("successes_per_spawn must be a positive integer")
    if successes_per_spawn != 1:
        raise ValueError("Real96 requires successes_per_spawn=1 per frozen plan item")
    target_successes = len(spawns) * successes_per_spawn
    max_attempts = batch_cfg.get("max_attempts")
    if not isinstance(max_attempts, int) or max_attempts < target_successes:
        raise ValueError("max_attempts must be an integer >= the planned success count")
    if base.get("operator_ui") is not True or base.get("result") != "pending":
        raise ValueError("continuous batch requires operator_ui=true and result=pending")
    if base.get("spawn_protocol_version") != REAL96_SPAWN_PROTOCOL_VERSION:
        raise ValueError(f"batch v4 requires spawn_protocol_version={REAL96_SPAWN_PROTOCOL_VERSION}")
    if base.get("collection_protocol_version") != REAL96_COLLECTION_PROTOCOL_VERSION:
        raise ValueError(
            f"batch v4 requires collection_protocol_version={REAL96_COLLECTION_PROTOCOL_VERSION}"
        )
    missing_batch_fields = [key for key in REAL96_BATCH_FIELDS if key not in base]
    if missing_batch_fields:
        raise ValueError(f"missing Real96 batch fields: {', '.join(missing_batch_fields)}")

    configs = []
    seen_spawn_ids = set()
    for spawn in spawns:
        if not isinstance(spawn, dict):
            raise ValueError("every spawn must be an object")
        missing = [field for field in SPAWN_FIELDS if field not in spawn]
        missing.extend(field for field in REAL96_PLAN_FIELDS if field not in spawn)
        if missing:
            raise ValueError(f"spawn missing fields: {', '.join(missing)}")
        if spawn["spawn_id"] in seen_spawn_ids:
            raise ValueError(f"duplicate spawn_id: {spawn['spawn_id']}")
        seen_spawn_ids.add(spawn["spawn_id"])
        cfg = _attempt_config(base, spawn, repetition=0, successes_per_spawn=successes_per_spawn)
        validate_config(cfg)
        if cfg["alignment_mode"] != "direct_absolute":
            raise ValueError("continuous batch v4 requires alignment_mode=direct_absolute")
        if cfg.get("max_relative_target") is not None:
            raise ValueError("continuous batch v4 requires max_relative_target=null")
        configs.append(cfg)

    roots = {Path(cfg["dataset_root"]).resolve() for cfg in configs}
    repo_ids = {cfg["repo_id"] for cfg in configs}
    session_ids = {cfg["session_id"] for cfg in configs}
    if len(roots) != 1 or len(repo_ids) != 1 or len(session_ids) != 1:
        raise ValueError("all batch episodes must share dataset_root, repo_id, and session_id")
    return configs


def _default_backend_factory(cfg: dict) -> Backend:
    return SyntheticBackend(cfg) if cfg["mode"] == "synthetic" else RealSO101Backend(cfg)


def _batch_manifest(
    *,
    batch_cfg: dict,
    configs: list[dict],
    batch_start: str,
    attempts: int,
    saved_episodes: int,
    success_counts: list[int],
    last_attempt: dict | None,
    complete: bool,
    accepted_episode_indices: list[int],
) -> dict:
    base = configs[0]
    stable_keys = (
        "repo_id",
        "operator_id",
        "session_id",
        "task_id",
        "task_version",
        "task",
        "task_spec_revision",
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
        "collection_protocol_version",
        "spawn_protocol_version",
        "success_annotation_source",
        "success_detection_mode",
        "control_hz",
        "camera_acquisition_fps",
        "record_fps",
        "formal_data",
        *REAL96_BATCH_FIELDS,
    )
    promoted_provenance_keys = (
        "backend",
        "control_mode",
        "collection_commit",
        "collection_repo_path",
        "collection_git_dirty",
        "lerobot_version",
        "lerobot_dataset_version",
        "python_version",
        "torch_version",
        "opencv_python_version",
        "joint_order",
        "task_frame",
        "alignment_reference",
        "camera_profile",
        "canonical_front",
        "spawn_contract",
        "gripper_alignment_mode",
        "action_mapping",
        "action_transform",
        "max_relative_target",
        "state_action_order",
        "camera_frame_evidence",
        "post_end_control_mode",
        "yaw_annotation_mode",
        "yaw_intended_range_deg",
        "yaw_sampling_method",
        "yaw_distribution_claim",
        "success_contract",
    )
    return {
        **{key: base[key] for key in stable_keys},
        **({key: last_attempt[key] for key in promoted_provenance_keys} if last_attempt is not None else {}),
        "collection_workflow_version": BATCH_WORKFLOW_VERSION,
        "batch_start_time": batch_start,
        "batch_end_time": utc_now() if complete else None,
        "last_update_time": utc_now(),
        "complete": complete,
        "attempt_count": attempts,
        "saved_episode_count": saved_episodes,
        "raw_attempt_episode_count": attempts,
        "accepted_dataset_episode_indices": accepted_episode_indices,
        "target_success_count": len(configs) * batch_cfg["successes_per_spawn"],
        "successes_per_spawn": batch_cfg["successes_per_spawn"],
        "max_attempts": batch_cfg["max_attempts"],
        "planned_spawns": [
            {
                **{field: spawn[field] for field in (*SPAWN_FIELDS, *REAL96_PLAN_FIELDS)},
                "target_successes": batch_cfg["successes_per_spawn"],
                "saved_successes": success_counts[index],
            }
            for index, spawn in enumerate(batch_cfg["spawns"])
        ],
        "advance_rule": "success advances; failure/discard retries the same spawn",
        "training_view_rule": (
            "raw v3 dataset retains every started attempt; accepted_dataset_episode_indices "
            "select exactly one SUCCESS per frozen plan_item"
        ),
        "inter_episode_control": "live absolute Leader-to-Follower reset with no dataset writes",
        "ready_pose_policy": "operator_visual_similar_ready_area_no_numeric_threshold",
        "last_attempt": last_attempt,
    }


def record_batch(
    batch_cfg: dict,
    *,
    backend_factory: Callable[[dict], Backend] | None = None,
    ui: OperatorUI | Any | None = None,
) -> Path:
    configs = validate_batch_config(batch_cfg)
    root = Path(configs[0]["dataset_root"]).resolve()
    successes_per_spawn = batch_cfg["successes_per_spawn"]
    target_successes = len(configs) * successes_per_spawn
    backend_factory = backend_factory or _default_backend_factory
    sample_count = round(float(configs[0]["episode_seconds"]) * FPS)
    ui = ui or OperatorUI(target_frames=sample_count)
    ui.open()

    dataset = None
    backend = None
    live_reset = None
    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    batch_start = utc_now()
    attempt_index = 0
    saved_episodes = 0
    spawn_index = 0
    success_counts = [0] * len(configs)
    last_attempt = None
    accepted_episode_indices: list[int] = []
    attempt_ordinals: dict[str, int] = {}
    previous_attempt_ids: dict[str, str] = {}
    operator_quit = False
    try:
        while saved_episodes < target_successes and attempt_index < batch_cfg["max_attempts"]:
            repetition = success_counts[spawn_index]
            cfg = _attempt_config(
                batch_cfg["base_config"],
                batch_cfg["spawns"][spawn_index],
                repetition,
                successes_per_spawn,
            )
            plan_item_id = cfg["plan_item_id"]
            attempt_ordinal = attempt_ordinals.get(plan_item_id, 0) + 1
            attempt_id = f"{plan_item_id}_attempt_{attempt_ordinal:02d}"
            cfg["attempt_id"] = attempt_id
            cfg["attempt_ordinal"] = attempt_ordinal
            cfg["previous_attempt_id"] = previous_attempt_ids.get(plan_item_id)
            ready_message = (
                f"{spawn_ui_summary(cfg)}\n"
                "Place cube at exact plan pose; open both grippers\n"
                "Bring arms to similar ready area; check view"
            )
            if backend is None:
                if not ui.wait_for_ready(last_frame, ready_message):
                    operator_quit = True
                    break
                cfg["yaw_randomization_confirmed"] = True
                ui.show_status(
                    last_frame,
                    status="CONNECTING",
                    message="Connecting with absolute joint mapping\nKeep both arms still",
                )
                backend = backend_factory(cfg)
                try:
                    backend.connect()
                except (ConnectionError, RuntimeError):
                    ui.show_connection_error(last_frame)
                    backend.close()
                    backend = None
                    continue
                try:
                    ui.wait_for_start(backend.preview_frame, message=spawn_ui_summary(cfg))
                except KeyboardInterrupt:
                    operator_quit = True
                    break
            else:
                if live_reset is None:
                    raise RuntimeError("connected batch is missing live reset follower")
                reset_message = (
                    f"{spawn_ui_summary(cfg)}\n"
                    "Follower LIVE (not recording); reset both arms to a similar ready area\n"
                    "Check view + gripper, then START"
                )
                if not ui.wait_for_next_start(live_reset.frame, reset_message):
                    operator_quit = True
                    break
                cfg["yaw_randomization_confirmed"] = True
                last_frame = live_reset.frame()
                live_reset.stop()
                live_reset = None

            if dataset is None:
                dataset = create_dataset(cfg)
            outcome = capture_episode(
                cfg,
                backend,
                dataset,
                ui,
                review_with_ui=False,
            )
            live_reset = LiveResetFollower(
                backend,
                float(cfg["control_hz"]),
                outcome.frame,
            )
            live_reset.start()
            result = ui.review_result(outcome.frame)
            outcome = replace(outcome, result=result, success=result == "success")
            ui.show_saving(outcome.frame, result=outcome.result)
            saved_to_training = outcome.result == "success"
            episode_index = attempt_index
            dataset.save_episode()
            provenance = episode_provenance(
                cfg,
                backend,
                outcome,
                episode_index=episode_index,
                attempt_index=attempt_index,
                saved_to_training=saved_to_training,
                collection_workflow_version=BATCH_WORKFLOW_VERSION,
            )
            provenance["post_end_control_mode"] = "live_follow_no_recording"
            provenance["attempt_id"] = attempt_id
            provenance["attempt_ordinal"] = attempt_ordinal
            provenance["previous_attempt_id"] = cfg["previous_attempt_id"]
            provenance["accepted_success_index"] = saved_episodes if saved_to_training else None
            provenance["all_attempts_retained"] = True
            write_json(
                root / f"provenance/attempts/attempt_{attempt_index:06d}.json",
                provenance,
            )
            write_json(
                root / f"provenance/episodes/episode_{attempt_index:06d}.json",
                provenance,
            )
            previous_attempt_ids[plan_item_id] = attempt_id
            attempt_ordinals[plan_item_id] = attempt_ordinal
            if saved_to_training:
                accepted_episode_indices.append(attempt_index)
                write_json(
                    root / f"provenance/accepted/accepted_{saved_episodes:06d}.json",
                    provenance,
                )
                saved_episodes += 1
                success_counts[spawn_index] += 1
                for offset in range(1, len(configs) + 1):
                    candidate = (spawn_index + offset) % len(configs)
                    if success_counts[candidate] < successes_per_spawn:
                        spawn_index = candidate
                        break
            last_attempt = provenance
            attempt_index += 1
            manifest = _batch_manifest(
                batch_cfg=batch_cfg,
                configs=configs,
                batch_start=batch_start,
                attempts=attempt_index,
                saved_episodes=saved_episodes,
                success_counts=success_counts,
                last_attempt=last_attempt,
                complete=saved_episodes >= target_successes,
                accepted_episode_indices=accepted_episode_indices,
            )
            write_json(root / "provenance/dataset.json", manifest)
            write_json(root / "provenance/session.json", manifest)
            next_message = (
                "Follower stays LIVE for next reset"
                if saved_to_training
                else "Follower stays LIVE; retry same spawn"
            )
            ui.show_attempt_complete(
                outcome.frame,
                result=outcome.result,
                saved_to_training=saved_to_training,
                next_message=next_message,
            )
            last_frame = live_reset.frame()
        complete = saved_episodes >= target_successes
        if complete and live_reset is not None:
            ui.wait_for_finish(live_reset.frame)
        if live_reset is not None:
            last_frame = live_reset.frame()
            live_reset.stop()
            live_reset = None
        if dataset is not None:
            ui.show_status(
                last_frame,
                status="SAVING",
                message="Finalizing dataset\nButtons locked; please wait",
            )
            dataset.finalize()
            manifest = _batch_manifest(
                batch_cfg=batch_cfg,
                configs=configs,
                batch_start=batch_start,
                attempts=attempt_index,
                saved_episodes=saved_episodes,
                success_counts=success_counts,
                last_attempt=last_attempt,
                complete=complete,
                accepted_episode_indices=accepted_episode_indices,
            )
            manifest["operator_quit"] = operator_quit
            write_json(root / "provenance/dataset.json", manifest)
            write_json(root / "provenance/session.json", manifest)
        return root
    except BaseException:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        raise
    finally:
        if live_reset is not None:
            live_reset.stop()
        if backend is not None:
            backend.close()
        ui.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    batch_cfg = json.loads(args.config.read_text())
    configs = validate_batch_config(batch_cfg)
    if args.validate_only:
        print(
            f"batch configuration valid; {len(configs)} spawns; "
            f"{len(configs) * batch_cfg['successes_per_spawn']} planned successes; no devices opened"
        )
        return
    print(record_batch(batch_cfg))


if __name__ == "__main__":
    main()
