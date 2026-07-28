from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from lerobot import __version__ as lerobot_version
from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
FPS = 20
REAL_ACK = "I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK"
SPAWN_PROTOCOL_VERSION = "picklift_spawn_v1"
RESULTS = {"pending", "success", "failure", "discard"}
REQUIRED = (
    "dataset_root",
    "repo_id",
    "operator_id",
    "session_id",
    "task_id",
    "task_version",
    "task",
    "real_world_setup_version",
    "camera_config_version",
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
    "spawn_region",
    "spawn_x_cm",
    "spawn_y_cm",
    "spawn_yaw_deg",
    "result",
    "formal_data",
    "termination_reason",
    "success",
)


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


def spawn_region_for(x_cm: float, y_cm: float) -> str:
    if not 20 <= x_cm <= 40:
        raise ValueError("spawn_x_cm must be within mat horizontal 20..40 cm")
    if not 15 <= y_cm <= 25:
        raise ValueError("spawn_y_cm must be within forward depth 15..25 cm")
    column = min(int((x_cm - 20) / (20 / 3)), 2) + 1
    row = min(int((y_cm - 15) / (10 / 3)), 2) + 1
    return f"r{row}c{column}"


def validate_config(cfg: dict) -> None:
    missing = [key for key in REQUIRED if key not in cfg or cfg[key] in ("", None)]
    if missing:
        raise ValueError(f"missing explicit configuration values: {', '.join(missing)}")
    if cfg.get("record_fps") != FPS:
        raise ValueError("record_fps must be exactly 20")
    if float(cfg.get("control_hz", 0)) < FPS:
        raise ValueError("control_hz must be >= 20")
    if int(cfg.get("camera_acquisition_fps", 0)) < FPS:
        raise ValueError("camera_acquisition_fps must be >= 20")
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
    yaw_deg = float(cfg["spawn_yaw_deg"])
    expected_region = spawn_region_for(x_cm, y_cm)
    if cfg["spawn_region"] != expected_region:
        raise ValueError(f"spawn_region does not match actual coordinates: expected {expected_region}")
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


class SyntheticBackend:
    def __init__(self) -> None:
        self.index = 0

    def connect(self) -> None:
        pass

    def read_pre_action(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.arange(6, dtype=np.float32) + self.index / 100
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = self.index % 255
        return state, image

    def requested_action(self) -> np.ndarray:
        return np.arange(6, dtype=np.float32) + 0.5

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self.index += 1
        return np.clip(action, -100, 100).astype(np.float32)

    def preview_frame(self) -> np.ndarray:
        return self.read_pre_action()[1]

    def close(self) -> None:
        pass


@dataclass
class RelativeRebaser:
    offset: np.ndarray | None = None

    def initialize(self, leader: np.ndarray, follower: np.ndarray) -> np.ndarray:
        leader = np.asarray(leader, dtype=np.float32)
        follower = np.asarray(follower, dtype=np.float32)
        if leader.shape != (6,) or follower.shape != (6,):
            raise ValueError("relative rebase requires two six-joint vectors")
        self.offset = follower - leader
        return self.apply(leader)

    def apply(self, leader: np.ndarray) -> np.ndarray:
        if self.offset is None:
            raise RuntimeError("relative rebase used before initialization")
        leader = np.asarray(leader, dtype=np.float32)
        if leader.shape != (6,):
            raise ValueError("leader action must contain six joints")
        return (leader + self.offset).astype(np.float32)


class RealSO101Backend:
    def __init__(self, cfg: dict):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO101Leader

        camera = OpenCVCameraConfig(
            index_or_path=cfg["camera_device"],
            width=640,
            height=480,
            fps=int(cfg["camera_acquisition_fps"]),
        )
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=cfg["follower_port"],
                id=cfg["robot_id"],
                cameras={"front": camera},
                use_degrees=True,
                max_relative_target=cfg.get("max_relative_target", 5.0),
            )
        )
        self.leader = SO101Leader(
            SO101LeaderConfig(port=cfg["leader_port"], id=cfg["leader_id"], use_degrees=True)
        )
        self.rebaser = RelativeRebaser()
        self.alignment_mode = cfg["alignment_mode"]
        self.startup_hold_s = float(cfg["startup_hold_s"])

    def _set_follower_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        self.robot.bus.sync_write("Torque_Enable", value, normalize=False, num_retry=2)
        actual = self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        if any(int(state) != value for state in actual.values()):
            raise RuntimeError(f"follower torque verification failed: expected {value}, got {actual}")

    def connect(self) -> None:
        # Deliberately bypass Robot.connect(): the generic SO implementation performs
        # configuration writes before establishing a no-jump goal. Existing setup is
        # audited separately; here we open buses, latch the present raw follower pose
        # as its goal, and only then enable torque.
        self.robot.bus.connect(handshake=True)
        self.leader.bus.connect(handshake=True)
        try:
            for camera in self.robot.cameras.values():
                camera.connect()
            follower_raw = self.robot.bus.sync_read("Present_Position", normalize=False)
            self.robot.bus.sync_write("Goal_Position", follower_raw, normalize=False)
            follower = self._read_follower_state()
            leader = self._read_leader_state()
            if self.alignment_mode == "relative_rebase":
                initial_command = self.rebaser.initialize(leader, follower)
                if not np.allclose(initial_command, follower, atol=1e-5):
                    raise RuntimeError("relative rebase failed zero-jump invariant")
            self._set_follower_torque(True)
            time.sleep(self.startup_hold_s)
        except BaseException:
            if self.robot.bus.is_connected:
                try:
                    self._set_follower_torque(False)
                finally:
                    self.robot.bus.disconnect(disable_torque=False)
            if self.leader.bus.is_connected:
                self.leader.bus.disconnect(disable_torque=False)
            raise

    def _read_follower_state(self) -> np.ndarray:
        state = self.robot.bus.sync_read("Present_Position")
        return np.asarray([state[j] for j in JOINTS], dtype=np.float32)

    def _read_leader_state(self) -> np.ndarray:
        action = self.leader.bus.sync_read("Present_Position")
        return np.asarray([action[j] for j in JOINTS], dtype=np.float32)

    def read_pre_action(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.robot.get_observation()
        state = np.asarray([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        image = np.asarray(obs["front"], dtype=np.uint8)
        if image.shape != (480, 640, 3):
            raise RuntimeError(f"front camera violated canonical shape: {image.shape}")
        return state, image

    def requested_action(self) -> np.ndarray:
        leader = self._read_leader_state()
        return self.rebaser.apply(leader) if self.alignment_mode == "relative_rebase" else leader

    def send_action(self, action: np.ndarray) -> np.ndarray:
        requested = {f"{joint}.pos": float(action[i]) for i, joint in enumerate(JOINTS)}
        sent = self.robot.send_action(requested)
        return np.asarray([sent[f"{j}.pos"] for j in JOINTS], dtype=np.float32)

    def preview_frame(self) -> np.ndarray:
        return np.asarray(self.robot.cameras["front"].read_latest(), dtype=np.uint8)

    def close(self) -> None:
        if self.robot.bus.is_connected:
            try:
                self._set_follower_torque(False)
            finally:
                self.robot.bus.disconnect(disable_torque=False)
        for camera in self.robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.leader.bus.is_connected:
            self.leader.bus.disconnect(disable_torque=False)


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
    backend = backend or (SyntheticBackend() if cfg["mode"] == "synthetic" else RealSO101Backend(cfg))
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
        spawn_summary = (
            f"{cfg['spawn_id']} | {cfg['spawn_region']}\n"
            f"x={cfg['spawn_x_cm']}  y={cfg['spawn_y_cm']}  yaw={cfg['spawn_yaw_deg']}"
        )
        ui.wait_for_start(backend.preview_frame, message=spawn_summary)
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
                    message=f"{cfg['spawn_id']} | {cfg['spawn_region']}\nMove Leader | END stops early",
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
        "backend": "real" if cfg["mode"] == "real" else "synthetic",
        "control_mode": "leader_follower",
        "spawn_protocol_version": SPAWN_PROTOCOL_VERSION,
        "collection_commit": git_commit(),
        "lerobot_version": lerobot_version,
        "lerobot_dataset_version": CODEBASE_VERSION,
        "control_hz": cfg["control_hz"],
        "camera_acquisition_fps": cfg["camera_acquisition_fps"],
        "record_fps": FPS,
        "joint_order": list(JOINTS),
        "canonical_front": {"width": 640, "height": 480, "color": "RGB"},
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
    common["configured_termination_reason"] = cfg["termination_reason"]
    common["termination_reason"] = actual_termination_reason
    common["configured_result"] = cfg["result"]
    common["result"] = actual_result
    common["success"] = actual_success
    write_json(root / "provenance/dataset.json", common)
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
