from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.picklift_v3.backend import RealSO101Backend, SyntheticBackend
from examples.picklift_v3.camera_profile import validate_camera_profile_config
from examples.picklift_v3.operator_ui import OperatorUI

PRACTICE_ACK = "I_UNDERSTAND_THIS_IS_LIVE_PRACTICE"


def validate_practice_config(cfg: dict) -> None:
    required = (
        "mode",
        "camera_device",
        "camera_profile_id",
        "camera_acquisition_fps",
        "robot_id",
        "follower_port",
        "leader_id",
        "leader_port",
        "control_hz",
        "alignment_mode",
        "startup_hold_s",
    )
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"missing practice configuration: {', '.join(missing)}")
    if cfg["mode"] not in {"real", "synthetic"}:
        raise ValueError("mode must be real or synthetic")
    validate_camera_profile_config(cfg)
    if float(cfg["control_hz"]) < 20:
        raise ValueError("control_hz must be >= 20")
    if cfg["alignment_mode"] not in {"direct_absolute", "relative_rebase"}:
        raise ValueError("unsupported alignment_mode")
    if cfg["mode"] == "real":
        if cfg.get("live_practice_ack") != PRACTICE_ACK:
            raise PermissionError("live practice acknowledgement is missing")
        for key in ("follower_port", "leader_port"):
            if not str(cfg[key]).startswith("/dev/serial/by-id/"):
                raise ValueError(f"{key} must use /dev/serial/by-id")


def run_practice(cfg: dict, backend=None, ui=None, max_seconds: float | None = None) -> dict:
    """Run live Leader-to-Follower control without creating or writing a dataset."""
    validate_practice_config(cfg)
    ui = ui or OperatorUI(target_frames=0)
    period = 1 / float(cfg["control_hz"])
    frames = 0
    started = time.monotonic()
    last_front = None
    ui.open()
    ui.show(
        np.zeros((480, 640, 3), dtype=np.uint8),
        status="WAITING",
        message="Connecting proven collection backend...",
        button_labels=("CONNECTING", "WAIT", "QUIT"),
    )
    print("PRACTICE_UI_READY", flush=True)
    backend = backend or (SyntheticBackend(cfg) if cfg["mode"] == "synthetic" else RealSO101Backend(cfg))
    backend.connect()
    print("PRACTICE_BACKEND_CONNECTED", flush=True)
    try:
        next_tick = time.perf_counter()
        while True:
            state, front = backend.read_pre_action()
            requested = backend.requested_action()
            backend.send_action(requested)
            last_front = front
            frames += 1
            elapsed = time.monotonic() - started
            command = ui.show(
                front,
                status="PRACTICE",
                elapsed_s=elapsed,
                frames=frames,
                message="NO DATA RECORDING\nDirect Leader -> Follower",
                button_labels=("RUNNING", "STOP", "QUIT"),
            )
            if command in {"stop", "quit"}:
                break
            if max_seconds is not None and elapsed >= max_seconds:
                break
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        backend.close()
        if last_front is not None:
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline:
                ui.show(
                    last_front,
                    status="STOPPED",
                    elapsed_s=time.monotonic() - started,
                    frames=frames,
                    message="Torque off | No data saved",
                    button_labels=("STOPPED", "STOPPED", "CLOSE"),
                )
                time.sleep(0.02)
        ui.close()
    return {"frames": frames, "elapsed_s": time.monotonic() - started, "data_recorded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 live practice with front camera preview")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    summary = run_practice(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
