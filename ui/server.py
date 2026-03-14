"""
LeRobot UI — FastAPI Backend

Manages subprocess lifecycle for lerobot CLI tools, camera streaming via
OpenCV MJPEG, hardware discovery, and a circular log buffer.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import platform
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="LeRobot UI Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MAX_LOG_LINES = 500

_state: dict[str, Any] = {
    "mode": "idle",          # idle | recording | teleoperation | evaluating
    "message": "Ready",
    "error": None,
    "episode_count": 0,
    "subprocess": None,      # active Popen
    "subprocess_lock": threading.Lock(),
    "logs": deque(maxlen=MAX_LOG_LINES),
    "log_lock": threading.Lock(),
    "cameras": {},           # name -> {cap, thread, frame, running}
    "camera_lock": threading.Lock(),
    "pending_cameras": [],   # list of camera dicts to restart after subprocess ends
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    with _state["log_lock"]:
        _state["logs"].append({"ts": _ts(), "msg": msg})


# ---------------------------------------------------------------------------
# Camera streaming
# ---------------------------------------------------------------------------

def _camera_reader(name: str, index_or_path: str | int) -> None:
    """Background thread: reads frames from a camera and stores the latest."""
    # Attempt numeric index first, then string path
    try:
        src: int | str = int(index_or_path)
    except (ValueError, TypeError):
        src = str(index_or_path)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        _log(f"[camera/{name}] Could not open {index_or_path}")
        with _state["camera_lock"]:
            if name in _state["cameras"]:
                _state["cameras"][name]["running"] = False
        return

    _log(f"[camera/{name}] Opened {index_or_path}")

    with _state["camera_lock"]:
        if name in _state["cameras"]:
            _state["cameras"][name]["cap"] = cap

    while True:
        with _state["camera_lock"]:
            if name not in _state["cameras"] or not _state["cameras"][name].get("running"):
                break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with _state["camera_lock"]:
            if name in _state["cameras"]:
                _state["cameras"][name]["frame"] = buf.tobytes()

    cap.release()
    _log(f"[camera/{name}] Stopped")


def _start_cameras(cameras: list[dict]) -> None:
    """Start MJPEG reader threads for a list of camera dicts."""
    with _state["camera_lock"]:
        # Stop any existing camera with same name
        for cam in cameras:
            name = cam["name"]
            if name in _state["cameras"]:
                _state["cameras"][name]["running"] = False

    time.sleep(0.1)  # let threads notice the stop flag

    with _state["camera_lock"]:
        for cam in cameras:
            name = cam["name"]
            entry: dict[str, Any] = {
                "cap": None,
                "thread": None,
                "frame": None,
                "running": True,
                "index_or_path": cam.get("index_or_path", 0),
            }
            t = threading.Thread(
                target=_camera_reader,
                args=(name, cam.get("index_or_path", 0)),
                daemon=True,
            )
            entry["thread"] = t
            _state["cameras"][name] = entry
            t.start()


def _stop_all_cameras() -> None:
    with _state["camera_lock"]:
        for name in list(_state["cameras"].keys()):
            _state["cameras"][name]["running"] = False
    _log("[cameras] All camera streams stopped")


def _mjpeg_generator(name: str):
    """Yield MJPEG frames for a given camera name."""
    while True:
        frame = None
        with _state["camera_lock"]:
            cam = _state["cameras"].get(name)
            if cam:
                frame = cam.get("frame")

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)  # ~30 fps ceiling


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _read_output(proc: subprocess.Popen, label: str) -> None:
    """Read stdout/stderr from a subprocess and feed into log buffer."""
    streams = []
    if proc.stdout:
        streams.append(proc.stdout)
    if proc.stderr:
        streams.append(proc.stderr)

    import select as _select

    while True:
        if proc.poll() is not None:
            # Drain remaining output
            for s in streams:
                for line in s:
                    _log(f"[{label}] {line.rstrip()}")
            break

        try:
            readable, _, _ = _select.select(streams, [], [], 0.1)
            for s in readable:
                line = s.readline()
                if line:
                    _log(f"[{label}] {line.rstrip()}")
        except Exception:
            break


def _watch_subprocess(label: str) -> None:
    """Watch the active subprocess; clean up state when it exits."""
    with _state["subprocess_lock"]:
        proc = _state["subprocess"]

    if proc is None:
        return

    proc.wait()
    rc = proc.returncode
    _log(f"[{label}] Process exited with code {rc}")

    with _state["subprocess_lock"]:
        if _state["subprocess"] is proc:
            _state["subprocess"] = None
            _state["mode"] = "idle"
            _state["message"] = f"Finished (exit {rc})"
            if rc not in (0, -15, -2):  # not clean exit / SIGTERM / SIGINT
                _state["error"] = f"Process exited with code {rc}"

    # Restart camera preview if cameras were pending
    pending = _state.get("pending_cameras", [])
    if pending:
        _log("[cameras] Restarting camera preview after subprocess ended")
        _start_cameras(pending)


def _launch(cmd: list[str], mode: str, label: str) -> None:
    """Launch a subprocess, stop cameras first, set state."""
    with _state["subprocess_lock"]:
        if _state["subprocess"] is not None and _state["subprocess"].poll() is None:
            raise RuntimeError("A process is already running. Stop it first.")

    # Stop cameras so the subprocess can access them
    _stop_all_cameras()

    _log(f"[server] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    with _state["subprocess_lock"]:
        _state["subprocess"] = proc
        _state["mode"] = mode
        _state["message"] = f"Running {label}..."
        _state["error"] = None

    # Pipe output
    t_out = threading.Thread(target=_read_output, args=(proc, label), daemon=True)
    t_out.start()

    # Watch for completion
    t_watch = threading.Thread(target=_watch_subprocess, args=(label,), daemon=True)
    t_watch.start()


# ---------------------------------------------------------------------------
# Hardware discovery
# ---------------------------------------------------------------------------

def _discover_robot_types() -> list[str]:
    try:
        from importlib.util import find_spec

        spec = find_spec("lerobot")
        if spec is None or spec.origin is None:
            return []
        robots_dir = Path(spec.origin).parent / "robots"
        return sorted(
            d.name
            for d in robots_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
    except Exception:
        return []


def _discover_teleop_types() -> list[str]:
    try:
        from importlib.util import find_spec

        spec = find_spec("lerobot")
        if spec is None or spec.origin is None:
            return []
        teleops_dir = Path(spec.origin).parent / "teleoperators"
        return sorted(
            d.name
            for d in teleops_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
    except Exception:
        return []


def _discover_serial_ports() -> list[str]:
    patterns = [
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]
    ports: list[str] = []
    for pat in patterns:
        ports.extend(glob.glob(pat))
    return sorted(set(ports))


def _discover_can_interfaces() -> list[str]:
    try:
        result = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        interfaces = []
        for line in result.stdout.splitlines():
            # Lines like: "3: can0: <...>"
            parts = line.strip().split(":")
            if len(parts) >= 2:
                iface = parts[1].strip()
                if iface.startswith("can"):
                    interfaces.append(iface)
        return sorted(interfaces)
    except Exception:
        return []


def _discover_cameras() -> list[dict]:
    """Scan OpenCV indices 0-9 and /dev/video* paths."""
    found: list[dict] = []
    checked: set = set()

    # Numeric indices
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append({"index_or_path": idx, "label": f"Camera {idx} (index {idx})"})
            checked.add(idx)
        cap.release()

    # /dev/video* paths on Linux
    for path in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            # Avoid duplicate if already found by index
            label = f"Camera ({path})"
            found.append({"index_or_path": path, "label": label})
        cap.release()

    return found


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CameraEntry(BaseModel):
    name: str
    type: str = "opencv"
    index_or_path: str | int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


class PreviewRequest(BaseModel):
    cameras: list[CameraEntry]


class RecordRequest(BaseModel):
    robot_type: str
    robot_port: str = ""
    robot_id: str = ""
    robot_cameras: list[CameraEntry] = []
    teleop_type: str
    teleop_port: str = ""
    teleop_id: str = ""
    repo_id: str
    single_task: str
    num_episodes: int = 10
    fps: int = 30
    episode_time_s: int = 60
    reset_time_s: int = 5
    push_to_hub: bool = True
    private: bool = False
    display_data: bool = False


class TeleopRequest(BaseModel):
    robot_type: str
    robot_port: str = ""
    robot_cameras: list[CameraEntry] = []
    teleop_type: str
    teleop_port: str = ""
    display_data: bool = False


class EvalRequest(BaseModel):
    policy_path: str
    env_type: str
    n_episodes: int = 10
    batch_size: int = 1
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helper: build cameras JSON arg
# ---------------------------------------------------------------------------

def _cameras_arg(cameras: list[CameraEntry]) -> str:
    d: dict[str, dict] = {}
    for cam in cameras:
        d[cam.name] = {
            "type": cam.type,
            "index_or_path": cam.index_or_path,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
        }
    return json.dumps(d)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status() -> dict:
    with _state["subprocess_lock"]:
        proc = _state["subprocess"]
        proc_running = proc is not None and proc.poll() is None

    with _state["camera_lock"]:
        active_cams = [
            name
            for name, cam in _state["cameras"].items()
            if cam.get("running") and cam.get("frame") is not None
        ]

    return {
        "mode": _state["mode"],
        "message": _state["message"],
        "error": _state["error"],
        "episode_count": _state["episode_count"],
        "process_running": proc_running,
        "active_cameras": active_cams,
    }


@app.get("/api/robots")
def list_robots() -> dict:
    return {"types": _discover_robot_types()}


@app.get("/api/teleops")
def list_teleops() -> dict:
    return {"types": _discover_teleop_types()}


@app.get("/api/hardware/serial-ports")
def serial_ports() -> dict:
    return {"ports": _discover_serial_ports()}


@app.get("/api/hardware/can-interfaces")
def can_interfaces() -> dict:
    return {"interfaces": _discover_can_interfaces()}


@app.get("/api/hardware/cameras")
def hardware_cameras() -> dict:
    return {"cameras": _discover_cameras()}


@app.post("/api/cameras/preview")
def start_preview(req: PreviewRequest) -> dict:
    cams = [c.model_dump() for c in req.cameras]
    _state["pending_cameras"] = cams
    _start_cameras(cams)
    return {"status": "ok", "started": len(cams)}


@app.post("/api/cameras/stop")
def stop_preview() -> dict:
    _stop_all_cameras()
    _state["pending_cameras"] = []
    return {"status": "ok"}


@app.get("/api/camera/stream/{name}")
def camera_stream(name: str):
    with _state["camera_lock"]:
        if name not in _state["cameras"]:
            raise HTTPException(status_code=404, detail=f"Camera '{name}' not active")
    return StreamingResponse(
        _mjpeg_generator(name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/record/start")
def start_record(req: RecordRequest) -> dict:
    cmd = ["lerobot-record"]

    cmd += [f"--robot.type={req.robot_type}"]
    if req.robot_port:
        cmd += [f"--robot.port={req.robot_port}"]
    if req.robot_id:
        cmd += [f"--robot.id={req.robot_id}"]
    if req.robot_cameras:
        cmd += [f"--robot.cameras={_cameras_arg(req.robot_cameras)}"]

    cmd += [f"--teleop.type={req.teleop_type}"]
    if req.teleop_port:
        cmd += [f"--teleop.port={req.teleop_port}"]
    if req.teleop_id:
        cmd += [f"--teleop.id={req.teleop_id}"]

    cmd += [
        f"--dataset.repo_id={req.repo_id}",
        f"--dataset.single_task={req.single_task}",
        f"--dataset.num_episodes={req.num_episodes}",
        f"--dataset.fps={req.fps}",
        f"--dataset.episode_time_s={req.episode_time_s}",
        f"--dataset.reset_time_s={req.reset_time_s}",
        f"--dataset.push_to_hub={'true' if req.push_to_hub else 'false'}",
        f"--dataset.private={'true' if req.private else 'false'}",
        f"--display_data={'true' if req.display_data else 'false'}",
    ]

    _state["pending_cameras"] = [c.model_dump() for c in req.robot_cameras]

    try:
        _launch(cmd, "recording", "lerobot-record")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": "ok"}


@app.post("/api/teleop/start")
def start_teleop(req: TeleopRequest) -> dict:
    cmd = ["lerobot-teleoperate"]

    cmd += [f"--robot.type={req.robot_type}"]
    if req.robot_port:
        cmd += [f"--robot.port={req.robot_port}"]
    if req.robot_cameras:
        cmd += [f"--robot.cameras={_cameras_arg(req.robot_cameras)}"]

    cmd += [f"--teleop.type={req.teleop_type}"]
    if req.teleop_port:
        cmd += [f"--teleop.port={req.teleop_port}"]

    cmd += [f"--display_data={'true' if req.display_data else 'false'}"]

    _state["pending_cameras"] = [c.model_dump() for c in req.robot_cameras]

    try:
        _launch(cmd, "teleoperation", "lerobot-teleoperate")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": "ok"}


@app.post("/api/eval/start")
def start_eval(req: EvalRequest) -> dict:
    cmd = [
        "lerobot-eval",
        f"--policy.path={req.policy_path}",
        f"--env.type={req.env_type}",
        f"--eval.n_episodes={req.n_episodes}",
        f"--eval.batch_size={req.batch_size}",
        f"--policy.device={req.device}",
    ]

    try:
        _launch(cmd, "evaluating", "lerobot-eval")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": "ok"}


@app.post("/api/process/stop")
def stop_process() -> dict:
    with _state["subprocess_lock"]:
        proc = _state["subprocess"]
        if proc is None or proc.poll() is not None:
            return {"status": "no_process"}
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    _log("[server] Sent SIGTERM to active process")
    return {"status": "ok"}


@app.post("/api/process/kill")
def kill_process() -> dict:
    with _state["subprocess_lock"]:
        proc = _state["subprocess"]
        if proc is None or proc.poll() is not None:
            return {"status": "no_process"}
        try:
            proc.send_signal(signal.SIGKILL)
        except ProcessLookupError:
            pass
    _log("[server] Sent SIGKILL to active process")
    return {"status": "ok"}


@app.get("/api/logs")
def get_logs() -> dict:
    with _state["log_lock"]:
        return {"logs": list(_state["logs"])}


@app.post("/api/counter/reset")
def reset_counter() -> dict:
    _state["episode_count"] = 0
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _log("[server] LeRobot UI backend starting on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
