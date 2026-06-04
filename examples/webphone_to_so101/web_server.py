"""FastAPI WebSocket server for web-based phone teleoperation.

Replaces AR/VR with DeviceOrientationEvent sensor data.
Provides a web UI with QR code, sensor status, gripper buttons,
sensitivity slider, and enable/disable toggle.
"""

import json
import logging
import math
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
import uvicorn.config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
STATIC_DIR = THIS_DIR / "static"
CERT_FILE = THIS_DIR / "cert.pem"
KEY_FILE = THIS_DIR / "key.pem"


def _get_local_ip() -> str:
    """Get the LAN IP address, preferring physical interfaces over virtual ones."""
    # Try hostname -I first (Linux), filters to common private IP ranges
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
        for ip in result.stdout.strip().split():
            if ip.startswith(("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                              "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                              "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                              "172.29.", "172.30.", "172.31.")):
                return ip
    except Exception:
        pass

    # Fallback: connect-based detection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _free_port(port: int) -> None:
    """Attempt to free a port by killing any process using it."""
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
            logger.info(f"Freed port {port}")
    except Exception:
        pass


def _generate_self_signed_cert() -> None:
    """Generate self-signed SSL certificate if it doesn't exist."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    logger.info(f"Generated self-signed cert: {CERT_FILE}, {KEY_FILE}")


class WebPhoneServer:
    """FastAPI server that receives device orientation data via WebSocket.

    Provides a get_action() interface compatible with the existing robot
    processing pipeline (same format as Phone.get_action()).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4443,
    ):
        self._host = host
        self._port = port
        self._app = FastAPI()

        # Allow sensor APIs on all pages (required for Generic Sensor API)
        @self._app.middleware("http")
        async def _sensor_headers(request, call_next):
            response = await call_next(request)
            response.headers["Permissions-Policy"] = (
                "accelerometer=self, gyroscope=self, magnetometer=self"
            )
            return response

        # Shared state (protected by _lock)
        self._lock = threading.Lock()
        self._latest_sensor: dict | None = None
        self._is_connected = False

        # Calibration
        self._calib_alpha: float | None = None
        self._calib_beta: float | None = None
        self._calib_gamma: float | None = None
        self._is_calibrated = False

        # Control state from client
        self._enabled = False
        self._gripper_open = True
        self._sensitivity = 1.0

        # Previous enabled state for rising-edge calibration
        self._prev_enabled = False

        self._setup_routes()

    # ── routes ────────────────────────────────────────────────────

    def _setup_routes(self) -> None:
        self._app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @self._app.get("/")
        async def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket client connected")
            with self._lock:
                self._is_connected = True

            try:
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    self._handle_message(msg)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            finally:
                with self._lock:
                    self._is_connected = False

    def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type != "sensor":
            return

        data = msg.get("data", {})
        with self._lock:
            self._latest_sensor = data

            # Update control state
            self._enabled = bool(data.get("enabled", False))
            self._gripper_open = not bool(data.get("gripper_vel", 0) > 0)
            self._sensitivity = float(data.get("sensitivity", 1.0))

            # Rising edge: recalibrate
            if self._enabled and not self._prev_enabled:
                if self._latest_sensor:
                    self._calib_alpha = float(self._latest_sensor.get("alpha", 0))
                    self._calib_beta = float(self._latest_sensor.get("beta", 0))
                    self._calib_gamma = float(self._latest_sensor.get("gamma", 0))
                    self._is_calibrated = True
                    logger.info(
                        f"Calibrated: alpha={self._calib_alpha:.1f}, "
                        f"beta={self._calib_beta:.1f}, gamma={self._calib_gamma:.1f}"
                    )
            self._prev_enabled = self._enabled

    # ── public interface ─────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    def get_action(self) -> dict:
        """Return action dict in same format as Phone.get_action().

        Always returns a valid dict so gripper commands pass through
        even when teleop is disabled or uncalibrated.
        """
        with self._lock:
            sensor = self._latest_sensor
            enabled = self._enabled
            gripper_open = self._gripper_open
            sensitivity = self._sensitivity
            calibrated = self._is_calibrated

        raw_inputs = {
            "move": True,
            "reservedButtonA": not gripper_open,
            "reservedButtonB": gripper_open,
        }

        if not sensor or not calibrated or not enabled:
            return {
                "phone.pos": np.zeros(3, dtype=np.float32),
                "phone.rot": Rotation.from_matrix(np.eye(3)),
                "phone.raw_inputs": raw_inputs,
                "phone.enabled": enabled and calibrated,
            }

        alpha = float(sensor.get("alpha", 0))
        beta = float(sensor.get("beta", 0))
        gamma = float(sensor.get("gamma", 0))

        # Relative angles from calibration, scaled by sensitivity
        d_alpha = math.radians(alpha - self._calib_alpha) * sensitivity
        d_beta = math.radians(beta - self._calib_beta) * sensitivity
        d_gamma = math.radians(gamma - self._calib_gamma) * sensitivity

        # Tait-Bryan ZXY → rotation matrix
        ca, sa = math.cos(d_alpha), math.sin(d_alpha)
        cb, sb = math.cos(d_beta), math.sin(d_beta)
        cg, sg = math.cos(d_gamma), math.sin(d_gamma)

        # R = Rz(alpha) * Rx(beta) * Ry(gamma)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        Ry = np.array([[cg, 0, sg], [0, 1, 0], [-sg, 0, cg]])

        R = Rz @ Rx @ Ry
        rot = Rotation.from_matrix(R)

        return {
            "phone.pos": np.zeros(3, dtype=np.float32),
            "phone.rot": rot,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": enabled,
        }

    def connect(self) -> None:
        """Start the server (non-blocking, runs in thread)."""
        _generate_self_signed_cert()
        local_ip = _get_local_ip()

        # Kill any lingering process on our port before starting
        _free_port(self._port)

        def _run():
            config = uvicorn.Config(
                self._app,
                host=self._host,
                port=self._port,
                ssl_keyfile=str(KEY_FILE),
                ssl_certfile=str(CERT_FILE),
                log_level="warning",
            )
            server = uvicorn.Server(config)
            # Enable SO_REUSEADDR for quick restart
            server.install_signal_handlers = lambda: None
            self._uvicorn_server = server
            server.run()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        # Give thread time to claim port
        time.sleep(0.5)

        print(f"\n{'=' * 60}")
        print(f"  WebPhone server started!")
        print(f"  Open in browser: https://{local_ip}:{self._port}")
        print(f"  Or scan the QR code on the page")
        print(f"{'=' * 60}\n")

    def disconnect(self) -> None:
        """Stop the server."""
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
            self._uvicorn_server.should_exit = True
