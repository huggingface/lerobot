#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud side: a minimal proof that WebRTCProxyRobot drives a REMOTE arm.

WebRTCProxyRobot is a drop-in lerobot ``Robot``: ``get_observation()`` returns the
remote arm's joints + camera (over WebRTC) and ``send_action()`` moves the remote
motors. Self-contained (no extra teleoperator dependency), two control front-ends:

- ``--mode web`` (default): serves ``panel.html`` from the stdlib HTTP server —
  a **live camera view** (MJPEG) + per-joint jog buttons + joint readout.
- ``--mode console``: keyboard jog in the terminal (1..6 select, +/- jog, q quit).

Either way it's the same plain ``send_action`` / ``get_observation`` loop. Seeing the
remote camera move when you press a button is the end-to-end proof the link works.

    Run after the relay + robot daemon are up (see README):
        uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py            # web
        uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py --mode console
"""

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig
from lerobot.robots.webrtc_proxy.proxy_robot import CameraLayoutMismatchError, WebRTCProxyRobot

SIGNALING_URL = "ws://127.0.0.1:8765/ws"
SESSION_ID = "so100"
FPS, WIDTH, HEIGHT = 30, 640, 480
WEB_PORT = 8088
JOG_DEG = 3.0
HOME_RATE_DPS = 30.0  # "reset to home" ramps to 0 at this deg/sec (gentle, not a snap)

_PANEL_HTML = (Path(__file__).parent / "panel.html").read_text(encoding="utf-8")

# Shared state: the control loop (single writer) touches the robot; the web handlers /
# console reader only read frames/joints and nudge targets.
_targets: dict[str, float] = {}
_latest_frame: dict[str, np.ndarray | None] = {"img": None}
_latest_joints: dict[str, float] = {}
_motors: list[str] = []
_cam_names: list[str] = []  # one or more cameras; the panel hconcats them into one view
_homing = {"on": False}  # when True, the control loop ramps every target toward 0
_sel = {"i": 0}
_running = True


# ---- web front-end --------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN002 - quiet
        pass

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path)
        if path.path == "/":
            self._bytes("text/html; charset=utf-8", _PANEL_HTML.encode())
        elif path.path == "/config":
            self._json({"motors": _motors, "step": JOG_DEG})
        elif path.path == "/state":
            self._json({m: round(float(v), 2) for m, v in _latest_joints.items()})
        elif path.path == "/stream":
            self._stream()
        elif path.path == "/jog":
            q = parse_qs(path.query)
            key = f"{q.get('motor', [''])[0]}.pos"
            if key in _targets:
                _homing["on"] = False  # manual jog cancels an in-progress home
                _targets[key] += float(q.get("delta", ["0"])[0])
            self._json({"ok": True})
        elif path.path == "/home":
            _homing["on"] = True  # ramp to 0 gradually in the control loop (HOME_RATE_DPS)
            self._json({"ok": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _bytes(self, content_type: str, body: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._bytes("application/json", json.dumps(obj).encode())

    def _stream(self):
        # MJPEG: the remote camera frames, live in the browser <img>.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while _running:
                img = _latest_frame["img"]
                if img is None:
                    time.sleep(0.05)
                    continue
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=70)
                data = buf.getvalue()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ---- console front-end ----------------------------------------------------
def _console_reader() -> None:
    global _running
    print("\ncommands: " + "  ".join(f"{i + 1}={m}" for i, m in enumerate(_motors)))
    print("          +/- jog selected joint   q quit\n")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "q":
            _running = False
            break
        if cmd.isdigit() and 1 <= int(cmd) <= len(_motors):
            _sel["i"] = int(cmd) - 1
        elif cmd in ("+", "="):
            _targets[f"{_motors[_sel['i']]}.pos"] += JOG_DEG
        elif cmd == "-":
            _targets[f"{_motors[_sel['i']]}.pos"] -= JOG_DEG
        sel = _motors[_sel["i"]]
        print(f"  -> {sel} target = {_targets[f'{sel}.pos']:.1f} deg")


# ---- shared control loop --------------------------------------------------
def _control_loop(robot: WebRTCProxyRobot) -> None:
    """Stream the absolute target every tick (arm tracks it AND the robot watchdog never
    safe-stops), and refresh the latest obs (joints + camera) for the front-end."""
    try:
        while _running:
            if _homing["on"]:  # ease every joint toward 0 by a bounded step per tick
                step = HOME_RATE_DPS / FPS
                done = True
                for k in _targets:
                    cur = _targets[k]
                    if abs(cur) <= step:
                        _targets[k] = 0.0
                    else:
                        _targets[k] = cur - step if cur > 0 else cur + step
                        done = False
                if done:
                    _homing["on"] = False
            robot.send_action(dict(_targets))  # action -> WebRTC -> robot motors
            o = robot.get_observation()  # joints + camera(s) <- WebRTC <- robot
            imgs = [o[n] for n in _cam_names if o.get(n) is not None]
            if imgs:
                try:
                    _latest_frame["img"] = imgs[0] if len(imgs) == 1 else np.concatenate(imgs, axis=1)
                except ValueError:  # mismatched heights — fall back to the first camera
                    _latest_frame["img"] = imgs[0]
            _latest_joints.update({m: float(o[f"{m}.pos"]) for m in _motors})
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        pass


def main() -> None:
    global _running
    parser = argparse.ArgumentParser(description="Cloud-side teleop for a remote SO-100 over WebRTC")
    parser.add_argument("--mode", choices=["web", "console"], default="web")
    parser.add_argument(
        "--session", default=SESSION_ID, help="session id == LiveKit room; must match the daemon's --session"
    )
    parser.add_argument("--web-port", type=int, default=WEB_PORT, help=f"web panel port (default {WEB_PORT})")
    parser.add_argument(
        "--transport", choices=["aiortc", "livekit"], default="aiortc", help="transport backend"
    )
    parser.add_argument("--signaling-url", default=SIGNALING_URL, help="WS relay URL (aiortc)")
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("SIGNALING_AUTH_TOKEN"),
        help="shared token for an authed signaling relay (aiortc; default $SIGNALING_AUTH_TOKEN)",
    )
    parser.add_argument(
        "--livekit-url", default=os.environ.get("LIVEKIT_URL"), help="LiveKit URL (default $LIVEKIT_URL)"
    )
    parser.add_argument(
        "--livekit-token", default=None, help="pre-signed JWT; omit to self-sign from api key/secret"
    )
    parser.add_argument(
        "--livekit-api-key", default=os.environ.get("LIVEKIT_API_KEY"), help="default $LIVEKIT_API_KEY"
    )
    parser.add_argument(
        "--livekit-api-secret", default=os.environ.get("LIVEKIT_API_SECRET"), help="$LIVEKIT_API_SECRET"
    )
    parser.add_argument("--livekit-identity", default="controller", help="this controller's LiveKit identity")
    parser.add_argument(
        "--cameras",
        default="front",
        help="camera set, comma-separated; must match the robot daemon's names. "
        f"Each is 'name' (default {WIDTH}x{HEIGHT}) or 'name:WxH'. e.g. 'front,wrist:640x480'",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    livekit_token = args.livekit_token
    if args.transport == "livekit":
        if not args.livekit_url:
            parser.error("--livekit-url (or $LIVEKIT_URL) is required for --transport livekit")
        if not livekit_token:
            if not args.livekit_api_key or not args.livekit_api_secret:
                parser.error(
                    "--transport livekit needs --livekit-token, or --livekit-api-key + "
                    "--livekit-api-secret (or $LIVEKIT_API_KEY/$LIVEKIT_API_SECRET) to self-sign"
                )
            from lerobot.robots.webrtc_proxy.transport_livekit import make_livekit_token

            # Room == session id; must match the daemon's --session.
            livekit_token = make_livekit_token(
                api_key=args.livekit_api_key,
                api_secret=args.livekit_api_secret,
                identity=args.livekit_identity,
                room=args.session,
            )

    # camera set must match the robot daemon (same names + per-camera height/width) so the
    # tiled video de-tiles correctly. 'name' => default res; 'name:WxH' => custom.
    cams_cfg: dict[str, WebRTCCameraSpec] = {}
    for item in (s.strip() for s in args.cameras.split(",")):
        if not item:
            continue
        if ":" in item:
            name, res = item.split(":", 1)
            w, h = (int(x) for x in res.lower().split("x"))
        else:
            name, h, w = item, HEIGHT, WIDTH
        cams_cfg[name.strip()] = WebRTCCameraSpec(height=h, width=w, fps=FPS)
    _cam_names[:] = list(cams_cfg)

    robot = WebRTCProxyRobot(
        WebRTCProxyRobotConfig(
            cameras=cams_cfg,
            signaling_url=args.signaling_url,
            signaling_token=args.auth_token,
            session_id=args.session,
            capture_fps=FPS,
            action_timeout_s=0.5,
            transport_backend=args.transport,
            livekit_url=args.livekit_url,
            livekit_token=livekit_token,
        )
    )
    robot.connect()
    _motors[:] = robot.motors
    try:
        obs = robot.get_observation()  # first obs validates the camera layout vs the robot
    except CameraLayoutMismatchError as e:
        # Friendly, actionable exit instead of a deep cv2.resize traceback.
        print(f"\n  ✗ {e}\n", file=sys.stderr)
        print(
            f"  This side:  --cameras {args.cameras}\n"
            "  Set the daemon's --cameras to the same names + resolutions (or fix this side),\n"
            "  then re-run. Tip: a single shared camera is the easiest to get working first.\n",
            file=sys.stderr,
        )
        robot.disconnect()
        sys.exit(2)
    _targets.update({f"{m}.pos": float(obs[f"{m}.pos"]) for m in _motors})  # hold current pose

    server = None
    if args.mode == "web":
        server = ThreadingHTTPServer(("0.0.0.0", args.web_port), _Handler)  # noqa: S104
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(
            f"\n  open http://localhost:{args.web_port}  — live remote camera + jog buttons. Ctrl-C to stop.\n"
        )
    else:
        threading.Thread(target=_console_reader, daemon=True).start()

    try:
        _control_loop(robot)
    finally:
        _running = False
        if server is not None:
            server.shutdown()
        robot.disconnect()


if __name__ == "__main__":
    main()
