#!/usr/bin/env python3
"""Simple HTTP service exposing RobotKinematics forward and inverse kinematics.

Run on a machine that has the `placo` dependency installed so RobotKinematics works:

    python rk_service.py --urdf-path /path/to/robot.urdf --port 8000

Endpoints:
    POST /fk
        Body: {"joint_positions": [deg, ...]}
        Response: {"pose": [[...], ...]}

    POST /ik
        Body: {
            "current_joint_positions": [deg, ...],
            "desired_pose": [[...], ...],
            "position_weight": 1.0,               # optional
            "orientation_weight": 0.01            # optional
        }
        Response: {"joint_positions": [deg, ...]}
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np

from lerobot.model.kinematics import RobotKinematics


LOGGER = logging.getLogger("rk_service")


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    """Serialize payload and send it as the HTTP response."""
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class KinematicsRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler exposing RobotKinematics operations."""

    kinematics: ClassVar[RobotKinematics]
    suppress_logging: ClassVar[bool] = False

    def log_message(self, format: str, *args: Any) -> None:  # noqa: D401 - `BaseHTTPRequestHandler` signature
        if not self.suppress_logging:
            super().log_message(format, *args)

    def do_GET(self) -> None:  # noqa: N802 D401 - `BaseHTTPRequestHandler` signature
        if self.path == "/healthz":
            _json_response(self, HTTPStatus.OK, {"status": "ok"})
        else:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

    def do_POST(self) -> None:  # noqa: N802 D401 - `BaseHTTPRequestHandler` signature
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.exception("Failed to decode request body as JSON")
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
            return

        if self.path == "/fk":
            self._handle_forward_kinematics(payload)
        elif self.path == "/ik":
            self._handle_inverse_kinematics(payload)
        else:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

    def _handle_forward_kinematics(self, payload: dict[str, Any]) -> None:
        try:
            joints = payload["joint_positions"]
            joint_array = np.asarray(joints, dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"Invalid joint_positions: {exc}"})
            return

        try:
            pose_matrix = self.kinematics.forward_kinematics(joint_array)
        except Exception as exc:  # placo may raise C++ exceptions that surface as generic Exception
            LOGGER.exception("Forward kinematics failed")
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"forward_kinematics failed: {exc}"})
            return

        _json_response(self, HTTPStatus.OK, {"pose": pose_matrix.tolist()})

    def _handle_inverse_kinematics(self, payload: dict[str, Any]) -> None:
        try:
            current = np.asarray(payload["current_joint_positions"], dtype=float)
            desired_pose = np.asarray(payload["desired_pose"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"error": f"Invalid current_joint_positions or desired_pose: {exc}"},
            )
            return

        if desired_pose.shape != (4, 4):
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"error": f"desired_pose must be 4x4 matrix, got shape {desired_pose.shape}"},
            )
            return

        position_weight = float(payload.get("position_weight", 1.0))
        orientation_weight = float(payload.get("orientation_weight", 0.01))

        try:
            joint_solution = self.kinematics.inverse_kinematics(
                current_joint_pos=current,
                desired_ee_pose=desired_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
        except Exception as exc:  # placo errors
            LOGGER.exception("Inverse kinematics failed")
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"inverse_kinematics failed: {exc}"})
            return

        _json_response(self, HTTPStatus.OK, {"joint_positions": joint_solution.tolist()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose RobotKinematics via a simple HTTP service.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind the HTTP server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Port for the HTTP server (default: 8081)")
    parser.add_argument("--urdf-path", default="Simulation/SO101/so101_new_calib.urdf", help="Path to the robot URDF file.")
    parser.add_argument(
        "--target-frame",
        default="gripper_frame_link",
        help="End-effector frame name (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=None,
        help="Optional list of joint names; defaults to URDF order when omitted.",
    )
    parser.add_argument(
        "--suppress-request-logs",
        action="store_true",
        help="Disable per-request logging emitted by the HTTP handler.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    LOGGER.info("Loading RobotKinematics with URDF: %s", args.urdf_path)
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=args.joint_names,
    )

    KinematicsRequestHandler.kinematics = kinematics
    KinematicsRequestHandler.suppress_logging = bool(args.suppress_request_logs)

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, KinematicsRequestHandler)
    LOGGER.info("RobotKinematics service listening on http://%s:%d", args.host, args.port)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down RobotKinematics service.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

