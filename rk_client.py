#!/usr/bin/env python3
"""Command line client for the RobotKinematics HTTP service."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urlopen(request) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def command_fk(args: argparse.Namespace) -> None:
    payload = {"joint_positions": args.joint_positions}
    response = _post_json(f"{args.url}/fk", payload)
    print(json.dumps(response, indent=2))


def command_ik(args: argparse.Namespace) -> None:
    pose_matrix = np.asarray(args.pose, dtype=float).reshape(4, 4).tolist()
    payload = {
        "current_joint_positions": args.current_joint_positions,
        "desired_pose": pose_matrix,
    }
    if args.position_weight is not None:
        payload["position_weight"] = args.position_weight
    if args.orientation_weight is not None:
        payload["orientation_weight"] = args.orientation_weight

    response = _post_json(f"{args.url}/ik", payload)
    print(json.dumps(response, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client for the RobotKinematics HTTP service.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8081",
        help="Base URL for the rk_service (default: http://127.0.0.1:8081)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    fk_parser = subparsers.add_parser("fk", help="Call forward kinematics.")
    fk_parser.add_argument(
        "--joint-positions",
        nargs="+",
        type=float,
        required=True,
        metavar="ANGLE_DEG",
        help="Joint positions in degrees.",
    )
    fk_parser.set_defaults(func=command_fk)

    ik_parser = subparsers.add_parser("ik", help="Call inverse kinematics.")
    ik_parser.add_argument(
        "--current-joint-positions",
        nargs="+",
        type=float,
        required=True,
        metavar="ANGLE_DEG",
        help="Current joint positions in degrees used as the IK seed.",
    )
    ik_parser.add_argument(
        "--pose",
        nargs=16,
        type=float,
        required=True,
        metavar="POSE_VALUE",
        help="Desired end-effector 4x4 pose matrix in row-major order (16 values).",
    )
    ik_parser.add_argument(
        "--position-weight",
        type=float,
        default=None,
        help="Optional position weight for IK.",
    )
    ik_parser.add_argument(
        "--orientation-weight",
        type=float,
        default=None,
        help="Optional orientation weight for IK.",
    )
    ik_parser.set_defaults(func=command_ik)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main(sys.argv[1:])

