#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Simple stereo vision demo using OpenCV cameras and YOLO detections."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


CONFIG_FILE = Path.home() / ".lerobot" / "stereo_vision.json"


def _load_config() -> dict | None:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return None


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg))


def _preview_camera(info: dict, duration: float = 3.0) -> None:
    """Preview a camera for a short time to help selection."""
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=info["id"]))
    try:
        cam.connect()
    except Exception as e:
        print(f"Failed to open camera {info.get('id')}: {e}")
        return

    win_name = f"Preview {info.get('id')}"
    start = time.time()
    try:
        while time.time() - start < duration:
            frame = cam.read()
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) != -1:
                break
    finally:
        cam.disconnect()
        cv2.destroyWindow(win_name)


def _interactive_setup() -> dict:
    print("=== Interactive stereo vision setup ===")
    available = OpenCVCamera.find_cameras()
    for i, info in enumerate(available):
        print(f"[{i}] {info.get('id')} - {info.get('name')}")
        _preview_camera(info)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    left = input("Left camera index or path: ")
    right = input("Right camera index or path: ")
    calibration = input("Stereo calibration .npz path (leave blank if none): ")
    cfg = {"left": left, "right": right, "calibration": calibration or None}
    _save_config(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    # Merged: Bessere Beschreibung vom codex-Branch, aber Logik vom main-Branch.
    parser = argparse.ArgumentParser(
        description="Test stereo vision with YOLO. Run with --setup the first time to store camera configuration."
    )
    parser.add_argument("--left", help="Left camera index or path")
    parser.add_argument("--right", help="Right camera index or path")
    parser.add_argument("--calibration", type=Path, help="Path to stereo calibration .npz file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive setup and ignore saved configuration",
    )

    args = parser.parse_args()

    # Logik zum Laden der Konfiguration und interaktiven Setup ist hier gekapselt.
    cfg = None if args.setup else _load_config()
    if cfg is None:
        cfg = _interactive_setup()

    # Befülle die Argumente aus der Konfiguration, falls sie nicht über die Kommandozeile gesetzt wurden.
    args.left = args.left or cfg.get("left")
    args.right = args.right or cfg.get("right")
    if args.calibration is None and cfg.get("calibration"):
        args.calibration = Path(cfg["calibration"])

    if args.left is None or args.right is None:
        parser.error("Left and right camera indices/paths must be provided")

    return args


def _parse_index_or_path(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def load_calibration(path: Path):
    data = np.load(str(path))
    required = {"M1", "D1", "M2", "D2", "R", "T"}
    if not required.issubset(set(data.keys())):
        raise ValueError(f"Calibration file must contain keys: {sorted(required)}")
    return data


def main() -> None:
    # Merged: Die main-Funktion wurde vereinfacht, da parse_args nun die Konfiguration übernimmt.
    args = parse_args()

    left_id = _parse_index_or_path(args.left)
    right_id = _parse_index_or_path(args.right)

    left_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=left_id, width=args.width, height=args.height))
    right_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=right_id, width=args.width, height=args.height))

    left_cam.connect()
    right_cam.connect()

    model = YOLO(args.model)

    calib = None
    q_matrix = None
    stereo_matcher = None
    if args.calibration:
        calib = load_calibration(args.calibration)
        m1, d1 = calib["M1"], calib["D1"]
        m2, d2 = calib["M2"], calib["D2"]
        r, t = calib["R"], calib["T"]
        img_size = (args.width, args.height)
        r1, r2, p1, p2, q_matrix, _, _ = cv2.stereoRectify(m1, d1, m2, d2, img_size, r, t)
        map1_l, map2_l = cv2.initUndistortRectifyMap(m1, d1, r1, p1, img_size, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(m2, d2, r2, p2, img_size, cv2.CV_32FC1)
        stereo_matcher = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)
    else:
        map1_l = map2_l = map1_r = map2_r = None

    try:
        while True:
            left_img = left_cam.read()
            right_img = right_cam.read()

            if calib is not None:
                left_rect = cv2.remap(left_img, map1_l, map2_l, interpolation=cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_img, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)
            else:
                left_rect, right_rect = left_img, right_img

            results = model(left_rect, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []

            if stereo_matcher is not None:
                gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
                disp = stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
                points_3d = cv2.reprojectImageTo3D(disp, q_matrix)
            else:
                disp = None
                points_3d = None

            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(left_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if points_3d is not None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    x_coord, y_coord, z_coord = points_3d[cy, cx]
                    dist = np.linalg.norm([x_coord, y_coord, z_coord])
                    cv2.putText(
                        left_rect,
                        f"{dist:.2f}m",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            combined = np.hstack((left_rect, right_rect))
            cv2.imshow("Stereo", combined)

            if cv2.waitKey(1) in [27, ord("q")]:
                break
    finally:
        left_cam.disconnect()
        right_cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()