"""Live ROI picker for UR10 HIL-SERL training.

Opens every camera declared in ``env.robot.cameras`` of an env JSON (same configs the
runtime reads), shows a live preview at the camera's NATIVE resolution, lets you draw
one rectangular ROI per camera, and prints a ``crop_params_dict`` snippet ready to paste
into the ``processor.image_preprocessing`` block of that JSON.

The crop coords are in pixels on the native frame -- exactly what
``ImageCropResizeProcessorStep`` expects (it crops first, then resizes to
``resize_size``). See ``src/lerobot/processor/hil_processor.py:217-222``.

Usage:
    python act_train/pick_crop_roi.py --config src/lerobot/rl/ur10_env_3cams.json

Controls per camera window:
    drag left mouse  -- draw rectangle
    c                -- confirm and move to next camera
    r                -- reset selection on current camera
    s                -- skip this camera (no crop entry written)
    q / ESC          -- abort
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


def _build_cameras(cam_specs: dict[str, dict]):
    """Instantiate cameras from the same dict layout the env JSON uses."""
    cams: dict[str, object] = {}
    for name, spec in cam_specs.items():
        spec = dict(spec)
        cam_type = spec.pop("type")
        if cam_type == "intelrealsense":
            cfg = RealSenseCameraConfig(**spec)
            cams[name] = RealSenseCamera(cfg)
        elif cam_type == "opencv":
            cfg = OpenCVCameraConfig(**spec)
            cams[name] = OpenCVCamera(cfg)
        else:
            raise ValueError(f"Unsupported camera type for ROI picker: {cam_type}")
    return cams


def _reset_realsense(cam_specs: dict[str, dict]) -> None:
    """Same hardware-reset preflight UR10Robot.connect runs."""
    serials = [
        s["serial_number_or_name"]
        for s in cam_specs.values()
        if s.get("type") == "intelrealsense"
    ]
    if not serials:
        return
    import pyrealsense2 as rs

    ctx = rs.context()
    reset_any = False
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) in serials:
            print(f"hardware_reset {dev.get_info(rs.camera_info.serial_number)}")
            dev.hardware_reset()
            reset_any = True
    if reset_any:
        time.sleep(5.0)


class _RoiSelector:
    """Mouse callback state for one camera window."""

    def __init__(self) -> None:
        self.drawing = False
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.roi: tuple[int, int, int, int] | None = None  # (top, left, height, width)

    def callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.current = (x, y)
            self.roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current = (x, y)
            assert self.start is not None
            x0, y0 = self.start
            x1, y1 = x, y
            top, bottom = sorted((y0, y1))
            left, right = sorted((x0, x1))
            self.roi = (top, left, bottom - top, right - left)

    def reset(self) -> None:
        self.__init__()


def _pick_roi(name: str, cam) -> tuple[int, int, int, int] | None:
    win = f"ROI [{name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sel = _RoiSelector()
    cv2.setMouseCallback(win, sel.callback)

    print(
        f"\n[{name}] drag to draw, c=confirm, r=reset, s=skip, q/ESC=abort"
    )

    while True:
        frame = cam.async_read()
        if frame is None:
            continue
        # OpenCVCamera returns RGB by default -- convert for cv2 display.
        if frame.ndim == 3 and frame.shape[2] == 3:
            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            disp = frame.copy()

        if sel.drawing and sel.start is not None and sel.current is not None:
            cv2.rectangle(disp, sel.start, sel.current, (0, 255, 0), 2)
        elif sel.roi is not None:
            top, left, h, w = sel.roi
            cv2.rectangle(disp, (left, top), (left + w, top + h), (0, 255, 0), 2)
            cv2.putText(
                disp,
                f"top={top} left={left} h={h} w={w}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return "ABORT"  # type: ignore[return-value]
        if key == ord("r"):
            sel.reset()
        if key == ord("s"):
            cv2.destroyWindow(win)
            return None
        if key == ord("c") and sel.roi is not None and sel.roi[2] > 0 and sel.roi[3] > 0:
            cv2.destroyWindow(win)
            return sel.roi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to env JSON whose env.robot.cameras block defines the cameras.",
    )
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = json.load(f)
    cam_specs = cfg["env"]["robot"]["cameras"]

    _reset_realsense(cam_specs)
    cams = _build_cameras(cam_specs)

    rois: dict[str, tuple[int, int, int, int]] = {}
    try:
        for name, cam in cams.items():
            cam.connect()
        for name, cam in cams.items():
            roi = _pick_roi(name, cam)
            if roi == "ABORT":  # type: ignore[comparison-overlap]
                print("aborted; nothing written")
                return 1
            if roi is None:
                print(f"[{name}] skipped")
                continue
            rois[f"observation.images.{name}"] = roi
    finally:
        for cam in cams.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        cv2.destroyAllWindows()

    print("\n--- paste this into env.processor.image_preprocessing ---\n")
    snippet = {
        "crop_params_dict": {k: list(v) for k, v in rois.items()},
        "resize_size": [128, 128],
    }
    print(json.dumps(snippet, indent=4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
