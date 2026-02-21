#!/usr/bin/env python3
"""
Simple camera viewer for one camera index.

Usage examples:
  python src/lerobot/scripts/lerobot_show_camera.py --robot.camera.index 0
  python -m lerobot.scripts.lerobot_show_camera --robot.camera.index 1 --fps 30

Press 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show camera frames for a single camera index.")
    parser.add_argument("--robot.camera.index", type=int, default=0,
                        help="Camera index (same as OpenCV VideoCapture index).")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Requested display frames per second (used to set wait/delay).")
    parser.add_argument("--width", type=int, default=0,
                        help="Optional: set capture width (pixels).")
    parser.add_argument("--height", type=int, default=0,
                        help="Optional: set capture height (pixels).")
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        help="Optional crop to apply to frames: top left height width (pixels).")
    parser.add_argument("--crop-file", type=str, default=None,
                        help="Optional JSON file containing a crop mapping (first entry will be used).")
    parser.add_argument("--window-name", default="Camera",
                        help="Window title for the viewer.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    idx = args.__dict__.get("robot.camera.index") if "robot.camera.index" in args.__dict__ else args.robot_camera_index

    try:
        idx = int(idx)
    except Exception:
        print("Invalid camera index:", idx, file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Unable to open camera index {idx}", file=sys.stderr)
        return 3

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = args.window_name
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    delay_ms = int(max(1, round(1000.0 / float(max(1.0, args.fps)))))
    last_print = time.time()
    frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # If capture temporarily fails, wait a bit and continue
                time.sleep(0.1)
                continue

            # Apply optional crop (top, left, height, width)
            if args.crop is not None:
                top, left, h, w = args.crop
                # guard bounds
                H, W = frame.shape[:2]
                top = max(0, min(top, H - 1))
                left = max(0, min(left, W - 1))
                h = max(1, min(h, H - top))
                w = max(1, min(w, W - left))
                frame = frame[top : top + h, left : left + w]

            cv2.imshow(window, frame)
            frames += 1

            # Print simple FPS once per 2 seconds
            if time.time() - last_print >= 2.0:
                print(f"Camera {idx} — approx display FPS: {frames / max(1e-6, time.time() - last_print):.1f}")
                last_print = time.time()
                frames = 0

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
