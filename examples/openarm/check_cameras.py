#!/usr/bin/env python
"""Grab one frame from each rollout camera and save labeled PNGs to confirm mapping.

Uses the *same* V4L2 + MJPG + resolution settings as the rollout command so what you
see is what the policy sees. Run in the lerobot312 env (has cv2):

    python examples/openarm/check_cameras.py

Then open the printed PNG paths and check that left_wrist / right_wrist are correct.
"""

from __future__ import annotations

import os

import cv2

# label -> (device, width, height) — matches the rollout --robot.cameras block
CAMERAS = {
    "left_wrist": ("/dev/video8", 1280, 720),
    "base": ("/dev/video6", 640, 480),
    "right_wrist": ("/dev/video4", 1280, 720),
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP_FRAMES = 12  # let auto-exposure/white-balance settle


def fourcc_str(cap) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def main() -> None:
    for label, (dev, w, h) in CAMERAS.items():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[{label}] {dev}: FAILED to open")
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, f = cap.read()
            if ok:
                frame = f
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = fourcc_str(cap)
        cap.release()

        if frame is None:
            print(f"[{label}] {dev}: opened ({aw}x{ah} {fc}) but no frame read")
            continue

        # Burn the label into the image so the saved file is self-identifying.
        cv2.putText(frame, f"{label}  {dev}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3, cv2.LINE_AA)
        out = os.path.join(OUT_DIR, f"cam_{label}.png")
        cv2.imwrite(out, frame)
        print(f"[{label}] {dev}: {aw}x{ah} {fc}  -> {out}")


if __name__ == "__main__":
    main()
