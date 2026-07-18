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

"""Lightweight side-by-side viewer for the G1 robot's ZMQ camera streams.

A fast cv2 alternative to ``--display_data`` (rerun), for eyeballing exactly what
the robot is streaming. Subscribes to the same ZMQ PUB server as teleop (PUB/SUB,
so it runs concurrently), and shows head + wrists in one window.

Example (run alongside teleop, no ``--display_data`` needed):

    python -m lerobot.robots.unitree_g1.view_cameras --server-address 172.18.130.111

Notes:
    - Needs a GUI-capable OpenCV build (``pip install opencv-python``, not the
      ``-headless`` variant) since it uses ``cv2.imshow``.
    - Camera names/resolutions default to what ``run_g1.sh`` streams.
"""

import argparse
import contextlib
import logging

import cv2
import numpy as np

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("g1_view_cameras")

# name -> (native width, native height) as served by run_g1.sh
DEFAULT_CAMERAS: dict[str, tuple[int, int]] = {
    "head_camera": (640, 480),
    "left_wrist": (1280, 720),
    "right_wrist": (1280, 720),
}


def _placeholder(name: str, pane_h: int) -> np.ndarray:
    img = np.zeros((pane_h, int(pane_h * 4 / 3), 3), dtype=np.uint8)
    cv2.putText(img, f"{name}: no frame", (10, pane_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-address", required=True, help="Robot IP running the ZMQ camera server")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--pane-height", type=int, default=360, help="Display height per camera pane (px)")
    p.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names to show (must match the server).",
    )
    args = p.parse_args()

    names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    cams: dict[str, ZMQCamera] = {}
    for name in names:
        w, h = DEFAULT_CAMERAS.get(name, (None, None))
        cfg = ZMQCameraConfig(
            server_address=args.server_address,
            port=args.port,
            camera_name=name,
            width=w,
            height=h,
            fps=args.fps,
            warmup_s=5,
        )
        cam = ZMQCamera(cfg)
        logger.info("Connecting to %s @ %s:%d ...", name, args.server_address, args.port)
        cam.connect()
        cams[name] = cam
    logger.info("All cameras connected. Press 'q' or ESC in the window to quit.")

    win = "G1 cameras (q/ESC to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            panes = []
            for name, cam in cams.items():
                frame = None
                with contextlib.suppress(Exception):
                    frame = cam.read_latest(max_age_ms=2000)
                if frame is None:
                    panes.append(_placeholder(name, args.pane_height))
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                scale = args.pane_height / bgr.shape[0]
                disp = cv2.resize(bgr, (int(bgr.shape[1] * scale), args.pane_height))
                cv2.putText(disp, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                panes.append(disp)

            if panes:
                cv2.imshow(win, cv2.hconcat(panes))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for cam in cams.values():
            with contextlib.suppress(Exception):
                cam.disconnect()


if __name__ == "__main__":
    main()
