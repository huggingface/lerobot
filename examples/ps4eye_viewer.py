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

"""ps4eye_viewer.py — Live frame viewer for the Sony PS4 Eye stereo camera.

Opens a PS4EyeCamera and displays frames in a window using OpenCV's imshow.
Press **q** or **Ctrl-C** to exit cleanly.

Usage::

    python examples/ps4eye_viewer.py --index 1 --eye left
    python examples/ps4eye_viewer.py --index 1 --eye right
    python examples/ps4eye_viewer.py --index 1 --eye both   # full panoramic

Two-eye demo (shared capture — device is opened only once)::

    python examples/ps4eye_viewer.py --index 1 --eye left  &
    python examples/ps4eye_viewer.py --index 1 --eye right

Find your device index first::

    lerobot-find-cameras ps4eye
"""

import argparse
import signal
import sys

import cv2

from lerobot.cameras.ps4eye import PS4EyeCamera, PS4EyeCameraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live viewer for the Sony PS4 Eye stereo camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="OpenCV device index for the PS4 Eye (use `lerobot-find-cameras ps4eye` to find it).",
    )
    parser.add_argument(
        "--eye",
        choices=["left", "right", "both"],
        default="left",
        help="Which stereo eye to display: left slice, right slice, or full panoramic.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3448,
        help="Raw panoramic frame width reported by the camera (3448 or 1748).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=808,
        help="Raw panoramic frame height reported by the camera (808 or 408).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target capture frame rate.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Warmup time in seconds (discards initial dark frames).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PS4EyeCameraConfig(
        index_or_path=args.index,
        eye=args.eye,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_s=int(args.warmup),
        color_mode="bgr",  # cv2.imshow expects BGR
    )

    cam = PS4EyeCamera(config)

    # Allow Ctrl-C to trigger a clean shutdown
    _running = [True]

    def _handle_sigint(signum, frame):  # noqa: ARG001
        _running[0] = False

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"Connecting to PS4 Eye index={args.index}, eye={args.eye} …")
    try:
        cam.connect()
    except ConnectionError as err:
        print(f"[ERROR] Could not open camera: {err}", file=sys.stderr)
        print("Tip: run `lerobot-find-cameras ps4eye` to list available devices.", file=sys.stderr)
        sys.exit(1)

    window_title = f"PS4 Eye [{args.eye}] — index {args.index}  |  press 'q' to quit"
    print(f"Streaming … {window_title}")

    try:
        while _running[0]:
            frame = cam.read()  # (H, W, 3) BGR numpy array

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
