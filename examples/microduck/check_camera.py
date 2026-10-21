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

"""Measure the Microduck camera over the LAN without connecting to robotd."""

import argparse
import asyncio
import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from lerobot.robots.microduck.camera_frame import CameraFrame
from lerobot.robots.microduck.webrtc import receive_video


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


async def measure(args: argparse.Namespace) -> bool:
    stop = threading.Event()
    received: list[float] = []
    media_times: list[float] = []
    latest: CameraFrame | None = None
    next_report = 0.0
    started = time.monotonic()
    preview = getattr(args, "preview", False)
    window = "Microduck camera - Q or Esc to quit"
    window_open = False

    def on_frame(frame: CameraFrame) -> None:
        nonlocal latest, next_report, window_open
        latest = frame
        received.append(frame.received_at)
        if frame.media_time_s is not None:
            media_times.append(frame.media_time_s)
        elapsed = received[-1] - received[0]
        if len(received) == 1:
            print(f"First upright RGB frame: {frame.image.shape}; startup {time.monotonic() - started:.2f}s")
        if elapsed >= next_report and elapsed > 0:
            print(
                f"{elapsed:5.1f}s  frames={len(received):4d}  delivered={(len(received) - 1) / elapsed:.1f} fps"
            )
            next_report = elapsed + 1
        if elapsed >= args.seconds:
            stop.set()
        if preview:
            if not window_open:
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                window_open = True
            height, width = frame.image.shape[:2]
            scale = min(1.0, 720 / height, 1280 / width)
            display = cv2.resize(frame.image, (round(width * scale), round(height * scale)))
            cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                stop.set()

    print(f"Connecting via {args.transport} to {args.host}; camera only (no robotd connection).", flush=True)
    try:
        if args.transport == "webrtc":
            await receive_video(
                f"ws://{args.host}:{args.signalling_port}",
                on_frame,
                stop,
                stall_timeout_s=args.timeout,
            )
        else:
            url = f"http://{args.host}:{args.camera_port}/frame"
            with requests.Session() as session:
                while not stop.is_set():
                    response = await asyncio.to_thread(session.get, url, timeout=args.timeout)
                    response.raise_for_status()
                    image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Snapshot endpoint returned an undecodable image")
                    on_frame(
                        CameraFrame(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), len(received) + 1, time.monotonic()
                        )
                    )
    finally:
        if window_open:
            cv2.destroyWindow(window)
            cv2.waitKey(1)

    if len(received) < 2 or latest is None:
        raise RuntimeError("Fewer than two frames received")
    gaps = np.diff(received)
    fps = (len(received) - 1) / (received[-1] - received[0])
    print(f"Delivered: {fps:.2f} fps across {len(received)} frames")
    print(
        f"Receive gaps: p50={np.percentile(gaps, 50) * 1000:.1f}ms "
        f"p95={np.percentile(gaps, 95) * 1000:.1f}ms max={max(gaps) * 1000:.1f}ms"
    )
    print(
        f"Time since last receive: {(time.monotonic() - latest.received_at) * 1000:.1f}ms (includes shutdown)"
    )
    media_ok = True
    if args.transport == "webrtc":
        media_ok = len(media_times) == len(received) and bool(np.all(np.diff(media_times) > 0))
        print(f"Media timestamps present and strictly increasing: {media_ok}")
        if media_ok:
            print(f"Media cadence: {(len(media_times) - 1) / (media_times[-1] - media_times[0]):.2f} fps")
    print("Receive gaps and relative media timestamps do not measure sensor-to-laptop latency.")
    if args.save_frame:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save_frame), cv2.cvtColor(latest.image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {args.save_frame}")
        print(f"Last upright frame: {args.save_frame.resolve()}")
    passed = fps >= args.min_fps and media_ok
    print(
        f"{'PASS' if passed else 'FAIL'}: minimum {args.min_fps:g} delivered fps"
        + (" and increasing media timestamps" if args.transport == "webrtc" else "")
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Duck LAN IP or hostname")
    parser.add_argument("--transport", choices=("webrtc", "snapshot"), default="webrtc")
    parser.add_argument("--signalling-port", type=int, default=8443)
    parser.add_argument("--camera-port", type=int, default=8080)
    parser.add_argument(
        "--seconds", type=positive, default=30, help="Measurement duration after the first frame"
    )
    parser.add_argument("--timeout", type=positive, default=10, help="Startup/stall or HTTP request timeout")
    parser.add_argument("--min-fps", type=positive, default=4, help="Minimum mean delivered rate to pass")
    parser.add_argument(
        "--preview", action="store_true", help="Show live video; Q/Esc or close the window to quit"
    )
    parser.add_argument("--save-frame", type=Path, help="Optional PNG/JPEG path for an upright RGB sample")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        return 0 if asyncio.run(measure(args)) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
