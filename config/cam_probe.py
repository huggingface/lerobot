#!/usr/bin/env python3
"""Measure a V4L2 camera's capture latency and buffer depth. No lerobot, no network.

    ./config/cam-probe.sh /dev/cam_left

Opens the device the way lerobot does (MJPG, 640x480, 30 fps) and runs
three measurements:

  1. rate     - read continuously for 3 s. Real fps, and per-read time.
  2. buffer   - stop reading for 5 s, then resume and count how many
                frames come back faster than a camera could possibly
                produce them. That count is the buffer depth: frames that
                were already queued, i.e. STALE. Multiply by the frame
                period for the worst-case latency this camera can hide.
  3. drain    - keep reading after the buffer empties and confirm the
                rate settles back to live.

A healthy V4L2 device holds 2-4 frames (under 150 ms at 30 fps). A number
in the tens or hundreds means something between the sensor and read() is
queueing deeply, and anything consuming these frames - the operator view,
a recording - is seeing the past.
"""

import argparse
import sys
import time

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-mjpg", action="store_true")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds to stop reading")
    ap.add_argument("--buffersize", type=int, default=None, help="try to set CAP_PROP_BUFFERSIZE")
    args = ap.parse_args()

    cv2.setNumThreads(1)  # as lerobot does
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"error: cannot open {args.device}", file=sys.stderr)
        return 1

    if not args.no_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.buffersize is not None:
        ok = cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
        print(f"CAP_PROP_BUFFERSIZE={args.buffersize}: {'accepted' if ok else 'REFUSED by driver'}")

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_s = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"opened {args.device}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} {fourcc_s} "
        f"{cap.get(cv2.CAP_PROP_FPS):g} fps, "
        f"CAP_PROP_BUFFERSIZE={cap.get(cv2.CAP_PROP_BUFFERSIZE):g}"
    )
    period = 1.0 / args.fps
    instant = period * 0.3  # anything faster than this did not come from the sensor

    # warm up
    for _ in range(10):
        cap.read()

    # 1. rate
    print("\n--- 1. live rate, 3 s ----------------------------------------")
    t0 = time.perf_counter()
    n = 0
    worst = 0.0
    while time.perf_counter() - t0 < 3.0:
        a = time.perf_counter()
        ok, _ = cap.read()
        d = time.perf_counter() - a
        if not ok:
            print("  read failed")
            return 1
        n += 1
        worst = max(worst, d)
    el = time.perf_counter() - t0
    print(f"  {n} frames in {el:.2f}s = {n / el:.1f} fps, worst read {worst * 1e3:.1f} ms")

    # 2. buffer
    print(f"\n--- 2. stop reading for {args.pause:g} s, then drain ---------------------")
    time.sleep(args.pause)
    stale = 0
    times = []
    for _ in range(600):
        a = time.perf_counter()
        ok, _ = cap.read()
        d = time.perf_counter() - a
        times.append(d)
        if d < instant:
            stale += 1
        else:
            break
    print(f"  {stale} frame(s) returned instantly (< {instant * 1e3:.0f} ms) = buffer depth")
    print(f"  worst-case hidden latency at this depth: {stale * period * 1e3:.0f} ms")
    print(f"  first reads (ms): {[round(t * 1e3, 1) for t in times[:8]]}")
    expected = int(args.pause * args.fps)
    if stale >= expected * 0.8:
        print(f"  !! that is ~all {expected} frames from the pause. NOTHING was dropped;")
        print(f"     this path queues without bound and will fall arbitrarily behind.")
    elif stale <= 4:
        print("  ok: normal V4L2 depth. Staleness is bounded to a few frames here.")

    # 3. drain
    print("\n--- 3. rate after draining, 2 s ------------------------------")
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 2.0:
        ok, _ = cap.read()
        n += 1
    el = time.perf_counter() - t0
    print(f"  {n / el:.1f} fps")

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
