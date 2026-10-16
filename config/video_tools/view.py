#!/usr/bin/env python3
"""The real cameras to the viewer, with no robot. Measures the video path.

This is the --watch half of a teleop run with the servos taken out: the
same three cameras, opened from the same yaml through the same lerobot
camera classes, read the way the robot reads them (async_read, latest
frame), and handed to the same log_rerun_data at the same image rate. What
it does not do is touch a motor bus, so it runs with the arms unpowered.

    ./config/video.sh view                    to the Mac viewer, 60 s
    ./config/video.sh view --seconds 300
    ./config/video.sh view --only left_wrist  one camera

It answers two questions.

1. Does video still cost the control loop anything?
   Every second it prints the loop rate and how long the two halves of a
   tick took: `observe` (reading every camera) and `log` (handing the frame
   set to the display). Before the async worker, `log` was ~55 ms and the
   loop ran at 15 Hz. Now `log` should be well under 1 ms and the loop
   should hold the requested rate. If it does not, the display is still
   in the loop.

2. How far behind is the picture?  (glass to glass)
   Each frame has elroy's wall clock burned into it at the moment it was
   read. That number is only comparable with the Mac's clock if the two
   are synchronised, so the honest measurement does not use it:

     a. On the Mac, run a millisecond clock in a terminal, font as large
        as you can make it (Cmd-+ a few times):
          python3 config/ms_clock.py
     b. Point a wrist camera at that terminal.
     c. Put the terminal and the rerun viewer side by side and take a
        screenshot (Cmd-Shift-4). The screenshot freezes both at the same
        instant: the live clock, and the clock as the camera saw it.
        Their difference is the whole pipeline - sensor, USB, driver
        buffer, read, JPEG, wifi, viewer decode and draw. Take three or
        four and keep the median.

   For reference: the wrists now run with a 2-frame driver buffer (<= 67 ms
   of that), and at the default 10 Hz image rate a frame can also wait up
   to 100 ms for its turn. 150-250 ms total is a healthy result; anything
   approaching a second means something is still queueing.

The viewer's `capture` timeline carries the same elroy timestamps, so the
gap between the newest frame and "now" on that timeline is the same lag
measured on elroy's clock, frame by frame.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def stamp(img: np.ndarray, text: str) -> np.ndarray:
    """Burn text into a COPY of the frame - never the camera's own buffer."""
    if cv2 is None:
        return img
    out = np.ascontiguousarray(img).copy()
    H = out.shape[0]
    scale = max(0.6, H / 300.0)
    pos = (8, H - 12)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(6 * scale))
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), int(2 * scale))
    return out


def wall_clock() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", nargs="?", default=None, help="viewer IP")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--config", default="config/cart-remote.yaml",
                    help="yaml whose robot.cameras are opened")
    ap.add_argument("--only", nargs="*", default=None, help="camera names to open (default: all)")
    ap.add_argument("--fps", type=float, default=30.0, help="loop rate, like the control loop")
    ap.add_argument("--image-fps", default=None,
                    help="images to the viewer per second; default is --watch's 10. "
                         "'all' = every tick, 0 = scalars only")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-stamp", action="store_true", help="do not burn elroy's clock into frames")
    args = ap.parse_args()

    if not args.host:
        print("error: give the viewer IP", file=sys.stderr)
        return 2

    # The same knob --watch turns, set before the visualizer is imported.
    if args.image_fps is None:
        os.environ.setdefault("LEROBOT_RERUN_IMAGE_FPS", "10")
    elif args.image_fps == "all":
        os.environ.pop("LEROBOT_RERUN_IMAGE_FPS", None)
    else:
        os.environ["LEROBOT_RERUN_IMAGE_FPS"] = str(args.image_fps)

    # Same yaml parsing as the self-test, so the cameras are configured
    # exactly as a run would configure them (buffer_size, fourcc, ...).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # config/, for cart_selftest
    from cart_selftest import load_config
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

    cfg, _ = load_config(Path(args.config))
    cam_cfgs = dict(cfg.cameras)
    if args.only:
        missing = [n for n in args.only if n not in cam_cfgs]
        if missing:
            print(f"error: no camera {missing} in {args.config}; it has {sorted(cam_cfgs)}", file=sys.stderr)
            return 2
        cam_cfgs = {n: cam_cfgs[n] for n in args.only}

    cams = make_cameras_from_configs(cam_cfgs)
    print(f"config  : {args.config}")
    print(f"cameras : {', '.join(cams)}")
    rate = os.environ.get("LEROBOT_RERUN_IMAGE_FPS")
    print(f"images  : {rate + ' Hz' if rate is not None else 'every tick'} to the viewer, "
          f"loop at {args.fps:g} Hz")
    for name, cam in cams.items():
        t = time.perf_counter()
        cam.connect()
        print(f"  {name:<12} connected in {time.perf_counter() - t:.1f}s")

    import rerun

    print(f"viewer  : {args.host}:{args.port} (rerun-sdk {rerun.__version__}; viewer must match)")
    init_rerun(session_name="cam_view", ip=args.host, port=args.port)
    print(f"\nrunning {args.seconds:g}s. Ctrl-C ends early.\n")

    period = 1.0 / args.fps
    obs_ms: list[float] = []
    log_ms: list[float] = []
    tick_ms: list[float] = []
    win_obs: list[float] = []
    win_log: list[float] = []
    win_n = 0
    t0 = time.perf_counter()
    last_report = t0
    n = 0
    try:
        while time.perf_counter() - t0 < args.seconds:
            tick = time.perf_counter()

            t = time.perf_counter()
            frames = {name: cam.async_read() for name, cam in cams.items()}
            o = (time.perf_counter() - t) * 1000

            if not args.no_stamp:
                clock = wall_clock()
                frames = {k: stamp(v, f"{k}  {clock}") for k, v in frames.items()}
            # One scalar so the blueprint has its plot row, as in a real run.
            obs = {**frames, "tick": float(n)}

            t = time.perf_counter()
            log_rerun_data(observation=obs, action={}, compress_images=True)
            lg = (time.perf_counter() - t) * 1000

            obs_ms.append(o); log_ms.append(lg)
            win_obs.append(o); win_log.append(lg); win_n += 1
            n += 1

            now = time.perf_counter()
            if now - last_report >= 1.0:
                print(f"  {now - t0:5.0f}s  loop {win_n / (now - last_report):5.1f} Hz   "
                      f"observe {statistics.mean(win_obs):5.1f} ms (max {max(win_obs):5.1f})   "
                      f"log {statistics.mean(win_log):5.2f} ms (max {max(win_log):6.2f})", flush=True)
                last_report = now
                win_obs.clear(); win_log.clear(); win_n = 0

            elapsed = time.perf_counter() - tick
            tick_ms.append(elapsed * 1000)
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print()
    finally:
        dur = time.perf_counter() - t0
        for cam in cams.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        shutdown_rerun()

    if not n:
        return 1

    def pct(xs, p):
        s = sorted(xs)
        return s[min(len(s) - 1, int(p * len(s)))]

    print()
    print("=" * 66)
    print(f"{n} ticks in {dur:.0f}s = {n / dur:.1f} Hz (asked for {args.fps:g})")
    print("=" * 66)
    print(f"observe  mean {statistics.mean(obs_ms):5.1f}  p99 {pct(obs_ms, .99):5.1f}  max {max(obs_ms):5.1f} ms")
    print(f"log      mean {statistics.mean(log_ms):5.2f}  p99 {pct(log_ms, .99):5.2f}  max {max(log_ms):5.2f} ms")
    print(f"tick     mean {statistics.mean(tick_ms):5.1f}  p99 {pct(tick_ms, .99):5.1f}  max {max(tick_ms):5.1f} ms")
    print()
    rate_ok = n / dur >= 0.95 * args.fps
    log_ok = pct(log_ms, .99) < 5.0
    print(("PASS" if rate_ok else "FAIL") + f"  loop held {n / dur:.1f} of {args.fps:g} Hz")
    print(("PASS" if log_ok else "FAIL") + f"  display cost p99 {pct(log_ms, .99):.2f} ms "
          "(was ~55 ms when it ran inside the loop)")
    if not rate_ok and log_ok:
        print("      The display is not the cost; `observe` is - a camera is slow to deliver.")
    print("\nNow the lag: the screenshot test in this script's header (./config/video.sh view --help).")
    return 0 if (rate_ok and log_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
