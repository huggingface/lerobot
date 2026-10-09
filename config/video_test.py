#!/usr/bin/env python3
"""Send a synthetic camera feed to a rerun viewer. No robot, no cameras.

The point is to take everything else out of the picture. A teleoperation run
that shows nothing on the viewer has four things that could be wrong at once -
the viewer command, the network path, the rerun version pair, and lerobot's
own display arguments - and you cannot tell them apart while the arms are
moving. This touches no hardware and goes through the same two lerobot
functions a real run does, so if a pattern appears here, video works, and
anything still missing during a run is lerobot's arguments and nothing else.

    ./config/video-test.sh                 # to the default viewer host
    ./config/video-test.sh 192.168.1.52    # somewhere else
    ./config/video-test.sh --local         # this machine's own desktop

Start the viewer first, on the machine that has the screen:

    uvx --from rerun-sdk==<version printed below> rerun

with no other arguments. Plain `rerun` opens the window AND listens on
0.0.0.0:9876 for exactly this kind of connection - those are one process,
not two. `--serve-grpc` is the other thing: a server with no window, which
is why it prints an invitation to go and start a viewer separately.
"""

import argparse
import math
import sys
import time

import numpy as np

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

W, H = 640, 480


def frame(t: float, phase: float) -> np.ndarray:
    """A moving bar on a colour gradient, with a frame counter block.

    Movement matters: a still image cannot tell a live stream apart from one
    frame that arrived and then nothing.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0, 255, W, dtype=np.uint8)
    img[:, :, 0] = xs[None, :]
    img[:, :, 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    img[:, :, 2] = int(127 + 127 * math.sin(t * 2 + phase))

    x = int((0.5 + 0.5 * math.sin(t * 1.5 + phase)) * (W - 60))
    img[:, x : x + 60, :] = 255
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("host", nargs="?", default=None, help="viewer IP")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--local", action="store_true", help="render on this machine")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-compress", action="store_true")
    args = ap.parse_args()

    try:
        import rerun

        ver = rerun.__version__
    except Exception:
        ver = "unknown"

    if args.local:
        print(f"sending to this machine's own desktop (rerun-sdk {ver})")
        init_rerun(session_name="video_test")
    else:
        if not args.host:
            print("error: give a viewer IP, or --local", file=sys.stderr)
            return 2
        print(f"sending to {args.host}:{args.port} (rerun-sdk {ver})")
        print(f"the viewer there must be {ver} too - start it with:")
        print(f"    uvx --from rerun-sdk=={ver} rerun")
        init_rerun(session_name="video_test", ip=args.host, port=args.port)

    print(f"{args.fps:g} fps for {args.seconds:g}s - ctrl-c to stop")
    print("expect three moving panels and two scalars that sweep.")

    period = 1.0 / args.fps
    t0 = time.perf_counter()
    n = 0
    try:
        while (t := time.perf_counter() - t0) < args.seconds:
            obs = {
                "cam_head": frame(t, 0.0),
                "cam_left": frame(t, 2.1),
                "cam_right": frame(t, 4.2),
                "left_arm_shoulder_pan.pos": 50.0 * math.sin(t),
            }
            act = {"left_arm_shoulder_pan.pos": 50.0 * math.sin(t + 0.2)}
            log_rerun_data(
                observation=obs, action=act, compress_images=not args.no_compress
            )
            n += 1
            if n % int(max(args.fps, 1)) == 0:
                print(f"  {n:5d} frames  {n / t:5.1f} Hz", flush=True)
            time.sleep(max(0.0, period - ((time.perf_counter() - t0) - t)))
    except KeyboardInterrupt:
        print()

    print(f"sent {n} frames.")
    print("Nothing on the viewer? Then it is the link, not lerobot:")
    print("  - viewer started as plain `rerun`, not `--serve-grpc`")
    print("  - same rerun version both ends")
    print("  - the viewer host's firewall allows incoming connections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
