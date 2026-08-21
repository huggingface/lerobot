#!/usr/bin/env python
"""Watch the G1's ZMQ camera stream in Rerun, from any machine on the network.

The image server publishes on a ZMQ PUB socket, so subscribing here is passive: it does not
disturb the robot client that is subscribed to the same stream.

    python view_g1_camera.py --robot-ip 172.18.130.111

Use it to check what the policy is actually being fed — wrong capture device, wrong
color order, upside-down mount, stale/frozen frames.
"""

import argparse
import time

import numpy as np
import rerun as rr

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig


def main() -> None:
    """Subscribe to the robot's image server and log every frame to Rerun."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="172.18.130.111")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    camera = ZMQCamera(
        ZMQCameraConfig(
            server_address=args.robot_ip,
            port=args.port,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_s=5,
        )
    )

    rr.init(f"g1_camera_{args.camera_name}", spawn=True)
    camera.connect()
    print(f"streaming {args.camera_name} from {args.robot_ip}:{args.port} - ctrl-c to stop")

    frames = 0
    started = time.perf_counter()
    try:
        while True:
            frame = camera.async_read(timeout_ms=1000)
            rr.set_time("wall", timestamp=time.time())
            rr.log(f"camera/{args.camera_name}", rr.Image(frame))
            # Channel means separate a real RGB/BGR swap from a merely odd-looking scene.
            for i, channel in enumerate("rgb"):
                rr.log(f"channel_mean/{channel}", rr.Scalars(float(np.mean(frame[:, :, i]))))

            frames += 1
            if frames % 30 == 0:
                print(f"{frames} frames, {frames / (time.perf_counter() - started):.1f} fps")
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
