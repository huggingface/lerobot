#!/usr/bin/env python3
"""
Image server for Unitree G1. Streams camera images over ZMQ.
Uses lerobot's OpenCVCamera for capture.
"""

import base64
import json
import time
from collections import deque

import cv2
import numpy as np
import zmq

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode


def encode_image(image: np.ndarray, quality: int = 80) -> str:
    """Encode RGB image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                             [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


class ImageServer:
    def __init__(self, config: dict, port: int = 5555):
        self.fps = config.get("fps", 30)
        self.cameras: dict[str, OpenCVCamera] = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            cam_config = OpenCVCameraConfig(
                index_or_path=cfg.get("device_id", 0),
                fps=self.fps,
                width=shape[1],
                height=shape[0],
                color_mode=ColorMode.RGB,
            )
            camera = OpenCVCamera(cam_config)
            camera.connect()
            self.cameras[name] = camera
            print(f"✓ {name}: {shape[1]}x{shape[0]}")

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        print(f"\n[ImageServer] Running on port {port}\n")

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        try:
            while True:
                t0 = time.time()

                # Build message
                message = {"timestamps": {}, "images": {}}
                for name, cam in self.cameras.items():
                    frame = cam.read()  # Returns RGB
                    message["timestamps"][name] = time.time()
                    message["images"][name] = encode_image(frame)

                # Send as JSON string
                try:
                    self.socket.send_string(json.dumps(message), zmq.NOBLOCK)
                except zmq.Again:
                    pass

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    print(f"FPS: {len(frame_times) / sum(frame_times):.1f}")

                sleep = (1.0 / self.fps) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    config = {
        "fps": 30,
        "cameras": {
            "head_camera": {"device_id": 4, "shape": [480, 640]}
        }
    }
    ImageServer(config, port=5555).run()
