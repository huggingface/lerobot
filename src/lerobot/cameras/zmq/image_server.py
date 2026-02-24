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

"""
Streams camera images over ZMQ.
Uses lerobot's OpenCVCamera for capture, encodes images to base64 and sends them over ZMQ.
"""

import base64
import contextlib
import json
import logging
import time
from collections import deque

import cv2
import numpy as np
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

logger = logging.getLogger(__name__)


def encode_image(image: np.ndarray, quality: int = 80) -> str:
    """Encode RGB image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
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
            logger.info(f"Camera {name}: {shape[1]}x{shape[0]}")

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(f"ImageServer running on port {port}")

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

                # Send as JSON string (suppress if buffer full)
                with contextlib.suppress(zmq.Again):
                    self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    logger.debug(f"FPS: {len(frame_times) / sum(frame_times):.1f}")

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
    logging.basicConfig(level=logging.INFO)
    config = {"fps": 30, "cameras": {"head_camera": {"device_id": 4, "shape": [480, 640]}}}
    ImageServer(config, port=5555).run()
