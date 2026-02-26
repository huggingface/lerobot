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

If the requested publish FPS is higher than the camera's native FPS, the server will
duplicate frames to maintain the publish rate. This allows high-frequency control loops
to receive frames at the requested rate without blocking.
"""

import base64
import contextlib
import json
import logging
import threading
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


class CameraCaptureThread:
    """Background thread that continuously captures and encodes frames from a camera."""

    def __init__(self, camera: OpenCVCamera, name: str):
        self.camera = camera
        self.name = name
        self.latest_encoded: str | None = None  # Pre-encoded JPEG as base64
        self.latest_timestamp: float = 0.0
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        """Start the capture thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _capture_loop(self):
        """Continuously capture and encode frames at the camera's native rate."""
        while self.running:
            try:
                frame = self.camera.read()  # Blocks at camera's native rate
                timestamp = time.time()
                # Encode immediately in capture thread (this is the slow part)
                encoded = encode_image(frame)
                with self.frame_lock:
                    self.latest_encoded = encoded
                    self.latest_timestamp = timestamp
            except Exception as e:
                logger.warning(f"Camera {self.name} capture error: {e}")
                time.sleep(0.01)

    def get_latest(self) -> tuple[str | None, float]:
        """Get the latest encoded frame and its timestamp."""
        with self.frame_lock:
            return self.latest_encoded, self.latest_timestamp


class ImageServer:
    def __init__(self, config: dict, port: int = 5555):
        self.fps = config.get("fps", 30)
        self.cameras: dict[str, OpenCVCamera] = {}
        self.capture_threads: dict[str, CameraCaptureThread] = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            # Don't pass fps to camera config - let it use native rate
            # The publish loop will handle frame duplication
            cam_config = OpenCVCameraConfig(
                index_or_path=cfg.get("device_id", 0),
                fps=None,  # Use camera's native rate
                width=shape[1],
                height=shape[0],
                color_mode=ColorMode.RGB,
            )
            camera = OpenCVCamera(cam_config)
            camera.connect()
            self.cameras[name] = camera
            logger.info(f"Camera {name}: {shape[1]}x{shape[0]}")

            # Create capture thread for this camera
            capture_thread = CameraCaptureThread(camera, name)
            self.capture_threads[name] = capture_thread

        # ZMQ PUB socket - minimal buffering to prevent lag
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)  # Only 1 message in send buffer
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(
            f"ImageServer running on port {port} at {self.fps} FPS (will duplicate frames if camera is slower)"
        )

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        # Start all capture threads
        for capture_thread in self.capture_threads.values():
            capture_thread.start()

        # Wait for first frames to be captured and encoded
        logger.info("Waiting for cameras to start capturing...")
        for name, capture_thread in self.capture_threads.items():
            while capture_thread.get_latest()[0] is None:
                time.sleep(0.01)
            logger.info(f"Camera {name} ready (capture + encode in background)")

        try:
            while True:
                t0 = time.time()

                # Build message using pre-encoded frames (fast - no encoding here)
                message = {"timestamps": {}, "images": {}}
                for name, capture_thread in self.capture_threads.items():
                    encoded, timestamp = capture_thread.get_latest()
                    if encoded is not None:
                        message["timestamps"][name] = timestamp
                        message["images"][name] = encoded  # Already encoded!

                # Send as JSON string (suppress if buffer full)
                with contextlib.suppress(zmq.Again):
                    self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    logger.debug(f"Publish FPS: {len(frame_times) / sum(frame_times):.1f}")

                # Sleep to maintain requested publish FPS
                sleep = (1.0 / self.fps) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            # Stop capture threads
            for capture_thread in self.capture_threads.values():
                capture_thread.stop()
            for cam in self.cameras.values():
                cam.disconnect()
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {"fps": 30, "cameras": {"head_camera": {"device_id": 4, "shape": [480, 640]}}}
    ImageServer(config, port=5555).run()
