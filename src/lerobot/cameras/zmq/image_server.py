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
import threading
import time
from collections import deque

import cv2
import numpy as np
import zmq

from ..camera import Camera
from ..configs import CameraConfig, ColorMode
from ..utils import make_cameras_from_configs

logger = logging.getLogger(__name__)


def encode_image(image: np.ndarray, quality: int = 80, is_rgb: bool = False) -> str:
    """Encode an image to a base64 JPEG string.

    ``cv2.imencode`` reads its input as BGR, so an RGB frame has to be converted first
    or the JPEG comes out with red and blue swapped.
    """
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


class CameraCaptureThread:
    """Background thread that continuously captures and encodes frames from a camera."""

    def __init__(
        self,
        camera: Camera,
        name: str,
        publish_size: tuple[int, int] | None = None,
        is_rgb: bool = False,
    ):
        self.camera = camera
        self.name = name
        # Optional (width, height) to shrink frames to before encoding. Capture resolution
        # is dictated by what the camera will negotiate, which can be far more than a
        # policy needs; publishing it unchanged just spends link bandwidth.
        self.publish_size = publish_size
        self.is_rgb = is_rgb
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
                if self.publish_size is not None and frame.shape[1::-1] != self.publish_size:
                    frame = cv2.resize(frame, self.publish_size, interpolation=cv2.INTER_AREA)
                # Encode immediately in capture thread (this is the slow part)
                encoded = encode_image(frame, is_rgb=self.is_rgb)
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
    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        fps: int = 30,
        port: int = 5555,
        publish_size: tuple[int, int] | None = None,
        open_attempts: int = 5,
        open_retry_delay_s: float = 2.0,
    ):
        # fps controls the publish loop rate (how often frames are sent over ZMQ), not the camera capture rate
        self.fps = fps
        # Flaky USB cameras intermittently fail the first open or first-frame read;
        # retry a few times before giving up.
        self.open_attempts = open_attempts
        self.open_retry_delay_s = open_retry_delay_s
        self.cameras: dict[str, Camera] = make_cameras_from_configs(cameras)
        self.capture_threads: dict[str, CameraCaptureThread] = {}
        self._stop = threading.Event()

        # If any camera fails to open, release the ones we already opened so the V4L2
        # devices get a clean STREAMOFF instead of staying busy until the next reboot.
        try:
            for name, camera in self.cameras.items():
                last_err: Exception | None = None
                for attempt in range(1, self.open_attempts + 1):
                    try:
                        camera.connect()
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        logger.warning(
                            "Camera %s open attempt %d/%d failed: %s",
                            name,
                            attempt,
                            self.open_attempts,
                            e,
                        )
                        with contextlib.suppress(Exception):
                            camera.disconnect()
                        if attempt < self.open_attempts:
                            time.sleep(self.open_retry_delay_s)
                if last_err is not None:
                    raise RuntimeError(
                        f"Camera {name} failed to open after {self.open_attempts} attempts"
                    ) from last_err

                published = publish_size or (cameras[name].width, cameras[name].height)
                logger.info(
                    f"Camera {name}: capture {cameras[name].width}x{cameras[name].height}, "
                    f"publish {published[0]}x{published[1]}"
                )
                # Only some camera types expose a color mode; anything else is assumed to
                # hand us BGR already, which is what the JPEG encoder wants.
                is_rgb = getattr(cameras[name], "color_mode", None) == ColorMode.RGB
                self.capture_threads[name] = CameraCaptureThread(camera, name, publish_size, is_rgb)
        except Exception:
            logger.exception("Failed to open cameras; releasing any already-opened devices.")
            self._release_cameras()
            raise

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(f"ImageServer running on port {port}")

    def stop(self) -> None:
        """Signal the publish loop to exit so ``run()`` reaches its cleanup.

        Call this from the owning process on shutdown (e.g. Ctrl-C) — otherwise a
        daemon thread running ``run()`` is killed abruptly and the cameras never get
        released, leaving the V4L2 devices wedged until reboot.
        """
        self._stop.set()

    def _release_cameras(self) -> None:
        """Stop capture threads and disconnect all cameras (safe to call twice).

        Ensures each V4L2 device gets a clean STREAMOFF/release so a failed or
        interrupted run doesn't leave devices busy until the next reboot.
        """
        for capture_thread in self.capture_threads.values():
            with contextlib.suppress(Exception):
                capture_thread.stop()
        for cam in self.cameras.values():
            with contextlib.suppress(Exception):
                cam.disconnect()

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        # Start all capture threads
        for capture_thread in self.capture_threads.values():
            capture_thread.start()

        # Wait for first frames to be captured and encoded
        logger.info("Waiting for cameras to start capturing...")
        for name, capture_thread in self.capture_threads.items():
            while capture_thread.get_latest()[0] is None and not self._stop.is_set():
                time.sleep(0.01)
            logger.info(f"Camera {name} ready (capture + encode in background)")

        try:
            while not self._stop.is_set():
                t0 = time.time()

                # Build message. Always include EVERY camera's latest frame so each message
                # is complete: clients pick their own stream by name, and a partial message
                # makes them fall back to another camera's image (cross-feed flicker).
                message = {"timestamps": {}, "images": {}}
                for name, capture_thread in self.capture_threads.items():
                    encoded, timestamp = capture_thread.get_latest()
                    if encoded is not None:
                        message["timestamps"][name] = timestamp
                        message["images"][name] = encoded

                # Send as JSON string (suppress if buffer full)
                if message["images"]:
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
            self._release_cameras()
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    from ..configs import Cv2Backends
    from ..opencv import OpenCVCameraConfig

    logging.basicConfig(level=logging.INFO)
    config = OpenCVCameraConfig(
        index_or_path=4, fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2
    )
    ImageServer({"head_camera": config}, fps=30, port=5555).run()
