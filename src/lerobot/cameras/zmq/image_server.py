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
from dataclasses import dataclass, field

import cv2
import numpy as np
import zmq

from ..camera import Camera
from ..configs import CameraConfig, ColorMode
from ..utils import make_cameras_from_configs

logger = logging.getLogger(__name__)

# How long an encoder waits on its camera before complaining. Long enough that a camera
# running below its configured fps stays quiet, short enough that an unplugged one is
# reported while the publish loop is still sending its last frame.
ENCODE_READ_TIMEOUT_MS = 1000


def encode_image(image: np.ndarray, quality: int = 80, is_rgb: bool = False) -> str:
    """Encode an image to a base64 JPEG string.

    ``cv2.imencode`` reads its input as BGR, so an RGB frame has to be converted first
    or the JPEG comes out with red and blue swapped.
    """
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


@dataclass
class FrameEncoder:
    """Keeps one camera's latest frame ready to publish, JPEG-encoded.

    Acquisition is the camera's own job: every ``Camera`` runs an internal read thread and
    ``async_read`` hands back whatever it last decoded, so this thread never touches the
    device. It exists for the encode, which is the expensive step and would otherwise run
    once per camera inside the publish loop and hold up every other stream.
    """

    camera: Camera
    name: str
    # Optional (width, height) to shrink frames to before encoding, for cameras that
    # cannot capture at the size we want to publish. Ask the camera first by setting
    # width/height in its config: that costs nothing, and OpenCVCamera raises rather than
    # quietly handing back a different size, so a request that survives connect() is one
    # the driver honoured. Resize here only once it has refused.
    publish_size: tuple[int, int] | None = None
    is_rgb: bool = False

    latest_encoded: str | None = field(default=None, init=False)  # base64 JPEG
    latest_timestamp: float = field(default=0.0, init=False)
    frame_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    running: bool = field(default=False, init=False)
    thread: threading.Thread | None = field(default=None, init=False)

    def start(self):
        """Start the encode thread."""
        self.running = True
        self.thread = threading.Thread(target=self._encode_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the encode thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _encode_loop(self):
        """Encode each new frame the camera produces, at the camera's own rate.

        ``async_read`` blocks until the camera's read thread publishes a frame we haven't
        taken yet, so a slow encode skips ahead to the newest frame rather than working
        through a backlog, and a stalled camera surfaces as a warning per timeout instead
        of a silently frozen stream.
        """
        while self.running:
            try:
                frame = self.camera.async_read(timeout_ms=ENCODE_READ_TIMEOUT_MS)
                timestamp = time.time()
                if self.publish_size is not None and frame.shape[1::-1] != self.publish_size:
                    frame = cv2.resize(frame, self.publish_size, interpolation=cv2.INTER_AREA)
                # Encode immediately in capture thread (this is the slow part)
                encoded = encode_image(frame, is_rgb=self.is_rgb)
                with self.frame_lock:
                    self.latest_encoded = encoded
                    self.latest_timestamp = timestamp
            except Exception as e:
                logger.warning(f"Camera {self.name} encode error: {e}")
                time.sleep(0.01)

    def get_latest(self) -> tuple[str | None, float]:
        """Get the latest encoded frame and its timestamp."""
        with self.frame_lock:
            return self.latest_encoded, self.latest_timestamp


@dataclass
class ImageServer:
    """Publishes every camera's latest JPEG frame on a single ZMQ PUB socket."""

    camera_configs: dict[str, CameraConfig]
    # Rate of the publish loop (how often frames go out over ZMQ), not of capture: each
    # camera reads at whatever rate it negotiated with the driver.
    fps: int = 30
    port: int = 5555
    publish_size: tuple[int, int] | None = None
    # Flaky USB cameras intermittently fail the first open or first-frame read;
    # retry a few times before giving up.
    open_attempts: int = 5
    open_retry_delay_s: float = 2.0

    cameras: dict[str, Camera] = field(default_factory=dict, init=False)
    encoders: dict[str, FrameEncoder] = field(default_factory=dict, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        self.cameras = make_cameras_from_configs(self.camera_configs)
        try:
            self._open_cameras()
        except Exception:
            # Release the devices we did open, so they get a clean STREAMOFF instead of
            # staying busy until the next reboot.
            logger.exception("Failed to open cameras; releasing any already-opened devices.")
            self._release_cameras()
            raise
        self._bind_publisher()

    def _open_cameras(self) -> None:
        for name, camera in self.cameras.items():
            self._connect_with_retries(name, camera)

            config = self.camera_configs[name]
            captured = (config.width, config.height)
            published = self.publish_size or captured
            logger.info(
                f"Camera {name}: capture {captured[0]}x{captured[1]}, publish {published[0]}x{published[1]}"
            )
            if published != captured and None not in captured:
                logger.warning(
                    "Camera %s captures %dx%d and publishes %dx%d, so every frame pays a CPU "
                    "resize. Capture at the publish size instead if this camera supports it.",
                    name,
                    *captured,
                    *published,
                )
            # Only some camera types expose a color mode; anything else is assumed to
            # hand us BGR already, which is what the JPEG encoder wants.
            is_rgb = getattr(config, "color_mode", None) == ColorMode.RGB
            self.encoders[name] = FrameEncoder(camera, name, self.publish_size, is_rgb)

    def _connect_with_retries(self, name: str, camera: Camera) -> None:
        last_err: Exception | None = None
        for attempt in range(1, self.open_attempts + 1):
            try:
                camera.connect()
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "Camera %s open attempt %d/%d failed: %s", name, attempt, self.open_attempts, e
                )
                with contextlib.suppress(Exception):
                    camera.disconnect()
                if attempt < self.open_attempts:
                    time.sleep(self.open_retry_delay_s)
        raise RuntimeError(f"Camera {name} failed to open after {self.open_attempts} attempts") from last_err

    def _bind_publisher(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"ImageServer running on port {self.port}")

    def stop(self) -> None:
        """Signal the publish loop to exit so ``run()`` reaches its cleanup.

        Call this from the owning process on shutdown (e.g. Ctrl-C) — otherwise a
        daemon thread running ``run()`` is killed abruptly and the cameras never get
        released, leaving the V4L2 devices wedged until reboot.
        """
        self._stop.set()

    def _release_cameras(self) -> None:
        """Stop the encoders and disconnect all cameras (safe to call twice).

        Ensures each V4L2 device gets a clean STREAMOFF/release so a failed or
        interrupted run doesn't leave devices busy until the next reboot.
        """
        for encoder in self.encoders.values():
            with contextlib.suppress(Exception):
                encoder.stop()
        for cam in self.cameras.values():
            with contextlib.suppress(Exception):
                cam.disconnect()

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        for encoder in self.encoders.values():
            encoder.start()

        logger.info("Waiting for cameras to start capturing...")
        for name, encoder in self.encoders.items():
            while encoder.get_latest()[0] is None and not self._stop.is_set():
                time.sleep(0.01)
            logger.info(f"Camera {name} ready (encoding in background)")

        try:
            while not self._stop.is_set():
                t0 = time.time()

                # Build message. Always include EVERY camera's latest frame so each message
                # is complete: clients pick their own stream by name, and a partial message
                # makes them fall back to another camera's image (cross-feed flicker).
                message = {"timestamps": {}, "images": {}}
                for name, encoder in self.encoders.items():
                    encoded, timestamp = encoder.get_latest()
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
