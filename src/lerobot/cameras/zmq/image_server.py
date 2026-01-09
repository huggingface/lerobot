#!/usr/bin/env python3

import base64
import json
import time
from collections import deque

import cv2
import numpy as np
import zmq


def encode_image(image: np.ndarray, quality: int = 80) -> str:
    """Encode image to base64 JPEG string (converts BGR→RGB to match MuJoCo sim)"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", image_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


class OpenCVCamera:
    def __init__(self, device_id, img_shape, fps):
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.read()[0]:
            raise RuntimeError(f"Failed to open camera {device_id}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class ImageServer:
    def __init__(self, config, port=5555):
        self.fps = config.get("fps", 30)
        self.cameras = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            self.cameras[name] = OpenCVCamera(cfg.get("device_id", 0), shape, self.fps)
            print(f"✓ {name}: {shape[1]}x{shape[0]}")

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        print(f"\n[ImageServer] Running on port {port} (JSON protocol)\n")

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        try:
            while True:
                t0 = time.time()

                # Build message (same format as MuJoCo sim & LeKiwi)
                message = {"timestamps": {}, "images": {}}
                for name, cam in self.cameras.items():
                    frame = cam.get_frame()
                    if frame is not None:
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
                cam.release()
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    config = {"fps": 30, "cameras": {"head_camera": {"type": "opencv", "device_id": 4, "shape": [480, 640]}}}
    ImageServer(config, port=5555).run()
