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
Example: End-to-End Real Video Streaming into LeRobot using Zero-Copy POSIX Shared Memory.

Demonstrates high-speed HD camera capture streaming directly into PyTorch policy inference.
"""

import sys
import time
import multiprocessing as mp
import numpy as np
import cv2
import torch

from lerobot.transport import (
    is_zerocopy_available,
    ZeroCopyPublisher,
    ZeroCopyDataset,
)

VIDEO_PATH = "input_robot_camera.mp4"
OUTPUT_VIDEO_PATH = "output_reconstructed_video.mp4"
CHANNEL_NAME = "/lerobot_video_stream"
WIDTH = 1280
HEIGHT = 720
FPS = 60
TOTAL_FRAMES = 120


def get_video_writer(filename, fps, width, height):
    """Create a VideoWriter compatible with headless Linux environments."""
    codecs = ["mp4v", "XVID", "MJPG", "avc1"]
    for c in codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        if out.isOpened():
            return out
    return cv2.VideoWriter(filename, 0, fps, (width, height))


def create_sample_video():
    """Generate a test HD video file with moving graphics and timestamp overlays."""
    print(f"[Video Generator] Creating test HD video ({WIDTH}x{HEIGHT} @ {FPS} FPS, {TOTAL_FRAMES} frames)...")
    out = get_video_writer(VIDEO_PATH, FPS, WIDTH, HEIGHT)

    for i in range(TOTAL_FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        color_val = (i * 2) % 255
        frame[:, :, 0] = color_val
        frame[:, :, 1] = (255 - color_val)
        frame[:, :, 2] = 128

        center_x = int(WIDTH / 2 + np.sin(i * 0.1) * 300)
        center_y = int(HEIGHT / 2 + np.cos(i * 0.1) * 200)
        cv2.circle(frame, (center_x, center_y), 60, (0, 255, 255), -1)

        text = f"LeRobot Zero-Copy Transport - Frame {i+1}/{TOTAL_FRAMES}"
        cv2.putText(frame, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        out.write(frame)

    out.release()


def video_producer_process(channel_name, num_frames, ready_event):
    """Producer Process: Reads MP4 video and writes frames directly into shared memory."""
    print(f"[Producer] Creating Publisher on channel {channel_name}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    publisher = ZeroCopyPublisher(channel_name=channel_name, num_slots=16)

    ready_event.set()
    time.sleep(0.1)

    frame_count = 0
    start_time = time.perf_counter()

    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        publisher.write_frame(frame)
        frame_count += 1
        time.sleep(0.005)

    cap.release()
    elapsed = time.perf_counter() - start_time
    print(f"[Producer] Finished streaming {frame_count} video frames in {elapsed:.4f}s ({frame_count/elapsed:.1f} FPS).")


def pytorch_consumer_process(channel_name, num_frames, ready_event, result_queue):
    """Consumer Process: Ingests zero-copy PyTorch Tensors from shared memory."""
    ready_event.wait()
    print(f"[PyTorch Consumer] Attaching Subscriber to channel {channel_name}...")
    dataset = ZeroCopyDataset(channel_name=channel_name, timeout_ms=5000, max_frames=num_frames)

    saved_frames = []
    received_count = 0
    start_time = time.perf_counter()

    for sample in dataset:
        pixel_values = sample["pixel_values"]
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.shape == (HEIGHT, WIDTH, 3)

        saved_frames.append(pixel_values.numpy().copy())
        del pixel_values
        del sample

        received_count += 1

    elapsed = time.perf_counter() - start_time

    print(f"[PyTorch Consumer] Ingested {received_count} PyTorch Tensors in {elapsed:.4f}s ({received_count/elapsed:.1f} FPS).")

    out = get_video_writer(OUTPUT_VIDEO_PATH, FPS, WIDTH, HEIGHT)
    for frame in saved_frames:
        out.write(frame)
    out.release()

    result_queue.put({
        "received_count": received_count,
        "fps": received_count / elapsed if elapsed > 0 else 0,
    })


def main():
    if not is_zerocopy_available():
        print("[ERROR] lerobot_ipc library not available. Install via `pip install lerobot_zerocopy_ipc`.")
        sys.exit(1)

    print("==================================================================")
    print(" LEROBOT REAL VIDEO ZERO-COPY IPC EXAMPLE")
    print("==================================================================")

    create_sample_video()

    ready_event = mp.Event()
    result_queue = mp.Queue()

    producer = mp.Process(target=video_producer_process, args=(CHANNEL_NAME, TOTAL_FRAMES, ready_event))
    consumer = mp.Process(target=pytorch_consumer_process, args=(CHANNEL_NAME, TOTAL_FRAMES, ready_event, result_queue))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    res = result_queue.get()
    print("==================================================================")
    print(f"[SUCCESS] Video Streaming Example Completed: Ingested {res['received_count']}/{TOTAL_FRAMES} frames at {res['fps']:.1f} FPS.")
    print("==================================================================")


if __name__ == "__main__":
    main()
