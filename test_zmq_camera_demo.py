#!/usr/bin/env python3
"""
Single-script ZMQ camera demo - runs server and client in one process
Demonstrates ZMQ camera functionality for PR reviewers
"""

import time
from threading import Thread
import cv2
import numpy as np
import zmq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from threading import Event

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from lerobot.cameras.configs import ColorMode


def create_test_pattern(width=640, height=480, frame_num=0):
    """Generate simple vertical color bars"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Classic SMPTE color bars
    colors = [
        (255, 255, 255),  # White
        (0, 255, 255),    # Yellow
        (0, 255, 255),    # Cyan
        (0, 255, 0),      # Green
        (255, 0, 255),    # Magenta
        (0, 0, 255),      # Red
        (255, 0, 0),      # Blue
    ]
    
    bar_width = width // len(colors)
    for i, color in enumerate(colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < len(colors) - 1 else width
        img[:, x_start:x_end] = color
    
    return img


def run_server(port=5550, stop_event=None):
    """Background thread: publishes test pattern frames via ZMQ"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.5)
    
    frame_num = 0
    while stop_event is None or not stop_event.is_set():
        frame = create_test_pattern(frame_num=frame_num)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        socket.send(buffer.tobytes())
        frame_num += 1
        time.sleep(1.0 / 30)  # 30 FPS
    
    socket.close()
    context.term()

port = 5550

#SERVER

stop_event = Event()
server_thread = Thread(target=run_server, args=(port, stop_event), daemon=True)
server_thread.start()
time.sleep(1)  # Let server initialize

#CLIENT

config = ZMQCameraConfig(
    server_address="localhost",
    port=port,
    color_mode=ColorMode.RGB,
    timeout_ms=5000
)
camera = ZMQCamera(config)
camera.connect()

# Setup matplotlib viewer
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('ZMQ Camera Demo - Server & Client in one script', fontweight='bold')
ax.axis('off')

frame = camera.read()
im = ax.imshow(frame)
text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                color='lime', fontweight='bold', family='monospace')
fig.tight_layout()

frame_count = [0]
fps_times = []

def update(frame_num):
    loop_start = time.time()
    frame = camera.read()
    frame_count[0] += 1
    
    im.set_data(frame)
    
    fps_times.append(time.time() - loop_start)
    if len(fps_times) > 30:
        fps_times.pop(0)
    fps = len(fps_times) / sum(fps_times) if fps_times else 0
    
    text.set_text(f'Frame: {frame_count[0]}\nFPS: {fps:.1f}')
    return [im, text]

try:
    anim = FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    camera.disconnect()
    stop_event.set()
    plt.close('all')
