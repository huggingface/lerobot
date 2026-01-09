0  #!/usr/bin/env python3
"""
ZMQ Camera Viewer - Display images from JSON-based ZMQ image servers
Uses matplotlib instead of OpenCV for display (works without GTK)
"""

import base64
import json
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zmq


def main():
    host = "172.18.129.215"  # "192.168.123.164" for real G1
    port = 5555

    print(f"Connecting to tcp://{host}:{port}...")

    # Setup ZMQ subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.CONFLATE, True)
    socket.connect(f"tcp://{host}:{port}")

    print("Connected! Waiting for images...")
    print("Close the window to quit\n")

    # Setup matplotlib for live updating
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    img_plot = None

    frame_count = 0
    last_print = time.time()

    try:
        while plt.fignum_exists(fig.number):
            try:
                # Receive JSON message
                msg = socket.recv_string()
                data = json.loads(msg)
            except zmq.Again:
                print("Timeout - no data received")
                plt.pause(0.1)
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

            images = data.get("images", {})

            if not images:
                print("No images in message")
                continue

            # Get first camera image
            cam_name = list(images.keys())[0]
            img_b64 = images[cam_name]

            try:
                frame_bytes = base64.b64decode(img_b64)
                img_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                frame_count += 1

                # Update plot
                if img_plot is None:
                    img_plot = ax.imshow(frame)
                    ax.axis("off")
                else:
                    img_plot.set_data(frame)

                # Update title with camera name and FPS
                now = time.time()
                if now - last_print > 1.0:
                    fps = frame_count / (now - last_print)
                    ax.set_title(f"{cam_name} | FPS: {fps:.1f} | Frame: {frame_count}")
                    print(f"Camera: {cam_name} | Shape: {frame.shape} | FPS: {fps:.1f}")
                    frame_count = 0
                    last_print = now

                fig.canvas.draw_idle()
                plt.pause(0.001)

            except Exception as e:
                print(f"Error decoding {cam_name}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        plt.close("all")
        socket.close()
        context.term()
        print("Done!")


if __name__ == "__main__":
    main()
