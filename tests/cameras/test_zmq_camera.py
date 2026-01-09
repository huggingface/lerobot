#!/usr/bin/env python3
"""
ZMQ Camera Viewer for unitree g1 (sim/real)
"""

import time

import matplotlib.pyplot as plt

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig


def main():
    host = "172.18.129.215"  # "192.168.123.164" for real G1
    port = 5555
    camera_name = "head_camera"

    print(f"Connecting to {camera_name} at tcp://{host}:{port}...")

    # Use ZMQCamera class
    config = ZMQCameraConfig(
        server_address=host,
        port=port,
        camera_name=camera_name,
    )
    camera = ZMQCamera(config)
    camera.connect()

    print("Waiting for images...")

    # Setup matplotlib for live updating
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    img_plot = None

    frame_count = 0
    last_print = time.time()

    try:
        while plt.fignum_exists(fig.number):
            try:
                frame = camera.async_read()
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
                    ax.set_title(f"{camera_name} | FPS: {fps:.1f} | Frame: {frame_count}")
                    print(f"Camera: {camera_name} | Shape: {frame.shape} | FPS: {fps:.1f}")
                    frame_count = 0
                    last_print = now

                fig.canvas.draw_idle()
                plt.pause(0.001)

            except TimeoutError:
                print("Timeout - no frame available")
                plt.pause(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        plt.close("all")
        camera.disconnect()
        print("Done!")

if __name__ == "__main__":
    main()
