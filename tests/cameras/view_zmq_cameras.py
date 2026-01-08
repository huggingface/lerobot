#!/usr/bin/env python3
"""
ZMQ Camera Viewer - Display images from JSON-based ZMQ image servers
Uses matplotlib instead of OpenCV for display (works without GTK)
"""

import base64
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import zmq


def decode_image(base64_str: str) -> np.ndarray:
    """Decode base64 JPEG string to numpy array"""
    import cv2
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # MuJoCo sim renders in RGB, so no conversion needed
    # For real cameras (BGR source), uncomment: img = img[:, :, ::-1]
    return img


def main():
    host = "localhost"  # Change to "172.18.129.215" for Unitree robot
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
            timestamps = data.get("timestamps", {})
            
            if not images:
                print("No images in message")
                continue
            
            # Get first camera image
            cam_name = list(images.keys())[0]
            img_b64 = images[cam_name]
            
            try:
                frame = decode_image(img_b64)
                frame_count += 1
                
                # Update plot
                if img_plot is None:
                    img_plot = ax.imshow(frame)
                    ax.axis('off')
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
        plt.close('all')
        socket.close()
        context.term()
        print("Done!")


if __name__ == "__main__":
    main()
