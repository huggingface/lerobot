#!/usr/bin/env python3
"""
ZMQ Camera Server - Publishes camera frames via ZeroMQ

This script captures frames from a camera (or generates test patterns) and
publishes them as JPEG-encoded images over a ZMQ PUB socket.

Usage:
    # Using a webcam (default camera index 0)
    python test_zmq_camera_server.py --port 5554
    
    # Using a specific camera
    python test_zmq_camera_server.py --camera 1 --port 5555
    
    # Using test pattern (no camera required)
    python test_zmq_camera_server.py --test-pattern --port 5554
    
    # Specify IP address and resolution
    python test_zmq_camera_server.py --address 192.168.1.100 --width 1280 --height 720
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import zmq


def create_test_pattern(width: int = 640, height: int = 480, frame_count: int = 0) -> np.ndarray:
    """
    Creates an animated test pattern image.
    
    Args:
        width: Image width
        height: Image height
        frame_count: Current frame number for animation
        
    Returns:
        BGR image array
    """
    # Create a colorful test pattern
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated gradient background
    for i in range(height):
        color_shift = int(((i + frame_count) % height) / height * 255)
        img[i, :] = [color_shift, 128, 255 - color_shift]
    
    # Moving circle
    center_x = int(width // 2 + 200 * np.sin(frame_count * 0.05))
    center_y = int(height // 2 + 100 * np.cos(frame_count * 0.05))
    cv2.circle(img, (center_x, center_y), 50, (0, 255, 0), -1)
    
    # Add text with timestamp
    text = f"ZMQ Test Pattern - Frame {frame_count}"
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cv2.putText(img, timestamp, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="ZMQ Camera Server - Publishes camera frames via ZeroMQ")
    parser.add_argument("--address", type=str, default="*", 
                        help="IP address to bind to (* for all interfaces)")
    parser.add_argument("--port", type=int, default=5550, 
                        help="Port number to publish on (default: 5554)")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Camera device index (default: 0)")
    parser.add_argument("--test-pattern", action="store_true", 
                        help="Use animated test pattern instead of camera")
    parser.add_argument("--width", type=int, default=640, 
                        help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, 
                        help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, 
                        help="Target frames per second (default: 30)")
    parser.add_argument("--quality", type=int, default=80, 
                        help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--display", action="store_true", 
                        help="Display the video feed locally")
    
    args = parser.parse_args()
    
    # Initialize ZMQ
    print(f"Initializing ZMQ publisher on tcp://{args.address}:{args.port}")
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://{args.address}:{args.port}")
    
    # Give ZMQ time to establish connections
    print("Waiting for subscribers to connect...")
    time.sleep(1)
    
    # Initialize camera or test pattern
    cap = None
    if not args.test_pattern:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {args.camera}")
            print("Falling back to test pattern mode")
            args.test_pattern = True
        else:
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            
            # Read actual properties (camera may not support requested values)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera opened successfully")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
    
    if args.test_pattern:
        print(f"Using test pattern mode")
        print(f"  Resolution: {args.width}x{args.height}")
        print(f"  FPS: {args.fps}")
    
    print("\n" + "="*60)
    print("SERVER RUNNING - Press Ctrl+C to stop")
    print("="*60)
    print(f"Publishing JPEG frames on tcp://{args.address}:{args.port}")
    if args.address == "*":
        print("\nClients can connect using your machine's IP address")
        print("Run 'hostname -I' or 'ipconfig' to find your IP")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    frame_times = []
    
    # JPEG encoding parameters
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture or generate frame
            if args.test_pattern:
                frame = create_test_pattern(args.width, args.height, frame_count)
                ret = True
            else:
                ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to capture frame")
                break
            
            # Resize if needed
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            
            # Encode frame as JPEG
            encode_start = time.time()
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            encode_time = (time.time() - encode_start) * 1000
            
            # Publish frame
            publish_start = time.time()
            socket.send(buffer.tobytes())
            publish_time = (time.time() - publish_start) * 1000
            
            frame_count += 1
            
            # Display locally if requested
            if args.display:
                # Add stats overlay
                display_frame = frame.copy()
                stats_text = f"Frame: {frame_count} | FPS: {len(frame_times):.1f}"
                cv2.putText(display_frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('ZMQ Camera Server', display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping server (user pressed 'q')...")
                    break
            
            # Calculate timing statistics
            loop_time = time.time() - loop_start
            frame_times.append(loop_time)
            
            # Keep only last second of frame times for FPS calculation
            if len(frame_times) > args.fps * 2:
                frame_times = frame_times[-args.fps:]
            
            # Print statistics every 2 seconds
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                actual_fps = len(frame_times) / sum(frame_times) if frame_times else 0
                avg_loop_time = np.mean(frame_times) * 1000 if frame_times else 0
                
                print(f"[{elapsed:6.1f}s] Frames: {frame_count:5d} | "
                      f"FPS: {actual_fps:5.1f} | "
                      f"Loop: {avg_loop_time:5.1f}ms | "
                      f"Encode: {encode_time:4.1f}ms | "
                      f"Publish: {publish_time:4.1f}ms")
                
                last_print_time = current_time
            
            # Control frame rate
            target_loop_time = 1.0 / args.fps
            sleep_time = target_loop_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\nStopping server (Ctrl+C pressed)...")
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        
        if args.display:
            cv2.destroyAllWindows()
        
        socket.close()
        context.term()
        
        # Print final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("SERVER STOPPED")
        print("="*60)
        print(f"Total frames published: {frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print("="*60)


if __name__ == "__main__":
    main()

