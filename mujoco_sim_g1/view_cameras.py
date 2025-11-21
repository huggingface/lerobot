#!/usr/bin/env python3
"""
Camera recorder for MuJoCo simulator
Subscribes to ZMQ camera feed and saves images to disk
"""
import argparse
import sys
import time
from pathlib import Path

# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from sim.sensor_utils import SensorClient


def main():
    parser = argparse.ArgumentParser(description="Save camera streams from MuJoCo simulator")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Simulator host address (default: localhost)")
    parser.add_argument("--port", type=int, default=5555,
                       help="ZMQ port (default: 5555)")
    parser.add_argument("--save-dir", type=str, default="./camera_recordings",
                       help="Directory to save images (default: ./camera_recordings)")
    parser.add_argument("--save-rate", type=int, default=5,
                       help="Save every Nth frame (default: 5 = ~6Hz at 30fps stream)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to save before exiting (default: unlimited)")
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("📷 MuJoCo Camera Recorder")
    print("="*60)
    print(f"🌐 Connecting to: tcp://{args.host}:{args.port}")
    print(f"💾 Saving to: {save_dir.absolute()}")
    print(f"⏬ Save rate: Every {args.save_rate} frames")
    if args.max_frames:
        print(f"🎬 Max frames: {args.max_frames}")
    print("="*60)
    print("\nWaiting for camera data...")
    print("Press Ctrl+C to stop recording\n")
    
    # Connect to sensor server
    client = SensorClient()
    client.start_client(server_ip=args.host, port=args.port)
    
    frame_count = 0
    saved_count = 0
    last_time = time.time()
    fps_display = 0
    
    # Track per-camera frame counts
    camera_frame_counts = {}
    
    try:
        while True:
            try:
                # Receive image data
                data = client.receive_message()
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps_display = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                
                # Process each camera
                for key, value in data.items():
                    if key == "timestamps":
                        continue
                    
                    # Initialize counter for this camera
                    if key not in camera_frame_counts:
                        camera_frame_counts[key] = 0
                        # Create subdirectory for each camera
                        (save_dir / key).mkdir(exist_ok=True)
                    
                    camera_frame_counts[key] += 1
                    
                    # Only save every Nth frame
                    if camera_frame_counts[key] % args.save_rate != 0:
                        continue
                    
                    # Decode image if it's a string (base64 encoded)
                    if isinstance(value, str):
                        from sim.sensor_utils import ImageUtils
                        img = ImageUtils.decode_image(value)
                    elif isinstance(value, np.ndarray):
                        img = value
                    else:
                        continue
                    
                    # Save image
                    filename = f"{key}_{saved_count:06d}.jpg"
                    save_path = save_dir / key / filename
                    cv2.imwrite(str(save_path), img)
                    
                    # Print progress
                    if saved_count % 10 == 0:
                        print(f"[{fps_display:5.1f} FPS] Saved {saved_count:4d} frames to {key}/")
                    
                    saved_count += 1
                    
                    # Check if we've reached max frames
                    if args.max_frames and saved_count >= args.max_frames:
                        print(f"\n✓ Reached max frames ({args.max_frames}). Stopping...")
                        return
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error receiving/saving frame: {e}")
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"\n\n✓ Recording stopped")
        print(f"📊 Total frames saved: {saved_count}")
        print(f"📁 Location: {save_dir.absolute()}")
    finally:
        client.stop_client()


if __name__ == "__main__":
    main()

