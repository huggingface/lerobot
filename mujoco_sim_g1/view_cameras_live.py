#!/usr/bin/env python3
"""
Live camera viewer for MuJoCo simulator using matplotlib
Works without X11/GTK - suitable for SSH sessions with X forwarding
"""
import argparse
import sys
import time
from pathlib import Path

# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sim.sensor_utils import SensorClient, ImageUtils


class CameraViewer:
    def __init__(self, host, port):
        self.client = SensorClient()
        self.client.start_client(server_ip=host, port=port)
        
        self.fig = None
        self.axes = {}
        self.images = {}
        self.text_objs = {}
        
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        
    def init_plot(self):
        """Initialize matplotlib figure and axes"""
        # Wait for first frame to know how many cameras we have
        print("Waiting for first frame to detect cameras...")
        data = self.client.receive_message()
        
        # Parse camera names - handle nested 'images' dict
        camera_names = []
        if "images" in data and isinstance(data["images"], dict):
            # Nested structure: data["images"]["camera_name"]
            camera_names = list(data["images"].keys())
        else:
            # Flat structure: data["camera_name"] directly
            camera_names = [k for k in data.keys() if k not in ["timestamps", "images"]]
        
        num_cameras = len(camera_names)
        
        if num_cameras == 0:
            print("No cameras found in stream!")
            return False
        
        print(f"Found {num_cameras} camera(s): {', '.join(camera_names)}")
        
        # Create subplots
        if num_cameras == 1:
            self.fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes_list = [ax]
        elif num_cameras == 2:
            self.fig, axes_list = plt.subplots(1, 2, figsize=(16, 6))
        else:
            rows = (num_cameras + 1) // 2
            self.fig, axes_list = plt.subplots(rows, 2, figsize=(16, 6 * rows))
            axes_list = axes_list.flatten()
        
        # Initialize each camera subplot
        for i, cam_name in enumerate(camera_names):
            ax = axes_list[i]
            ax.set_title(f"{cam_name}", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Get image data from nested or flat structure
            if "images" in data and cam_name in data["images"]:
                img_data = data["images"][cam_name]
            elif cam_name in data:
                img_data = data[cam_name]
            else:
                img_data = cam_name  # Use the actual data if it's the value
            
            # Decode first image
            if isinstance(img_data, str):
                img = ImageUtils.decode_image(img_data)
            elif isinstance(img_data, np.ndarray):
                img = img_data
            else:
                print(f"Warning: Unknown image format for {cam_name}: {type(img_data)}")
                continue
            
            # Check if image is valid
            if img is None or not isinstance(img, np.ndarray):
                print(f"Warning: Invalid image data for {cam_name}")
                continue
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image
            im = ax.imshow(img_rgb)
            self.images[cam_name] = im
            self.axes[cam_name] = ax
            
            # Add FPS text
            text = ax.text(0.02, 0.98, 'FPS: 0.0', 
                          transform=ax.transAxes,
                          fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                          color='lime',
                          fontweight='bold')
            self.text_objs[cam_name] = text
        
        # Hide unused subplots
        if num_cameras < len(axes_list):
            for i in range(num_cameras, len(axes_list)):
                axes_list[i].axis('off')
        
        self.fig.tight_layout()
        return True
    
    def update_frame(self, frame_num):
        """Update function for animation"""
        try:
            # Receive new frame
            data = self.client.receive_message()
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
            
            # Update each camera
            for cam_name in self.images.keys():
                # Get image data from nested or flat structure
                if "images" in data and cam_name in data["images"]:
                    img_data = data["images"][cam_name]
                elif cam_name in data:
                    img_data = data[cam_name]
                else:
                    continue
                
                # Decode image
                if isinstance(img_data, str):
                    img = ImageUtils.decode_image(img_data)
                elif isinstance(img_data, np.ndarray):
                    img = img_data
                else:
                    continue
                
                # Check if image is valid
                if img is None or not isinstance(img, np.ndarray):
                    continue
                
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Update image
                self.images[cam_name].set_data(img_rgb)
                
                # Update FPS text
                self.text_objs[cam_name].set_text(f'FPS: {self.fps:.1f}')
            
        except Exception as e:
            print(f"Error updating frame: {e}")
        
        return list(self.images.values()) + list(self.text_objs.values())
    
    def start(self, interval=33):
        """Start the live viewer"""
        if not self.init_plot():
            return
        
        print(f"\n{'='*60}")
        print("📹 Live camera viewer started!")
        print("Close the window or press Ctrl+C to exit")
        print(f"{'='*60}\n")
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=interval,  # ms between frames
            blit=True,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping viewer...")
        finally:
            self.client.stop_client()
            plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Live camera viewer for MuJoCo simulator")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Simulator host address (default: localhost)")
    parser.add_argument("--port", type=int, default=5555,
                       help="ZMQ port (default: 5555)")
    parser.add_argument("--interval", type=int, default=33,
                       help="Update interval in ms (default: 33 = ~30fps)")
    args = parser.parse_args()
    
    print("="*60)
    print("📷 MuJoCo Live Camera Viewer (matplotlib)")
    print("="*60)
    print(f"🌐 Connecting to: tcp://{args.host}:{args.port}")
    print(f"⏱️  Update interval: {args.interval}ms (~{1000/args.interval:.0f} fps)")
    print("="*60)
    
    viewer = CameraViewer(host=args.host, port=args.port)
    viewer.start(interval=args.interval)


if __name__ == "__main__":
    main()

