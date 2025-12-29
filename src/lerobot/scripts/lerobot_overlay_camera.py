#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Camera overlay tool for precise item positioning.

Overlays a live camera feed on top of a reference image to help position
items in the exact same location. Useful for reproducible robot setups.

Open http://localhost:8000 to view the overlay.
Use the slider to adjust the blend between live feed and reference image.
Press Ctrl+C to stop.

Example:

```shell
# Overlay camera feed on reference image
lerobot-overlay-camera --camera-id 128422270679 --reference-image /path/to/reference.png

# Specify camera resolution
lerobot-overlay-camera --camera-id 128422270679 --reference-image /path/to/image.png --width 1280 --height 720

# Use a different port
lerobot-overlay-camera --camera-id 128422270679 --reference-image /path/to/image.png --port 9000
```
"""

import argparse
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global state
camera_state: dict = {}
running = True
blend_alpha = 0.5  # 0.0 = full reference, 1.0 = full live feed


def load_reference_image(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """Load and resize reference image to match camera resolution."""
    logger.info(f"Loading reference image: {image_path}")
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to match camera resolution
    img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    logger.info(f"Reference image resized to {target_width}x{target_height}")
    
    return img_resized


def create_camera(camera_id: str, width: int, height: int, fps: int) -> RealSenseCamera:
    """Create and connect to a RealSense camera."""
    logger.info(f"Connecting to RealSense camera: {camera_id}")
    
    config = RealSenseCameraConfig(
        serial_number_or_name=camera_id,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.BGR,
    )
    camera = RealSenseCamera(config)
    camera.connect(warmup=True)
    logger.info(f"Connected to RealSense camera: {camera_id}")
    
    return camera


def blend_frames(live_frame: np.ndarray, reference_frame: np.ndarray, alpha: float) -> np.ndarray:
    """Blend live frame with reference image.
    
    Args:
        live_frame: Current camera frame
        reference_frame: Reference image
        alpha: Blend factor (0.0 = full reference, 1.0 = full live)
    
    Returns:
        Blended frame
    """
    return cv2.addWeighted(live_frame, alpha, reference_frame, 1 - alpha, 0)


def add_overlay_info(frame: np.ndarray, alpha: float, fps: float) -> np.ndarray:
    """Add information overlay to the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Blend info
    live_pct = int(alpha * 100)
    ref_pct = int((1 - alpha) * 100)
    
    cv2.putText(frame, f"Live: {live_pct}% | Reference: {ref_pct}%", (20, 35), font, 0.6, (0, 255, 136), 1)
    cv2.putText(frame, f"FPS: {fps:.1f} | {w}x{h}", (20, 55), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Adjust blend with slider in browser", (20, 80), font, 0.45, (150, 150, 150), 1)
    
    return frame


def camera_capture_thread():
    """Thread to continuously capture frames from the camera."""
    global running, blend_alpha
    
    cam = camera_state["camera"]
    reference = camera_state["reference"]
    
    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0
    
    while running:
        try:
            live_frame = cam.read()
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - last_fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = current_time
            
            # Blend frames
            blended = blend_frames(live_frame, reference, blend_alpha)
            
            # Add overlay info
            blended_with_info = add_overlay_info(blended, blend_alpha, fps)
            
            # Store frame
            with camera_state["lock"]:
                camera_state["frame"] = blended_with_info
                camera_state["fps"] = fps
                
        except Exception as e:
            logger.warning(f"Error reading from camera: {e}")
            time.sleep(0.1)


def generate_html_page() -> str:
    """Generate HTML page with camera overlay and controls."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>LeRobot Camera Overlay</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: linear-gradient(145deg, #0d0d0d 0%, #1a1a2e 40%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        header {
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(12px);
            padding: 18px 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #00ff88;
            box-shadow: 0 4px 30px rgba(0, 255, 136, 0.1);
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        h1 {
            font-size: 1.5em;
            font-weight: 400;
            letter-spacing: 2px;
            color: #00ff88;
            text-shadow: 0 0 25px rgba(0, 255, 136, 0.4);
        }
        .icon {
            font-size: 1.8em;
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 20px;
            background: rgba(0, 0, 0, 0.4);
            padding: 12px 20px;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .slider-label {
            font-size: 0.85em;
            color: #888;
            min-width: 80px;
        }
        .slider-value {
            font-size: 0.9em;
            color: #00ff88;
            font-weight: bold;
            min-width: 45px;
            text-align: right;
        }
        input[type="range"] {
            -webkit-appearance: none;
            width: 200px;
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(90deg, #ff6b6b 0%, #00ff88 100%);
            outline: none;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            transition: transform 0.1s;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        .preset-btns {
            display: flex;
            gap: 8px;
        }
        .preset-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #444;
            color: #ccc;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75em;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }
        .preset-btn:hover {
            background: rgba(0, 255, 136, 0.2);
            border-color: #00ff88;
            color: #fff;
        }
        main {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 30px;
        }
        .video-container {
            background: rgba(15, 15, 25, 0.9);
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid #333;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 255, 136, 0.05);
            transition: border-color 0.3s;
        }
        .video-container:hover {
            border-color: #00ff88;
        }
        .video-container img {
            display: block;
            max-width: 90vw;
            max-height: 75vh;
            background: #111;
        }
        .status-bar {
            background: rgba(0, 0, 0, 0.6);
            padding: 12px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8em;
            color: #666;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff88; }
            50% { opacity: 0.5; box-shadow: 0 0 5px #00ff88; }
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #444;
            font-size: 0.75em;
        }
        .help-text {
            margin-top: 8px;
            font-size: 0.7em;
            color: #555;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span class="icon">📷</span>
            <h1>OVERLAY CAMERA</h1>
        </div>
        <div class="controls">
            <div class="slider-container">
                <span class="slider-label">Reference</span>
                <input type="range" id="blendSlider" min="0" max="100" value="50">
                <span class="slider-label">Live</span>
                <span class="slider-value" id="blendValue">50%</span>
            </div>
            <div class="preset-btns">
                <button class="preset-btn" onclick="setBlend(0)">Ref Only</button>
                <button class="preset-btn" onclick="setBlend(25)">25%</button>
                <button class="preset-btn" onclick="setBlend(50)">50%</button>
                <button class="preset-btn" onclick="setBlend(75)">75%</button>
                <button class="preset-btn" onclick="setBlend(100)">Live Only</button>
            </div>
        </div>
    </header>
    <main>
        <div class="video-container">
            <img id="streamImg" src="/stream" alt="Camera Overlay">
            <div class="status-bar">
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Streaming</span>
                </div>
                <span>Position items to match the reference image</span>
            </div>
        </div>
    </main>
    <footer>
        LeRobot Camera Overlay Tool • Press Ctrl+C in terminal to stop
        <div class="help-text">
            Tip: Use 50% blend to see both live feed and reference simultaneously
        </div>
    </footer>
    <script>
        const slider = document.getElementById('blendSlider');
        const valueDisplay = document.getElementById('blendValue');
        const streamImg = document.getElementById('streamImg');
        
        function setBlend(value) {
            slider.value = value;
            updateBlend();
        }
        
        function updateBlend() {
            const value = slider.value;
            valueDisplay.textContent = value + '%';
            
            // Update stream URL with new blend value
            fetch('/set_blend?alpha=' + (value / 100));
        }
        
        slider.addEventListener('input', updateBlend);
        
        // Refresh stream periodically to ensure smooth updates
        setInterval(() => {
            const newSrc = '/stream?' + new Date().getTime();
            // Only update src if there's a change to avoid flicker
        }, 100);
    </script>
</body>
</html>'''
    return html


class OverlayStreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for overlay streaming."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        global running, blend_alpha
        
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == "/" or path == "/index.html":
            # Serve HTML page
            content = generate_html_page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
            
        elif path == "/set_blend":
            # Update blend alpha
            try:
                alpha = float(query.get("alpha", [0.5])[0])
                blend_alpha = max(0.0, min(1.0, alpha))
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            except ValueError:
                self.send_error(400, "Invalid alpha value")
                
        elif path == "/stream":
            # Stream MJPEG
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            
            while running:
                with camera_state["lock"]:
                    frame = camera_state.get("frame")
                
                if frame is not None:
                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpeg_bytes = jpeg.tobytes()
                    
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                
                time.sleep(0.033)  # ~30 FPS
        else:
            self.send_error(404, "Not found")


def run_overlay_camera(
    camera_id: str,
    reference_image: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    port: int = 8000,
    initial_blend: float = 0.5,
):
    """
    Run camera overlay preview with web streaming.
    
    Args:
        camera_id: RealSense camera serial number or name.
        reference_image: Path to reference image.
        width: Camera width resolution.
        height: Camera height resolution.
        fps: Camera FPS.
        port: HTTP server port.
        initial_blend: Initial blend value (0.0 = reference, 1.0 = live).
    """
    global camera_state, running, blend_alpha
    
    blend_alpha = initial_blend
    
    # Load reference image
    try:
        reference = load_reference_image(reference_image, width, height)
    except Exception as e:
        logger.error(f"Failed to load reference image: {e}")
        return
    
    # Connect to camera
    try:
        camera = create_camera(camera_id, width, height, fps)
    except Exception as e:
        logger.error(f"Failed to connect to camera: {e}")
        return
    
    # Initialize state
    camera_state = {
        "camera": camera,
        "reference": reference,
        "frame": None,
        "fps": 0.0,
        "lock": threading.Lock(),
    }
    
    # Start capture thread
    capture_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    capture_thread.start()
    
    # Wait for first frame
    time.sleep(0.5)
    
    # Setup signal handler
    def signal_handler(sig, frame):
        global running
        print("\n\nShutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start HTTP server
    server = HTTPServer(("0.0.0.0", port), OverlayStreamHandler)
    server.timeout = 1
    
    print(f"\n{'='*55}")
    print(f"📷 Camera overlay preview running!")
    print(f"   Open in browser: http://localhost:{port}")
    print(f"   Camera: {camera_id} ({width}x{height} @ {fps}fps)")
    print(f"   Reference: {reference_image}")
    print(f"{'='*55}")
    print("Use the slider in the browser to adjust blend.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while running:
            server.handle_request()
    finally:
        print("\nCleaning up...")
        running = False
        server.server_close()
        
        try:
            if camera.is_connected:
                camera.disconnect()
                logger.info(f"Disconnected camera: {camera_id}")
        except Exception as e:
            logger.error(f"Error disconnecting camera: {e}")
        
        camera_state.clear()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay live camera feed on a reference image for precise item positioning."
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        required=True,
        help="RealSense camera serial number or name.",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        required=True,
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width resolution (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height resolution (default: 480).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port (default: 8000).",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.5,
        help="Initial blend value: 0.0 = reference only, 1.0 = live only (default: 0.5).",
    )
    
    args = parser.parse_args()
    
    run_overlay_camera(
        camera_id=args.camera_id,
        reference_image=args.reference_image,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port,
        initial_blend=args.blend,
    )


if __name__ == "__main__":
    main()

