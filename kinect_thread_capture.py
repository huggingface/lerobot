# kinect_thread_capture.py
import time
import threading
import sys
import numpy as np

# Enable display
import cv2

from pylibfreenect2 import (
    Freenect2,
    SyncMultiFrameListener,
    FrameType,
    Registration,
    Frame,
)

# Try fastest pipelines first
def make_best_pipeline():
    try:
        from pylibfreenect2 import CudaPacketPipeline
        return CudaPacketPipeline()
    except Exception:
        pass
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        return OpenCLPacketPipeline(-1)
    except Exception:
        pass
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        return OpenGLPacketPipeline(False)
    except Exception:
        pass
    from pylibfreenect2 import CpuPacketPipeline
    return CpuPacketPipeline()

class KinectCapture:
    def __init__(self):
        # Optional FPS limiter
        self.target_fps = None  # don't limit FPS
        self._target_period = (1.0 / self.target_fps) if self.target_fps else None
        self._next_tick = time.perf_counter()

        # Optional display and copying
        self.show = True  # display with OpenCV windows
        self.copy_frames = False  # set True if you need persistent copies per frame
        # Rotation for both RGB and depth displays
        self.rotation_flag = cv2.ROTATE_90_CLOCKWISE

        self.pipeline = make_best_pipeline()

        self.fn = Freenect2()
        n = self.fn.enumerateDevices()
        if n == 0:
            raise RuntimeError("No Kinect v2 device found")

        self.device = self.fn.openDefaultDevice(self.pipeline)
        if self.device is None:
            raise RuntimeError("Failed to open Kinect v2 device")

        # Listen to color + depth
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        # Registration (for alignment and undistortion)
        ir_params = self.device.getIrCameraParams()
        color_params = self.device.getColorCameraParams()
        self.registration = Registration(ir_params, color_params)

        # Preallocated numpy buffers (depth only). Color is read as BGRA without conversion.
        self.depth_buf = np.empty((424, 512), dtype=np.float32)      # depth in mm

        # Shared state
        self._stop = threading.Event()
        self._thread = None

        self.last_fps = 0.0
        self._sec_count = 0
        self._sec_start = time.perf_counter()

        # Latest frame copies (optional; avoid copying if not needed)
        self.latest_bgr = None
        self.latest_depth_mm = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="KinectCaptureLoop", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.device.stop()
        self.device.close()

    def _loop(self):
        while not self._stop.is_set():
            frames = self.listener.waitForNewFrame()
            try:
                color = frames["color"]
                depth = frames["depth"]

                # Color: keep BGRA (no BGRA→BGR conversion) and Depth: memcpy
                color_bgra = color.asarray()
                depth_mm = depth.asarray_optimized(self.depth_buf)

                # Optionally keep copies for consumers
                if self.copy_frames:
                    self.latest_bgr = color_bgra.copy()
                    self.latest_depth_mm = depth_mm.copy()
                else:
                    # Warning: these reference reusable buffers and will change next loop
                    self.latest_bgr = color_bgra
                    self.latest_depth_mm = depth_mm

                # FPS accounting (per second)
                self._sec_count += 1
                now = time.perf_counter()
                if now - self._sec_start >= 1.0:
                    self.last_fps = self._sec_count / (now - self._sec_start)
                    print(f"FPS (post-transform): {self.last_fps:.1f}")
                    self._sec_count = 0
                    self._sec_start = now

                # Display
                if self.show:
                    try:
                        # Overlay FPS (per-second post-transform rate)
                        overlay_text = f"FPS: {self.last_fps:.1f}"

                        # Prepare display frames (rotate both RGB and depth)
                        disp_color = cv2.rotate(color_bgra, self.rotation_flag)
                        disp_depth = cv2.rotate((depth_mm / 4500.0).astype(np.float32), self.rotation_flag)

                        cv2.putText(disp_color, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                        cv2.putText(disp_depth, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (1.0,1.0,1.0), 2)

                        cv2.imshow("RGB (raw 1920x1080, BGRA)", disp_color)
                        cv2.imshow("Depth raw (normalized)", disp_depth)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self._stop.set()
                    except cv2.error:
                        # HighGUI may be unavailable in this environment
                        pass

            finally:
                self.listener.release(frames)

def main():
    cap = KinectCapture()
    cap.start()
    print("Capturing... press Ctrl+C to stop")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        print("Stopped.")

if __name__ == "__main__":
    main()