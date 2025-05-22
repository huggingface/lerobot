import cv2
import pyrealsense2 as rs
import numpy as np

def capture_realsense_color():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    
    try:
        while True:
            # Wait for a coherent color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("Error: No color frame received")
                return None

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # # Show image
            # cv2.imshow('RealSense Color Stream', color_image)
            # cv2.waitKey(0)  # Wait for a key press
            # cv2.destroyAllWindows()

    finally:
        # Stop streaming
        pipeline.stop()

def main():
    # Ensure RealSense camera is connected
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense device found")
        return

    # Capture and display color image
    capture_realsense_color()

if __name__ == "__main__":
    main()