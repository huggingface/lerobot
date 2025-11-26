import cv2
from pathlib import Path

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Camera Configuration - 2 cameras at NATIVE resolution @ 30 FPS
# 
# WORKING: 2 cameras on Bus 003 at full native resolution:
# - /dev/video0: Microsoft LifeCam - 1280x720 @ 30fps (native HD 720p) ✓
# - /dev/video2: Lenovo Camera - 1920x1080 @ 30fps (native Full HD 1080p) ✓
# - /dev/video6: UVC Camera - DISABLED (3rd camera exceeds USB bandwidth)
#
# ⚠️ TO ENABLE ALL 3 CAMERAS AT NATIVE RESOLUTION:
# Cameras MUST be on DIFFERENT USB BUSES (not just different ports on same bus!)
# 
# STEP 1: Test each USB port to find which bus it's on:
#   1. Unplug all cameras
#   2. Plug ONE camera into a port
#   3. Run: lsusb -t | grep -B2 "Driver=uvcvideo"  
#   4. Note the "Bus XXX" number
#   5. Repeat for each USB port on your computer (front panel, rear, USB 3.0 ports)
#
# STEP 2: Find 3 ports on DIFFERENT bus numbers (e.g., Bus 001, Bus 002, Bus 003)
#
# STEP 3: Plug the 3 cameras into those different buses
#
# STEP 4: Verify: lsusb -t (should show 3 different "Bus XXX" numbers)
#
# STEP 5: Run script - if device paths changed, update them with v4l2-ctl --list-devices
#
# STEP 6: Uncomment camera_front_high lines throughout this file

camera_config_front_low = OpenCVCameraConfig(
    index_or_path='/dev/video0',
    fps=30.0,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG",
)

camera_config_top = OpenCVCameraConfig(
    index_or_path='/dev/video2',
    fps=30.0,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG",
)

camera_config_front_high = OpenCVCameraConfig(
    index_or_path='/dev/video6',  # video6 = Video Capture, video7 = Metadata only
    fps=30.0,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG",
    warmup_s=2,
)

# Create output directory
output_dir = Path("outputs/captured_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Instantiate and connect cameras
camera_front_low = OpenCVCamera(camera_config_front_low)
camera_top = OpenCVCamera(camera_config_top)
# camera_front_high = OpenCVCamera(camera_config_front_high)  # DISABLED - see note below

print("Connecting cameras...")
camera_front_low.connect()
print("  ✓ Front Low connected")
camera_top.connect()
print("  ✓ Top connected")
# camera_front_high.connect()  # DISABLED
# print("  ✓ Front High connected")

# ⚠️ THIRD CAMERA DISABLED: All 3 cameras on Bus 003 = insufficient bandwidth
# TO ENABLE THIRD CAMERA: Move it to a different USB bus (see instructions in comments above)

# Read and save frames
print(f"\n✓ 2 cameras active! Streaming... Press 'q' to quit or Ctrl+C")
print(f"Saving frames to {output_dir}\n")
try:
    frame_count = 0
    while True:
        # Read frames from active cameras
        frame_front_low = camera_front_low.async_read(timeout_ms=500)
        frame_top = camera_top.async_read(timeout_ms=500)

        # Convert RGB to BGR for OpenCV saving and display
        frame_front_low_bgr = cv2.cvtColor(frame_front_low, cv2.COLOR_RGB2BGR)
        frame_top_bgr = cv2.cvtColor(frame_top, cv2.COLOR_RGB2BGR)

        # Display frames in real-time windows
        cv2.imshow('Front Camera Low', frame_front_low_bgr)
        cv2.imshow('Top Camera', frame_top_bgr)

        # Save frames periodically (every 30 frames)
        if frame_count % 30 == 0:
            front_low_path = output_dir / f"front_low_frame_{frame_count:06d}.png"
            top_path = output_dir / f"top_frame_{frame_count:06d}.png"

            cv2.imwrite(str(front_low_path), frame_front_low_bgr)
            cv2.imwrite(str(top_path), frame_top_bgr)
            
            print(
                f"Frame {frame_count} - Front Low: {frame_front_low.shape}, "
                f"Top: {frame_top.shape} - Saved"
            )
        
        frame_count += 1

        # Wait 1ms and check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    camera_front_low.disconnect()
    camera_top.disconnect()
    # camera_front_high.disconnect()  # DISABLED
    cv2.destroyAllWindows()
    print(f"\nCaptured {frame_count} frames total")
