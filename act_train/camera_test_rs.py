import time
import cv2
import numpy as np
import pyrealsense2 as rs

UVC_INDEX = 12
WIDTH, HEIGHT, FPS = 640, 480, 30
TILE = 320

ctx = rs.context()
rs_devices = list(ctx.query_devices())
if len(rs_devices) < 2:
    print(f"Warning: expected 2 RealSense devices, found {len(rs_devices)}")

print("Resetting RealSense devices...")
serials = [d.get_info(rs.camera_info.serial_number) for d in rs_devices]
for d in rs_devices:
    d.hardware_reset()
time.sleep(5)

ctx = rs.context()
rs_devices = [d for d in ctx.query_devices()
              if d.get_info(rs.camera_info.serial_number) in serials]

pipelines = []
for dev in rs_devices:
    serial = dev.get_info(rs.camera_info.serial_number)
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe = rs.pipeline()
    pipe.start(cfg)
    pipelines.append((serial, pipe))
    print(f"Started RealSense {serial}")

uvc = cv2.VideoCapture(UVC_INDEX, cv2.CAP_V4L2)
uvc.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
uvc.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
if not uvc.isOpened():
    print(f"Warning: could not open UVC camera at /dev/video{UVC_INDEX}")

print("Press 'q' to exit.")

try:
    while True:
        frames = []

        for serial, pipe in pipelines:
            ok, fs = pipe.try_wait_for_frames(timeout_ms=200)
            color = fs.get_color_frame() if ok else None
            if color:
                frames.append(np.asanyarray(color.get_data()))
            else:
                frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

        ret, uvc_frame = uvc.read()
        if not ret:
            uvc_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frames.append(uvc_frame)

        tiles = [cv2.resize(f, (TILE, TILE)) for f in frames]
        placeholder = np.zeros_like(tiles[0])
        top = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], placeholder))
        grid = np.vstack((top, bottom))

        cv2.imshow("Multi-Camera Live Feed", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for _, pipe in pipelines:
        pipe.stop()
    uvc.release()
    cv2.destroyAllWindows()
