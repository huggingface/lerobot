import time
import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
TILE = 320

ctx = rs.context()
rs_devices = list(ctx.query_devices())
if len(rs_devices) < 3:
    print(f"Warning: expected 3 RealSense devices, found {len(rs_devices)}")

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

        # Pad to 4 tiles if fewer cameras are connected.
        while len(frames) < 4:
            frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

        tiles = [cv2.resize(f, (TILE, TILE)) for f in frames[:4]]
        top = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], tiles[3]))
        grid = np.vstack((top, bottom))

        cv2.imshow("Multi-Camera Live Feed", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for _, pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()
