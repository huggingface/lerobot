import cv2
import os
import time
import numpy as np
from datetime import datetime

# --- Settings ---
CAM_INDEX = 0                # change if your cam is not at index 0
WIDTH, HEIGHT = 1280, 720
CALIB_FILE = "calib_fisheye_1280x720.npz"
OUT_DIR = "captures_undistorted"
PREFIX = "undist"

SHOW_ORIGINAL = False   # set True to display side-by-side
SAVE_ORIGINAL = False   # set True to also save original images

os.makedirs(OUT_DIR, exist_ok=True)

# --- Load calibration ---
data = np.load(CALIB_FILE, allow_pickle=True)
K, D = data["K"], data["D"]

# Optional: adjust new camera matrix to crop/zoom slightly
new_K = K.copy()

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, R=np.eye(3), P=new_K, size=(WIDTH, HEIGHT), m1type=cv2.CV_16SC2
)

# --- Open camera ---
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)  # force V4L2 backend
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Warmup
time.sleep(0.2)
for _ in range(3):
    cap.read()

print("Controls: SPACE = save photo, q = quit")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Failed to read frame")
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if SHOW_ORIGINAL:
        canvas = np.hstack([frame, undist])
        cv2.imshow("Original | Undistorted", canvas)
    else:
        cv2.imshow("Undistorted", undist)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        undist_path = os.path.join(OUT_DIR, f"{PREFIX}_{ts}_undist.png")
        cv2.imwrite(undist_path, undist)
        print("Saved:", undist_path)

        if SAVE_ORIGINAL:
            orig_path = os.path.join(OUT_DIR, f"{PREFIX}_{ts}_orig.png")
            cv2.imwrite(orig_path, frame)
            print("Saved:", orig_path)

cap.release()
cv2.destroyAllWindows()
