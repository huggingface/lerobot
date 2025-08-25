import numpy as np
import cv2

CALIB_FILE = "calib_fisheye_1280x720.npz"
CAM_INDEX = 10

# --- Load calibration ---
data = np.load(CALIB_FILE, allow_pickle=True)
K = data["K"]
D = data["D"]
WIDTH = int(data["WIDTH"])
HEIGHT = int(data["HEIGHT"])

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Precompute undistortion maps
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, R=np.eye(3), P=K, size=(WIDTH, HEIGHT), m1type=cv2.CV_16SC2
)

def adjust_brightness(img, factor=1.0):
    """Increase brightness by multiplying pixel values."""
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def zoom_center(img, zoom=1.0):
    """Zoom into the center of an image and resize back."""
    h, w = img.shape[:2]
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

print("Press q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    # Undistort
    undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Apply brightness + zoom
    undist_mod = adjust_brightness(undist, factor=1.0)  # +20% brightness
    undist_mod = zoom_center(undist_mod, zoom=1.0)      # 1.5x zoom

    # Side-by-side view
    side = np.hstack([frame, undist_mod])
    cv2.putText(side, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(side, "Undistorted + Bright + Zoom", (WIDTH+20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("USB Camera @1280x720", side)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
