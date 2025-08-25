import numpy as np
import cv2

CALIB_FILE = "calib_fisheye_1280x720.npz"
CAM_INDEX = 0
ALPHA = 1.0  # 0..1: crop more (0) vs keep FOV (1)

data = np.load(CALIB_FILE, allow_pickle=True)
K = data["K"]
D = data["D"]
WIDTH = int(data["WIDTH"])
HEIGHT = int(data["HEIGHT"])

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Optionally scale new camera matrix (keeps “pinhole-like” view while minimizing black edges)
balance = float(ALPHA)  # 0..1
new_K = K.copy()
# You can scale focal length down a touch to reduce black borders if desired:
# new_K[0,0] *= 0.98
# new_K[1,1] *= 0.98

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, R=np.eye(3), P=new_K, size=(WIDTH, HEIGHT), m1type=cv2.CV_16SC2
)

print("Press q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    side = np.hstack([frame, undist])
    cv2.putText(side, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(side, "Undistorted (pinhole)", (WIDTH+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Fisheye -> Pinhole @ 1280x720 (side-by-side)", side)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
