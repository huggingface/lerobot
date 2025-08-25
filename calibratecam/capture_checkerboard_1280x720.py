import cv2, os, time

# --- Settings ---
CAM_INDEX = 4                 # change if needed
WIDTH, HEIGHT = 1280, 720
SAVE_DIR = "calib_images"
EXPOSURE_HINT = None          # e.g. -6 for some UVC cams, or None to skip
# INNER corner count of your printed board (NOT squares!)
CHECKERBOARD = (9, 6)         # ==> 9 by 6 inner corners (print ~10x7 squares)

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # more reliable on many USB cams
# Try increasing brightness/exposure/gain
'''
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)   # range usually [0.0,1.0] but depends on driver
cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # smaller values = brighter (common UVC quirk)
cap.set(cv2.CAP_PROP_GAIN, 10)          # optional extra boost
'''


if EXPOSURE_HINT is not None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (varies by driver)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(EXPOSURE_HINT))

print("Press SPACE to save an image, q to quit.")
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame.")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    cv2.putText(display, f"Found corners: {ret}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if ret else (0,0,255), 2)

    if ret:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)

    cv2.imshow("Capture @ 1280x720", display)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        path = os.path.join(SAVE_DIR, f"img_{i:03d}.png")
        cv2.imwrite(path, frame)
        print("Saved:", path)
        i += 1
        time.sleep(0.2)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
