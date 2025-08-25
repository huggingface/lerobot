import glob, os
import numpy as np
import cv2

# --- Settings ---
IMAGE_DIR = "calib_images"
CHECKERBOARD = (9, 6)           # inner corners (must match capture script)
SQUARE_SIZE = 1.0               # any units; affects translation scale only
WIDTH, HEIGHT = 1280, 720       # target resolution
OUT_FILE = "calib_fisheye_1280x720.npz"

# Collect image paths
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
assert images, f"No images found in {IMAGE_DIR}"

# Prepare object points for one view
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # 3D points
imgpoints = []   # 2D points

for f in images:
    img = cv2.imread(f)
    if img is None:
        continue
    # Down/resize if needed; ensure calibration matches view resolution
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # Corner refinement
        corners = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        imgpoints.append(corners)
        objpoints.append(objp)
    else:
        print("Corners not found:", f)

print(f"Using {len(imgpoints)} valid images.")

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs, tvecs = [], []

# Fisheye calibration flags
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    image_size=(WIDTH, HEIGHT),
    K=K, D=D, rvecs=rvecs, tvecs=tvecs,
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("RMS reprojection error:", rms)
print("K (intrinsics):\n", K)
print("D (distortion):\n", D.ravel())

np.savez(OUT_FILE, K=K, D=D, WIDTH=WIDTH, HEIGHT=HEIGHT, checkerboard=CHECKERBOARD, square_size=SQUARE_SIZE)
print("Saved calibration to", OUT_FILE)
