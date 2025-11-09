import cv2

def find_cameras(max_tested=10):
    """Return a list of available camera indices."""
    available_cams = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

def capture_from_cameras(camera_indices):
    """Capture one frame from each available camera."""
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Camera {idx} failed to open.")
            continue
        ret, frame = cap.read()
        if ret:
            filename = f"camera_{idx}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved image from camera {idx} as {filename}")
        else:
            print(f"Failed to capture from camera {idx}")
        cap.release()

if __name__ == "__main__":
    cameras = find_cameras()
    if cameras:
        print(f"Available cameras: {cameras}")
        capture_from_cameras(cameras)
    else:
        print("No cameras found.")

