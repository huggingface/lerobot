import cv2

def test_resolutions(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Common resolutions to test
    resolutions = [
        (320, 240),
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]

    print(f"--- Testing Camera {camera_index} ---")
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        status = "MATCH" if (actual_w == w and actual_h == h) else "FAILED (Forced to {}x{})".format(int(actual_w), int(actual_h))
        print(f"Requested: {w}x{h} -> Actual: {int(actual_w)}x{int(actual_h)} [{status}]")

    cap.release()

if __name__ == "__main__":
    # Change index if your iPhone is not the default (0)
    test_resolutions(camera_index=2)
