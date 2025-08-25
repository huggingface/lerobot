#!/usr/bin/env python

import cv2


def main() -> None:
    device_path = "/dev/video8"
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Failed to open camera at {device_path}")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        raise SystemExit(1)

    h, w = frame.shape[:2]
    print(f"Device: {device_path}")
    print(f"Resolution: {w}x{h}")
    print(f"FPS: {fps:.2f}")

    window_name = "Webcam (/dev/video4)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.imshow(window_name, frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


