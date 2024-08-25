import cv2
import time

cap1 = cv2.VideoCapture('/dev/video4')  # Your custom device node
cap2 = cv2.VideoCapture('/dev/video6')

# Check if cameras opened successfully
if not cap1.isOpened():
    print("Failed to open camera 4")
if not cap2.isOpened():
    print("Failed to open camera 6")

cap1.set(cv2.CAP_PROP_FPS, 5)
cap2.set(cv2.CAP_PROP_FPS, 5)

# Read from cameras
for i in range(100):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("error, exiting")
        break

    print(f"read {i+1} frames")
