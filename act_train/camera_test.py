import cv2
import numpy as np

cam_ids = [11, 12, 14]
caps = [cv2.VideoCapture(i) for i in cam_ids]

# Verification: Check if all cameras opened
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Warning: Could not open camera {cam_ids[i]}")

print("Press 'q' to exit.")

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"{fps} frames per second")

        if not ret:
            # Create a black placeholder if a frame is missing
            frame = np.zeros((640, 360, 3), dtype=np.uint8)
        
        # Resize to the desired resolution
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)

    placeholder = np.zeros_like(frames[0])
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], placeholder))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Multi-Camera Live Feed", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
