import cv2
import os

save_path = "captured_image.jpg"  # Local path to save

for device_id in range(40):  # 0 to 39
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        continue  # Skip if camera not available
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(save_path, frame)
        print(f"Captured image from /dev/video{device_id} and saved to {save_path}")
        break  # Stop after first successful capture
else:
    print("No available camera found from /dev/video0 to /dev/video39.")

