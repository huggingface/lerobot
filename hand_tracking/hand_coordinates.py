#!/usr/bin/env python3.10
# filepath: /Users/omarahmed/Desktop/Projects/Tech Europe - AI Hackathon/hand_coordinates.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np


model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,  # Detect up to 2 hands
    min_hand_detection_confidence=0.1,  # Lower threshold for better detection
    min_hand_presence_confidence=0.1,   # Lower threshold for tracking
    min_tracking_confidence=0.3         # Lower threshold for tracking
)

cap = cv.VideoCapture('training_vids/gripPush1.mp4')

# Get video properties for frame timing
fps = cap.get(cv.CAP_PROP_FPS)
frame_time_ms = int(1000 / fps) if fps > 0 else 33  # Default to ~30fps if fps detection fails

frame_count = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
     
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int(frame_count * 1000 / fps) if fps > 0 else frame_count * 33
        
        # Detect hand landmarks
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Only draw landmarks for thumb and index finger
                thumb_index_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Wrist, thumb joints, index joints
                
                for idx in thumb_index_landmarks:
                    if idx < len(hand_landmarks):
                        landmark = hand_landmarks[idx]
                        # Convert normalized coordinates to pixel coordinates
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        
                        # Draw landmark points
                        cv.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                        
                        # Draw landmark numbers (optional)
                        cv.putText(annotated_frame, str(idx), (x + 5, y - 5), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw connections between landmarks
                connections = [
                    # Thumb (complete structure)
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    # Index finger (complete structure)
                    (0, 5), (5, 6), (6, 7), (7, 8)
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    start_landmark = hand_landmarks[start_idx]
                    end_landmark = hand_landmarks[end_idx]
                    
                    start_x = int(start_landmark.x * frame.shape[1])
                    start_y = int(start_landmark.y * frame.shape[0])
                    end_x = int(end_landmark.x * frame.shape[1])
                    end_y = int(end_landmark.y * frame.shape[0])
                    
                    cv.line(annotated_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            
            # Print coordinates when hand is detected
            if len(detection_result.hand_landmarks) > 0:
                wrist = detection_result.hand_landmarks[0][0]  # Wrist is landmark 0
                index_tip = detection_result.hand_landmarks[0][8]  # Index finger tip
                thumb_tip = detection_result.hand_landmarks[0][4]  # Thumb tip
                
                print(f"Frame {frame_count}: Wrist({wrist.x:.3f}, {wrist.y:.3f}), "
                      f"Index({index_tip.x:.3f}, {index_tip.y:.3f}), "
                      f"Thumb({thumb_tip.x:.3f}, {thumb_tip.y:.3f})")
        else:
            # Print when no hand is detected
            print(f"Frame {frame_count}: No hand detected")
        
        # Display the annotated frame
        cv.imshow('Hand Tracking', annotated_frame)
        
        # Exit on 'q' key press
        if cv.waitKey(frame_time_ms) == ord('q'):
            break
        
        frame_count += 1
 
cap.release()
cv.destroyAllWindows()

