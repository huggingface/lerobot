# Hand Tracking Project - AI Hackathon

## Overview
This project implements real-time hand tracking using MediaPipe for gesture recognition and analysis. The system focuses on tracking thumb and index finger movements for gesture classification.

## Files Structure
```
hand_tracking/
├── hand_coordinates.py      # Main hand tracking script using MediaPipe
├── hand_landmarker.task     # MediaPipe hand landmark detection model
├── training_vids/           # Training video files for gesture analysis
│   ├── gripPush1.mp4       # Grip and push gestures
│   ├── gripPush2.mp4
│   ├── pickup1.mp4         # Pickup gestures  
│   ├── pickup2.mp4
│   ├── push1.mp4           # Push gestures
│   └── push2.mp4
└── README.md               # This documentation
```

## Features
- Real-time hand landmark detection (21 points)
- Focus on thumb and index finger tracking
- Optimized detection thresholds for better accuracy
- Coordinate extraction for gesture analysis
- Visual feedback with landmark visualization

## Requirements
```bash
pip install mediapipe opencv-python numpy
```

## Usage
```bash
cd hand_tracking
python hand_coordinates.py
```

## Configuration
The script is configured with optimized detection parameters:
- `min_hand_detection_confidence=0.1` - Lower threshold for better detection
- `min_hand_presence_confidence=0.1` - Lower threshold for tracking
- `min_tracking_confidence=0.3` - Tracking confidence
- `num_hands=2` - Detect up to 2 hands

## Output
The script outputs normalized coordinates (0.0-1.0) for key landmarks:
- Wrist (landmark 0)
- Index finger tip (landmark 8)
- Thumb tip (landmark 4)

Example output:
```
Frame 92: Wrist(0.175, 0.912), Index(0.178, 0.516), Thumb(0.167, 0.576)
Frame 93: No hand detected
Frame 94: Wrist(0.170, 0.994), Index(0.166, 0.593), Thumb(0.149, 0.637)
```

## AI Hackathon Context
This project was developed for the Tech Europe AI Hackathon, focusing on human-to-robot gesture communication using computer vision and machine learning techniques.

## Integration with LeRobot
This hand tracking module can be integrated with the LeRobot framework for:
- Teleoperation control using hand gestures
- Training data collection for robot manipulation tasks
- Real-time gesture-based robot control interfaces
