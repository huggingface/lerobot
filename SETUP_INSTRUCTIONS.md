# Local Hardware Setup

This document records the detected hardware configuration for this machine.

## Cameras

Two OpenCV cameras were detected via V4L2:

| Camera | Device     | Type   | Backend | Format | Fourcc | Resolution | FPS |
|--------|------------|--------|---------|--------|--------|------------|-----|
| #0     | /dev/video0 | OpenCV | V4L2    | 0.0    | YUYV   | 640x480    | 30  |
| #1     | /dev/video2 | OpenCV | V4L2    | 0.0    | YUYV   | 640x480    | 30  |

## Motors / Arms

| Role     | Port          |
|----------|---------------|
| Follower | /dev/ttyACM0  |
| Leader   | /dev/ttyACM1  |
