from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

camera = OpenCVCamera(config)
camera.connect()
frame = camera.async_read(timeout_ms=500)
print(frame.shape)
camera.disconnect()
