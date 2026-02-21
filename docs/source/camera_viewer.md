## Camera viewer

This page documents a small convenience script to preview a single camera attached to the host.

Script location: `src/lerobot/scripts/lerobot_show_camera.py`

Usage examples

```bash
# show camera index 0 at default 15 FPS
python src/lerobot/scripts/lerobot_show_camera.py --robot.camera.index 0

# show camera index 1 at 30 FPS and request 1280x720 resolution
python src/lerobot/scripts/lerobot_show_camera.py --robot.camera.index 1 --fps 30 --width 1280 --height 720

# run as a module (if package is installed / PYTHONPATH set)
python -m lerobot.scripts.lerobot_show_camera --robot.camera.index 0 --fps 20
```

Notes

- Press `q` or ESC in the viewer window to quit.
- If running on macOS via a remote session, the GUI window might not appear; run locally or enable GUI forwarding.
- If OpenCV fails to open the device, try other indices (0, 1, ...), check camera permissions, or verify that the device driver is healthy.

Options supported by the script:

- `--robot.camera.index`: camera index (integer, default 0)
- `--fps`: requested display FPS (default 15)
- `--width`, `--height`: optional capture resolution hints (driver dependent)
- `--window-name`: window title (default `Camera`)
