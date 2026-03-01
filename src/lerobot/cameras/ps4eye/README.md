# PS4 Eye Stereo Camera Driver

The Sony PS4 Eye is an inexpensive stereo USB camera (~$10–15 used) that enumerates
as a standard **UVC / V4L2** device. This driver wraps OpenCV's `VideoCapture` and
adds stereo frame slicing so the left and right eyes can be used as **two independent
cameras** inside the lerobot pipeline — without opening the device twice.

---

## Hardware

| Spec | Value |
|------|-------|
| Interface | USB 2.0 (High Speed) |
| Sensor | Dual OV7725 (1/4″ CMOS) |
| Max resolution | 3448 × 808 px (panoramic, both eyes side-by-side) |
| Frame rates | Up to 60 fps (full-res) / 120 fps (half-res) |
| Field of view | ~75° (wide) or ~56° (narrow, hardware lens) |
| Built-in audio | 4-mic array |
| Typical price | $10–15 USD on eBay / Amazon (used PS4 accessory) |

---

## Supported Resolutions

| Width | Height | Max FPS | Notes |
|-------|--------|---------|-------|
| 3448  | 808    | 60      | Full-resolution — **recommended** |
| 1748  | 408    | 120     | Half-resolution (high-speed mode) |

> [!NOTE]
> Both eyes share the same `index_or_path` and must use the **same** resolution.
> Mixing resolutions across instances will raise a `RuntimeError` at connect time.

---

## Stereo Crop Geometry

After the device delivers a panoramic frame, the driver slices out the individual eye views:

### 3448 × 808 (full-resolution)

```
┌───────────────────────────────────────────────────────────────┐
│ raw frame: 3448 × 808                                         │
│  ┌────────────────────────┐     ┌────────────────────────┐   │
│  │  left eye crop         │     │  right eye crop         │   │
│  │  cols 64 → 1328        │     │  cols 1328 → 2592       │   │
│  │  rows 0  → 800         │     │  rows 0   → 800         │   │
│  │  → 1264 × 800 px       │     │  → 1264 × 800 px        │   │
│  └────────────────────────┘     └────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### 1748 × 408 (half-resolution)

| Eye   | Columns   | Rows   | Output    |
|-------|-----------|--------|-----------|
| left  | 32 → 668  | 0→400  | 636 × 400 |
| right | 668 → 1304 | 0→400 | 636 × 400 |

---

## Linux Setup — Firmware

On Linux the PS4 Eye requires a firmware upload before it will enumerate at the
correct panoramic resolution. Without this step the camera may appear as a 640×480
webcam or not open at all.

A ready-to-use firmware upload script lives in the R2B Safety Lab repo:

👉 **[ps4-firmware-update.py](https://github.com/rafafelixphd/r2b_safety_lab/blob/main/playground/scripts/ps4-firmware-update.py)**

Run it **once** after plugging in the camera (root/sudo usually required):

```bash
sudo python ps4-firmware-update.py
```

Then unplug and re-plug the USB cable. The device should now enumerate with the
full 3448 × 808 panoramic resolution.

> [!NOTE]
> macOS and Windows do **not** require a firmware upload. The camera enumerates
> correctly as a UVC device out of the box.

---

## Find Your Device Index

Use the built-in camera discovery tool to find the device index or path:

```bash
lerobot-find-cameras ps4eye
```

Example output (macOS):

```
--- Detected Cameras ---
Camera #0:
  Name: PS4 Eye @ 1
  Type: PS4Eye
  Id: 1
  Backend api: AVFOUNDATION
  Default stream profile:
    Width: 1748
    Height: 408
    Fps: 30.00003
--------------------
```

Pass the `Id` value as `index_or_path` in your configuration.

---

## Python Quick-Start

```python
from lerobot.cameras.ps4eye import PS4EyeCamera, PS4EyeCameraConfig

cameras = {
    "left":  PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="left",  width=3448, height=808, fps=30)),
    "right": PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="right", width=3448, height=808, fps=30)),
}

for cam in cameras.values():
    cam.connect()

left_img  = cameras["left"].read()   # shape: (800, 1264, 3)  — RGB
right_img = cameras["right"].read()  # shape: (800, 1264, 3)  — RGB

print(left_img.shape, right_img.shape)

for cam in cameras.values():
    cam.disconnect()
```

### Single-eye or panoramic

```python
# Single eye (default: left)
cam = PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="left"))

# Full unsplit panoramic frame
cam = PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="both"))
```

---

## CLI Teleoperation Example

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXX \
  --robot.id=my_robot \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYY \
  --teleop.id=my_leader \
  --robot.cameras='{
      left:  {type: ps4eye, index_or_path: 1, width: 3448, height: 808, fps: 30, eye: left},
      right: {type: ps4eye, index_or_path: 1, width: 3448, height: 808, fps: 30, eye: right}
  }' \
  --display_data=true
```

---

## Live Viewer

A minimal command-line viewer is provided in `examples/ps4eye_viewer.py`:

```bash
# View the left eye
python examples/ps4eye_viewer.py --index 1 --eye left

# View the full panoramic frame
python examples/ps4eye_viewer.py --index 1 --eye both

# View both eyes in separate windows (shared capture — device opened once)
python examples/ps4eye_viewer.py --index 1 --eye left  &
python examples/ps4eye_viewer.py --index 1 --eye right
```

Press **`q`** or **Ctrl-C** to exit.

---

## Tips & Known Limitations

| Topic | Notes |
|-------|-------|
| **Linux firmware** | Required on most Linux distros; see the firmware section above |
| **macOS** | Works out of the box via AVFoundation; no firmware needed |
| **Windows** | UVC driver works; index probing searches indices 0–59 |
| **warmup_s** | Default is 1 s. The first few frames from a cold start may be dark; warmup discards them |
| **Shared capture** | Both instances *must* use the same `index_or_path`, `width`, `height`, and `fps` |
| **Thread safety** | The shared `VideoCapture` is read from a single daemon thread; each eye instance has its own processed-frame buffer |
| **USB bandwidth** | At 3448×808 @ 60 fps the camera saturates a USB 2.0 HS bus — avoid sharing the bus with other high-bandwidth devices |
