"""Pre-flight check — tile all cameras into one window and report hw acceleration. Press Q to quit."""
import os
import glob
import platform
import cv2
import numpy as np

TILE_W, TILE_H = 320, 240


def detect_device():
    """Auto-detect the best available torch device. Returns 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _probe_indices():
    """Return camera indices to try. On Linux, read /dev/video* to skip dead probing."""
    if platform.system() == "Linux":
        devs = sorted(glob.glob("/dev/video*"))
        return [int(d.replace("/dev/video", "")) for d in devs] if devs else list(range(8))
    return list(range(8))


def _open_camera(idx):
    """Open camera with low-latency settings. Returns (cap, idx) or None."""
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap


caps = {}
for i in _probe_indices():
    cap = _open_camera(i)
    if cap is not None:
        caps[i] = cap
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Index {i}: READ OK  ({w}x{h})")
    else:
        print(f"Index {i}: not available")

if not caps:
    print("No cameras found.")
    exit(1)

n = len(caps)
cols = min(n, 3)
rows = (n + cols - 1) // cols

device = detect_device()
print(f"HW acceleration: {device}")
print(f"\nShowing {n} camera(s) tiled. Press Q to quit.")

while True:
    canvas = np.zeros((rows * TILE_H, cols * TILE_W, 3), dtype=np.uint8)
    for i, (idx, cap) in enumerate(sorted(caps.items())):
        ret, frame = cap.read()
        if not ret:
            continue
        r, c = divmod(i, cols)
        small = cv2.resize(frame, (TILE_W, TILE_H))
        label = f"cam {idx}"
        cv2.putText(small, label, (5, TILE_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = small

    cv2.imshow("cameras", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
