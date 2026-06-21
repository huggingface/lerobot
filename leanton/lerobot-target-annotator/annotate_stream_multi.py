"""
LeRobot Multi-Stream Annotator — ZMQ Publisher
==============================================
Publishes N simultaneous ZMQ streams (default 3: overview + wrist + target patch)
for SmolVLA's camera1/camera2/camera3 slots. Every stream is an equal citizen
running the same code, differentiated only by its mode.

Modes:
  detect      — YOLO detection → DetectionBuffer → annotated ZMQ output
  passthrough — raw camera feed → ZMQ output (no YOLO, no buffer)
  patch       — static template: snapshot at freeze, replay until unfreeze, black otherwise

Modifier-key dispatch:
  No modifier        → stream 0 (overview)
  Shift              → stream 1 (wrist)
  Ctrl+Shift         → stream 2 (target patch)

SmolVLA integration:
  camera1 ← annotated (port 5555, 640×480)
  camera2 ← wrist     (port 5556, 640×480)
  camera3 ← target_patch (port 5557, 640×480)

Setup:
    conda activate lerobot
    pip install ultralytics pyzmq

Run:
    # Default 3-stream setup (no arguments)
    python annotate_stream_multi.py

    # Custom 2-stream
    python annotate_stream_multi.py --modes passthrough,detect --cameras 0,2

    # Legacy v5 single-stream compat
    python annotate_stream_multi.py --legacy
    python annotate_stream_multi.py --legacy --camera 0 --passthrough

Controls:
    TAB — cycle active stream (which stream receives keystrokes)
    F   — freeze / unfreeze target on active stream
    P   — toggle passthrough / annotated (detect streams only)
    [ ] / PgUp PgDn — cycle through detected objects
    A   — toggle show-all secondary detections
    C   — cycle camera (camera-backed streams only)
    R   — toggle raw view (display only)
    S   — save snapshot of active stream pane
    Z   — define pick zone (detect streams only)
    X   — define exclusion zone (detect streams only)
    Q   — quit (auto-saves state)

State file: annotate_stream_multi_state.json (multi-stream format).
Pass --fresh to skip loading saved state.
"""

# ═════════════════════════════════════════════════════════════
# CONFIG  ← all tuneable parameters live here
# ═════════════════════════════════════════════════════════════

# ── Camera ──────────────────────────────────────────────────
DEFAULT_DEVICE    = "auto"       # "auto" | "cuda" | "mps" | "cpu"
DEFAULT_CAMERA    = None         # None = auto-detect (skips built-in)
SKIP_INDICES      = (0,)
MAX_PROBE_INDEX   = 8

# ── Model ────────────────────────────────────────────────────
MODEL_PATH        = "yolov8s-worldv2.pt"
CONF              = 0.05
DEFAULT_CLASSES   = ["small object", "desk object", "cloth toy",]

# ── Detection area guard ─────────────────────────────────────
# Reject detections whose bounding-box area exceeds this fraction
# of the total frame area.  Prevents YOLO-World from hallucinating
# a full-screen "desk object" / "cloth toy" blob after a scene
# change (e.g. removing an object from a cluttered table).
MAX_BBOX_AREA_RATIO = 0.65

# ── Stabilization buffer ─────────────────────────────────────
# Rolling median buffer that smooths raw YOLO detections into a stable bbox.
# v4 state machine: FILLING → STABLE → HOLD (8s) → COOLDOWN (8s) → FILLING
#   FILLING  — buffer accumulating, bbox not yet drawn on ZMQ
#   STABLE   — ≥80% full, bbox committed (green bar, drawn on ZMQ)
#   HOLD     — stable but detections dropped or don't match; holds bbox for
#              HYSTERESIS_SECS (arm occlusion tolerance), then releases instantly
#   COOLDOWN — target confirmed lost; suppresses all detections for
#              LOSS_COOLDOWN_SECS to prevent false re-lock on empty table
#   WAITING  — buffer empty, nothing in scene
BUFFER_SIZE       = 15     # samples held for temporal median smoothing
IOU_GATE          = 0.4    # min IoU overlap to consider a new detection the same object
STABLE_FILL       = 0.8    # fraction of buffer that must be filled before bbox is committed
STALE_FRAMES      = 20     # (FILLING phase only) consecutive empty frames before reset
HYSTERESIS_SECS   = 8.0    # HOLD duration: committed bbox persists through occlusions

# ── Loss cooldown ────────────────────────────────────────────
# After a committed target is lost (HOLD expired → reset), incoming detections
# are suppressed for this many seconds to prevent false positives on an empty
# table from immediately re-locking the buffer.
LOSS_COOLDOWN_SECS = 8.0

# ── ZMQ publisher ────────────────────────────────────────────
ZMQ_JPEG_QUALITY  = 85            # 0-100; higher = larger frame, lower latency loss

# ── Window ──────────────────────────────────────────────────────
# Initial display window size (WxH). The window is resizable so you
# can stretch it after launch.  Pick something that fits a 1080p
# screen with room for a terminal underneath.
INIT_WINDOW_W     = 640
INIT_WINDOW_H     = 540

# ── Multi-stream defaults ────────────────────────────────────
DEFAULT_MODES      = "detect,passthrough,patch"  # default 3-stream setup
DEFAULT_CAMERAS    = "0,2,"       # overview=/dev/video0, wrist=/dev/video2, patch=none
DEFAULT_ZMQ_PORTS  = "5555,5556,5557"
DEFAULT_ZMQ_NAMES  = "annotated,wrist,target_patch"
DEFAULT_ZMQ_RES    = "640x480,640x480,640x480"
DEFAULT_CAPTURE_RES = "1024x768,640x480,"  # empty for patch (no camera)
DEFAULT_PATCH_SOURCES = ",,0"     # patch derives from stream 0 (overview)

# ── Bounding-box expansion ────────────────────────────────────
# Expand the committed bbox by this fraction of each dimension
# before drawing on the ZMQ stream and storing as frozen_bbox.
# 0.05 = 5% added to each side (10% total per axis).
# Set to 0.0 to disable.  The operator HUD always shows the
# raw (unexpanded) detection bbox.
BBOX_EXPAND_RATIO = 0.0

# Runtime override set from CLI --bbox-expand (module-level so draw_annotated_stream can read it)
_bbox_expand_ratio = BBOX_EXPAND_RATIO

# ── Patch crop top bias ───────────────────────────────────────
# Extra padding applied ONLY to the TOP edge of the patch crop, as a fraction
# of the bbox height. YOLO-World (generic open-vocab classes like "small
# object" / "cloth toy") tends to box the object's body and under-cover the
# top, so the frozen bbox cuts the object's top off. This extends the crop
# upward to recover it. Set 0.0 to disable. Tune with --patch-top-pad.
PATCH_TOP_PAD_RATIO = 0.10

# Runtime override set from CLI --patch-top-pad
_patch_top_pad_ratio = PATCH_TOP_PAD_RATIO

# ── Colors (BGR) ─────────────────────────────────────────────
COLOR_RUNNING     = (180, 100, 255)
COLOR_STABLE      = (0, 230, 80)
COLOR_FROZEN      = (0, 200, 255)
COLOR_STALE       = (80, 80, 80)

# ═════════════════════════════════════════════════════════════

import cv2
import sys
import os
import glob
import platform
import time
import base64
import json
import argparse
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


# ── ZMQ publisher ─────────────────────────────────────────────────────────────

def setup_zmq(port):
    import zmq
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 5)    # drop old frames under backpressure
    sock.setsockopt(zmq.LINGER, 0)    # don't block on close
    sock.bind(f"tcp://*:{port}")
    return ctx, sock


def zmq_publish(sock, camera_name, frame_bgr, quality=ZMQ_JPEG_QUALITY, timestamp=None):
    """Encode frame as JPEG and publish in LeRobot ZMQCamera wire format.

    ZMQCamera decodes with cv2.imdecode (returns BGR) but performs no
    BGR→RGB conversion, while all other LeRobot cameras return RGB.
    Sending RGB here ensures the decoded bytes match what downstream expects.

    If timestamp is None, uses time.monotonic(). Pass a timestamp to inherit
    from another stream (e.g. patch inherits source stream's timestamp).
    """
    import zmq
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _, jpg = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    encoded = base64.b64encode(jpg).decode("utf-8")
    ts = timestamp if timestamp is not None else time.monotonic()
    msg = json.dumps({
        "timestamps": {camera_name: ts},
        "images":     {camera_name: encoded},
    })
    try:
        sock.send_string(msg, zmq.NOBLOCK)
    except zmq.Again:
        pass   # no subscriber yet — drop silently, never block the main loop


# ── Camera ────────────────────────────────────────────────────────────────────

def _probe_indices():
    """Return camera indices to try. On Linux, read /dev/video* to skip dead probing."""
    if platform.system() == "Linux":
        devs = sorted(glob.glob("/dev/video*"))
        return [int(d.replace("/dev/video", "")) for d in devs] if devs else list(range(MAX_PROBE_INDEX + 1))
    return list(range(MAX_PROBE_INDEX + 1))


def probe_cameras():
    available = []
    for idx in _probe_indices():
        cap = _open_camera_raw(idx)
        if cap is not None:
            available.append(idx)
            cap.release()
    return available


def _open_camera_raw(idx, width=640, height=480):
    """Open camera with V4L2 backend — used for probing and sustained capture.
    Returns cap or None."""
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap


def open_camera(idx, width=640, height=480):
    """Open camera for sustained capture."""
    return _open_camera_raw(idx, width, height)


def probe_max_capture_res(cap, current_w=640, current_h=480):
    """Probe an already-open V4L2 camera for its maximum capture resolution.

    Tries progressively higher resolutions.  V4L2 cameras clamp unsupported
    high requests down to their hardware ceiling, so we overshoot and read
    back what the camera actually delivered.

    Returns (width, height).  Falls back to *current_w* × *current_h* if
    every probe candidate fails.
    """
    # Highest-first candidates — first one whose frame is delivered wins
    candidates = [
        (1920, 1080),
        (1600, 1200),
        (1280, 960),
        (1280, 720),
    ]

    for w, h in candidates:
        if w <= current_w and h <= current_h:
            continue          # no point trying lower than what we already have
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Flush stale frames left over from the previous mode
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            actual_h, actual_w = frame.shape[:2]
            if actual_w >= 640 and actual_h >= 480:
                print(f"    max capture resolution → {actual_w}×{actual_h} "
                      f"(requested {w}×{h})")
                return (actual_w, actual_h)

    # Nothing higher worked — restore original and return it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_h)
    for _ in range(3):
        cap.read()
    print(f"    max capture resolution → {current_w}×{current_h} (fallback)")
    return (current_w, current_h)


# ── State persistence ─────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).with_name("annotate_stream_multi_state.json")


def load_state():
    """Load multi-stream state from STATE_FILE.
    Returns the state dict, or None ONLY if no state file exists (first run).
    On ANY parse or structure error, exits immediately — the file is NEVER
    overwritten; the user must fix it manually.
    """
    if not STATE_FILE.exists():
        return None

    try:
        data = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError as e:
        print(f"[FATAL] State file {STATE_FILE.name} is malformed JSON: {e}")
        print(f"        The file has NOT been touched. Fix it manually, then restart.")
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"[FATAL] State file {STATE_FILE.name} is not a JSON object "
              f"(got {type(data).__name__}).")
        print(f"        The file has NOT been touched. Fix it manually, then restart.")
        sys.exit(1)

    if "streams" not in data:
        print(f"[FATAL] State file {STATE_FILE.name} is missing the 'streams' key.")
        print(f"        The file has NOT been touched. Fix it manually, then restart.")
        sys.exit(1)

    return data


def reinit_state():
    """Back up the existing state file and signal a fresh start.
    The backup is named with a timestamp so multiple re-inits stack safely.
    Returns None (same as first-run)."""
    if STATE_FILE.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = STATE_FILE.with_name(f"{STATE_FILE.stem}.bak_{ts}.json")
        backup.write_text(STATE_FILE.read_text())
        print(f"Backed up {STATE_FILE.name} → {backup.name}")
        print("Starting with fresh state (will be saved on first explicit action or quit).")
    else:
        print("No state file to back up — starting fresh.")
    return None


def save_state(streams):
    """Save complete multi-stream configuration and runtime state."""
    data = {"streams": {}}
    for s in streams:
        entry = {
            "mode": s.cfg.mode,
            "zmq_port": s.cfg.zmq_port,
            "zmq_camera_name": s.cfg.zmq_camera_name,
            "zmq_width": s.cfg.zmq_width,
            "zmq_height": s.cfg.zmq_height,
        }
        if s.cfg.camera_idx is not None:
            entry["camera"] = s.cfg.camera_idx
        if s.cfg.capture_width is not None:
            entry["capture_width"] = s.cfg.capture_width
        if s.cfg.capture_height is not None:
            entry["capture_height"] = s.cfg.capture_height
        if s.cfg.mode == "patch" and s.cfg.source_stream is not None:
            entry["source_stream"] = s.cfg.source_stream
        if s.cfg.mode == "detect":
            # Never fall back to DEFAULT_CLASSES — if cfg.classes is somehow
            # empty/None for a detect stream, that's a bug we should see.
            entry["classes"] = list(s.cfg.classes) if s.cfg.classes else list(DEFAULT_CLASSES)
            entry["pick_zone"] = list(s.pick_zone) if s.pick_zone else None
            entry["excl_zone"] = list(s.excl_zone) if s.excl_zone else None
            entry["frozen"] = s.frozen
            entry["frozen_bbox"] = list(s.frozen_bbox) if s.frozen_bbox else None
            entry["frozen_label"] = s.frozen_label if s.frozen else ""
            print(f"[save_state] {s.cfg.name}: classes={entry['classes']}")
        data["streams"][s.cfg.name] = entry
    STATE_FILE.write_text(json.dumps(data, indent=2))


# ── Model ─────────────────────────────────────────────────────────────────────

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


def load_model(device, classes):
    from ultralytics import YOLOWorld
    actual_device = detect_device() if device == "auto" else device
    print(f"Loading {MODEL_PATH} on {actual_device}...")
    model = YOLOWorld(MODEL_PATH)
    try:
        model.to(actual_device)
    except Exception:
        print(f"  {actual_device} unavailable, falling back to CPU")
        actual_device = "cpu"
        model.to("cpu")
    model.set_classes(classes)
    print(f"Ready. Device: {actual_device}  |  Classes: {classes}")
    return model, actual_device


# ── Manual ROI selector (reuses the live main window; Qt-safe) ───────────────

# The main composite window, kept open for the whole session. The ROI selector
# draws on THIS window instead of creating a fresh one: on the Qt/Wayland
# backend a newly created window fails to get a valid handler, so attaching a
# mouse callback to it crashes with "NULL window handler". The main window is
# already realized (it is imshow'd every loop iteration), so its handler is
# always valid.
MAIN_WINDOW = "Annotation Stream v6"


def _manual_select_roi(window_name, frame):
    """Interactive ROI selection drawn on an ALREADY-REALIZED window.
    Returns (x, y, w, h) like cv2.selectROI, or (0,0,0,0) on cancel.

    `window_name` must be the main display window that is already being
    imshow'd every loop iteration (MAIN_WINDOW in multi-stream mode, or the
    v5 window in compat mode). We must NOT create a fresh window here: on the
    Qt/Wayland backend a newly created window fails to get a valid handler, so
    binding a mouse callback to it crashes with "NULL window handler". The
    main display window is already live, so its handler is always valid.

    Mouse coordinates are reported in the shown image's pixel space, so the
    returned ROI is in the same coordinate system as `frame`."""
    roi_state = {"drawing": False, "start": None, "rect": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_state["drawing"] = True
            roi_state["start"] = (x, y)
            roi_state["rect"] = None
        elif event == cv2.EVENT_MOUSEMOVE and roi_state["drawing"]:
            sx, sy = roi_state["start"]
            roi_state["rect"] = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
        elif event == cv2.EVENT_LBUTTONUP:
            roi_state["drawing"] = False
            if roi_state["start"]:
                sx, sy = roi_state["start"]
                roi_state["rect"] = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))

    display = frame.copy()
    cv2.setMouseCallback(window_name, on_mouse)

    print("  Drag to select region. SPACE/ENTER=confirm  ESC=c=cancel")
    while True:
        show = display.copy()
        if roi_state["rect"] is not None:
            rx, ry, rw, rh = roi_state["rect"]
            if rw > 0 and rh > 0:
                cv2.rectangle(show, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.imshow(window_name, show)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):  # Enter or Space: confirm
            break
        elif k in (27, ord('c')):  # Esc or 'c': cancel
            roi_state["rect"] = (0, 0, 0, 0)
            break
        elif k == ord('q'):
            roi_state["rect"] = (0, 0, 0, 0)
            break

    # Detach the mouse callback so it stops firing during normal operation.
    cv2.setMouseCallback(window_name, lambda *a: None)
    if roi_state["rect"] is None:
        return (0, 0, 0, 0)
    return roi_state["rect"]


# ── Zone helpers ──────────────────────────────────────────────────────────────

def _passes_zones(cx, cy, pick_zone, excl_zone):
    if pick_zone is not None:
        px1, py1, px2, py2 = pick_zone
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            return False
    if excl_zone is not None:
        ex1, ey1, ex2, ey2 = excl_zone
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            return False
    return True


def _draw_dashed_rect(frame, x1, y1, x2, y2, color, label):
    for i in range(x1, x2, 16):
        cv2.line(frame, (i, y1), (min(i + 8, x2), y1), color, 1)
        cv2.line(frame, (i, y2), (min(i + 8, x2), y2), color, 1)
    for i in range(y1, y2, 16):
        cv2.line(frame, (x1, i), (x1, min(i + 8, y2)), color, 1)
        cv2.line(frame, (x2, i), (x2, min(i + 8, y2)), color, 1)
    cv2.putText(frame, label, (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_zone_overlays(frame, pick_zone, excl_zone):
    if pick_zone is not None:
        _draw_dashed_rect(frame, *pick_zone, (0, 200, 60), "PICK ZONE")
    if excl_zone is not None:
        _draw_dashed_rect(frame, *excl_zone, (0, 60, 220), "EXCL ZONE")


# ── Detection ─────────────────────────────────────────────────────────────────

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def raw_detect(frame, model, pick_zone, excl_zone):
    results = model(frame, conf=CONF, verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    if boxes is None or len(boxes) == 0:
        return []

    h, w = frame.shape[:2]
    frame_area = h * w
    max_area = MAX_BBOX_AREA_RATIO * frame_area

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        if not _passes_zones((x1 + x2) // 2, (y1 + y2) // 2, pick_zone, excl_zone):
            continue
        area = (x2 - x1) * (y2 - y1)

        # v5: reject detections whose bbox covers too much of the frame.
        # Full-screen "desk object" / "cloth toy" hallucinations (often low
        # confidence) appear after scene changes (e.g. removing an object).
        # A genuine small-object detection never fills >65% of the frame.
        if area > max_area:
            continue

        cls_id = int(boxes.cls[i].item())
        label = names[cls_id] if isinstance(names, list) else names.get(cls_id, "?")
        conf = float(boxes.conf[i].item())
        detections.append((area, x1, y1, x2, y2, label, conf))
    detections.sort(key=lambda d: d[0], reverse=True)
    return detections


# ── DetectionBuffer ───────────────────────────────────────────────────────────

class DetectionBuffer:
    def __init__(self, size=BUFFER_SIZE, iou_gate=IOU_GATE,
                 stale_limit=STALE_FRAMES, hysteresis=HYSTERESIS_SECS,
                 loss_cooldown=LOSS_COOLDOWN_SECS):
        self._buf           = deque(maxlen=size)
        self._size          = size
        self._iou_gate      = iou_gate
        self._stale_limit   = stale_limit
        self._hysteresis    = hysteresis
        self._loss_cooldown = loss_cooldown
        self._stale_count   = 0
        self._last_label    = ""
        self._stable_at     = None
        self._last_valid_at = None
        self._lost_at       = None

    def reset(self):
        """Normal reset — enforces cooldown if target was committed before loss."""
        was_committed = self.is_stable or self.in_hysteresis
        self._buf.clear()
        self._stale_count   = 0
        self._last_label    = ""
        self._stable_at     = None
        self._last_valid_at = None
        if was_committed:
            self._lost_at = time.monotonic()

    def hard_reset(self):
        """Operator-forced reset — no cooldown, accept next detection immediately."""
        self._buf.clear()
        self._stale_count   = 0
        self._last_label    = ""
        self._stable_at     = None
        self._last_valid_at = None
        self._lost_at       = None

    def _protected(self):
        return (self._last_valid_at is not None and
                self.is_stable and
                time.monotonic() - self._last_valid_at < self._hysteresis)

    def _in_cooldown(self):
        return (self._lost_at is not None and
                time.monotonic() - self._lost_at < self._loss_cooldown)

    def _accept(self, box, label):
        self._buf.append(box)
        self._last_label    = label
        self._last_valid_at = time.monotonic()
        self._lost_at       = None   # accepted a detection — clear cooldown
        if self.is_stable and self._stable_at is None:
            self._stable_at = time.monotonic()

    def update(self, detection):
        # ── No detection this frame ──────────────────────────────
        if detection is None:
            self._stale_count += 1
            if self._protected():
                self._stale_count = 0   # HOLD: reset stale debt
                return self.median()
            # HOLD expired with empty scene → immediate reset → COOLDOWN
            self.reset()
            return self.median()

        # ── Detection suppressed during COOLDOWN ──────────────────
        if self._in_cooldown():
            return self.median()

        # ── Detection received ────────────────────────────────────
        self._stale_count = 0
        _, x1, y1, x2, y2, label, _ = detection
        new_box = (x1, y1, x2, y2)
        current = self.median()

        if current is None or _iou(new_box, current) >= self._iou_gate:
            # Match → accept into buffer (or seed fresh buffer)
            self._accept(new_box, label)
        else:
            if self._protected():
                # HOLD: non-matching detection tolerated, clock not refreshed
                return self.median()
            # HOLD expired + non-matching detection → immediate reset + accept
            self.reset()
            self._accept(new_box, label)

        return self.median()

    def median(self):
        if not self._buf:
            return None
        return tuple(int(v) for v in np.median(np.array(self._buf), axis=0))

    @property
    def fill(self): return len(self._buf) / self._size

    @property
    def is_stable(self): return self.fill >= STABLE_FILL

    @property
    def is_stale(self): return self._stale_count > 0 and len(self._buf) == 0

    @property
    def in_hysteresis(self): return self._protected()

    @property
    def in_cooldown(self): return self._in_cooldown()

    @property
    def label(self): return self._last_label


# ── Drawing: ZMQ output ───────────────────────────────────────────────────────

def expand_bbox(bbox, ratio, frame_width, frame_height):
    """Expand bbox symmetrically by `ratio` of each dimension.
    When expansion hits a frame edge, the opposite edge is shifted to
    keep the object centered in the visible region (no asymmetric drift).
    Returns (x1, y1, x2, y2) or None if bbox is None."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * ratio), int(bh * ratio)

    # Ideal expansion, then shift to stay centered when clamped
    nx1, ny1 = x1 - dx, y1 - dy
    nx2, ny2 = x2 + dx, y2 + dy

    # Left edge
    if nx1 < 0:
        shift = -nx1
        nx1 = 0
        nx2 = min(frame_width, nx2 + shift)
    # Top edge
    if ny1 < 0:
        shift = -ny1
        ny1 = 0
        ny2 = min(frame_height, ny2 + shift)
    # Right edge
    if nx2 > frame_width:
        shift = nx2 - frame_width
        nx2 = frame_width
        nx1 = max(0, nx1 - shift)
    # Bottom edge
    if ny2 > frame_height:
        shift = ny2 - frame_height
        ny2 = frame_height
        ny1 = max(0, ny1 - shift)

    return (nx1, ny1, nx2, ny2)


def crop_and_letterbox(frame, bbox, out_w, out_h, expand_ratio=0.0, top_pad_ratio=0.0):
    """Crop `bbox` from `frame`, optionally pad symmetrically, then resize
    PRESERVING ASPECT RATIO and center on a black (out_w x out_h) canvas.

    The crop is scaled by min(out_w/cw, out_h/ch) so it fits entirely inside
    the output; leftover space is black padding split symmetrically, keeping
    the object centered. This replaces the old "force-square then resize"
    path, which squashed the object when square-making hit a frame edge and
    biased the crop toward whichever side had more room.

    `expand_ratio` adds context padding equal to that fraction of each bbox
    dimension on every side (preserves aspect ratio away from frame edges).

    `top_pad_ratio` adds EXTRA padding to the TOP edge only (fraction of bbox
    height), to recover the object top that YOLO-World routinely under-covers.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if expand_ratio and expand_ratio > 0:
        x1 -= int(bw * expand_ratio); x2 += int(bw * expand_ratio)
        y1 -= int(bh * expand_ratio); y2 += int(bh * expand_ratio)
    if top_pad_ratio and top_pad_ratio > 0:
        y1 -= int(bh * top_pad_ratio)   # extend top upward only
    # Clamp to frame; object stays fully inside, edges just lose a little context
    x1 = max(0, min(int(x1), fw))
    y1 = max(0, min(int(y1), fh))
    x2 = max(0, min(int(x2), fw))
    y2 = max(0, min(int(y2), fh))
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)
    crop = frame[y1:y2, x1:x2]
    # Fit inside output preserving aspect ratio (the anti-squash step)
    scale = min(out_w / cw, out_h / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x_off = (out_w - new_w) // 2
    y_off = (out_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_annotated_stream(frame, bbox, buf, frozen):
    """
    Clean frame with bbox rectangle only when target is committed.
    No HUD, no zone overlays, no stability bar.
    Always returns a frame (clean if no committed target).
    Bbox is expanded by the runtime `_bbox_expand_ratio` on each side
    for the ZMQ stream (the policy benefits from extra context around the object).
    """
    out = frame.copy()
    if bbox is None or not (frozen or buf.is_stable):
        return out
    if _bbox_expand_ratio > 0:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = expand_bbox(bbox, _bbox_expand_ratio, w, h)
    else:
        x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_STABLE, 3)
    return out


# ── Drawing: operator HUD ─────────────────────────────────────────────────────

def draw_target_hud(frame, bbox, label, conf, buf, frozen):
    """Operator overlay — bbox with label and coordinate readout."""
    if bbox is None or not (frozen or buf.is_stable):
        return
    color = COLOR_FROZEN if frozen else COLOR_STABLE
    tag   = f"FROZEN: {label}" if frozen else f"STABLE: {label} ({conf:.0%})"
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text_y = y1 - 10 if y1 > 25 else y2 + 20
    cv2.putText(frame, tag, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    cv2.putText(frame, f"bbox [{x1},{y1},{x2},{y2}]",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def draw_secondary(frame, detections):
    for _, x1, y1, x2, y2, label, conf in detections[1:]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)


def draw_stability_bar(frame, buf, frozen):
    h, w = frame.shape[:2]
    bar_w, bar_h = 120, 10
    bx, by = w - bar_w - 10, 50
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
    if frozen:
        fill_color, fill_px, label = COLOR_FROZEN, bar_w, "FROZEN"
    elif buf.in_cooldown:
        remaining  = LOSS_COOLDOWN_SECS - (time.monotonic() - buf._lost_at)
        fill_color, fill_px, label = COLOR_STALE, 0, f"COOLDOWN {remaining:.1f}s"
    elif buf.is_stale:
        fill_color, fill_px, label = COLOR_STALE, 0, "WAITING"
    elif buf.in_hysteresis:
        remaining  = HYSTERESIS_SECS - (time.monotonic() - buf._last_valid_at)
        fill_color = COLOR_STABLE
        fill_px    = int(buf.fill * bar_w)
        label      = f"HOLD {remaining:.0f}s"
    else:
        fill_px    = int(buf.fill * bar_w)
        fill_color = COLOR_STABLE if buf.is_stable else COLOR_RUNNING
        label      = "STABLE" if buf.is_stable else f"{int(buf.fill * 100)}%"
    if fill_px > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill_px, by + bar_h), fill_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (120, 120, 120), 1)
    cv2.putText(frame, label, (bx, by - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, fill_color, 1)


def draw_detect_hud(frame, stream, detections, fps, device):
    """Full HUD for detect-mode streams — top bar, stability bar, zones, info."""
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    mode_label = "[ FROZEN ]  YOLO-World" if stream.frozen else f"YOLO-World ({device})"
    mode_color = COLOR_FROZEN if stream.frozen else COLOR_RUNNING
    cv2.putText(frame, mode_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    cv2.putText(frame, f"{fps:.1f} fps  |  proc:{stream.proc_ms:.0f}ms",
                (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, stream.info_str, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if stream.show_all:
        cv2.putText(frame, "[ALL]", (w - 60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Camera indicator
    if stream.cfg.camera_idx is not None:
        cam_str = f"CAM [{stream.cfg.camera_idx}]"
        cv2.putText(frame, cam_str, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)


def draw_passthrough_hud(frame, stream, fps):
    """Minimal HUD for passthrough-mode streams."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(frame, "PASSTHROUGH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RUNNING, 1)
    cv2.putText(frame, f"{fps:.1f} fps  |  proc:{stream.proc_ms:.0f}ms",
                (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if stream.cfg.camera_idx is not None:
        cam_str = f"CAM [{stream.cfg.camera_idx}]"
        cv2.putText(frame, cam_str, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)


def draw_patch_hud(frame, stream, fps):
    """Bare HUD for patch-mode streams — label only."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(frame, "PATCH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FROZEN, 1)
    cv2.putText(frame, f"{fps:.1f} fps  |  proc:{stream.proc_ms:.0f}ms",
                (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ── Stream dataclasses ────────────────────────────────────────────────────────

@dataclass
class StreamConfig:
    id: int                          # 0, 1, 2, ... (position in streams list)
    name: str                        # "overview", "wrist", "target_patch"
    mode: str                        # "detect" | "passthrough" | "patch"
    zmq_port: int                    # e.g. 5555, 5556, 5557
    zmq_camera_name: str             # "annotated", "wrist", "target_patch"
    zmq_width: int                   # output width for ZMQ (e.g. 640, 256)
    zmq_height: int                  # output height for ZMQ (e.g. 480, 256)

    # ── Camera-backed modes (detect, passthrough) ──
    camera_idx: int | None = None    # /dev/video index
    capture_width: int | None = None # acquisition width before downscale (e.g. 1024)
    capture_height: int | None = None

    # ── detect mode only ──
    classes: list | None = None      # YOLO classes (None → DEFAULT_CLASSES)
    passthrough_start: bool = False  # start in passthrough? (runtime toggle via P)

    # ── patch mode only ──
    source_stream: int | None = None # stream id whose freeze event drives this patch


@dataclass
class StreamState:
    cfg: StreamConfig

    # ── Camera (None for patch mode) ──
    cap: cv2.VideoCapture | None = None
    capture_frame_hi: np.ndarray | None = None  # latest hi-res frame (for patch cropping)
    actual_capture_w: int | None = None         # from cap.get(CAP_PROP_FRAME_WIDTH)
    actual_capture_h: int | None = None         # from cap.get(CAP_PROP_FRAME_HEIGHT)

    # ── ZMQ ──
    zmq_ctx: object | None = None
    zmq_sock: object | None = None
    last_publish_monotonic: float = 0.0

    # ── patch mode: cached snapshot ──
    patch_cache: np.ndarray | None = None  # snapshot at freeze, replayed until unfreeze

    # ── detect-mode runtime state ──
    buf: DetectionBuffer | None = None
    frozen: bool = False
    frozen_bbox: tuple | None = None
    frozen_label: str = ""
    frozen_conf: float = 0.0
    passthrough_active: bool = False       # runtime toggle (P key) on detect streams
    show_raw: bool = False
    show_all: bool = False
    selected_det_idx: int | None = None
    prev_sel_idx: int | None = None
    pick_zone: tuple | None = None
    excl_zone: tuple | None = None
    detections: list = field(default_factory=list)

    # ── Per-stream timing (processing latency, not independent FPS) ──
    frame_count: int = 0
    proc_ms: float = 0.0
    info_str: str = "Initialising..."
    display: np.ndarray | None = None      # last rendered frame for compositing
    is_active: bool = False                 # highlighted in composite (Tab-cycled)


# ── Composite window ──────────────────────────────────────────────────────────

def build_composite(streams, shared_top_bar=""):
    """Stack all stream displays vertically into one composite frame.
    Letterbox/pad panes to common width. Pre-allocates canvas if possible
    (uses a cached buffer on a mutable container to avoid ~2MB alloc per frame)."""
    if not hasattr(build_composite, "_canvas"):
        build_composite._canvas = None
        build_composite._last_shape = None

    panes = []
    for s in streams:
        if s.display is None:
            # Black placeholder if stream hasn't produced a frame yet
            panes.append(np.zeros((s.cfg.zmq_height, s.cfg.zmq_width, 3), dtype=np.uint8))
        else:
            panes.append(s.display)

    # Determine common width (max pane width) and total height
    common_w = max(p.shape[1] for p in panes)
    total_h = sum(p.shape[0] for p in panes) + 26 * (len(panes))  # 26px label per pane
    separator_h = 2  # thin separator line between panes
    total_h += separator_h * (len(panes) - 1)

    # Shared bar heights
    top_bar_h = 24
    hint_bar_h = 24
    total_h += top_bar_h + hint_bar_h

    # Pre-allocate or reuse canvas
    canvas_shape = (total_h, common_w, 3)
    if build_composite._canvas is None or build_composite._last_shape != canvas_shape:
        build_composite._canvas = np.zeros(canvas_shape, dtype=np.uint8)
        build_composite._last_shape = canvas_shape

    canvas = build_composite._canvas
    canvas.fill(0)

    # ── Shared top bar ──
    cv2.rectangle(canvas, (0, 0), (common_w, top_bar_h), (30, 30, 30), -1)
    cv2.putText(canvas, shared_top_bar, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y = top_bar_h

    for i, (pane, stream) in enumerate(zip(panes, streams)):
        ph, pw = pane.shape[:2]
        # Letterbox: center pane horizontally, pad with black
        x_offset = (common_w - pw) // 2

        # Stream label bar
        bar_color = (80, 60, 20) if stream.is_active else (40, 40, 40)
        cv2.rectangle(canvas, (0, y), (common_w, y + 26), bar_color, -1)
        # Active indicator
        if stream.is_active:
            cv2.rectangle(canvas, (0, y), (common_w, y + 26), (0, 220, 255), 2)
        label_text = f" {stream.cfg.name.upper()}  [{stream.cfg.mode}]  —  ZMQ :{stream.cfg.zmq_port}  \"{stream.cfg.zmq_camera_name}\"  —  proc:{stream.proc_ms:.1f}ms"
        if stream.is_active:
            label_text = "▶" + label_text
        # Mode color
        if stream.cfg.mode == "detect":
            label_color = COLOR_FROZEN if stream.frozen else COLOR_STABLE
        elif stream.cfg.mode == "patch":
            label_color = (200, 150, 0)  # amber
        else:
            label_color = (200, 200, 200)
        cv2.putText(canvas, label_text, (6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1)
        y += 26

        # Pane
        canvas[y:y + ph, x_offset:x_offset + pw] = pane
        y += ph

        # Separator (except after last pane)
        if i < len(panes) - 1:
            cv2.line(canvas, (0, y), (common_w, y), (60, 60, 60), 1)
            y += separator_h

    # ── Hint bar ──
    cv2.rectangle(canvas, (0, y), (common_w, y + hint_bar_h), (30, 30, 30), -1)
    hint = "TAB=cycle stream  |  Q=quit  P=passthrough  F=freeze  []=cycle  C=cam  A=all  R=raw  S=snap  Z=pick-zone  X=excl-zone"
    cv2.putText(canvas, hint, (10, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    return canvas


# ── Per-stream processing ─────────────────────────────────────────────────────

def process_stream(stream, streams, shared_model, no_zmq):
    """Process one stream for one iteration. Mode dispatch is explicit."""

    if stream.cfg.mode == "detect":
        # ── Acquire ──
        ret, frame_hi = stream.cap.read()
        if not ret:
            # Camera failure — reopen
            stream.cap.release()
            stream.cap = open_camera(stream.cfg.camera_idx,
                                     stream.cfg.capture_width or 640,
                                     stream.cfg.capture_height or 480)
            if stream.cap is None:
                return  # skip this iteration
            return  # try again next frame

        stream.capture_frame_hi = frame_hi
        h, w = frame_hi.shape[:2]
        if (w, h) != (stream.cfg.zmq_width, stream.cfg.zmq_height):
            frame = cv2.resize(frame_hi, (stream.cfg.zmq_width, stream.cfg.zmq_height))
        else:
            frame = frame_hi

        # ── Detection ──
        if stream.passthrough_active:
            output = frame
        elif stream.frozen:
            output = draw_annotated_stream(frame, stream.frozen_bbox, stream.buf, frozen=True)
        else:
            stream.detections = raw_detect(frame, shared_model, stream.pick_zone, stream.excl_zone)

            # Manual selection override
            if stream.selected_det_idx is not None:
                if not stream.detections:
                    stream.selected_det_idx = None
                elif stream.selected_det_idx >= len(stream.detections):
                    stream.selected_det_idx = None

            if stream.selected_det_idx != stream.prev_sel_idx:
                stream.buf.hard_reset()

            # Pick best_match
            if stream.selected_det_idx is not None:
                best_match = stream.detections[stream.selected_det_idx]
            else:
                best_match = None
                if stream.detections:
                    current = stream.buf.median()
                    if current is not None:
                        for d in stream.detections:
                            _, x1, y1, x2, y2, _, _ = d
                            if _iou((x1, y1, x2, y2), current) >= IOU_GATE:
                                best_match = d
                                break
                    if best_match is None:
                        best_match = stream.detections[0]

            median_bbox = stream.buf.update(best_match)

            # Update label/conf for HUD
            if best_match is not None:
                _, _, _, _, _, disp_label, disp_conf = best_match
            else:
                disp_label, disp_conf = stream.buf.label, 0.0

            stream.frozen_label = disp_label
            stream.frozen_conf = disp_conf
            output = draw_annotated_stream(frame, median_bbox, stream.buf, frozen=False)

        # ── ZMQ publish ──
        if not no_zmq and stream.zmq_sock is not None:
            ts = time.monotonic()
            zmq_publish(stream.zmq_sock, stream.cfg.zmq_camera_name, output, timestamp=ts)
            stream.last_publish_monotonic = ts

        # ── Build display ──
        display = frame.copy()

        if not stream.passthrough_active:
            if not stream.show_raw:
                if not stream.frozen and stream.show_all and len(stream.detections) > 1:
                    draw_secondary(display, stream.detections)
                if stream.frozen:
                    median_bbox = stream.frozen_bbox
                else:
                    median_bbox = stream.buf.median()
                draw_target_hud(display, median_bbox, stream.frozen_label, stream.frozen_conf,
                                stream.buf, stream.frozen)

            # Info string
            if stream.frozen:
                stream.info_str = f"FROZEN  bbox {stream.frozen_bbox}"
            elif stream.buf.in_hysteresis:
                remaining = HYSTERESIS_SECS - (time.monotonic() - stream.buf._last_valid_at)
                stream.info_str = f"Holding committed target — releasing in {remaining:.0f}s"
            elif stream.buf.in_cooldown:
                remaining = LOSS_COOLDOWN_SECS - (time.monotonic() - stream.buf._lost_at)
                stream.info_str = f"Cooldown — suppressing re-detection ({remaining:.1f}s)"
            elif stream.buf.is_stale:
                stream.info_str = "Waiting for object..."
            elif stream.buf.median() is None:
                stream.info_str = "No objects detected"
            else:
                n = len(stream.detections)
                sel_tag = f"  [SEL {stream.selected_det_idx+1}/{n}]" if stream.selected_det_idx is not None else ""
                stable_tag = "  [STABLE]" if stream.buf.is_stable else f"  [{int(stream.buf.fill*100)}%]"
                stream.info_str = f"{n} object{'s' if n != 1 else ''} detected{sel_tag}{stable_tag}"

            if stream.show_raw:
                stream.info_str = "[RAW] " + stream.info_str
        else:
            stream.info_str = "PASSTHROUGH — raw feed"

        if not stream.passthrough_active and not stream.show_raw:
            draw_stability_bar(display, stream.buf, stream.frozen)
            draw_zone_overlays(display, stream.pick_zone, stream.excl_zone)

        stream.prev_sel_idx = stream.selected_det_idx
        stream.display = display

    elif stream.cfg.mode == "passthrough":
        ret, frame = stream.cap.read()
        if not ret:
            stream.cap.release()
            stream.cap = open_camera(stream.cfg.camera_idx,
                                     stream.cfg.zmq_width,
                                     stream.cfg.zmq_height)
            if stream.cap is None:
                return
            return

        # Resize if capture dims differ from ZMQ dims
        h, w = frame.shape[:2]
        if (w, h) != (stream.cfg.zmq_width, stream.cfg.zmq_height):
            frame = cv2.resize(frame, (stream.cfg.zmq_width, stream.cfg.zmq_height))

        if not no_zmq and stream.zmq_sock is not None:
            ts = time.monotonic()
            zmq_publish(stream.zmq_sock, stream.cfg.zmq_camera_name, frame, timestamp=ts)
            stream.last_publish_monotonic = ts

        stream.display = frame
        stream.info_str = "PASSTHROUGH"

    elif stream.cfg.mode == "patch":
        source = streams[stream.cfg.source_stream]
        if source.frozen and source.frozen_bbox is not None:
            # ── On freeze: snapshot once into cache ──
            if source.capture_frame_hi is not None and stream.patch_cache is None:
                # Scale raw bbox from ZMQ coords → capture coords.
                # MUST use capture_frame_hi.shape, NOT cap.get(CAP_PROP_*): USB
                # cameras routinely deliver a different resolution than they
                # report (request 1024x768, actually hand you 1280x720 while
                # cap.get still says 768). A vertical scale computed from the
                # reported (too-large) height maps the bbox TOP too far down
                # the frame, cropping the object's top off, while the bottom
                # just clamps to the real frame edge, leaving empty space
                # below. Scaling from the actual array we crop fixes both.
                hi_h, hi_w = source.capture_frame_hi.shape[:2]
                sx = hi_w / source.cfg.zmq_width
                sy = hi_h / source.cfg.zmq_height
                bx1, by1, bx2, by2 = source.frozen_bbox
                cx1 = int(bx1 * sx); cy1 = int(by1 * sy)
                cx2 = int(bx2 * sx); cy2 = int(by2 * sy)
                # Crop preserving the object's natural aspect ratio, then
                # letterbox onto a black output-size canvas. Replaces the old
                # "force square + resize" path, which (a) squashed the object
                # whenever the square-making couldn't reach a frame edge, and
                # (b) biased the crop toward whichever side had more room
                # (the "more cropped on top" symptom). Letterbox keeps the
                # detector's aspect ratio and centers the object symmetrically.
                stream.patch_cache = crop_and_letterbox(
                    source.capture_frame_hi,
                    (cx1, cy1, cx2, cy2),
                    stream.cfg.zmq_width, stream.cfg.zmq_height,
                    expand_ratio=_bbox_expand_ratio,
                    top_pad_ratio=_patch_top_pad_ratio,
                )
            patch = (stream.patch_cache if stream.patch_cache is not None
                     else np.zeros((stream.cfg.zmq_height, stream.cfg.zmq_width, 3), dtype=np.uint8))
        else:
            # ── Unfrozen: clear cache, publish black ──
            stream.patch_cache = None
            patch = np.zeros((stream.cfg.zmq_height, stream.cfg.zmq_width, 3), dtype=np.uint8)

        # Inherit source's timestamp so camera3 aligns with camera2/camera1
        ts = source.last_publish_monotonic if source.last_publish_monotonic else time.monotonic()
        if not no_zmq and stream.zmq_sock is not None:
            zmq_publish(stream.zmq_sock, stream.cfg.zmq_camera_name, patch, timestamp=ts)
        stream.last_publish_monotonic = ts
        stream.display = patch
        stream.info_str = "PATCH" if stream.patch_cache is not None else "PATCH (black)"


# ── Key dispatch ──────────────────────────────────────────────────────────────

def dispatch_key(key_ext, key_ascii, target, streams, available_cams, no_zmq):
    """Route a keypress to the targeted stream's handler.
    Extended keys (PgUp, arrows) checked via key_ext first,
    ASCII keys checked via key_ascii (already shift-normalized).
    """

    s = target
    cam_streams = [st for st in streams if st.cfg.mode in ("detect", "passthrough")]

    # ── Quit ──
    if key_ascii == ord('q'):
        return False  # signal main loop to exit

    # ── Passthrough toggle (detect streams only) ──
    elif key_ascii == ord('p'):
        if s.cfg.mode == "detect" and s.buf is not None:
            s.passthrough_active = not s.passthrough_active
            s.selected_det_idx = None
            # Unfreeze unconditionally — a frozen target makes no sense during
            # passthrough, and coming out of passthrough the old freeze is stale
            # (scene may have changed).  This also clears the patch stream
            # immediately instead of leaving the previous cached patch on screen.
            if s.frozen:
                s.frozen = False
                s.frozen_bbox = None
                s.frozen_label = ""
                s.frozen_conf = 0.0
            s.buf.hard_reset()
            verb = "ON — raw camera feed, YOLO bypassed" if s.passthrough_active else "OFF — YOLO annotation resumed"
            print(f"[{s.cfg.name}] Passthrough {verb}")
        else:
            print(f"[{s.cfg.name}] Passthrough toggle unavailable (mode={s.cfg.mode})")

    # ── Freeze (detect streams only) ──
    elif key_ascii == ord('f'):
        if s.cfg.mode != "detect":
            pass  # no-op on passthrough/patch
        elif s.passthrough_active:
            pass
        elif s.frozen:
            s.frozen = False
            s.frozen_bbox = None
            s.frozen_label = ""
            s.frozen_conf = 0.0
            s.buf.hard_reset()
            s.selected_det_idx = None
            save_state(streams)
            print(f"[{s.cfg.name}] Unfrozen. Buffer reset. State saved.")
        else:
            m = s.buf.median()
            if m:
                s.frozen = True
                s.frozen_bbox = m
                s.frozen_label = s.buf.label
                s.frozen_conf = getattr(s, 'frozen_conf', 0.0)
                save_state(streams)
                print(f"[{s.cfg.name}] Frozen: {s.frozen_bbox}  label={s.frozen_label}  State saved.")
            else:
                print(f"[{s.cfg.name}] Nothing to freeze — no stable detection yet.")
        s.selected_det_idx = None

    # ── Cycle objects: [ or PageUp (detect streams only) ──
    elif key_ext in (ord('['), 0xFF55, 0x210000) or key_ascii == ord('['):
        if s.cfg.mode != "detect" or s.passthrough_active:
            print(f"[{s.cfg.name}] Object cycling unavailable (mode={s.cfg.mode})")
        elif s.detections:
            n = len(s.detections)
            if s.selected_det_idx is None:
                s.selected_det_idx = 0
            else:
                s.selected_det_idx = (s.selected_det_idx - 1) % n
            print(f"[{s.cfg.name}] Selected object {s.selected_det_idx+1}/{n}  "
                  f"label={s.detections[s.selected_det_idx][5]}")

    # ── Cycle objects: ] or PageDown (detect streams only) ──
    elif key_ext in (ord(']'), 0xFF56, 0x220000) or key_ascii == ord(']'):
        if s.cfg.mode != "detect" or s.passthrough_active:
            print(f"[{s.cfg.name}] Object cycling unavailable (mode={s.cfg.mode})")
        elif s.detections:
            n = len(s.detections)
            if s.selected_det_idx is None:
                s.selected_det_idx = 0
            else:
                s.selected_det_idx = (s.selected_det_idx + 1) % n
            print(f"[{s.cfg.name}] Selected object {s.selected_det_idx+1}/{n}  "
                  f"label={s.detections[s.selected_det_idx][5]}")

    # ── Show-all toggle (detect streams only) ──
    elif key_ascii == ord('a'):
        if s.cfg.mode == "detect" and not s.passthrough_active:
            s.show_all = not s.show_all
            n = len(s.detections) if s.detections else 0
            state = "ON" if s.show_all else "OFF"
            print(f"[{s.cfg.name}] Show-all {state}  ({n} objects detected)")
        else:
            print(f"[{s.cfg.name}] Show-all unavailable (mode={s.cfg.mode})")

    # ── Raw view toggle (detect streams only) ──
    elif key_ascii == ord('r'):
        if s.cfg.mode == "detect" and not s.passthrough_active:
            s.show_raw = not s.show_raw
            state = "ON" if s.show_raw else "OFF"
            print(f"[{s.cfg.name}] Raw view {state}")
        else:
            print(f"[{s.cfg.name}] Raw view unavailable (mode={s.cfg.mode})")

    # ── Snapshot ──
    elif key_ascii == ord('s'):
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if s.display is not None:
            path = snapshot_dir / f"{s.cfg.name}_{ts}.jpg"
            cv2.imwrite(str(path), s.display)
            print(f"[{s.cfg.name}] Snapshot: {path}")
        else:
            print(f"[{s.cfg.name}] Snapshot skipped — no display frame yet")

    # ── Pick zone (always routes to first detect stream, regardless of active) ──
    elif key_ascii == ord('z'):
        # Auto-route to first detect stream that isn't in passthrough mode.
        # Zones are a detect-only concept; silently ignoring Z on a passthrough/patch
        # stream breaks the v5 workflow where Z always "just works".
        zone_target = next((st for st in streams
                           if st.cfg.mode == "detect" and not st.passthrough_active), None)
        if zone_target is None:
            print("Pick zone unavailable — no active detect stream.")
        elif zone_target.pick_zone is not None:
            zone_target.pick_zone = None
            save_state(streams)
            print(f"[{zone_target.cfg.name}] Pick zone cleared. State saved.")
        else:
            print(f"[{zone_target.cfg.name}] Draw PICK zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
            zone_frame = (zone_target.display if zone_target.display is not None
                          else np.zeros((480, 640, 3), dtype=np.uint8))
            roi = _manual_select_roi(MAIN_WINDOW, zone_frame)
            zone_target.pick_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
            if zone_target.pick_zone:
                save_state(streams)
            print(f"[{zone_target.cfg.name}] Pick zone: {zone_target.pick_zone}" if zone_target.pick_zone
                  else f"[{zone_target.cfg.name}] Cancelled.")

    # ── Exclusion zone (always routes to first detect stream, regardless of active) ──
    elif key_ascii == ord('x'):
        zone_target = next((st for st in streams
                           if st.cfg.mode == "detect" and not st.passthrough_active), None)
        if zone_target is None:
            print("Exclusion zone unavailable — no active detect stream.")
        elif zone_target.excl_zone is not None:
            zone_target.excl_zone = None
            save_state(streams)
            print(f"[{zone_target.cfg.name}] Exclusion zone cleared. State saved.")
        else:
            print(f"[{zone_target.cfg.name}] Draw EXCLUSION zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
            zone_frame = (zone_target.display if zone_target.display is not None
                          else np.zeros((480, 640, 3), dtype=np.uint8))
            roi = _manual_select_roi(MAIN_WINDOW, zone_frame)
            zone_target.excl_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
            if zone_target.excl_zone:
                save_state(streams)
            print(f"[{zone_target.cfg.name}] Exclusion zone: {zone_target.excl_zone}" if zone_target.excl_zone
                  else f"[{zone_target.cfg.name}] Cancelled.")

    # ── Camera cycle (camera-backed streams only) ──
    # Arrow left (81), Arrow right (83), 'c' key
    elif key_ext in (ord('c'), 81, 83, 0xFF51, 0xFF53) or key_ascii in (ord('c'),):
        if s.cfg.mode == "patch":
            pass  # patch has no camera
        elif len(available_cams) > 1 and s.cfg.camera_idx is not None:
            # Determine direction
            if key_ext in (81, 0xFF51):  # left arrow
                direction = -1
            else:  # right arrow (83, 0xFF53) or 'c'
                direction = 1
            cur_cam = s.cfg.camera_idx
            if cur_cam in available_cams:
                pos = available_cams.index(cur_cam)
            else:
                pos = 0
            # Find next available camera not in use by another stream
            for _ in range(len(available_cams)):
                pos = (pos + direction) % len(available_cams)
                new_idx = available_cams[pos]
                # Check if already assigned to another stream
                conflict = any(st.cfg.camera_idx == new_idx and st is not s
                              for st in cam_streams)
                if not conflict:
                    break
            new_cap = open_camera(new_idx,
                                 s.cfg.capture_width or 640,
                                 s.cfg.capture_height or 480)
            if new_cap:
                s.cap.release()
                s.cap = new_cap
                s.cfg.camera_idx = new_idx
                s.actual_capture_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                s.actual_capture_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if s.buf:
                    s.buf.reset()
                s.selected_det_idx = None
                save_state(streams)
                print(f"[{s.cfg.name}] Camera → {new_idx}")
            else:
                print(f"[{s.cfg.name}] Cannot open camera {new_idx}")

    return True  # continue main loop


# ── CLI parsing ───────────────────────────────────────────────────────────────

def parse_res_list(arg_str):
    """Parse a comma-separated list of WxH resolutions. Empty entries → (None, None)."""
    if not arg_str:
        return []
    parts = arg_str.split(",")
    results = []
    for p in parts:
        p = p.strip()
        if not p:
            results.append((None, None))
        elif "x" in p:
            w, h = p.split("x", 1)
            results.append((int(w.strip()), int(h.strip())))
        else:
            results.append((None, None))
    return results


def parse_int_list(arg_str, allow_empty=True):
    """Parse comma-separated ints. Empty entries → None."""
    if not arg_str:
        return []
    results = []
    for p in arg_str.split(","):
        p = p.strip()
        if not p:
            results.append(None)
        else:
            results.append(int(p))
    return results


def parse_str_list(arg_str):
    """Parse comma-separated strings. Empty entries → empty string."""
    if not arg_str:
        return []
    return [s.strip() for s in arg_str.split(",")]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-World multi-stream annotation — N-stream ZMQ publisher")
    # ── Multi-stream arguments ──
    parser.add_argument("--modes", default=DEFAULT_MODES,
                        help="Comma-separated stream modes: detect,passthrough,patch. "
                             "Use --legacy for v5 single-stream compat.")
    parser.add_argument("--cameras", default=None,
                        help="Comma-separated camera indices per stream (empty for patch). "
                             f"Default: {DEFAULT_CAMERAS}")
    parser.add_argument("--zmq-ports", default=None,
                        help=f"Comma-separated ZMQ ports. Default: {DEFAULT_ZMQ_PORTS}")
    parser.add_argument("--zmq-names", default=None,
                        help=f"Comma-separated ZMQ camera_names. Default: {DEFAULT_ZMQ_NAMES}")
    parser.add_argument("--zmq-res", default=None,
                        help="Comma-separated ZMQ WxH per stream. "
                             f"Default: {DEFAULT_ZMQ_RES}")
    parser.add_argument("--capture-res", default=None,
                        help="Comma-separated capture WxH per stream (empty for patch). "
                             f"Default: {DEFAULT_CAPTURE_RES}")
    parser.add_argument("--patch-sources", default=None,
                        help=f"Comma-separated source stream idx per stream. Default: {DEFAULT_PATCH_SOURCES}")

    # ── Shared arguments ──
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--classes", default=None, help="Comma-separated class list (applied to ALL detect streams)")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print available cameras and exit")
    parser.add_argument("--no-zmq", action="store_true", help="Disable ALL ZMQ publishers")
    parser.add_argument("--passthrough", action="store_true",
                        help="v5 compat: start in passthrough mode on single detect stream")
    parser.add_argument("--fresh", action="store_true", help="Skip loading saved state")
    parser.add_argument("--init-state-file", action="store_true",
                        help="Back up the existing state file and start fresh. "
                             "Use this when the state file is corrupted and you "
                             "cannot fix it manually.")
    parser.add_argument("--bbox-expand", type=float, default=BBOX_EXPAND_RATIO,
                        help=f"Expand committed bbox by this fraction per side "
                             f"(default {BBOX_EXPAND_RATIO}, 0.0=disable)")
    parser.add_argument("--patch-top-pad", type=float, default=PATCH_TOP_PAD_RATIO,
                        help=f"Extra padding on the TOP edge of the patch crop only, "
                             f"as a fraction of bbox height. Recovers the object top that "
                             f"YOLO-World under-covers (default {PATCH_TOP_PAD_RATIO}, 0.0=disable)")
    parser.add_argument("--debug-keys", action="store_true",
                        help="Print raw waitKeyEx() values to diagnose modifier-bit issues")

    # ── Legacy v5 single-stream mode ──
    parser.add_argument("--legacy", action="store_true",
                        help="Run in v5 single-stream compat mode (legacy). "
                             "Without this flag, the default is multi-stream.")
    parser.add_argument("--camera", type=int, default=None,
                        help="legacy: camera index for single-stream mode")
    parser.add_argument("--zmq-port", type=int, default=None,
                        help="legacy: ZMQ port for single-stream mode")
    parser.add_argument("--zmq-name", default=None,
                        help="legacy: ZMQ camera_name for single-stream mode")

    args = parser.parse_args()

    # Apply CLI-overridable config
    global _bbox_expand_ratio, _patch_top_pad_ratio
    _bbox_expand_ratio = args.bbox_expand
    _patch_top_pad_ratio = args.patch_top_pad

    # ═══════════════════════════════════════════════════════
    # Determine mode: multi-stream (default) OR legacy v5
    # ═══════════════════════════════════════════════════════
    if args.legacy:
        # ────────────────────────────────────────────────
        # Legacy v5 single-stream backward-compat mode
        # ────────────────────────────────────────────────
        print("Scanning cameras...")
        available = probe_cameras()
        if not available:
            print("[ERROR] No cameras found.")
            sys.exit(1)
        print(f"Available cameras: {available}  (0 = built-in on Mac)")
        if args.list:
            sys.exit(0)

        state = reinit_state() if args.init_state_file else (None if args.fresh else load_state())
        if state:
            print(f"Loaded state: {json.dumps(state, indent=2)[:200]}...")

        # Resolve camera
        if args.camera is not None:
            cam_idx = args.camera
        elif state and "streams" in state:
            # Try to get camera from first detect stream in state
            for sname, sdata in state.get("streams", {}).items():
                if sdata.get("camera") is not None and sdata["camera"] in available:
                    cam_idx = sdata["camera"]
                    break
            else:
                candidates = [c for c in available if c not in SKIP_INDICES]
                cam_idx = candidates[0] if candidates else available[0]
        else:
            candidates = [c for c in available if c not in SKIP_INDICES]
            cam_idx = candidates[0] if candidates else available[0]

        cap = open_camera(cam_idx)
        if cap is None:
            print(f"[ERROR] Cannot open camera {cam_idx}")
            sys.exit(1)
        actual_cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {cam_idx} open at {actual_cap_w}×{actual_cap_h}")

        # ZMQ setup
        zmq_port = args.zmq_port or 5555
        zmq_name = args.zmq_name or "annotated"
        if args.no_zmq:
            zmq_sock = None
            zmq_port_active = None
            print("ZMQ disabled.")
        else:
            zmq_ctx, zmq_sock = setup_zmq(zmq_port)
            zmq_port_active = zmq_port
            print(f"ZMQ publisher bound on tcp://*:{zmq_port}  camera_name='{zmq_name}'")
            print(f"  lerobot config: {{\"type\": \"zmq\", \"server_address\": \"localhost\", "
                  f"\"port\": {zmq_port}, \"camera_name\": \"{zmq_name}\"}}")

        # Model / passthrough
        if args.passthrough:
            print("PASSTHROUGH mode — raw camera feed, no YOLO detection.")
            model = None
            active_device = "cpu"
            buf = None
            frozen = False
            classes = []
        else:
            classes = ([c.strip() for c in args.classes.split(",")] if args.classes
                       else DEFAULT_CLASSES)
            model, active_device = load_model(args.device, classes)
            buf = DetectionBuffer()
            frozen = False

        passthrough_active = args.passthrough
        frozen_bbox = None
        frozen_label = ""
        frozen_conf = 0.0

        show_raw = False
        show_all = args.show_all
        selected_det_idx = None
        prev_sel_idx = None

        pick_zone = None
        excl_zone = None
        if state and "streams" in state:
            for sdata in state.get("streams", {}).values():
                if sdata.get("pick_zone"):
                    pick_zone = tuple(sdata["pick_zone"])
                if sdata.get("excl_zone"):
                    excl_zone = tuple(sdata["excl_zone"])
                if sdata.get("frozen") and sdata.get("frozen_bbox"):
                    frozen = True
                    frozen_bbox = tuple(sdata["frozen_bbox"])
                    frozen_label = sdata.get("frozen_label", "")
                    if buf and frozen_bbox:
                        for _ in range(BUFFER_SIZE):
                            buf._buf.append(frozen_bbox)
                    print(f"Auto-frozen: {frozen_bbox}  label={frozen_label}")
                break  # v5 compat: only first stream

        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)

        frame_count = 0
        fps_timer = cv2.getTickCount()
        fps = 0.0
        buf_read = deque(maxlen=30)
        buf_infer = deque(maxlen=30)
        buf_total = deque(maxlen=30)
        avg_read = avg_infer = avg_total = 0.0
        info_str = "Initialising..."

        print("\nControls: Q=quit  P=passthrough  F=freeze  [/]=cycle  C=cam  A=all  R=raw  S=snap  Z=pick-zone  X=excl-zone\n")

        # Make the v5 window resizable
        cv2.namedWindow("Annotation Stream v5", cv2.WINDOW_NORMAL)

        # ── v5 compat main loop ──
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret:
                cap.release()
                cap = open_camera(cam_idx)
                if cap is None:
                    break
                continue

            frame_count += 1
            elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = cv2.getTickCount()

            detections = []
            if passthrough_active:
                median_bbox = None
                disp_label = ""
                disp_conf = 0.0
                info_str = "PASSTHROUGH — raw feed"
            elif frozen:
                median_bbox = frozen_bbox
                disp_label = frozen_label
                disp_conf = frozen_conf
            else:
                detections = raw_detect(frame, model, pick_zone, excl_zone)
                if selected_det_idx is not None:
                    if not detections:
                        selected_det_idx = None
                    elif selected_det_idx >= len(detections):
                        selected_det_idx = None
                if selected_det_idx != prev_sel_idx:
                    buf.hard_reset()
                if selected_det_idx is not None:
                    best_match = detections[selected_det_idx]
                else:
                    best_match = None
                    if detections:
                        current = buf.median()
                        if current is not None:
                            for d in detections:
                                _, x1, y1, x2, y2, _, _ = d
                                if _iou((x1, y1, x2, y2), current) >= IOU_GATE:
                                    best_match = d
                                    break
                        if best_match is None:
                            best_match = detections[0]
                median_bbox = buf.update(best_match)
                if best_match is not None:
                    _, _, _, _, _, disp_label, disp_conf = best_match
                else:
                    disp_label, disp_conf = buf.label, 0.0

            # ZMQ
            if passthrough_active:
                annotated = frame
            else:
                annotated = draw_annotated_stream(frame, median_bbox, buf, frozen)
            if zmq_sock is not None:
                zmq_publish(zmq_sock, zmq_name, annotated)

            # HUD
            display = frame.copy()
            if passthrough_active:
                pass
            else:
                if not show_raw:
                    if not frozen and show_all and len(detections) > 1:
                        draw_secondary(display, detections)
                    draw_target_hud(display, median_bbox, disp_label, disp_conf, buf, frozen)
                if frozen:
                    info_str = f"FROZEN  bbox {frozen_bbox}"
                elif buf.in_hysteresis:
                    remaining = HYSTERESIS_SECS - (time.monotonic() - buf._last_valid_at)
                    info_str = f"Holding committed target — releasing in {remaining:.0f}s"
                elif buf.in_cooldown:
                    remaining = LOSS_COOLDOWN_SECS - (time.monotonic() - buf._lost_at)
                    info_str = f"Cooldown — suppressing re-detection ({remaining:.1f}s)"
                elif buf.is_stale:
                    info_str = "Waiting for object..."
                elif median_bbox is None:
                    info_str = "No objects detected"
                else:
                    n = len(detections)
                    sel_tag = f"  [SEL {selected_det_idx+1}/{n}]" if selected_det_idx is not None else ""
                    stable_tag = "  [STABLE]" if buf.is_stable else f"  [{int(buf.fill*100)}%]"
                    info_str = f"{n} object{'s' if n != 1 else ''} detected{sel_tag}{stable_tag}"
                if show_raw:
                    info_str = "[RAW] " + info_str

            if not passthrough_active and not show_raw:
                draw_stability_bar(display, buf, frozen)
                draw_zone_overlays(display, pick_zone, excl_zone)

            # Draw v5-style HUD
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 38), (20, 20, 20), -1)
            mode_label = "[ FROZEN ]  YOLO-World" if frozen else f"YOLO-World ({active_device})"
            mode_color = COLOR_FROZEN if frozen else COLOR_RUNNING
            cv2.putText(display, mode_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
            cv2.putText(display, f"{fps:.1f} fps  |  r:{avg_read:.0f}  i:{avg_infer:.0f}  t:{avg_total:.0f} ms",
                        (w - 310, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, info_str, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            if show_all:
                cv2.putText(display, "[ALL]", (w - 60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            if zmq_port_active is not None:
                zmq_label = f"ZMQ:{zmq_port_active}"
                cv2.putText(display, zmq_label, (w - 90, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 100), 1)
            cam_str = "CAM [" + "  ".join(f">{c}<" if c == cam_idx else str(c) for c in available) + "]"
            cv2.putText(display, cam_str, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
            hint = "Q=quit  P=passthrough  F=freeze  [/]=cycle  C=cam  A=all  R=raw  S=snap  Z=pick-zone  X=excl-zone"
            cv2.putText(display, hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

            cv2.imshow("Annotation Stream v5", display)

            prev_sel_idx = selected_det_idx

            t2 = time.perf_counter()
            buf_read.append((t1 - t0) * 1000)
            buf_infer.append((t2 - t1) * 1000)
            buf_total.append((t2 - t0) * 1000)
            avg_read = sum(buf_read) / len(buf_read)
            avg_infer = sum(buf_infer) / len(buf_infer)
            avg_total = sum(buf_total) / len(buf_total)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Save state in multi-stream format (loadable by load_state)
                # "annotated" key matches DEFAULT_ZMQ_NAMES for multi-stream restore
                data = {
                    "streams": {
                        "annotated": {
                            "camera": cam_idx,
                            "capture_width": 1024,
                            "capture_height": 768,
                            "classes": list(classes) if isinstance(classes, (list, tuple)) else [classes],
                            "pick_zone": list(pick_zone) if pick_zone else None,
                            "excl_zone": list(excl_zone) if excl_zone else None,
                            "frozen": frozen,
                            "frozen_bbox": list(frozen_bbox) if frozen_bbox else None,
                            "frozen_label": frozen_label if frozen else "",
                        }
                    }
                }
                STATE_FILE.write_text(json.dumps(data, indent=2))
                break
            elif key == ord('p'):
                if model is not None:
                    passthrough_active = not passthrough_active
                    selected_det_idx = None
                    if passthrough_active:
                        buf.hard_reset()
                        print("Passthrough ON — raw camera feed, YOLO bypassed")
                    else:
                        buf.hard_reset()
                        print("Passthrough OFF — YOLO annotation resumed")
                else:
                    print("Passthrough toggle unavailable — model not loaded (start without --passthrough)")
            elif key == ord('f'):
                if passthrough_active:
                    pass
                elif frozen:
                    frozen = False
                    frozen_bbox = None
                    frozen_label = ""
                    frozen_conf = 0.0
                    buf.hard_reset()
                    print("Unfrozen. Buffer reset.")
                else:
                    m = buf.median()
                    if m:
                        frozen = True
                        frozen_bbox = m
                        frozen_label = buf.label
                        frozen_conf = disp_conf
                        print(f"Frozen: {frozen_bbox}  label={frozen_label}")
                    else:
                        print("Nothing to freeze — no stable detection yet.")
                selected_det_idx = None
            elif key == ord('[') or key == 0xFF55 or key == 0x210000:
                if not passthrough_active and detections:
                    n = len(detections)
                    if selected_det_idx is None:
                        selected_det_idx = 0
                    else:
                        selected_det_idx = (selected_det_idx - 1) % n
                    print(f"Selected object {selected_det_idx+1}/{n}  label={detections[selected_det_idx][5]}")
            elif key == ord(']') or key == 0xFF56 or key == 0x220000:
                if not passthrough_active and detections:
                    n = len(detections)
                    if selected_det_idx is None:
                        selected_det_idx = 0
                    else:
                        selected_det_idx = (selected_det_idx + 1) % n
                    print(f"Selected object {selected_det_idx+1}/{n}  label={detections[selected_det_idx][5]}")
            elif key == ord('a'):
                if not passthrough_active:
                    show_all = not show_all
                    n = len(detections) if detections else 0
                    state = "ON" if show_all else "OFF"
                    print(f"Show-all {state}  ({n} objects detected)")
                else:
                    print("Show-all unavailable while passthrough is active")
            elif key == ord('r'):
                if not passthrough_active:
                    show_raw = not show_raw
                    state = "ON" if show_raw else "OFF"
                    print(f"Raw view {state}")
                else:
                    print("Raw view unavailable while passthrough is active")
            elif key == ord('s'):
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = snapshot_dir / f"cam{cam_idx}_{ts_str}.jpg"
                cv2.imwrite(str(path), display)
                print(f"Snapshot: {path}")
            elif key == ord('z'):
                if passthrough_active:
                    pass
                elif pick_zone is not None:
                    pick_zone = None
                    print("Pick zone cleared.")
                else:
                    print("Draw PICK zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
                    roi = _manual_select_roi("Annotation Stream v5", frame)
                    pick_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
                    print(f"Pick zone: {pick_zone}" if pick_zone else "Cancelled.")
            elif key == ord('x'):
                if passthrough_active:
                    pass
                elif excl_zone is not None:
                    excl_zone = None
                    print("Exclusion zone cleared.")
                else:
                    print("Draw EXCLUSION zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
                    roi = _manual_select_roi("Annotation Stream v5", frame)
                    excl_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
                    print(f"Exclusion zone: {excl_zone}" if excl_zone else "Cancelled.")
            elif key in (ord('c'), 83, 81):
                if len(available) > 1:
                    pos = available.index(cam_idx) if cam_idx in available else 0
                    next_idx = available[(pos + (1 if key != 81 else -1)) % len(available)]
                    new_cap = open_camera(next_idx)
                    if new_cap:
                        cap.release()
                        cap, cam_idx = new_cap, next_idx
                        buf.reset()
                        print(f"Camera → {cam_idx}")

        cap.release()
        if zmq_sock is not None:
            zmq_sock.close()
            zmq_ctx.term()
        cv2.destroyAllWindows()
        print("Done.")
        return

    # ═══════════════════════════════════════════════════════
    # Multi-stream mode
    # ═══════════════════════════════════════════════════════
    mode_list = [m.strip() for m in args.modes.split(",")]
    n_streams = len(mode_list)
    print(f"Multi-stream mode: {n_streams} streams  modes={mode_list}")

    # Parse multi-stream args
    cameras    = parse_int_list(args.cameras or DEFAULT_CAMERAS)
    zmq_ports  = parse_int_list(args.zmq_ports or DEFAULT_ZMQ_PORTS)
    zmq_names  = parse_str_list(args.zmq_names or DEFAULT_ZMQ_NAMES)
    zmq_res    = parse_res_list(args.zmq_res or DEFAULT_ZMQ_RES)
    cap_res    = parse_res_list(args.capture_res or DEFAULT_CAPTURE_RES)
    patch_srcs = parse_int_list(args.patch_sources or DEFAULT_PATCH_SOURCES)

    # Pad to n_streams
    while len(cameras) < n_streams: cameras.append(None)
    while len(zmq_ports) < n_streams: zmq_ports.append(5555 + len(zmq_ports))
    while len(zmq_names) < n_streams: zmq_names.append(f"stream_{len(zmq_names)}")
    while len(zmq_res) < n_streams: zmq_res.append((640, 480))
    while len(cap_res) < n_streams: cap_res.append((None, None))
    while len(patch_srcs) < n_streams: patch_srcs.append(None)

    # Validate
    if len(mode_list) != n_streams:
        print("[ERROR] Inconsistent stream counts in arguments.")
        sys.exit(1)

    # Scan cameras
    print("Scanning cameras...")
    available = probe_cameras()
    if not available:
        print("[ERROR] No cameras found.")
        sys.exit(1)
    print(f"Available cameras: {available}  (0 = built-in on Mac)")
    if args.list:
        sys.exit(0)

    # Load state
    state = reinit_state() if args.init_state_file else (None if args.fresh else load_state())
    if state:
        print("Loaded multi-stream state.")

    # Resolve classes
    if args.classes:
        shared_classes = [c.strip() for c in args.classes.split(",")]
        print(f"Classes from --classes flag: {shared_classes}")
    elif state:
        # Get classes from first detect stream in state
        shared_classes = None
        for sdata in state.get("streams", {}).values():
            if sdata.get("classes"):
                shared_classes = sdata["classes"]
                break
        if shared_classes is None:
            shared_classes = DEFAULT_CLASSES
            print(f"Classes: no classes in state file, using defaults: {shared_classes}")
        else:
            print(f"Classes from state file: {shared_classes}")
    else:
        shared_classes = DEFAULT_CLASSES
        print(f"Classes: no state file, using defaults: {shared_classes}")

    # Build StreamConfigs — prefer saved-state camera over CLI defaults
    stream_names = zmq_names  # "annotated", "wrist", "target_patch" from DEFAULT_ZMQ_NAMES

    configs = []
    for i in range(n_streams):
        mode = mode_list[i]
        name = stream_names[i] if i < len(stream_names) else f"stream_{i}"

        zmq_w, zmq_h = zmq_res[i] if i < len(zmq_res) and zmq_res[i][0] is not None else (640, 480)
        cap_w, cap_h = cap_res[i] if i < len(cap_res) else (None, None)

        cam_idx = cameras[i] if i < len(cameras) else None
        src = patch_srcs[i] if i < len(patch_srcs) else None

        # Prefer saved camera from state file over CLI default
        if state and "streams" in state:
            sdata = state["streams"].get(name, {})
            if sdata.get("camera") is not None and mode in ("detect", "passthrough"):
                cam_idx = sdata["camera"]
            # Also restore ZMQ config from state
            if sdata.get("zmq_port") is not None:
                zmq_ports[i] = sdata["zmq_port"]
            if sdata.get("zmq_width") is not None:
                zmq_w = sdata["zmq_width"]
            if sdata.get("zmq_height") is not None:
                zmq_h = sdata["zmq_height"]
            if sdata.get("capture_width") is not None:
                cap_w = sdata["capture_width"]
            if sdata.get("capture_height") is not None:
                cap_h = sdata["capture_height"]

        cfg = StreamConfig(
            id=i,
            name=name,
            mode=mode,
            zmq_port=zmq_ports[i] if i < len(zmq_ports) else (5555 + i),
            zmq_camera_name=zmq_names[i] if i < len(zmq_names) else f"stream_{i}",
            zmq_width=zmq_w,
            zmq_height=zmq_h,
            camera_idx=cam_idx,
            capture_width=cap_w,
            capture_height=cap_h,
            classes=list(shared_classes) if mode == "detect" else None,
            passthrough_start=args.passthrough if mode == "detect" else False,
            source_stream=src,
        )
        configs.append(cfg)

    # ── Validate ──
    # All detect streams must share same classes (single YOLO instance)
    detect_classes = [tuple(cfg.classes) for cfg in configs if cfg.mode == "detect" and cfg.classes]
    if len(set(detect_classes)) > 1:
        raise ValueError(
            "Per-stream class lists require per-stream YOLO instances (not yet implemented). "
            "Use --classes to set a common class list for all detect streams."
        )

    # Patch streams must appear after their source (fixes R2-B2)
    for cfg in configs:
        if cfg.mode == "patch" and cfg.source_stream is not None:
            if cfg.id <= cfg.source_stream:
                raise ValueError(
                    f"Patch stream '{cfg.name}' (idx {cfg.id}) must appear after "
                    f"its source stream (idx {cfg.source_stream}). "
                    f"Check --modes ordering and --patch-sources."
                )

    # ── Build StreamState instances ──
    streams = []
    for cfg in configs:
        s = StreamState(cfg=cfg)
        if cfg.mode in ("detect", "passthrough"):
            # Auto-assign camera if None
            # Validate / auto-assign camera
            assigned = {st.cfg.camera_idx for st in streams if st.cfg.camera_idx is not None}
            if cfg.camera_idx is None or cfg.camera_idx not in available:
                if cfg.camera_idx is not None:
                    print(f"  Stream {cfg.id} '{cfg.name}': camera {cfg.camera_idx} not available, auto-assigning...")
                # Pick first available not already assigned
                for c in available:
                    if c not in SKIP_INDICES and c not in assigned:
                        cfg.camera_idx = c
                        break
                else:
                    cfg.camera_idx = available[0]  # fallback
            print(f"  Stream {cfg.id} '{cfg.name}' [{cfg.mode}]: opening camera {cfg.camera_idx}...")
            cap = open_camera(cfg.camera_idx,
                             cfg.capture_width or 640,
                             cfg.capture_height or 480)
            if cap is None:
                print(f"  [ERROR] Cannot open camera {cfg.camera_idx}")
                sys.exit(1)
            s.cap = cap
            s.actual_capture_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            s.actual_capture_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"    open at {s.actual_capture_w}×{s.actual_capture_h} "
                  f"→ ZMQ {cfg.zmq_width}×{cfg.zmq_height}")

            # ── Bump to max capture resolution (better patch crops) ──
            max_w, max_h = probe_max_capture_res(cap, s.actual_capture_w, s.actual_capture_h)
            if (max_w, max_h) != (s.actual_capture_w, s.actual_capture_h):
                cfg.capture_width = max_w
                cfg.capture_height = max_h
                s.actual_capture_w = max_w
                s.actual_capture_h = max_h
                print(f"    ↑ bumped capture to {max_w}×{max_h} "
                      f"(ZMQ stays at {cfg.zmq_width}×{cfg.zmq_height})")

            if cfg.mode == "detect":
                s.buf = DetectionBuffer()
                s.passthrough_active = cfg.passthrough_start
        else:
            print(f"  Stream {cfg.id} '{cfg.name}' [{cfg.mode}]: derived from stream {cfg.source_stream}")

        # ZMQ setup
        if not args.no_zmq:
            s.zmq_ctx, s.zmq_sock = setup_zmq(cfg.zmq_port)
            print(f"    ZMQ :{cfg.zmq_port}  name='{cfg.zmq_camera_name}'")
        streams.append(s)

    # ── Restore runtime state ──
    # (camera assignments were already applied during config build above)
    if state and "streams" in state:
        for s in streams:
            sdata = state["streams"].get(s.cfg.name, {})
            if not sdata:
                continue
            if s.cfg.mode == "detect":
                if sdata.get("pick_zone"):
                    s.pick_zone = tuple(sdata["pick_zone"])
                if sdata.get("excl_zone"):
                    s.excl_zone = tuple(sdata["excl_zone"])
                if sdata.get("classes") and not args.classes:
                    # Only restore classes from state if the user didn't
                    # provide --classes on the CLI (CLI always wins).
                    s.cfg.classes = sdata["classes"]
                # Restore frozen state
                if sdata.get("frozen") and sdata.get("frozen_bbox"):
                    s.frozen = True
                    s.frozen_bbox = tuple(sdata["frozen_bbox"])
                    s.frozen_label = sdata.get("frozen_label", "")
                    # Pre-fill buffer
                    if s.buf and s.frozen_bbox:
                        for _ in range(BUFFER_SIZE):
                            s.buf._buf.append(s.frozen_bbox)
                    print(f"  [{s.cfg.name}] Auto-frozen: {s.frozen_bbox}  label={s.frozen_label}")
            # selected_det_idx always reset to None on restore (fixes R1-#12)
            s.selected_det_idx = None

    # ── Load shared model ──
    has_detect = any(s.cfg.mode == "detect" and not s.cfg.passthrough_start for s in streams)
    if has_detect:
        shared_model, active_device = load_model(args.device, shared_classes)
    else:
        shared_model = None
        active_device = "cpu"
        print("No detect streams with model — YOLO not loaded.")

    # Setup snapshot dir
    snapshot_dir = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)

    fps_timer = cv2.getTickCount()
    frame_count = 0
    fps = 0.0

    # Save initial config on FIRST RUN only (no state file existed).
    # Never overwrite an existing state file on startup — the user may have
    # hand-edited it, and only explicit actions (F, Z, X, C, Q) should write.
    if not args.no_zmq and state is None:
        print("[save_state] First run — writing initial config to state file")
        save_state(streams)

    print(f"\nMulti-stream annotator ready. {n_streams} streams. "
          f"TAB=cycle active stream  "
          f"Q=quit  [no mod]:S0  [SHIFT]:S1  [CTRL+SHIFT]:S2\n")

    active_stream_idx = 0  # which stream receives keystrokes (cycled via Tab)

    # ── Main loop ──
    while True:
        for stream in streams:
            t0 = time.perf_counter()
            process_stream(stream, streams, shared_model, args.no_zmq)
            stream.proc_ms = (time.perf_counter() - t0) * 1000

        frame_count += 1
        elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = cv2.getTickCount()

        # Apply per-stream HUD overlays onto each pane before compositing
        for s in streams:
            if s.display is not None:
                if s.cfg.mode == "detect":
                    draw_detect_hud(s.display, s, s.detections, fps, active_device)
                elif s.cfg.mode == "passthrough":
                    draw_passthrough_hud(s.display, s, fps)
                elif s.cfg.mode == "patch":
                    draw_patch_hud(s.display, s, fps)

        # Mark active stream before compositing
        for i, s in enumerate(streams):
            s.is_active = (i == active_stream_idx)

        # Build composite with HUD-applied panes
        model_label = f"YOLO-World ({active_device})" if shared_model else "No YOLO"
        shared_bar = f"{model_label}  {fps:.1f} fps  [ACTIVE: {streams[active_stream_idx].cfg.name}]"
        composite = build_composite(streams, shared_bar)

        # Highlight active stream pane (bright border on its label bar)
        # build_composite draws labels; we overlay a highlight here
        # (The active indicator is in shared_bar; per-pane highlight via border)

        # Create resizable window on first frame (no resizeWindow —
        # it fights manual resizes by snapping back on some backends)
        if frame_count == 1:
            cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(MAIN_WINDOW, composite)

        # ── Key dispatch ──
        key = cv2.waitKeyEx(1)
        modifiers = key & 0xFF0000

        if args.debug_keys and key != -1 and key != 0:
            print(f"[DEBUG] raw=0x{key:06X} ({key})  modifiers=0x{modifiers:06X}  "
                  f"low_byte=0x{(key & 0xFF):02X} ('{chr(key & 0xFF) if 32 <= (key & 0xFF) < 127 else '?'}')  "
                  f"ext=0x{(key & ~0x300000):06X}  active=S{active_stream_idx}")

        # ── Tab: cycle active stream (platform-agnostic targeting) ──
        key_ext = key & ~0x300000  # strip modifier bits only, never & 0xFFFF (fixes R2-A4)
        if key_ext == 0x09 or (key & 0xFF) == ord('\t'):  # Tab (raw or ASCII)
            active_stream_idx = (active_stream_idx + 1) % len(streams)
            print(f"Active stream → {active_stream_idx} ({streams[active_stream_idx].cfg.name} "
                  f"[{streams[active_stream_idx].cfg.mode}])")
            continue

        # Target selection: modifier bits first (works on GTK), fall back to active_stream
        if modifiers == 0x300000:         # Ctrl+Shift → stream 2
            target_idx = min(2, len(streams) - 1)
        elif modifiers == 0x100000:       # Shift → stream 1
            target_idx = min(1, len(streams) - 1)
        else:                              # no modifier → active stream
            target_idx = active_stream_idx

        # Guard: if target_idx exceeds available streams, fall back to last
        if target_idx >= len(streams):
            target_idx = len(streams) - 1

        target = streams[target_idx]

        # ── Normalize shifted letter keys (fixes R2-A2) ──
        if modifiers == 0x100000:                     # Shift only, no Ctrl
            low_byte = key & 0xFF
            if ord('A') <= low_byte <= ord('Z'):
                key_ascii = low_byte | 0x20           # 'F' → 'f'
            else:
                key_ascii = low_byte
        else:
            key_ascii = key & 0xFF

        if not dispatch_key(key_ext, key_ascii, target, streams, available, args.no_zmq):
            break  # quit

    # ── Cleanup ──
    for s in streams:
        if s.cap is not None:
            s.cap.release()
        if s.zmq_sock is not None:
            s.zmq_sock.close()
            s.zmq_ctx.term()
    cv2.destroyAllWindows()

    # Save state on exit (unless --no-zmq prevented tracking)
    if not args.no_zmq:
        save_state(streams)
    print("Done.")


if __name__ == "__main__":
    main()
