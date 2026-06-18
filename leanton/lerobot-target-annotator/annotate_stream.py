"""
LeRobot Target Annotator — ZMQ Publisher  [v5]
===============================================
Streams the annotated overview frame to LeRobot's native ZMQCamera.

v5 changes (from v4):
  - MAX_BBOX_AREA_RATIO (0.65): reject detections whose bbox covers >65% of the frame
    before they enter the stabilization buffer. Prevents YOLO-World's low-confidence
    full-screen "desk object" / "cloth toy" hallucinations after scene changes (e.g.
    removing an object from a cluttered table). The guard is applied in raw_detect()
    so the spurious detection never seeds or contaminates the buffer.

v4 state machine (simplified from v3):
  FILLING → STABLE → HOLD (8s countdown) → release → COOLDOWN (8s) → FILLING
  HOLD tolerates arm occlusions for HYSTERESIS_SECS, then releases immediately.
  Challenger buffer and stale-frame gating removed for committed targets.

v3 changes (from v2):
  - HYSTERESIS_SECS: 1.0 → 8.0
  - Protected state resets stale_count
  - CHALLENGER_SIZE: 8 → 90 (now dead code in v4)

Two display surfaces, one ZMQ stream:
  ZMQ stream      — clean frame + bbox rectangle only; consumed by LeRobot
  imshow window   — full operator HUD: stability bar, zones, fps, hint bar

The ZMQ frame is published every loop iteration:
  - Before target commits:  raw clean frame (no bbox)
  - After target commits:   frame with bbox rectangle drawn (green / cyan)

LeRobot camera config for this stream:
  {"type": "zmq", "server_address": "localhost", "port": 5555,
   "camera_name": "annotated", "width": 640, "height": 480}

Setup:
    conda activate lerobot
    pip install ultralytics pyzmq

Run:
    python annotate_stream.py
    python annotate_stream.py --zmq-port 5556 --zmq-name wrist_annotated
    python annotate_stream.py --no-zmq          # disable ZMQ (annotation only)
    python annotate_stream.py --camera 1 --classes "cup,bottle,toy"
    python annotate_stream.py --passthrough      # raw camera passthrough, no YOLO

Controls:
    P   — toggle passthrough / annotated (requires model loaded)
    F   — freeze / unfreeze target (buffer resets on unfreeze)
    [ ] / PgUp PgDn — cycle through detected objects
    A   — toggle show-all secondary detections
    C   — next camera
    R   — toggle raw view (display only — detection + ZMQ continue)
    S   — save snapshot
    Z   — define pick zone / clear
    X   — define exclusion zone / clear
    Q   — quit (auto-saves state)

State file: annotate_stream_state.json (camera, classes, zones, last frozen bbox).
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
RESOLUTION        = (640, 480)

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

# ── Challenger buffer (dead code in v4) ──────────────────────
# v4 bypasses the challenger entirely for committed targets — when HOLD expires
# the buffer resets immediately, no gradual takeover.  Kept for reference only.
CHALLENGER_SIZE   = 90
CHALLENGER_FILL   = 0.75

# ── Loss cooldown ────────────────────────────────────────────
# After a committed target is lost (HOLD expired → reset), incoming detections
# are suppressed for this many seconds to prevent false positives on an empty
# table from immediately re-locking the buffer.
# object from being annotated mid-episode and polluting the policy.
LOSS_COOLDOWN_SECS = 8.0

# ── ZMQ publisher ────────────────────────────────────────────
ZMQ_PORT          = 5555          # must match lerobot ZMQCameraConfig port
ZMQ_CAMERA_NAME   = "annotated"   # must match lerobot ZMQCameraConfig camera_name
ZMQ_JPEG_QUALITY  = 85            # 0-100; higher = larger frame, lower latency loss

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


# ── ZMQ publisher ─────────────────────────────────────────────────────────────

def setup_zmq(port):
    import zmq
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 5)    # drop old frames under backpressure
    sock.setsockopt(zmq.LINGER, 0)    # don't block on close
    sock.bind(f"tcp://*:{port}")
    return ctx, sock


def zmq_publish(sock, camera_name, frame_bgr, quality=ZMQ_JPEG_QUALITY):
    """Encode frame as JPEG and publish in LeRobot ZMQCamera wire format.

    ZMQCamera decodes with cv2.imdecode (returns BGR) but performs no
    BGR→RGB conversion, while all other LeRobot cameras return RGB.
    Sending RGB here ensures the decoded bytes match what downstream expects.
    """
    import zmq
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _, jpg = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    encoded = base64.b64encode(jpg).decode("utf-8")
    msg = json.dumps({
        "timestamps": {camera_name: time.monotonic()},
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


def _open_camera_raw(idx, resolution=RESOLUTION):
    """Open camera with V4L2 backend — used for probing only. Returns cap or None."""
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap


def open_camera(idx, resolution=RESOLUTION):
    """Open camera for sustained capture. Same low-latency path as _open_camera_raw."""
    return _open_camera_raw(idx, resolution)


# ── State persistence ─────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).with_name("annotate_stream_state.json")


def save_state(cam_idx, classes, pick_zone, excl_zone, frozen, frozen_bbox, frozen_label):
    data = {
        "camera": cam_idx,
        "classes": classes,
        "pick_zone": list(pick_zone) if pick_zone else None,
        "excl_zone": list(excl_zone) if excl_zone else None,
        "frozen": frozen,
        "frozen_bbox": list(frozen_bbox) if frozen_bbox else None,
        "frozen_label": frozen_label if frozen else "",
    }
    STATE_FILE.write_text(json.dumps(data, indent=2))


def load_state():
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        if data.get("pick_zone"):
            data["pick_zone"] = tuple(data["pick_zone"])
        if data.get("excl_zone"):
            data["excl_zone"] = tuple(data["excl_zone"])
        if data.get("frozen_bbox"):
            data["frozen_bbox"] = tuple(data["frozen_bbox"])
        return data
    except (json.JSONDecodeError, KeyError):
        return None


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

class _ChallengerBuffer:
    def __init__(self, size=CHALLENGER_SIZE, iou_gate=IOU_GATE):
        self._buf      = deque(maxlen=size)
        self._size     = size
        self._iou_gate = iou_gate

    def reset(self): self._buf.clear()

    def update(self, box):
        current = self._median()
        if current is not None and _iou(box, current) < self._iou_gate:
            self._buf.clear()
        self._buf.append(box)

    def _median(self):
        if not self._buf:
            return None
        return tuple(int(v) for v in np.median(np.array(self._buf), axis=0))

    @property
    def ready(self):
        return (len(self._buf) / self._size) >= CHALLENGER_FILL

    def drain(self):
        boxes = list(self._buf)
        self._buf.clear()
        return boxes


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
        self._challenger    = _ChallengerBuffer()

    def reset(self):
        """Normal reset — enforces cooldown if target was committed before loss."""
        was_committed = self.is_stable or self.in_hysteresis
        self._buf.clear()
        self._stale_count   = 0
        self._last_label    = ""
        self._stable_at     = None
        self._last_valid_at = None
        self._challenger.reset()
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
        self._challenger.reset()

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
        self._challenger.reset()
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


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_annotated_stream(frame, bbox, buf, frozen):
    """
    Stream 3 — what LeRobot records via ZMQCamera.
    Clean frame with bbox rectangle only when target is committed.
    No HUD, no zone overlays, no stability bar.
    Always returns a frame (clean if no committed target).
    """
    out = frame.copy()
    if bbox is None or not (frozen or buf.is_stable):
        return out
    # Always render the bbox in STABLE green on the ZMQ stream.
    # The policy sees this frame — it shouldn't know or care whether the
    # operator froze the target.  Freeze vs stable distinction is for the
    # operator HUD (draw_target_hud) only.
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_STABLE, 3)
    return out


def draw_target_hud(frame, bbox, label, conf, buf, frozen):
    """Stream 4 operator overlay — bbox with label and coordinate readout."""
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


def draw_hud(frame, cam_idx, available, fps, info, show_all, frozen, zmq_port, read_ms=0.0, infer_ms=0.0, total_ms=0.0, device=""):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    mode_label = "[ FROZEN ]  YOLO-World" if frozen else f"YOLO-World ({device})"
    mode_color = COLOR_FROZEN if frozen else COLOR_RUNNING
    cv2.putText(frame, mode_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    cv2.putText(frame, f"{fps:.1f} fps  |  r:{read_ms:.0f}  i:{infer_ms:.0f}  t:{total_ms:.0f} ms",
                (w - 310, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, info, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if show_all:
        cv2.putText(frame, "[ALL]", (w - 60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    # ZMQ status indicator
    if zmq_port is not None:
        zmq_label = f"ZMQ:{zmq_port}"
        cv2.putText(frame, zmq_label, (w - 90, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 100), 1)
    cam_str = "CAM [" + "  ".join(f">{c}<" if c == cam_idx else str(c) for c in available) + "]"
    cv2.putText(frame, cam_str, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    hint = "Q=quit  P=passthrough  F=freeze  [/]=cycle  C=cam  A=all  R=raw  S=snap  Z=pick-zone  X=excl-zone"
    cv2.putText(frame, hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO-World annotation stream v5 — ZMQ publisher")
    parser.add_argument("--camera",   type=int, default=DEFAULT_CAMERA)
    parser.add_argument("--device",   default=DEFAULT_DEVICE, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--classes",  default=None, help="Comma-separated class list")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--list",     action="store_true", help="Print available cameras and exit")
    parser.add_argument("--zmq-port", type=int, default=ZMQ_PORT,
                        help=f"ZMQ PUB port (default {ZMQ_PORT})")
    parser.add_argument("--zmq-name", default=ZMQ_CAMERA_NAME,
                        help=f"Camera name in ZMQ messages, must match lerobot camera_name (default '{ZMQ_CAMERA_NAME}')")
    parser.add_argument("--no-zmq",   action="store_true", help="Disable ZMQ publisher")
    parser.add_argument("--passthrough", action="store_true",
                        help="Bypass YOLO entirely — publish raw camera feed to ZMQ. "
                             "No model loaded, no detection. For deployment scenarios "
                             "where annotation is not needed (e.g. unload task).")
    parser.add_argument("--fresh",    action="store_true", help="Skip loading saved state")
    args = parser.parse_args()

    print("Scanning cameras...")
    available = probe_cameras()
    if not available:
        print("[ERROR] No cameras found.")
        sys.exit(1)
    print(f"Available cameras: {available}  (0 = built-in on Mac)")
    if args.list:
        sys.exit(0)

    # Load saved state (unless --fresh)
    state = None if args.fresh else load_state()
    if state:
        print(f"Loaded state: camera={state.get('camera')}, classes={state.get('classes')}, "
              f"pick_zone={state.get('pick_zone')}, excl_zone={state.get('excl_zone')}, "
              f"frozen={state.get('frozen')}  (use --fresh to skip)")

    if args.camera is not None:
        start_idx = args.camera
    elif state and state.get("camera") is not None and state["camera"] in available:
        start_idx = state["camera"]
    else:
        candidates = [c for c in available if c not in SKIP_INDICES]
        start_idx  = candidates[0] if candidates else available[0]

    cap = open_camera(start_idx)
    if cap is None:
        print(f"[ERROR] Cannot open camera {start_idx}")
        sys.exit(1)
    cam_idx = start_idx
    print(f"Camera {cam_idx} open at {int(cap.get(3))}×{int(cap.get(4))}")

    # ZMQ setup
    zmq_sock = None
    zmq_port_active = None
    if not args.no_zmq:
        zmq_ctx, zmq_sock = setup_zmq(args.zmq_port)
        zmq_port_active   = args.zmq_port
        print(f"ZMQ publisher bound on tcp://*:{args.zmq_port}  camera_name='{args.zmq_name}'")
        print(f"  lerobot config: {{\"type\": \"zmq\", \"server_address\": \"localhost\", "
              f"\"port\": {args.zmq_port}, \"camera_name\": \"{args.zmq_name}\"}}")
    else:
        print("ZMQ disabled.")

    # ── Passthrough mode: skip YOLO entirely ──────────────────────────────────
    if args.passthrough:
        print("PASSTHROUGH mode — raw camera feed, no YOLO detection.")
        model      = None
        active_device = "cpu"
        buf        = None
        frozen     = False
        classes    = []
    else:
        classes = ([c.strip() for c in args.classes.split(",")] if args.classes
                   else state.get("classes") if state and state.get("classes")
                   else DEFAULT_CLASSES)
        model, active_device = load_model(args.device, classes)
        buf    = DetectionBuffer()
        frozen = False
    passthrough_active = args.passthrough  # runtime toggle via 'P' key
    frozen_bbox  = None
    frozen_label = ""
    frozen_conf  = 0.0
    disp_conf    = 0.0

    show_raw       = False
    show_all       = args.show_all
    selected_det_idx = None   # None=auto, 0..N-1=manual selection via [/] or PgUp/PgDn
    prev_sel_idx     = None   # track changes to force buffer reset on switch
    pick_zone    = state.get("pick_zone") if state else None
    excl_zone    = state.get("excl_zone") if state else None

    # Auto-restore frozen bbox from saved state (skip in passthrough — no detection)
    if not args.passthrough and state and state.get("frozen") and state.get("frozen_bbox"):
        frozen       = True
        frozen_bbox  = state["frozen_bbox"]
        frozen_label = state.get("frozen_label", "")
        # Pre-fill buffer so stability bar is green from the start
        for _ in range(BUFFER_SIZE):
            buf._buf.append(frozen_bbox)
        print(f"Auto-frozen: {frozen_bbox}  label={frozen_label}")
    snapshot_dir = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)
    frame_count  = 0
    fps_timer    = cv2.getTickCount()
    fps          = 0.0
    buf_read     = deque(maxlen=30)
    buf_infer    = deque(maxlen=30)
    buf_total    = deque(maxlen=30)
    avg_read     = avg_infer = avg_total = 0.0
    info_str     = "Initialising..."

    print("\nControls: Q=quit  P=passthrough  F=freeze  [/]=cycle  C=cam  A=all  R=raw  S=snap  Z=pick-zone  X=excl-zone\n")

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
            fps         = frame_count / elapsed
            frame_count = 0
            fps_timer   = cv2.getTickCount()

        # ── compute current bbox ───────────────────────────────────────────────
        detections = []   # always defined for the HUD block below
        if passthrough_active:
            median_bbox = None
            disp_label  = ""
            disp_conf   = 0.0
            info_str    = "PASSTHROUGH — raw feed"
        elif frozen:
            median_bbox = frozen_bbox
            disp_label  = frozen_label
            disp_conf   = frozen_conf
        else:
            detections  = raw_detect(frame, model, pick_zone, excl_zone)
            # ── manual selection override ──────────────────────────────────
            if selected_det_idx is not None:
                if not detections:
                    selected_det_idx = None   # no objects, release selection
                elif selected_det_idx >= len(detections):
                    selected_det_idx = None   # list shortened, release selection
            # Force-reset buffer when selection changes — no cooldown,
            # the operator explicitly chose a different target.
            if selected_det_idx != prev_sel_idx:
                buf.hard_reset()
            # ── pick best_match ────────────────────────────────────────────
            if selected_det_idx is not None:
                best_match = detections[selected_det_idx]
            else:
                # Prefer the detection that matches the current buffer to avoid
                # rank oscillation when two objects of similar size swap rank0
                # every frame — each resetting the other's challenger.
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

        # ── stream 3: annotated frame for ZMQ ─────────────────────────────────
        if passthrough_active:
            # Passthrough: publish raw frame directly, no annotation overlay
            annotated = frame
        else:
            annotated = draw_annotated_stream(frame, median_bbox, buf, frozen)
        if zmq_sock is not None:
            zmq_publish(zmq_sock, args.zmq_name, annotated)

        # ── stream 4: operator HUD ────────────────────────────────────────────
        display = frame.copy()

        if passthrough_active:
            # Minimal HUD — just fps, camera indicator, and passthrough label
            pass
        else:
            # Draw bbox overlays only when not in raw view
            if not show_raw:
                if not frozen and show_all and len(detections) > 1:
                    draw_secondary(display, detections)
                draw_target_hud(display, median_bbox, disp_label, disp_conf, buf, frozen)
            # Detection state info — always computed (raw view hides overlays, not state)
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
                n          = len(detections)
                sel_tag    = f"  [SEL {selected_det_idx+1}/{n}]" if selected_det_idx is not None else ""
                stable_tag = "  [STABLE]" if buf.is_stable else f"  [{int(buf.fill*100)}%]"
                info_str   = f"{n} object{'s' if n != 1 else ''} detected{sel_tag}{stable_tag}"
            if show_raw:
                info_str = "[RAW] " + info_str

        if not passthrough_active and not show_raw:
            draw_stability_bar(display, buf, frozen)
            draw_zone_overlays(display, pick_zone, excl_zone)
        draw_hud(display, cam_idx, available, fps, info_str, show_all, frozen, zmq_port_active,
                 avg_read, avg_infer, avg_total, active_device)
        cv2.imshow("Annotation Stream v5", display)

        prev_sel_idx = selected_det_idx  # track for change detection next frame

        t2 = time.perf_counter()
        buf_read.append((t1 - t0) * 1000)
        buf_infer.append((t2 - t1) * 1000)
        buf_total.append((t2 - t0) * 1000)
        avg_read  = sum(buf_read)  / len(buf_read)
        avg_infer = sum(buf_infer) / len(buf_infer)
        avg_total = sum(buf_total) / len(buf_total)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_state(cam_idx, classes, pick_zone, excl_zone, frozen, frozen_bbox, frozen_label)
            break

        elif key == ord('p'):
            if model is not None:
                passthrough_active = not passthrough_active
                selected_det_idx  = None
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
                pass  # no detection → nothing to freeze
            elif frozen:
                frozen = False
                frozen_bbox = None
                frozen_label = ""
                frozen_conf = 0.0
                buf.hard_reset()
                save_state(cam_idx, classes, pick_zone, excl_zone, False, None, "")
                print("Unfrozen. Buffer reset. State saved.")
            else:
                m = buf.median()
                if m:
                    frozen       = True
                    frozen_bbox  = m
                    frozen_label = buf.label
                    frozen_conf  = disp_conf
                    save_state(cam_idx, classes, pick_zone, excl_zone, True, m, buf.label)
                    print(f"Frozen: {frozen_bbox}  label={frozen_label}  State saved.")
                else:
                    print("Nothing to freeze — no stable detection yet.")
            selected_det_idx = None   # unfreeze also releases manual selection

        elif key == ord('[') or key == 0xFF55 or key == 0x210000:   # [ or PageUp
            if not passthrough_active and detections:
                n = len(detections)
                if selected_det_idx is None:
                    selected_det_idx = 0
                else:
                    selected_det_idx = (selected_det_idx - 1) % n
                print(f"Selected object {selected_det_idx+1}/{n}  label={detections[selected_det_idx][5]}")
            elif passthrough_active:
                print("Selection unavailable — passthrough active")

        elif key == ord(']') or key == 0xFF56 or key == 0x220000:   # ] or PageDown
            if not passthrough_active and detections:
                n = len(detections)
                if selected_det_idx is None:
                    selected_det_idx = 0
                else:
                    selected_det_idx = (selected_det_idx + 1) % n
                print(f"Selected object {selected_det_idx+1}/{n}  label={detections[selected_det_idx][5]}")
            elif passthrough_active:
                print("Selection unavailable — passthrough active")

        elif key == ord('a'):
            if not passthrough_active:
                show_all = not show_all

        elif key == ord('r'):
            if not passthrough_active:
                show_raw = not show_raw

        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = snapshot_dir / f"cam{cam_idx}_{ts}.jpg"
            cv2.imwrite(str(path), display)
            print(f"Snapshot: {path}")

        elif key == ord('z'):
            if passthrough_active:
                pass
            elif pick_zone is not None:
                pick_zone = None
                save_state(cam_idx, classes, None, excl_zone, frozen, frozen_bbox, frozen_label)
                print("Pick zone cleared. State saved.")
            else:
                print("Draw PICK zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
                roi = cv2.selectROI("Annotation Stream v5", frame,
                                    fromCenter=False, showCrosshair=True)
                pick_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
                if pick_zone:
                    save_state(cam_idx, classes, pick_zone, excl_zone, frozen, frozen_bbox, frozen_label)
                print(f"Pick zone: {pick_zone}" if pick_zone else "Cancelled.")

        elif key == ord('x'):
            if passthrough_active:
                pass
            elif excl_zone is not None:
                excl_zone = None
                save_state(cam_idx, classes, pick_zone, None, frozen, frozen_bbox, frozen_label)
                print("Exclusion zone cleared. State saved.")
            else:
                print("Draw EXCLUSION zone: drag, SPACE/ENTER to confirm, ESC=cancel.")
                roi = cv2.selectROI("Annotation Stream v5", frame,
                                    fromCenter=False, showCrosshair=True)
                excl_zone = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]) if roi[2] > 0 else None
                if excl_zone:
                    save_state(cam_idx, classes, pick_zone, excl_zone, frozen, frozen_bbox, frozen_label)
                print(f"Exclusion zone: {excl_zone}" if excl_zone else "Cancelled.")

        elif key in (ord('c'), 83, 81):
            if len(available) > 1:
                pos      = available.index(cam_idx) if cam_idx in available else 0
                next_idx = available[(pos + (1 if key != 81 else -1)) % len(available)]
                new_cap  = open_camera(next_idx)
                if new_cap:
                    cap.release()
                    cap, cam_idx = new_cap, next_idx
                    buf.reset()
                    save_state(cam_idx, classes, pick_zone, excl_zone, frozen, frozen_bbox, frozen_label)
                    print(f"Camera → {cam_idx}")

    cap.release()
    if zmq_sock is not None:
        zmq_sock.close()
        zmq_ctx.term()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
