#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Zero-copy RealSense depth sharing between two processes on the same machine.

``pyrealsense2`` often cannot be imported in the interpreter that runs the policy (e.g. a conda
python whose wheel needs a newer glibc than the robot's Jetson has). The writer therefore runs in
whichever interpreter *can* import it and publishes each depth frame into a single mmap'd file under
``/dev/shm``; the reader mmaps the same file. No sockets, no serialization: one memcpy per frame.

Writer (any python with numpy + pyrealsense2)::

    python depth_shm.py --width 480 --height 270 --fps 60

Reader (policy process)::

    reader = DepthShmReader()
    depth_mm, meta = reader.read(timeout_s=2.0)  # blocks until a frame newer than the last one

The header is protected by a seqlock: the writer bumps ``seq`` to an odd value before copying the
frame and to the next even value after, so the reader retries whenever it observes a torn read.

This module deliberately imports only the standard library and numpy at import time so the writer
can run from a bare venv.
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct
import time

import numpy as np

DEFAULT_PATH = "/dev/shm/g1_depth"  # nosec B108
DEFAULT_TOKENS_PATH = "/dev/shm/g1_tokens"  # nosec B108

# seq(u64) t_capture(f64) width(u32) height(u32) fx fy cx cy(f32 x4) depth_scale(f32) pad(u32)
_HEADER = struct.Struct("<QdIIfffffI")
HEADER_BYTES = 64
assert _HEADER.size <= HEADER_BYTES

# Policy-side contract (mirrors lerobot.robots.unitree_g1.controllers.depth_dodge; duplicated here so
# the writer can run from a bare venv that has onnxruntime-gpu but neither torch nor lerobot).
POLICY_DEPTH_SHAPE = (64, 96)
POLICY_VFOV_DEG = 45.0
THEIA_SIZE = 224
THEIA_TOKENS = (197, 192)
DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0


def _file_size(width: int, height: int) -> int:
    return HEADER_BYTES + width * height * 2


def policy_crop_bounds(
    fx: float, fy: float, cx: float, cy: float, h: int, w: int
) -> tuple[int, int, int, int]:
    """Rows/cols of the sensor image that match the policy camera (64x96 pinhole, 45 deg vFOV)."""
    import math  # noqa: PLC0415

    vfov = math.radians(POLICY_VFOV_DEG)
    hfov = 2 * math.atan(POLICY_DEPTH_SHAPE[1] / POLICY_DEPTH_SHAPE[0] * math.tan(vfov / 2))
    half_h, half_w = fy * math.tan(vfov / 2), fx * math.tan(hfov / 2)
    y0, y1 = max(int(round(cy - half_h)), 0), min(int(round(cy + half_h)), h)
    x0, x1 = max(int(round(cx - half_w)), 0), min(int(round(cx + half_w)), w)
    return y0, y1, x0, x1


def depth_to_theia_image_np(
    depth_m: np.ndarray, near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M
) -> np.ndarray:
    """numpy/cv2 twin of ``depth_dodge.depth_to_theia_image`` (255 near, 1 far, 0 invalid; bilinear to
    224x224). Differs from the torch version by at most 1 grey level on ~1% of pixels (resize rounding)."""
    import cv2  # noqa: PLC0415

    depth = np.asarray(depth_m, np.float32).astype(np.float16).astype(np.float32)
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    value = np.round(np.clip(1.0 + 254.0 * (far - depth) / (far - near), 1.0, 255.0))
    gray = np.where(valid, value, 0.0).astype(np.float32)
    x = cv2.resize(gray, (THEIA_SIZE, THEIA_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# Percept file: header + Theia tokens f32[197,192] + policy depth f32[64,96] (metres, for stats/debug).
# seq(u64) t_capture(f64) theia_ms(f32) pad(f32)
_PHEADER = struct.Struct("<Qdff")
_TOKENS_BYTES = THEIA_TOKENS[0] * THEIA_TOKENS[1] * 4
_PDEPTH_BYTES = POLICY_DEPTH_SHAPE[0] * POLICY_DEPTH_SHAPE[1] * 4
PERCEPT_FILE_BYTES = HEADER_BYTES + _TOKENS_BYTES + _PDEPTH_BYTES


class PerceptShmWriter:
    """Publishes (Theia tokens, 64x96 policy depth) per frame; same seqlock protocol as the depth file."""

    def __init__(self, path: str = DEFAULT_TOKENS_PATH):
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, PERCEPT_FILE_BYTES)
        self._mm = mmap.mmap(fd, PERCEPT_FILE_BYTES)
        os.close(fd)
        self._tokens: np.ndarray = np.ndarray(THEIA_TOKENS, np.float32, self._mm, HEADER_BYTES)
        self._depth: np.ndarray = np.ndarray(
            POLICY_DEPTH_SHAPE, np.float32, self._mm, HEADER_BYTES + _TOKENS_BYTES
        )
        self._seq = 0
        self._write_header(0.0, 0.0)

    def _write_header(self, t_capture: float, theia_ms: float) -> None:
        self._mm[: _PHEADER.size] = _PHEADER.pack(self._seq, t_capture, theia_ms, 0.0)

    def write(self, tokens: np.ndarray, depth_m: np.ndarray, t_capture: float, theia_ms: float) -> None:
        self._seq += 1
        self._write_header(t_capture, theia_ms)
        np.copyto(self._tokens, tokens)
        np.copyto(self._depth, depth_m)
        self._seq += 1
        self._write_header(t_capture, theia_ms)

    def close(self) -> None:
        self._mm.close()


class PerceptShmReader:
    def __init__(self, path: str = DEFAULT_TOKENS_PATH, wait_s: float = 10.0):
        deadline = time.monotonic() + wait_s
        while not (os.path.exists(path) and os.path.getsize(path) >= PERCEPT_FILE_BYTES):
            if time.monotonic() > deadline:
                raise TimeoutError(f"no percept writer at {path} after {wait_s:.0f}s")
            time.sleep(0.05)
        fd = os.open(path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, PERCEPT_FILE_BYTES, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        self._tokens: np.ndarray = np.ndarray(THEIA_TOKENS, np.float32, self._mm, HEADER_BYTES)
        self._depth: np.ndarray = np.ndarray(
            POLICY_DEPTH_SHAPE, np.float32, self._mm, HEADER_BYTES + _TOKENS_BYTES
        )
        self._last_seq = 0

    def _header(self) -> tuple[int, dict]:
        seq, t, theia_ms, _ = _PHEADER.unpack(self._mm[: _PHEADER.size])
        return seq, {"t_capture": t, "theia_ms": theia_ms}

    def try_read(self) -> tuple[np.ndarray, np.ndarray, dict] | None:
        """Return (tokens f32[197,192], depth_m f32[64,96], meta) for a new complete frame, else None."""
        seq0, meta = self._header()
        if seq0 & 1 or seq0 == self._last_seq or seq0 == 0:
            return None
        tokens, depth = self._tokens.copy(), self._depth.copy()
        seq1, _ = self._header()
        if seq1 != seq0:
            return None
        self._last_seq = seq0
        meta["seq"] = seq0
        return tokens, depth, meta

    def read(self, timeout_s: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
        deadline = time.monotonic() + timeout_s
        while True:
            got = self.try_read()
            if got is not None:
                return got
            if time.monotonic() > deadline:
                raise TimeoutError(f"no new percept within {timeout_s}s")
            time.sleep(0.0005)

    def close(self) -> None:
        self._mm.close()


class DepthShmWriter:
    def __init__(self, width: int, height: int, intrinsics: dict, path: str = DEFAULT_PATH):
        self.width, self.height, self.path = width, height, path
        self.intr = intrinsics
        size = _file_size(width, height)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, size)
        self._mm = mmap.mmap(fd, size)
        os.close(fd)
        self._frame: np.ndarray = np.ndarray(
            (height, width), dtype=np.uint16, buffer=self._mm, offset=HEADER_BYTES
        )
        self._seq = 0
        self._write_header(0.0)

    def _write_header(self, t_capture: float) -> None:
        self._mm[: _HEADER.size] = _HEADER.pack(
            self._seq,
            t_capture,
            self.width,
            self.height,
            self.intr["fx"],
            self.intr["fy"],
            self.intr["cx"],
            self.intr["cy"],
            self.intr["depth_scale"],
            0,
        )

    def write(self, depth: np.ndarray, t_capture: float) -> None:
        self._seq += 1  # odd: frame being written
        self._write_header(t_capture)
        np.copyto(self._frame, depth)
        self._seq += 1  # even: frame complete
        self._write_header(t_capture)

    def close(self) -> None:
        self._mm.close()


class DepthShmReader:
    """Reads frames published by :class:`DepthShmWriter`; safe to poll from any thread."""

    def __init__(self, path: str = DEFAULT_PATH, wait_s: float = 10.0):
        deadline = time.monotonic() + wait_s
        while not (os.path.exists(path) and os.path.getsize(path) >= HEADER_BYTES):
            if time.monotonic() > deadline:
                raise TimeoutError(f"no depth writer at {path} after {wait_s:.0f}s")
            time.sleep(0.05)
        fd = os.open(path, os.O_RDONLY)
        try:
            # Wait for the writer to publish its geometry (width/height non-zero).
            while True:
                head = os.pread(fd, _HEADER.size, 0)
                _, _, w, h, *_ = _HEADER.unpack(head)
                if w and h and os.path.getsize(path) >= _file_size(w, h):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"depth writer at {path} never published a frame size")
                time.sleep(0.05)
            self.width, self.height = w, h
            self._mm = mmap.mmap(fd, _file_size(w, h), prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        self._frame: np.ndarray = np.ndarray((h, w), dtype=np.uint16, buffer=self._mm, offset=HEADER_BYTES)
        self._last_seq = 0
        self.intrinsics = self._header()[1]

    def _header(self) -> tuple[int, dict]:
        seq, t, w, h, fx, fy, cx, cy, scale, _ = _HEADER.unpack(self._mm[: _HEADER.size])
        return seq, {"t_capture": t, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": scale}

    def try_read(self) -> tuple[np.ndarray, dict] | None:
        """Return (depth uint16 HxW copy, meta) if a new complete frame is available, else None."""
        seq0, meta = self._header()
        if seq0 & 1 or seq0 == self._last_seq or seq0 == 0:
            return None
        out = self._frame.copy()
        seq1, _ = self._header()
        if seq1 != seq0:  # torn: writer got in between, let the caller retry
            return None
        self._last_seq = seq0
        meta["seq"] = seq0
        return out, meta

    def read(self, timeout_s: float = 2.0) -> tuple[np.ndarray, dict]:
        deadline = time.monotonic() + timeout_s
        while True:
            got = self.try_read()
            if got is not None:
                return got
            if time.monotonic() > deadline:
                raise TimeoutError(f"no new depth frame within {timeout_s}s")
            time.sleep(0.0005)

    def close(self) -> None:
        self._mm.close()


class BallMasker:
    """Keep only the orange ball's depth; every other pixel becomes ``background_mm``.

    Colour is aligned onto the depth grid, the ball is found by HSV segmentation + a round,
    depth-backed, plausibly sized blob (same detector as ``perception/track.py`` in the cannon demo),
    and the depth inside its enclosing disc is passed through untouched. Mirrors the sim's
    ``--ball-only`` ablation: the policy sees the ball at its true range against a flat background.
    """

    def __init__(self, fx: float, args: argparse.Namespace):
        import cv2  # noqa: PLC0415

        self.cv2 = cv2
        self.fx = fx
        self.low = np.array(args.hsv_low, np.uint8)
        self.high = np.array(args.hsv_high, np.uint8)
        self.radius = args.radius
        self.size_tolerance = args.size_tolerance
        self.min_area = args.min_area
        self.min_circularity = args.min_circularity
        self.background_mm = 0 if args.background == "invalid" else int(round(args.far_m * 1000))
        # Sim throws teleport the ball in at 1.8-2.8 m from the root (parked 8 m up in between), so the
        # policy never saw a ball further than that. Detections beyond this range are hidden.
        self.max_range = args.ball_max_range
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.last: dict | None = None
        self.last_far: dict | None = None  # best detection that was hidden by the range gate
        self.last_mask: np.ndarray | None = None

    def mask(self, color_bgr: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        if self.low[0] <= self.high[0]:
            m = cv2.inRange(hsv, self.low, self.high)
        else:  # hue wraps through red
            m = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, self.low[1], self.low[2]], np.uint8), self.high),
                cv2.inRange(hsv, self.low, np.array([179, self.high[1], self.high[2]], np.uint8)),
            )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)

    def detect(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> dict | None:
        cv2 = self.cv2
        self.last_mask = self.mask(color_bgr)
        contours = cv2.findContours(self.last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        best = None
        self.rejects: list[str] = []  # why the largest orange blobs were not accepted (for --trace)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            (u, v), r_px = cv2.minEnclosingCircle(contour)
            circ = area / max(np.pi * r_px**2, 1.0)
            if circ < self.min_circularity:
                self.rejects.append(f"{int(area)}px circ={circ:.2f}")
                continue
            inner = np.zeros(depth_mm.shape, np.uint8)
            cv2.circle(inner, (int(u), int(v)), max(int(0.7 * r_px), 2), 255, -1)
            inside = depth_mm[(inner > 0) & (depth_mm > 0)]
            if inside.size < 10:
                self.rejects.append(f"{int(area)}px no-depth")
                continue
            centre_m = float(np.median(inside)) / 1000.0 + self.radius
            metric_r = r_px * centre_m / self.fx
            size_err = abs(metric_r - self.radius) / self.radius if self.radius > 0 else 0.0
            if self.radius > 0 and size_err > self.size_tolerance:
                self.rejects.append(f"{int(area)}px r={metric_r * 100:.0f}cm@{centre_m:.1f}m")
                continue
            score = size_err + 0.5 * (1.0 - circ)
            if best is None or score < best["score"]:
                best = {"u": u, "v": v, "r_px": r_px, "z": centre_m, "circ": circ, "score": score}
        return best

    def apply(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        det = self.detect(color_bgr, depth_mm)
        self.last_far = None
        if det is not None and self.max_range > 0 and det["z"] > self.max_range:
            self.last_far, det = det, None
        self.last = det
        out = np.full_like(depth_mm, self.background_mm)
        if det is not None:
            disc = np.zeros(depth_mm.shape, np.uint8)
            self.cv2.circle(disc, (int(det["u"]), int(det["v"])), max(int(det["r_px"]), 1), 255, -1)
            keep = disc > 0
            out[keep] = depth_mm[keep]
        return out


class MjpegView:
    """Tiny multipart/x-mixed-replace server: open http://<robot>:<port> in a browser on the laptop.

    Rendering + JPEG encoding run on their own thread so the capture/Theia loop only pays for a
    handful of array copies, and nothing at all while no browser is connected.
    """

    def __init__(self, port: int):
        import threading  # noqa: PLC0415
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._clients = 0
        self._pending: tuple | None = None
        self._pending_cond = threading.Condition()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    body = (
                        b"<html><body style='margin:0;background:#111'>"
                        b"<img src='/stream' style='width:100vw;image-rendering:pixelated'></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                with view._cond:
                    view._clients += 1
                try:
                    while True:
                        with view._cond:
                            view._cond.wait(timeout=1.0)
                            data = view._jpeg
                        if data is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(data)).encode()
                            + b"\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    with view._cond:
                        view._clients -= 1

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)  # nosec B104
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def wanted(self) -> bool:
        """True while at least one browser is attached; the caller skips even the copies otherwise."""
        return self._clients > 0

    def submit(self, render, *arrays) -> None:
        """Queue ``render(*arrays)`` for the worker; drops the frame if the previous one is still rendering."""
        with self._pending_cond:
            if self._pending is not None:
                return
            self._pending = (render, tuple(a.copy() if isinstance(a, np.ndarray) else a for a in arrays))
            self._pending_cond.notify()

    def _worker(self) -> None:
        import cv2  # noqa: PLC0415

        while True:
            with self._pending_cond:
                while self._pending is None:
                    self._pending_cond.wait()
                render, arrays = self._pending
                self._pending = None
            ok, buf = cv2.imencode(".jpg", render(*arrays), [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                with self._cond:
                    self._jpeg = buf.tobytes()
                    self._cond.notify_all()


def render_panel(
    color_bgr, depth_mm, masked_mm, det, far, mask, policy_img, crop, far_m, stats: str
) -> np.ndarray:
    """2x2: colour + detection | published depth ; HSV mask | what Theia gets (224x224 uint8).
    Takes plain snapshots (no live objects) so it can run on the view's worker thread."""
    import cv2  # noqa: PLC0415

    h, w = depth_mm.shape

    def turbo(mm):
        d = np.clip(mm.astype(np.float32) / (far_m * 1000.0), 0, 1)
        img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[mm == 0] = 0
        return img

    def label(img, text, y=None):
        cv2.putText(
            img,
            text,
            (4, y or img.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    color = color_bgr.copy()
    if crop is not None:
        y0, y1, x0, x1 = crop
        cv2.rectangle(color, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 1)
    if det is not None:
        c, r = (int(det["u"]), int(det["v"])), int(det["r_px"])
        cv2.circle(color, c, r, (0, 255, 0), 2)
        cv2.putText(
            color,
            f"{det['z']:.2f}m r={det['r_px']:.0f}px",
            (c[0] + r + 4, c[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if far is not None:
        c, r = (int(far["u"]), int(far["v"])), int(far["r_px"])
        cv2.circle(color, c, r, (0, 140, 255), 1)
        cv2.putText(
            color,
            f"hidden {far['z']:.2f}m",
            (c[0] + r + 4, c[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )
    label(color, stats, 14)
    label(color, "colour (aligned to depth) + detection")
    pub = turbo(masked_mm)
    label(pub, "depth published to the policy")
    mask = cv2.cvtColor(mask if mask is not None else np.zeros((h, w), np.uint8), cv2.COLOR_GRAY2BGR)
    label(mask, "HSV mask")
    if policy_img is not None:
        pol = cv2.resize(policy_img, (h, h), interpolation=cv2.INTER_NEAREST)
        pol = cv2.cvtColor(pol, cv2.COLOR_GRAY2BGR)
        pad = np.zeros((h, w - h, 3), np.uint8)
        pol = np.hstack([pol, pad])
    else:
        pol = np.zeros((h, w, 3), np.uint8)
    label(pol, "Theia input 224x224 (255 near, 1 far, 0 invalid)")
    return np.vstack([np.hstack([color, pub]), np.hstack([mask, pol])])


def main() -> None:
    import pyrealsense2 as rs  # only the writer needs it

    ap = argparse.ArgumentParser(description="Publish RealSense depth into shared memory")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--serial", default=None)
    g = ap.add_argument_group("ball-only masking (needs opencv; opens the colour stream too)")
    g.add_argument("--ball-only", action="store_true", help="publish only the ball's depth pixels")
    g.add_argument(
        "--background",
        choices=("far", "invalid"),
        default="far",
        help="non-ball pixels: far (= --far-m, policy uint8 1) or invalid (0 mm, uint8 0)",
    )
    g.add_argument("--far-m", type=float, default=6.0, help="policy's DEPTH_FAR_M")
    g.add_argument("--color-res", default="424x240", help="colour WxH; this D435i does 424x240@60")
    g.add_argument("--hsv-low", type=int, nargs=3, default=(3, 120, 70), metavar=("H", "S", "V"))
    g.add_argument("--hsv-high", type=int, nargs=3, default=(22, 255, 255), metavar=("H", "S", "V"))
    g.add_argument("--radius", type=float, default=0.12, help="ball radius m; 0 disables the size gate")
    g.add_argument("--size-tolerance", type=float, default=0.6)
    g.add_argument("--min-area", type=int, default=25)
    g.add_argument("--min-circularity", type=float, default=0.5)
    g.add_argument(
        "--ball-max-range",
        type=float,
        default=0.0,
        help="hide detections further than this (m); 0 = off. Sim throws start at 1.8-2.8 m",
    )
    g.add_argument(
        "--trace", action="store_true", help="print one line per frame while a ball/orange blob is around"
    )
    g.add_argument(
        "--view-port",
        type=int,
        default=0,
        help="serve a live MJPEG view (colour + detection, masked depth, mask, policy input) on this port",
    )
    t = ap.add_argument_group("onboard Theia (needs opencv + onnxruntime; publishes tokens, not just depth)")
    t.add_argument(
        "--theia", default=None, help="theia_image.onnx path; tokenize every frame here (GPU if available)"
    )
    t.add_argument("--tokens-path", default=DEFAULT_TOKENS_PATH)
    t.add_argument(
        "--theia-provider", default="auto", help="CUDAExecutionProvider | CPUExecutionProvider | auto"
    )
    args = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if args.ball_only:
        cw, ch = (int(x) for x in args.color_res.split("x"))
        cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, args.fps)
    profile = pipeline.start(cfg)
    dev = profile.get_device()
    # Shortest sensor->host queue: drop frames rather than buffer them.
    depth_sensor = dev.first_depth_sensor()
    for sensor in dev.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            sensor.set_option(rs.option.frames_queue_size, 1)
    vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = vsp.get_intrinsics()
    intrinsics = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "depth_scale": depth_sensor.get_depth_scale(),
    }
    writer = DepthShmWriter(intr.width, intr.height, intrinsics, args.path)
    print(
        f"depth {intr.width}x{intr.height}@{args.fps} -> {args.path}  fx={intr.fx:.1f} fy={intr.fy:.1f} "
        f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}",
        flush=True,
    )
    masker = align = None
    if args.ball_only:
        masker = BallMasker(intr.fx, args)
        align = rs.align(rs.stream.depth)  # colour -> depth grid
        print(
            f"BALL-ONLY: hsv {tuple(args.hsv_low)}..{tuple(args.hsv_high)} r={args.radius}m "
            f"background={args.background} ({masker.background_mm} mm)",
            flush=True,
        )
    theia = pwriter = None
    if args.theia:
        import cv2  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        available = ort.get_available_providers()
        if args.theia_provider == "auto":
            providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []
            providers.append("CPUExecutionProvider")
        else:
            providers = [args.theia_provider, "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 3
        theia = ort.InferenceSession(args.theia, so, providers=providers)
        y0, y1, x0, x1 = policy_crop_bounds(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.height, intr.width)
        pwriter = PerceptShmWriter(args.tokens_path)
        print(
            f"THEIA onboard: {args.theia} on {theia.get_providers()[0]}; crop rows {y0}:{y1} cols {x0}:{x1} "
            f"-> {POLICY_DEPTH_SHAPE} -> tokens {THEIA_TOKENS} -> {args.tokens_path}",
            flush=True,
        )
        black = depth_to_theia_image_np(np.zeros(POLICY_DEPTH_SHAPE, np.float32))[None]
        for _ in range(5):  # warm up (CUDA allocs, kernel selection)
            theia.run(None, {"image": black})

    # t_capture: the sensor's global-time stamp (host epoch, includes exposure->USB->host) when the
    # camera provides it, else host arrival. The reader compares against time.time().
    def capture_time(frame) -> float:
        if frame.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
            return frame.get_timestamp() / 1000.0
        return time.time()

    view = None
    if args.view_port:
        if not args.ball_only:
            raise SystemExit("--view-port needs --ball-only (it shows the colour stream and the detection)")
        view = MjpegView(args.view_port)
        print(f"live view: http://<robot-ip>:{args.view_port}", flush=True)
    crop = (y0, y1, x0, x1) if theia is not None else None

    n, hits, t_report = 0, 0, time.monotonic()
    theia_ms = cam_lat_ms = 0.0
    ball_seen, n_seen, n_missing = False, 0, 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:  # "Frame didn't arrive within 2000": USB hiccup, restart the stream
                print(f"!!! RealSense stalled ({e}); restarting the pipeline", flush=True)
                pipeline.stop()
                time.sleep(0.5)
                pipeline.start(cfg)
                continue
            if align is not None:
                frames = align.process(frames)
            depth = frames.get_depth_frame()
            if not depth:
                continue
            t_cap = capture_time(depth)
            cam_lat_ms = 0.9 * cam_lat_ms + 0.1 * (time.time() - t_cap) * 1e3
            raw_mm = np.asanyarray(depth.get_data())
            depth_mm, color_bgr = raw_mm, None
            if masker is not None:
                color = frames.get_color_frame()
                if not color:
                    continue
                color_bgr = np.asanyarray(color.get_data())
                depth_mm = masker.apply(color_bgr, raw_mm)
                hits += masker.last is not None
                # Detection trace: transitions always, every frame with --trace. This is what tells us
                # whether the policy saw a continuous ball or a flickering one during a throw.
                det, far = masker.last, masker.last_far
                seen = det is not None
                if seen != ball_seen:
                    if det is not None:
                        print(
                            f"[{t_cap % 1000:7.3f}] ball APPEARED z={det['z']:.2f}m r={det['r_px']:.0f}px "
                            f"circ={det['circ']:.2f} after {n_missing} frames without",
                            flush=True,
                        )
                    else:
                        why = (
                            f"range {far['z']:.2f}m"
                            if far
                            else (", ".join(masker.rejects[:2]) or "no orange blob")
                        )
                        print(f"[{t_cap % 1000:7.3f}] ball LOST after {n_seen} frames ({why})", flush=True)
                    ball_seen, n_seen, n_missing = seen, 0, 0
                n_seen += seen
                n_missing += not seen
                if args.trace and (seen or far or masker.rejects):
                    if det is not None:
                        print(
                            f"[{t_cap % 1000:7.3f}] z={det['z']:.2f} px=({det['u']:.0f},{det['v']:.0f}) "
                            f"r={det['r_px']:.0f} circ={det['circ']:.2f}",
                            flush=True,
                        )
                    else:
                        why = f"range {far['z']:.2f}m" if far else ", ".join(masker.rejects[:3])
                        print(f"[{t_cap % 1000:7.3f}] miss: {why}", flush=True)
            writer.write(depth_mm, t_cap)
            policy_img = None
            if theia is not None and pwriter is not None:
                t0 = time.perf_counter()
                roi = depth_mm[y0:y1, x0:x1]
                small = cv2.resize(
                    roi, (POLICY_DEPTH_SHAPE[1], POLICY_DEPTH_SHAPE[0]), interpolation=cv2.INTER_NEAREST
                )
                depth_m = small.astype(np.float32) * 1e-3
                policy_img = depth_to_theia_image_np(depth_m)
                tokens = theia.run(None, {"image": policy_img[None]})[0][0]
                tokens = tokens.astype(np.float16).astype(np.float32)  # training cached fp16 features
                dt_ms = (time.perf_counter() - t0) * 1e3
                theia_ms = 0.9 * theia_ms + 0.1 * dt_ms
                pwriter.write(tokens, depth_m, t_cap, dt_ms)
            n += 1
            # ~15 fps, rendered on the view's thread
            if view is not None and masker is not None and view.wanted and n % 4 == 0:
                status = f"cam->host {cam_lat_ms:.0f}ms  theia {theia_ms:.1f}ms"
                view.submit(
                    render_panel,
                    color_bgr,
                    raw_mm,
                    depth_mm,
                    masker.last,
                    masker.last_far,
                    masker.last_mask,
                    policy_img,
                    crop,
                    args.far_m,
                    status,
                )
            if time.monotonic() - t_report > 5.0:
                dt = time.monotonic() - t_report
                msg = f"{n / dt:.1f} fps  cam->host {cam_lat_ms:.0f}ms"
                if theia is not None:
                    msg += f"  theia {theia_ms:.1f}ms"
                if masker is not None:
                    msg += f"  ball in {100 * hits / max(n, 1):.0f}% of frames"
                    if masker.last:
                        msg += f"  last px=({masker.last['u']:.0f},{masker.last['v']:.0f}) z={masker.last['z']:.2f}m"
                print(msg, flush=True)
                n, hits, t_report = 0, 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        writer.close()
        if pwriter is not None:
            pwriter.close()


if __name__ == "__main__":
    main()
