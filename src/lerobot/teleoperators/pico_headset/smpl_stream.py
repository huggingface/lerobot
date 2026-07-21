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

"""Live SMPL stream as a SONIC reference motion (drop-in for ``SmplMotion``).

Instead of reading an ``.npz`` clip, this pulls per-frame SMPL joints live off the
``rt/smpl`` ZMQ channel published by the GEAR PICO teleop
(``gear_sonic/scripts/pico_manager_thread_server.py``). Each message carries one
frame of **canonical** (root-orientation-removed) SMPL local joints ``(24, 3)`` --
the exact per-frame format ``SmplMotion`` emits -- so this class exposes the same
``step() -> (720,)`` window interface and can be handed to ``sonic.py`` (or the
``pico_headset`` teleoperator) wherever a reference motion is expected
(``encode_mode == 2``).

Transport mirrors the Unitree SDK socket bridge (``unitree_sdk2_socket.py``): a ZMQ
``SUB`` socket with ``CONFLATE`` (keep only the latest frame) subscribed to the
``rt/smpl`` topic, JSON payloads.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import deque

import numpy as np
import zmq

from .smpl_constants import (
    JOINT_DIM,
    LOCO_N_AXES,
    LOCO_N_BTN,
    N_JOINTS,
    SMPL_OBS_DIM,
    VR3_ORN_DIM,
    VR3_POS_DIM,
    WINDOW,
)

logger = logging.getLogger(__name__)

SMPL_TOPIC = "rt/smpl"
DEFAULT_SMPL_HOST = "127.0.0.1"
DEFAULT_SMPL_PORT = 5560


class SmplStream:
    """Live ``rt/smpl`` consumer with the ``SmplMotion`` interface.

    Args:
        host: publisher host (the laptop running pico_manager_thread_server.py).
        port: publisher port for the ``rt/smpl`` channel.
        fps: nominal source rate, only used for status/reporting.
        stale_after_s: log a warning if no fresh frame arrives within this window.
        loop: accepted for API parity with ``SmplMotion`` (ignored; a stream never ends).
    """

    def __init__(
        self,
        host: str = DEFAULT_SMPL_HOST,
        port: int = DEFAULT_SMPL_PORT,
        fps: float = 50.0,
        stale_after_s: float = 0.5,
        loop: bool = True,
    ):
        self.host = host
        self.port = port
        self.fps = float(fps)
        self.loop = loop
        self.stale_after_s = stale_after_s

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)  # keep only the most recent frame
        self._sock.connect(f"tcp://{host}:{port}")
        # Single-frame JSON messages (topic embedded in payload); CONFLATE does not
        # support multipart, so subscribe to everything on this dedicated port.
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

        # Rolling window, oldest -> newest (matches SmplMotion.window layout).
        self._buf: deque[np.ndarray] = deque(maxlen=WINDOW)
        self._last_frame = np.zeros((N_JOINTS, JOINT_DIM), np.float32)
        # Latest root/torso pose (updated every received frame).
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0], np.float32)  # (w, x, y, z)
        self.root_transl = np.zeros(3, np.float32)
        # Latest sparse 3-point VR targets (encode_mode 1), if the producer sends them.
        self.vr3_pos = np.zeros(VR3_POS_DIM, np.float32)
        self.vr3_orn = np.tile([1.0, 0.0, 0.0, 0.0], VR3_ORN_DIM // 4).astype(np.float32)
        self._got_vr3 = False
        self._last_vr3_t = 0.0
        # Latest controller-stick locomotion (encode_mode 1): [lx, ly, rx, ry] + [A,B,X,Y].
        self.loco_axes = np.zeros(LOCO_N_AXES, np.float32)
        self.loco_buttons = np.zeros(LOCO_N_BTN, np.float32)
        self._got_loco = False
        self._last_loco_t = 0.0
        self._last_index = -1
        self._last_recv_t = 0.0
        self._warned_stale = False
        self._got_first = False

    # -- SmplMotion-compatible attributes ------------------------------------
    @property
    def num_frames(self) -> int:
        """Streams are unbounded; report 0 (kept for API parity)."""
        return 0

    @property
    def done(self) -> bool:
        """A live stream never finishes."""
        return False

    @property
    def has_data(self) -> bool:
        """True once at least one real frame has been received."""
        return self._got_first

    @property
    def has_vr3(self) -> bool:
        """True once the producer has sent at least one 3-point VR frame."""
        return self._got_vr3

    @property
    def has_fresh_vr3(self) -> bool:
        """True when a 3-point VR frame arrived within ``stale_after_s``.

        Unlike :attr:`has_data`, this is independent of the SMPL window, so the
        controller-state source (head + controllers only, empty SMPL) still drives
        ``encode_mode 1`` without a whole-body reference.
        """
        if not self._got_vr3:
            return False
        if not self.stale_after_s:
            return True
        return (time.time() - self._last_vr3_t) <= self.stale_after_s

    @property
    def has_fresh_loco(self) -> bool:
        """True when controller-stick locomotion arrived within ``stale_after_s``."""
        if not self._got_loco:
            return False
        if not self.stale_after_s:
            return True
        return (time.time() - self._last_loco_t) <= self.stale_after_s

    @property
    def seconds_since_last(self) -> float:
        """Wall-clock seconds since the last real frame (inf before the first)."""
        if not self._got_first:
            return float("inf")
        return time.time() - self._last_recv_t

    @property
    def is_stale(self) -> bool:
        """True when the stream has gone silent past ``stale_after_s``.

        Consumers use this to stop feeding a frozen pose and let the controller
        fall back to a safe standing/locomotion mode.
        """
        if not self._got_first or not self.stale_after_s:
            return False
        return self.seconds_since_last > self.stale_after_s

    def reset(self):
        self._buf.clear()
        self._got_first = False
        self._got_vr3 = False
        self._last_vr3_t = 0.0
        self._got_loco = False
        self._last_loco_t = 0.0

    # -- core ----------------------------------------------------------------
    def _drain_latest(self) -> np.ndarray | None:
        """Return the newest available (24, 3) frame, or None if nothing new.

        CONFLATE already keeps only the last message, but we poll non-blocking so
        the 50 Hz control loop never stalls waiting on the headset.
        """
        frame = None
        while dict(self._poller.poll(0)).get(self._sock) == zmq.POLLIN:
            payload = self._sock.recv()
            data = json.loads(payload.decode("utf-8")).get("data", {})

            # Sparse 3-point VR targets (encode_mode 1). Parsed independently of the
            # SMPL window so the controller-state source (head + controllers, empty
            # SMPL) is still handled.
            vp = data.get("vr3_pos")
            vo = data.get("vr3_orn")
            if vp is not None and vo is not None and len(vp) == VR3_POS_DIM and len(vo) == VR3_ORN_DIM:
                self.vr3_pos = np.asarray(vp, np.float32)
                self.vr3_orn = np.asarray(vo, np.float32)
                self._got_vr3 = True
                self._last_vr3_t = time.time()

            # Controller-stick locomotion (encode_mode 1), also independent of SMPL.
            la = data.get("loco_axes")
            lb = data.get("loco_buttons")
            if la is not None and lb is not None and len(la) == LOCO_N_AXES and len(lb) == LOCO_N_BTN:
                self.loco_axes = np.asarray(la, np.float32)
                self.loco_buttons = np.asarray(lb, np.float32)
                self._got_loco = True
                self._last_loco_t = time.time()

            # SMPL whole-body window (encode_mode 2), optional on this stream.
            joints = np.asarray(data.get("smpl_joints_local", []), np.float32)
            if joints.size != N_JOINTS * JOINT_DIM:
                continue
            frame = joints.reshape(N_JOINTS, JOINT_DIM)
            self._last_index = int(data.get("frame_index", self._last_index + 1))
            rq = data.get("root_quat")
            if rq is not None and len(rq) == 4:
                self.root_quat = np.asarray(rq, np.float32)
            rt = data.get("root_transl")
            if rt is not None and len(rt) == 3:
                self.root_transl = np.asarray(rt, np.float32)
        return frame

    def step(self) -> np.ndarray:
        """Advance one control tick, returning the current 720-vec window.

        If no new headset frame arrived this tick we hold the last one, so the
        policy keeps tracking the latest pose rather than snapping to zero.
        """
        frame = self._drain_latest()
        now = time.time()

        if frame is not None:
            self._last_frame = frame
            self._last_recv_t = now
            self._warned_stale = False
            if not self._got_first:
                # Pre-fill the window so the first send is a full, coherent clip.
                self._buf.extend([frame.copy() for _ in range(WINDOW)])
                self._got_first = True
            else:
                self._buf.append(frame)
        elif self._got_first:
            # No fresh frame: repeat the most recent to keep the window moving.
            self._buf.append(self._last_frame.copy())
            if (
                self.stale_after_s
                and not self._warned_stale
                and (now - self._last_recv_t) > self.stale_after_s
            ):
                logger.warning(
                    "[SmplStream] no %s frame for %.2fs (holding last pose)",
                    SMPL_TOPIC,
                    now - self._last_recv_t,
                )
                self._warned_stale = True

        if not self._got_first:
            return np.zeros(SMPL_OBS_DIM, np.float32)
        # Flatten to (720,): frames oldest->newest, joint-major within a frame
        # [f0_j0_xyz, f0_j1_xyz, ..., f9_j23_xyz] — matches SmplMotion.window.
        return np.concatenate(list(self._buf), dtype=np.float32).reshape(-1)

    def close(self):
        with contextlib.suppress(Exception):
            self._poller.unregister(self._sock)
        self._sock.close(0)
