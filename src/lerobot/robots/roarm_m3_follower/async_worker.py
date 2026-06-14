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

"""Background serial-I/O thread for the RoArm-M3 follower.

The ``roarm_sdk`` serial calls block ~35-50 ms each. The synchronous record/eval loop
would be capped well below 20 fps by a blocking get/set inside the loop, so this worker
moves ALL serial I/O for the follower onto one background thread: it owns reads AND writes
in a single thread, so a read and a write never race on the same serial port (no locking
needed between the main thread and the worker). No reference arm needs this because their
CAN/gRPC transports are fast; the JSON-over-serial transport here makes it worth it.

Gripper-B force (opt-in via ``force_control_gripper``): the bundled "move all joints"
command (``joints_angle_ctrl``, firmware T:122 / SyncWritePosEx) zeroes the
GOAL_TIME/GOAL_TORQUE registers for every servo. On a Waveshare Gripper B (CF-3512, CFSCL)
that zeroes the constant-current force, so the jaw won't hold. When force control is
enabled, the worker issues a second, gripper-only write (``joint_angle_ctrl(joint=6)``,
firmware T:121) right after the bundle, which re-applies the gripper torque. With force
control off, only the single bundled write is sent (the common case).
"""

from __future__ import annotations

import contextlib
import logging
import queue as _queue
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class AsyncArmWorker:
    """Single background thread that owns ALL serial I/O for one follower arm.

    Main-thread API (non-blocking, ~0 ms):
        worker.write(goal_pos, speed, acc)  - queue a command, returns instantly
        worker.get_pos()                    - return the latest cached joint state
    """

    def __init__(
        self,
        arm,
        name: str,
        read_interval_s: float = 0.15,
        force_control_gripper: bool = False,
    ):
        self._arm = arm
        self._name = name
        self._read_interval = read_interval_s
        self._force_control_gripper = force_control_gripper
        self._write_q: _queue.Queue = _queue.Queue(maxsize=1)
        self._pos: np.ndarray | None = None
        self._pos_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"AsyncArmWorker-{name}")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def write(self, goal_pos: list, speed: int, acc: int) -> None:
        """Non-blocking write. Drops the previous command if the queue is full."""
        cmd = (goal_pos, speed, acc)
        if self._write_q.full():
            with contextlib.suppress(_queue.Empty):
                self._write_q.get_nowait()
        with contextlib.suppress(_queue.Full):
            self._write_q.put_nowait(cmd)

    def get_pos(self) -> np.ndarray | None:
        with self._pos_lock:
            return None if self._pos is None else self._pos.copy()

    def _execute_write(self, goal_pos: list, speed: int, acc: int) -> None:
        """Perform one write: the bundled joint command, plus -- when
        force_control_gripper is set -- a gripper-only command (firmware T:121) that
        restores the CF-3512 clamping force the bundled write zeroes. Pulled out of the
        run loop so it can be unit-tested without the background thread."""
        self._arm.joints_angle_ctrl(angles=goal_pos, speed=speed, acc=acc)
        if self._force_control_gripper and len(goal_pos) >= 6:
            self._arm.joint_angle_ctrl(joint=6, angle=float(goal_pos[5]), speed=speed, acc=acc)

    def _run(self):
        last_read = 0.0
        while not self._stop_event.is_set():
            # Priority 1: execute a pending write.
            try:
                goal_pos, speed, acc = self._write_q.get(timeout=0.005)
                self._execute_write(goal_pos, speed, acc)
            except _queue.Empty:
                pass

            # Priority 2: periodic joint read.
            now = time.perf_counter()
            if now - last_read >= self._read_interval:
                try:
                    pos = self._arm.joints_angle_get()
                    if pos is not None:
                        with self._pos_lock:
                            self._pos = np.array(pos, dtype=np.float32)
                    last_read = now
                except Exception as exc:  # pragma: no cover - serial flakiness
                    logger.debug(f"AsyncArmWorker [{self._name}] read error: {exc}")
