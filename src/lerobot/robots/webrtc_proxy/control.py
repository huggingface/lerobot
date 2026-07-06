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

"""Control plane: cloud-driven device onboarding over a reliable RPC channel.

Port + camera IDs are *robot-local* physical identifiers that are meaningless in the
cloud, so the cloud never stores them. Instead the cloud drives discovery over the
``control`` DataChannel and the robot answers from its own OS:

    cloud  ──RpcRequest{list_ports}──▶  robot (ControlServer -> DeviceInventory)
    cloud  ◀──RpcResponse{result}─────  robot

The crux vs. the local ``lerobot-find-port`` CLI: that tool blocks on ``input()``
while the human unplugs the bus, but here the human is at the robot and the
orchestrator is in the cloud. So ``find_port`` becomes an *event-driven* two-step —
``find_port_begin`` snapshots the ports, the human unplugs (prompted by the cloud
UI), then ``find_port_result`` diffs to the port that disappeared. No shared stdin.

``DeviceInventory`` is the seam between this transport and the OS: M3 ships a
``SyntheticInventory`` (loopback-testable); a real ``LocalDeviceInventory`` wrapping
``lerobot.scripts.lerobot_find_port`` / ``lerobot_find_cameras`` lands with M2/M4
hardware bring-up.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any, Protocol

import numpy as np

from .protocol import RpcRequest, RpcResponse

logger = logging.getLogger(__name__)


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 75) -> str:
    """RGB uint8 frame -> base64 JPEG string (small enough for the control DataChannel)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_jpeg_b64(b64: str) -> np.ndarray:
    """Inverse of :func:`_encode_jpeg_b64`: base64 JPEG -> RGB uint8 ndarray."""
    import base64
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


@contextlib.contextmanager
def _silence_native_stderr():
    """Redirect OS-level fd 2 to /dev/null for the duration of the block.

    lerobot's ``OpenCVCamera.find_cameras`` brute-force-opens every camera index, and
    OpenCV/AVFoundation writes "out device of bound / camera failed to initialize" for
    each nonexistent one straight to the C stderr — Python logging can't intercept it.
    RealSense probing on a machine without a device also logs noisily. This swallows
    both around the (one-shot, onboarding) enumeration call. Process-wide for the brief
    call window, so it is intentionally narrow.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


class FindPortError(RuntimeError):
    """Raised when a find_port diff is ambiguous (0 or >1 ports changed)."""


class DeviceInventory(Protocol):
    """OS-facing device enumeration. Implementations run on the robot."""

    def list_ports(self) -> list[str]: ...

    def list_cameras(self) -> list[dict[str, Any]]: ...

    def grab_frame(self, cam_id: Any, width: int, height: int) -> np.ndarray:
        """Open camera ``cam_id``, return one RGB uint8 frame (for an onboarding preview)."""
        ...


class SyntheticInventory:
    """In-memory inventory for loopback tests / the synthetic capture agent.

    Mirrors the real shapes: ``list_ports`` returns serial-port strings; cameras
    carry a *stable* identifier (``index_or_path`` for opencv, ``serial`` for
    realsense) plus a human-readable name so a person can map them to logical roles.
    ``simulate_unplug`` lets a test reproduce the find_port unplug step.
    """

    def __init__(
        self,
        ports: list[str] | None = None,
        cameras: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ports: list[str] = list(
            ports if ports is not None else ["/dev/tty.usbmodem-FAKE-A", "/dev/tty.usbmodem-FAKE-B"]
        )
        self._cameras: list[dict[str, Any]] = list(
            cameras
            if cameras is not None
            else [
                {"type": "opencv", "index_or_path": 0, "name": "FaceTime HD Camera"},
                {"type": "opencv", "index_or_path": 1, "name": "USB Camera"},
            ]
        )

    def list_ports(self) -> list[str]:
        return list(self._ports)

    def list_cameras(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._cameras]

    def grab_frame(self, cam_id: Any, width: int = 320, height: int = 240) -> np.ndarray:
        # Distinct flat colour per id so previews are visually distinguishable in tests.
        seed = int(cam_id) if str(cam_id).lstrip("-").isdigit() else abs(hash(str(cam_id)))
        img = np.empty((height, width, 3), dtype=np.uint8)
        img[:] = ((seed * 70) % 256, (seed * 40 + 80) % 256, 120)
        return img

    def simulate_unplug(self, port: str) -> None:
        if port in self._ports:
            self._ports.remove(port)


class LocalDeviceInventory:
    """Real robot-side inventory: enumerates the actual serial ports + cameras.

    Wraps lerobot's own discovery so it sees exactly what the stock
    ``lerobot-find-port`` / ``lerobot-find-cameras`` CLIs see. Imports are lazy so
    ``control.py`` still loads without the hardware/camera extras; a missing extra
    surfaces as a control-RPC error to the cloud rather than an import failure.

    Note: ``list_cameras`` probes camera indices (opens each briefly), so it is a
    one-shot onboarding call, not something to poll.
    """

    def list_ports(self) -> list[str]:
        from lerobot.scripts.lerobot_find_port import find_available_ports

        return find_available_ports()

    def list_cameras(self) -> list[dict[str, Any]]:
        # Keep the import *inside* the hush: importing it loads cv2 (libavdevice), whose
        # objc duplicate-class warning vs. av also goes to C-level stderr — along with
        # OpenCV's index-scan noise. One-shot onboarding call, so the brief redirect is fine.
        with _silence_native_stderr():
            from lerobot.scripts.lerobot_find_cameras import (
                find_all_opencv_cameras,
                find_all_realsense_cameras,
            )

            # Each dict carries a stable id ('id' = opencv index_or_path / realsense serial).
            return [*find_all_opencv_cameras(), *find_all_realsense_cameras()]

    def grab_frame(self, cam_id: Any, width: int = 320, height: int = 240) -> np.ndarray:
        """Open this opencv camera at its *native* resolution, read one frame, close it,
        then downscale to a small preview. Onboarding preview only — fails if the camera
        is already in use (e.g. currently being streamed).

        We deliberately don't request a capture size: OpenCVCamera errors if the camera
        can't apply it. Resize happens in software afterwards.
        """
        import cv2

        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        index_or_path = int(cam_id) if str(cam_id).isdigit() else cam_id
        with _silence_native_stderr():
            cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index_or_path))
            cam.connect(warmup=True)
            try:
                frame = cam.read()  # RGB uint8, native resolution
            finally:
                cam.disconnect()
            frame = cv2.resize(frame, (width, height))
        return np.ascontiguousarray(frame)


class ControlServer:
    """robot side: receives RpcRequests on the control channel and answers them.

    Stateful only for the find_port begin/result handshake (it stashes the
    pre-unplug port snapshot between the two calls).
    """

    def __init__(self, inventory: DeviceInventory, on_camera_plan=None) -> None:
        self.inventory = inventory
        # Called with the cloud's desired camera spec {width,height,fps} at session start.
        self._on_camera_plan = on_camera_plan
        self._channel = None
        self._ports_before: list[str] | None = None

    def attach(self, channel) -> None:  # noqa: ANN001 (transport Channel)
        self._channel = channel
        channel.on_message(self._on_message)

    def _on_message(self, raw: str) -> None:
        try:
            req = RpcRequest.from_json(raw)
        except Exception:
            logger.exception("ControlServer: malformed RpcRequest")
            return
        try:
            result = self._dispatch(req)
            resp = RpcResponse(id=req.id, ok=True, result=result)
        except Exception as e:  # report failures back to the cloud, don't crash the loop
            resp = RpcResponse(id=req.id, ok=False, error=f"{type(e).__name__}: {e}")
        if self._channel is not None and self._channel.is_open:
            self._channel.send(resp.to_json())

    def _dispatch(self, req: RpcRequest) -> Any:
        if req.method == "list_ports":
            return {"ports": self.inventory.list_ports()}
        if req.method == "list_cameras":
            return {"cameras": self.inventory.list_cameras()}
        if req.method == "grab_camera":
            width = int(req.params.get("width", 320))
            height = int(req.params.get("height", 240))
            frame = self.inventory.grab_frame(req.params["id"], width, height)
            return {"jpeg_b64": _encode_jpeg_b64(frame), "width": frame.shape[1], "height": frame.shape[0]}
        if req.method == "set_camera_plan":
            # Cloud declares the obs size it wants; the robot resizes/encodes to it.
            if self._on_camera_plan is not None:
                self._on_camera_plan(req.params)
            return {"applied": req.params}
        if req.method == "find_port_begin":
            self._ports_before = self.inventory.list_ports()
            return {"ports": list(self._ports_before)}
        if req.method == "find_port_result":
            if self._ports_before is None:
                raise FindPortError("find_port_result called before find_port_begin")
            after = self.inventory.list_ports()
            diff = sorted(set(self._ports_before) - set(after))
            self._ports_before = None
            if len(diff) == 1:
                return {"port": diff[0]}
            raise FindPortError(f"expected exactly one disconnected port, found {diff}")
        raise ValueError(f"unknown control method {req.method!r}")


class ControlClient:
    """Cloud side: issues RpcRequests and awaits the matching RpcResponse by id."""

    def __init__(self) -> None:
        self._channel = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}

    def attach(self, channel) -> None:  # noqa: ANN001 (transport Channel)
        self._channel = channel
        channel.on_message(self._on_message)

    def _on_message(self, raw: str) -> None:
        try:
            resp = RpcResponse.from_json(raw)
        except Exception:
            logger.exception("ControlClient: malformed RpcResponse")
            return
        fut = self._pending.pop(resp.id, None)
        if fut is None or fut.done():
            return
        if resp.ok:
            fut.set_result(resp.result)
        else:
            fut.set_exception(RuntimeError(resp.error or "control RPC failed"))

    async def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        if self._channel is None or not self._channel.is_open:
            raise RuntimeError("control channel not open")
        self._next_id += 1
        req = RpcRequest(id=self._next_id, method=method, params=params or {})
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req.id] = fut
        self._channel.send(req.to_json())
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(req.id, None)
