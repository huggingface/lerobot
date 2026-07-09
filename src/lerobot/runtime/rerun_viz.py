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

"""Rerun live visualisation for the interactive runtime (real-robot camera view).

Starts a headless rerun gRPC server + web viewer so a remote operator can watch
the robot's cameras (and state / subtask) over SSH by forwarding two ports and
opening the web viewer in a browser. Logging is best-effort — a rerun failure
never interrupts robot control.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED = False


def start_rerun(app_name: str = "lerobot_runtime", grpc_port: int = 9876, web_port: int = 9090) -> bool:
    """Init rerun and serve a headless gRPC + web viewer. Returns True on success."""
    global _ENABLED
    try:
        import rerun as rr  # noqa: PLC0415

        rr.init(app_name)
        url = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=url)
        _ENABLED = True
        # Open the viewer with the data URL as a query param so it auto-connects
        # to the gRPC stream (plain http://host:web_port shows only the welcome
        # screen — the web app needs the ?url= to know where the data is).
        view_url = f"http://localhost:{web_port}/?url={url}"
        print(
            f"[runtime] rerun live view: {view_url}\n"
            f"          (over SSH forward both ports: "
            f"ssh -L {web_port}:localhost:{web_port} -L {grpc_port}:localhost:{grpc_port} <host>)",
            flush=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[runtime] could not start rerun: %s", exc)
        print(f"[runtime] WARNING: rerun unavailable ({exc})", flush=True)
        return False


def log_cameras(robot: Any) -> None:
    """Log every robot camera's latest buffered frame (cheap async_read).

    Called every control tick for a smooth live view (best-effort)."""
    if not _ENABLED:
        return
    try:
        import rerun as rr  # noqa: PLC0415

        cams = getattr(robot, "cameras", None) or {}
        for name, cam in cams.items():
            try:
                frame = cam.async_read(timeout_ms=1)
                rr.log(f"cameras/{name}", rr.Image(frame))
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("[runtime] rerun camera log failed: %s", exc)


def log_robot_frame(
    raw_obs: dict[str, Any],
    camera_keys: list[str],
    state: dict[str, float] | None = None,
    task: str | None = None,
    subtask: str | None = None,
) -> None:
    """Log camera images + optional state/task/subtask for one step (best-effort)."""
    if not _ENABLED:
        return
    try:
        import numpy as np  # noqa: PLC0415
        import rerun as rr  # noqa: PLC0415

        for cam in camera_keys:
            img = raw_obs.get(cam)
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                rr.log(f"cameras/{cam}", rr.Image(img))
        if state:
            for name, val in state.items():
                try:
                    rr.log(f"state/{name}", rr.Scalars(float(val)))
                except Exception:  # noqa: BLE001
                    pass
        if task:
            rr.log("prompt/task", rr.TextLog(task))
        if subtask:
            rr.log("prompt/subtask", rr.TextLog(subtask))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[runtime] rerun log failed: %s", exc)
