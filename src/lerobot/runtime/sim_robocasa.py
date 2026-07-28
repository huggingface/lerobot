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

"""RoboCasa backend for interactive language-conditioned rollouts.

It reuses the eval observation/action pipeline while prompts control a persistent selected scene.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.io_utils import StreamingVideoWriter
from lerobot.utils.video_annotation import annotate_frame

logger = logging.getLogger(__name__)


def _short_cam_name(cam: str) -> str:
    """Human-friendly view label for a RoboCasa camera name."""
    c = cam.replace("robot0_", "")
    return {
        "agentview_left": "left",
        "agentview_right": "right",
        "eye_in_hand": "wrist",
    }.get(c, c)


def _label_panel(img: np.ndarray, label: str) -> np.ndarray:
    """Draw a small camera-view label in the bottom-left corner of a panel."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return img
    y = img.shape[0] - 6
    cv2.putText(img, label, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return img


# Two workers avoid broken single-worker EGL rendering; only env 0 is displayed.
_SIM_N_ENVS = 2


def create_sim_env(
    *,
    task: str,
    split: str | None,
    obj_registries: list[str],
    seed: int | None,
    render_size: int = 384,
) -> tuple[Any, dict]:
    """Create and reset the vectorized RoboCasa environment before CUDA initializes.

    Two workers keep EGL stable, while only env 0 is driven and displayed.
    """
    from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig  # noqa: PLC0415

    # The policy resizes inputs, so render_size only affects display quality and cost.
    env_cfg = RoboCasaEnvConfig(
        task=task,
        split=split,
        obj_registries=list(obj_registries),
        observation_height=render_size,
        observation_width=render_size,
    )
    # Keep one kitchen alive across sequential prompts.
    envs = env_cfg.create_envs(
        n_envs=_SIM_N_ENVS,
        use_async_envs=True,
        terminate_on_success=False,
        horizon=100_000,
    )
    env = envs[next(iter(envs))][0]
    logger.info("[sim] resetting RoboCasa scene task=%r split=%r (n_envs=%d)", task, split, _SIM_N_ENVS)
    seeds = None if seed is None else [seed + i for i in range(_SIM_N_ENVS)]
    obs, _ = env.reset(seed=seeds)
    return env, obs


def start_mjpeg_server(port: int, get_frame: Callable[[], np.ndarray | None]) -> Any:
    """Start an MJPEG server that shows a placeholder until ``get_frame`` returns frames."""
    import io  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    _placeholder = Image.new("RGB", (256, 256), (17, 17, 17))

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='margin:0;background:#111;text-align:center'>"
                    b"<img src='/stream' style='max-width:100vw;max-height:100vh;"
                    b"image-rendering:pixelated'></body></html>"
                )
                return
            if self.path != "/stream":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = get_frame()
                    buf = io.BytesIO()
                    img = Image.fromarray(frame) if frame is not None else _placeholder
                    img.save(buf, format="JPEG", quality=80)
                    data = buf.getvalue()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

    try:
        # Bind all interfaces intentionally so the viewer remains reachable
        # through the documented SSH port-forwarding workflow.
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)  # nosec B104
    except OSError as exc:
        logger.warning("[sim] could not start live stream on port %d: %s", port, exc)
        print(f"[runtime] WARNING: live stream port {port} unavailable ({exc})", flush=True)
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="sim-mjpeg").start()
    print(
        f"[runtime] live view: http://localhost:{port}  "
        f"(over SSH: ssh -L {port}:localhost:{port} <host>) — loading until scene is ready",
        flush=True,
    )
    return server


class RoboCasaSimBackend:
    """Expose a RoboCasa environment through the runtime observation/action contract.

    The environment must be created before the policy initializes CUDA.
    """

    def __init__(
        self,
        *,
        env: Any,
        last_obs: dict,
        task: str,
        seed: int | None,
        device: str,
        preprocessor: Any,
        postprocessor: Any,
        record: bool = True,
        output_dir: str | None = None,
        view_cams: list[str] | None = None,
    ) -> None:
        self.env = env
        self._last_obs = last_obs
        self._scene_task = task
        self._view_cams = view_cams or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
            "robot0_agentview_right",
        ]
        self.device = torch.device(device) if isinstance(device, str) else device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.seed = seed
        self.record = record
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/runtime_sim")

        self._video_writer: StreamingVideoWriter | None = None
        self._video_path: Path | None = None
        self._live_counter = 0
        self._latest_frame: np.ndarray | None = None
        self._stream_server: Any = None
        self._reset_count = 0
        # Bind these after runtime construction for live annotations.
        self._task_getter: Callable[[], str | None] | None = None
        self._subtask_getter: Callable[[], str | None] | None = None
        self._memory_getter: Callable[[], str | None] | None = None
        logger.info("[sim] scene ready — task_description=%r", self._scene_description())

    def bind_runtime(self, runtime: Any) -> None:
        """Wire live task/subtask/memory getters from the runtime state."""
        self._task_getter = lambda: runtime.state.get("task")
        self._subtask_getter = lambda: runtime.state.language_context.get("subtask")
        self._memory_getter = lambda: (runtime.state.get("language_context") or {}).get("memory")

    def _scene_description(self) -> str:
        try:
            return str(self.env.get_attr("task_description")[0]) or self._scene_task
        except Exception:  # noqa: BLE001
            return self._scene_task

    def _current_task(self) -> str:
        task = self._task_getter() if self._task_getter else None
        return task or self._scene_description() or self._scene_task

    def reset_scene(self) -> None:
        """Re-roll the kitchen: reset the env to a fresh scene (new layout/style).

        Uses a new seed each call so ``/reset`` explores different kitchens.
        """
        self._reset_count += 1
        n = self.env.num_envs
        if self.seed is None:
            seeds = None
        else:
            base = self.seed + self._reset_count * 1000
            seeds = [base + i for i in range(n)]
        obs, _ = self.env.reset(seed=seeds)
        self._last_obs = obs
        logger.info("[sim] scene reset (#%d)", self._reset_count)

    def _env0_obs(self) -> dict:
        """Slice env 0 out of the batched vec-env observation (batch of 1)."""
        raw = self._last_obs or {}
        pixels = raw.get("pixels")
        out: dict[str, Any] = {}
        if isinstance(pixels, dict):
            out["pixels"] = {k: np.asarray(v)[0:1] for k, v in pixels.items()}
        agent_pos = raw.get("agent_pos")
        if agent_pos is not None:
            out["agent_pos"] = np.asarray(agent_pos)[0:1]
        return out

    def observation_provider(self) -> dict | None:
        from lerobot.envs.utils import preprocess_observation  # noqa: PLC0415

        try:
            obs = preprocess_observation(self._env0_obs())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sim] preprocess_observation failed: %s", exc)
            return None
        # The adapter later replaces this recipe input with its generated subtask.
        obs["task"] = [self._current_task()]
        if self.preprocessor is not None:
            try:
                obs = self.preprocessor(obs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[sim] preprocessor failed: %s", exc)
                return None
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.items()
            if isinstance(k, str) and k.startswith("observation.")
        }

    def action_executor(self, action: Any) -> None:
        try:
            if self.postprocessor is not None:
                action = self.postprocessor(action)
            if isinstance(action, torch.Tensor):
                if action.ndim > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
                action = action.detach().to("cpu").numpy()
            # Tile env 0's action because the extra workers exist only for EGL stability.
            action_row = np.asarray(action, dtype=np.float32).reshape(-1)
            action_np = np.tile(action_row, (self.env.num_envs, 1))
            obs, _reward, terminated, truncated, _info = self.env.step(action_np)
            self._last_obs = obs
            self._capture_frame()
            # AsyncVectorEnv resets terminated sub-environments automatically.
            if bool(np.any(terminated)) or bool(np.any(truncated)):
                logger.info("[sim] episode ended — scene auto-reset")
        except Exception as exc:  # noqa: BLE001
            logger.error("[sim] env.step failed: %s", exc, exc_info=True)

    def _multiview_frame(self) -> np.ndarray | None:
        """Label and compose env 0's existing observation views without extra rendering."""
        pixels = (self._last_obs or {}).get("pixels")
        if not isinstance(pixels, dict) or not pixels:
            return None
        panels: list[np.ndarray] = []
        for cam in self._view_cams:
            v = pixels.get(cam)
            if v is None:
                continue
            img = np.asarray(v)
            if img.ndim == 4:  # (n_envs, H, W, C) -> env 0
                img = img[0]
            if img.ndim != 3 or img.shape[-1] != 3:
                continue
            panels.append(_label_panel(np.ascontiguousarray(img.astype(np.uint8)), _short_cam_name(cam)))
        if not panels:
            return None
        h = min(p.shape[0] for p in panels)
        panels = [p[:h] for p in panels]
        return np.concatenate(panels, axis=1)

    def _capture_frame(self) -> None:
        frame = self._multiview_frame()
        if frame is None:  # fallback to single env.render()
            try:
                rendered = self.env.call("render")[0]
                if isinstance(rendered, np.ndarray) and rendered.ndim == 3:
                    frame = rendered
            except Exception as exc:  # noqa: BLE001
                logger.debug("[sim] render failed: %s", exc)
        if frame is None:
            return
        subtask = self._subtask_getter() if self._subtask_getter else None
        memory = self._memory_getter() if self._memory_getter else None
        annotated = annotate_frame(
            frame,
            (("Task", self._current_task()), ("Subtask", subtask), ("Memory", memory)),
        )
        self._latest_frame = annotated  # served by the live MJPEG stream
        self._write_live_frame(annotated)
        if self.record:
            self._write_recording_frame(annotated)

    def _write_recording_frame(self, frame: np.ndarray) -> None:
        try:
            if self._video_writer is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._video_path = self.output_dir / f"sim_{stamp}.mp4"
                fps = int((getattr(self.env, "metadata", None) or {}).get("render_fps", 20))
                self._video_writer = StreamingVideoWriter(self._video_path, fps)
            self._video_writer.add_frame(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sim] video encoding failed: %s", exc)
            self.record = False

    def _write_live_frame(self, frame: np.ndarray) -> None:
        """Write a rolling latest.png every few frames for live viewing over SSH.

        Open ``{output_dir}/latest.png`` in an editor/viewer and refresh to watch
        the rollout in near-real-time without a GUI window. Written atomically
        (temp + replace) so a reader never sees a half-written file.
        """
        self._live_counter += 1
        if self._live_counter % 3 != 0:
            return
        try:
            import os  # noqa: PLC0415

            from PIL import Image  # noqa: PLC0415

            self.output_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.output_dir / ".latest.tmp.png"
            Image.fromarray(frame).save(tmp)
            os.replace(tmp, self.output_dir / "latest.png")
        except Exception as exc:  # noqa: BLE001
            logger.debug("[sim] live frame write failed: %s", exc)

    def _flush_video(self) -> None:
        if self._video_writer is None:
            return
        writer = self._video_writer
        self._video_writer = None
        try:
            writer.close()
            logger.info("[sim] wrote video (%d frames) to %s", writer.frames_written, self._video_path)
            print(f"[runtime] sim video saved to {self._video_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sim] video close failed: %s", exc)

    def attach_stream_server(self, server: Any) -> None:
        """Attach an already-running MJPEG server so disconnect() can stop it."""
        self._stream_server = server

    def disconnect(self) -> None:
        """Match the robot backend's cleanup contract."""
        if self._stream_server is not None:
            try:
                self._stream_server.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[sim] stream server shutdown raised %s", exc)
        self._flush_video()
        try:
            self.env.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[sim] env.close raised %s", exc)
