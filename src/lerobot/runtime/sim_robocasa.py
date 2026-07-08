"""RoboCasa simulation backend for the interactive language runtime.

Lets an operator type open-ended prompts (``/action <prompt>``) and have a
language-conditioned policy (e.g. PI052) execute them inside a RoboCasa mujoco
kitchen scene. The observation/action pipeline mirrors ``lerobot-eval`` exactly
so behaviour matches offline evaluation; only the *source* of observations and
the *sink* of actions differ from the real-robot backend, which is left
untouched.

A RoboCasa episode always instantiates a concrete scene (objects + layout) from
its task name, so ``--sim.task`` selects the scene while the prompt typed at the
prompt drives what the policy is asked to do inside it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def _overlay_text(
    frame: np.ndarray, task: str | None, subtask: str | None, memory: str | None
) -> np.ndarray:
    """Draw task / subtask / memory lines onto an (H, W, 3) uint8 frame.

    Best-effort: returns the frame unchanged if OpenCV is unavailable.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return frame

    lines = [f"{label}: {val}" for label, val in
             (("Task", task), ("Subtask", subtask), ("Memory", memory)) if val]
    if not lines:
        return frame

    img = np.ascontiguousarray(frame)
    font, scale, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 6
    max_width = img.shape[1] - 2 * margin
    y = 18
    for text in lines:
        # naive width-based wrap so long memory strings stay on-frame
        words, cur = text.split(), ""
        wrapped: list[str] = []
        for w in words:
            cand = f"{cur} {w}".strip()
            if cv2.getTextSize(cand, font, scale, 1)[0][0] > max_width and cur:
                wrapped.append(cur)
                cur = w
            else:
                cur = cand
        wrapped.append(cur)
        for line in wrapped:
            cv2.putText(img, line, (margin, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (margin, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
    return img


# RoboCasa's MuJoCo EGL offscreen renderer produces garbled/static frames when
# only ONE worker env is running (reproducible with lerobot-eval --batch_size=1).
# With >=2 workers the renderer is stable. We therefore run the interactive sim
# with a small vec env, drive env 0 with the policy, and ignore the rest.
_SIM_N_ENVS = 2


def create_sim_env(
    *,
    task: str,
    split: str | None,
    obj_registries: list[str],
    seed: int | None,
    render_size: int = 384,
) -> tuple[Any, dict]:
    """Create + reset a RoboCasa AsyncVectorEnv (n_envs=_SIM_N_ENVS), return (env, obs).

    MUST be called BEFORE the policy initialises CUDA in the parent process, so
    the forkserver workers don't inherit a CUDA context (which corrupts EGL).
    Uses >=2 workers because single-worker EGL rendering is broken on this stack
    (garbled frames) — the same reason lerobot-eval renders cleanly only at
    batch_size>=2. Only env 0 is driven/displayed.
    """
    from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig  # noqa: PLC0415

    # Higher-res observation cameras => higher-quality display. The policy is
    # unaffected: its preprocessor resizes images to 224 and VISUAL norm is
    # identity, so only render cost (not behaviour) changes with render_size.
    env_cfg = RoboCasaEnvConfig(
        task=task,
        split=split,
        obj_registries=list(obj_registries),
        observation_height=render_size,
        observation_width=render_size,
    )
    # Persistent kitchen: never end/reset on task success, and use a huge horizon
    # so the scene doesn't truncate. The user drives it with sequential prompts.
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
    """Start an MJPEG server serving frames from ``get_frame()`` on ``port``.

    Started early (before the ~60s policy load) so the port listens immediately
    and browsers get a page instead of connection-refused. ``get_frame`` returns
    the latest annotated frame or None (a "waiting" placeholder is shown until
    frames arrive). The server thread only reads/encodes frames — no CUDA/EGL —
    so it never affects rendering. Returns the server (for shutdown) or None.
    """
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
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
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
    """Drive a single RoboCasa gym env from the language runtime.

    Exposes ``observation_provider`` / ``action_executor`` closures matching the
    runtime's injected-callable contract, plus ``disconnect`` so the shared
    ``_run_autonomous`` cleanup path can close the env (and flush the video).

    The env must be created via :func:`create_sim_env` *before* the policy
    touches CUDA (see that function's note on the EGL/CUDA fork hazard).
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
        # Camera views to composite into the display frame (order = left→right).
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

        self._frames: list[np.ndarray] = []
        self._live_counter = 0
        self._latest_frame: np.ndarray | None = None
        self._stream_server: Any = None
        self._reset_count = 0
        # State getters wired after the runtime exists (bind_runtime), so the
        # video overlay can show the live task/subtask/memory.
        self._task_getter: Callable[[], str | None] | None = None
        self._subtask_getter: Callable[[], str | None] | None = None
        self._memory_getter: Callable[[], str | None] | None = None
        logger.info("[sim] scene ready — task_description=%r", self._scene_description())

    def bind_runtime(self, runtime: Any) -> None:
        """Wire live task/subtask/memory getters from the runtime state."""
        self._task_getter = lambda: runtime.state.get("task")
        self._subtask_getter = lambda: runtime.state.get("current_subtask")
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
        # ``task`` feeds the recipe RenderMessagesStep; the PI052 adapter
        # overwrites the language tokens with its generated subtask before the
        # action forward pass, so this only needs to be present, not exact.
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
            # Only env 0 is policy-driven; tile its action across all workers so
            # env.step gets a full (n_envs, action_dim) batch. The extra workers
            # exist only to keep MuJoCo's EGL renderer stable (single-worker
            # rendering is broken); their rollouts are ignored.
            action_row = np.asarray(action, dtype=np.float32).reshape(-1)
            action_np = np.tile(action_row, (self.env.num_envs, 1))
            obs, _reward, terminated, truncated, _info = self.env.step(action_np)
            self._last_obs = obs
            if self.record:
                self._capture_frame()
            # AsyncVectorEnv auto-resets a sub-env after it terminates, so the
            # scene continues on its own — no manual reset needed here.
            if bool(np.any(terminated)) or bool(np.any(truncated)):
                logger.info("[sim] episode ended — scene auto-reset")
        except Exception as exc:  # noqa: BLE001
            logger.error("[sim] env.step failed: %s", exc, exc_info=True)

    def _frontal_obs_image(self) -> np.ndarray | None:
        """Return the current front agent-view camera image (H, W, 3) uint8.

        Uses the observation the policy already consumes rather than a separate
        ``env.render()`` call: the render path's camera is intermittently
        corrupted by the offscreen EGL context, whereas the policy's obs images
        come straight through the eval pipeline and stay clean.
        """
        pixels = (self._last_obs or {}).get("pixels")
        if not isinstance(pixels, dict) or not pixels:
            return None
        cam = "robot0_agentview_left" if "robot0_agentview_left" in pixels else next(iter(pixels))
        img = np.asarray(pixels[cam])
        if img.ndim == 4:  # vec env batches to (1, H, W, C)
            img = img[0]
        if img.ndim != 3 or img.shape[-1] != 3:
            return None
        return img.astype(np.uint8)

    def _multiview_frame(self) -> np.ndarray | None:
        """Composite the configured camera views (env 0) side by side, labeled.

        Uses the policy's own high-res observation images (env.step already
        rendered them), so there's no extra render cost and orientation matches.
        """
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
        annotated = _overlay_text(frame, self._current_task(), subtask, memory)
        self._frames.append(annotated)
        self._latest_frame = annotated  # served by the live MJPEG stream
        self._write_live_frame(annotated)

    def _write_live_frame(self, frame: np.ndarray) -> None:
        """Write a rolling latest.png every few frames for live viewing over SSH.

        Open ``{output_dir}/latest.png`` in an editor/viewer and refresh to watch
        the rollout in near-real-time without a GUI window. Written atomically
        (temp + replace) so a reader never sees a half-written file.
        """
        if not self.record:
            return
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
        if not self.record or not self._frames:
            return
        from datetime import datetime  # noqa: PLC0415

        from lerobot.utils.io_utils import write_video  # noqa: PLC0415

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"sim_{stamp}.mp4"
        fps = int((getattr(self.env, "metadata", None) or {}).get("render_fps", 20))
        try:
            write_video(str(path), np.stack(self._frames), fps)
            logger.info("[sim] wrote video (%d frames) to %s", len(self._frames), path)
            print(f"[runtime] sim video saved to {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sim] write_video failed: %s", exc)

    def attach_stream_server(self, server: Any) -> None:
        """Attach an already-running MJPEG server so disconnect() can stop it."""
        self._stream_server = server

    def disconnect(self) -> None:
        """Match the robot backend's cleanup contract (called by _run_autonomous)."""
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
