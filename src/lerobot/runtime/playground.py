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

"""Browser controls for an interactive language-conditioned rollout.

The server intentionally uses the standard library so the runtime does not gain
another required web-framework dependency.  It owns no policy or simulator
state: commands are queued for the simulator's main thread, where inference and
MuJoCo rendering already run safely.
"""

from __future__ import annotations

import io
import ipaddress
import json
import queue
import socket
import threading
import time
from base64 import b64encode
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import torch
from PIL import Image


@dataclass
class PlaygroundCommand:
    """One browser request awaiting execution on the simulator thread."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    completed: threading.Event = field(default_factory=threading.Event, repr=False)
    result: dict[str, Any] | None = None
    error: str | None = None


class PlaygroundController:
    """Thread-safe bridge between the HTTP server and a live runtime."""

    def __init__(
        self,
        *,
        policy_path: str,
        benchmark: str = "robocasa",
        blog_url: str = "",
        planner: Any | None = None,
    ) -> None:
        self.policy_path = policy_path
        self.benchmark = benchmark
        self.blog_url = blog_url
        self.planner = planner
        self.runtime: Any | None = None
        self.sim_backend: Any | None = None
        self._commands: queue.Queue[PlaygroundCommand] = queue.Queue()
        self._messages: list[dict[str, Any]] = []
        self._message_id = 0
        self._lock = threading.RLock()
        self.started_at = time.time()

    def attach(self, runtime: Any, sim_backend: Any) -> None:
        """Attach live objects after policy construction has completed."""
        with self._lock:
            self.runtime = runtime
            self.sim_backend = sim_backend

    def enqueue(self, kind: str, payload: dict[str, Any] | None = None) -> PlaygroundCommand:
        command = PlaygroundCommand(kind=kind, payload=payload or {})
        self._commands.put(command)
        return command

    def next_command(self) -> PlaygroundCommand | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def finish(
        self, command: PlaygroundCommand, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        command.result = result or {}
        command.error = error
        command.completed.set()

    def add_message(
        self, role: str, text: str, *, image_url: str | None = None, kind: str = "chat"
    ) -> dict[str, Any]:
        with self._lock:
            self._message_id += 1
            message = {
                "id": self._message_id,
                "role": role,
                "text": text,
                "kind": kind,
                "created_at": time.time(),
            }
            if image_url:
                message["image_url"] = image_url
            self._messages.append(message)
            self._messages = self._messages[-100:]
            return message

    def snapshot(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            state: dict[str, Any] = {}
        else:
            with runtime.state.lock:
                state = {
                    "mode": runtime.state.get("mode", "paused"),
                    "task": runtime.state.get("task") or "",
                    "language_context": dict(runtime.state.get("language_context") or {}),
                    "queued_actions": len(runtime.state.get("action_queue") or []),
                    "actions_dispatched": int(runtime.state.get("actions_dispatched") or 0),
                    "revision": int(runtime.state.get("revision") or 0),
                }
        with self._lock:
            messages = list(self._messages)
        return {
            "connected": runtime is not None,
            "policy_path": self.policy_path,
            "benchmark": self.benchmark,
            "blog_url": self.blog_url,
            "uptime_seconds": max(0, int(time.time() - self.started_at)),
            "state": state,
            "messages": messages,
            "capabilities": {
                "benchmarks": {
                    "robocasa": {"available": True, "label": "RoboCasa"},
                    "libero": {"available": False, "label": "LIBERO"},
                    "robotwin": {"available": False, "label": "RoboTwin"},
                },
                "vqa": True,
                "remote_image": True,
                "planner": {
                    "available": self.planner is not None,
                    "model": getattr(self.planner, "model_path", None),
                    "loaded": bool(getattr(self.planner, "loaded", False)),
                },
            },
        }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _asset_bytes(name: str) -> bytes:
    return files("lerobot.runtime.playground_assets").joinpath(name).read_bytes()


def _observation_image(observation: dict[str, Any] | None) -> Image.Image:
    """Convert the first image tensor in a policy observation to RGB."""
    if not observation:
        raise ValueError("the runtime has no current observation")
    source = next(
        (
            value
            for key, value in observation.items()
            if isinstance(key, str)
            and ("image" in key or "pixel" in key)
            and isinstance(value, torch.Tensor)
            and value.ndim in {3, 4}
        ),
        None,
    )
    if source is None:
        raise ValueError("this policy observation has no image input")
    sample = source[0] if source.ndim == 4 else source
    if sample.shape[0] in {1, 3, 4}:
        sample = sample[:3].permute(1, 2, 0)
    sample = sample.detach().to(device="cpu", dtype=torch.float32)
    if float(sample.max()) > 1.5:
        sample = sample / 255
    if float(sample.min()) < 0:
        sample = (sample + 1) / 2
    array = (sample.clamp(0, 1).numpy() * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


class LazyQwenPlanner:
    """Lazily load Qwen3.5 as an optional visual high-level planner."""

    def __init__(self, model_path: str = "Qwen/Qwen3.5-2B", *, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self.model: Any | None = None
        self.processor: Any | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: PLC0415

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForMultimodalLM.from_pretrained(self.model_path, dtype="auto").to(device)
            self.model.eval()

    def plan(
        self,
        goal: str,
        observation: dict[str, Any] | None,
        *,
        system_prompt: str,
    ) -> str:
        """Return the next low-level subtask from the current view and goal."""
        self._ensure_loaded()
        image = _observation_image(observation)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_url = f"data:image/png;base64,{b64encode(buffer.getvalue()).decode()}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {
                        "type": "text",
                        "text": f"High-level goal: {goal}\nReturn only the next concrete robot subtask.",
                    },
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.processor.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def _validate_remote_image_url(image_url: str) -> str:
    """Reject non-HTTP and local/private targets before fetching a VQA image."""
    parsed = urlsplit(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("image URL must use http or https")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for info in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError("image URL must resolve to a public address")
    return image_url


class _SafeImageRedirectHandler(HTTPRedirectHandler):
    """Apply the same public-address restriction to every redirect hop."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        return super().redirect_request(req, fp, code, msg, headers, _validate_remote_image_url(newurl))


def replace_observation_image_from_url(
    observation: dict[str, Any] | None,
    image_url: str,
    *,
    max_bytes: int = 12_000_000,
) -> dict[str, Any]:
    """Replace the first tensor image input with a safely fetched remote image."""
    if not observation:
        raise ValueError("the runtime has no current observation")
    request = Request(  # noqa: S310
        _validate_remote_image_url(image_url),
        headers={"User-Agent": "LeRobot-Playground/1.0"},
    )
    with build_opener(_SafeImageRedirectHandler()).open(request, timeout=8) as response:  # noqa: S310
        content_type = str(response.headers.get("Content-Type") or "")
        if not content_type.lower().startswith("image/"):
            raise ValueError("remote URL did not return an image")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("remote image is larger than 12 MB")
    with Image.open(io.BytesIO(data)) as opened:
        rgb = opened.convert("RGB")

    target_key = next(
        (
            key
            for key, value in observation.items()
            if isinstance(key, str)
            and ("image" in key or "pixel" in key)
            and isinstance(value, torch.Tensor)
            and value.ndim in {3, 4}
        ),
        None,
    )
    if target_key is None:
        raise ValueError("this policy observation has no replaceable image input")
    target = observation[target_key]
    batched = target.ndim == 4
    sample = target[0] if batched else target
    channel_first = sample.shape[0] in {1, 3, 4}
    height, width = (
        (sample.shape[-2], sample.shape[-1]) if channel_first else (sample.shape[0], sample.shape[1])
    )
    rgb = rgb.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    replacement = torch.from_numpy(np.asarray(rgb).copy()).to(dtype=torch.float32) / 255
    if torch.is_floating_point(target) and float(target.min()) < 0:
        replacement = replacement * 2 - 1
    if channel_first:
        replacement = replacement.permute(2, 0, 1)
    replacement = replacement.to(device=target.device, dtype=target.dtype)
    if batched:
        replacement = replacement.unsqueeze(0).expand(target.shape[0], *replacement.shape)
    updated = dict(observation)
    updated[target_key] = replacement
    return updated


def start_playground_server(
    port: int,
    get_frame: Any,
    controller: PlaygroundController,
) -> ThreadingHTTPServer | None:
    """Serve the full-screen playground, its API, and the live MJPEG stream."""

    placeholder = Image.new("RGB", (640, 400), (15, 18, 22))
    assets = {
        "/": ("text/html; charset=utf-8", _asset_bytes("index.html")),
        "/index.html": ("text/html; charset=utf-8", _asset_bytes("index.html")),
        "/playground.css": ("text/css; charset=utf-8", _asset_bytes("playground.css")),
        "/playground.js": ("text/javascript; charset=utf-8", _asset_bytes("playground.js")),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "LeRobotPlayground/1"

        def log_message(self, *_args: Any) -> None:
            pass

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            data = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in assets:
                content_type, data = assets[path]
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/state":
                self._send_json(HTTPStatus.OK, controller.snapshot())
                return
            if path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if path != "/stream":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = get_frame()
                    image = Image.fromarray(frame) if frame is not None else placeholder
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=82)
                    data = buffer.getvalue()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path not in {"/api/command", "/api/chat"}:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                payload = self._read_json()
                kind = str(payload.get("kind") or ("chat" if path == "/api/chat" else "")).strip()
                if kind not in {"action", "pause", "reset", "chat", "vqa", "planner"}:
                    raise ValueError(f"unsupported command: {kind!r}")
                command = controller.enqueue(kind, payload)
                timeout = 600 if kind == "planner" else (180 if kind in {"chat", "vqa"} else 15)
                if not command.completed.wait(timeout):
                    self._send_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "runtime did not respond"})
                    return
                if command.error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": command.error})
                    return
                self._send_json(HTTPStatus.OK, {"ok": True, **(command.result or {})})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)  # nosec B104
    except OSError:
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="lerobot-playground").start()
    return server
