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
"""Shared Qwen-VL client.

The pipeline uses a single shared VLM across modules. vLLM is preferred when
available (high throughput, JSON-guided decoding); transformers is the
fallback. A ``stub`` backend is used for unit tests so fixtures never call
into a real model.

The client speaks one method, :meth:`VlmClient.generate_json`, which:

- accepts a list of OpenAI/HF-style multimodal messages,
- requests JSON output from the server,
- batches requests transparently,
- and reprompts once on a JSON parse failure with an inline correction
  message before raising.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .config import VlmConfig


class VlmClient(Protocol):
    """Protocol every backend must implement."""

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        """Generate one JSON-decoded response per messages list."""


@dataclass
class StubVlmClient:
    """Deterministic stub used in unit tests.

    A test passes a callable that maps the *last user message text* (or, if
    that is empty, the full message list) to a JSON-serializable response.
    """

    responder: Callable[[Sequence[dict[str, Any]]], Any]

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        return [self.responder(list(messages)) for messages in messages_batch]


def _strip_to_json(text: str) -> Any:
    text = text.strip()
    # Strip <think>...</think> blocks (Qwen3 Thinking style)
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = (text[:start] + text[end:]).strip()
    # Strip ```json ... ``` fences from chat-tuned backbones
    if text.startswith("```"):
        first = text.find("\n")
        last = text.rfind("```")
        if first != -1 and last != -1 and last > first:
            text = text[first + 1 : last].strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    # Fall back to extracting the first balanced {...} block.
    obj_text = _extract_first_json_object(text)
    if obj_text is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(obj_text)


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces in
    string literals. Returns ``None`` if no balanced block is found."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        # Note: ``escape`` is always False here — the ``if escape`` branch
        # above already handled and reset it.
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass
class _GenericTextClient:
    """Wraps any text-generation callable in JSON-mode + one-retry semantics."""

    generate_text: Callable[[Sequence[Sequence[dict[str, Any]]], int, float], list[str]]
    config: VlmConfig

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        max_tok = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        raw = self.generate_text(messages_batch, max_tok, temp)
        out: list[Any] = []
        for messages, text in zip(messages_batch, raw, strict=True):
            try:
                out.append(_strip_to_json(text))
                continue
            except (ValueError, json.JSONDecodeError):
                pass
            retry = list(messages) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. "
                        "Reply with strictly valid JSON, no prose, no fences."
                    ),
                },
            ]
            retry_text = self.generate_text([retry], max_tok, temp)[0]
            try:
                out.append(_strip_to_json(retry_text))
            except (ValueError, json.JSONDecodeError):
                # After retry: log preview and return None instead of crashing
                # the whole pipeline. Modules treat None as "skip".
                preview = retry_text.strip().replace("\n", " ")[:200]
                print(
                    f"[vlm] WARNING: failed to parse JSON after retry; preview: {preview!r}",
                    flush=True,
                )
                out.append(None)
        return out


def make_vlm_client(config: VlmConfig) -> VlmClient:
    """Build the shared VLM client.

    Only the ``openai`` backend is supported for now. The shipped workflow
    is Hugging Face Jobs (``examples/annotations/run_hf_job.py``): it boots
    a vLLM server inside the ``vllm/vllm-openai`` image and the pipeline
    talks to it over the OpenAI-compatible API (``--vlm.backend=openai``,
    optionally auto-spawning the server via ``auto_serve`` /
    ``serve_command``). The former in-process ``vllm`` / ``transformers``
    backends were removed to keep the support surface to the HF Jobs path.

    For ``stub``, construct :class:`StubVlmClient` directly with a responder
    callable; it is rejected here to make accidental misuse obvious.
    """
    if config.backend == "openai":
        return _make_openai_client(config)
    if config.backend == "stub":
        raise ValueError(
            "Use StubVlmClient(...) directly for the stub backend; make_vlm_client builds real clients."
        )
    if config.backend in {"vllm", "transformers"}:
        raise ValueError(
            f"backend={config.backend!r} (in-process local model) is not supported for now — "
            "only backend='openai' (the Hugging Face Jobs flow) is. Run the pipeline via "
            "examples/annotations/run_hf_job.py, which serves the model with vLLM in the "
            "vllm/vllm-openai image and talks to it over the OpenAI-compatible API."
        )
    raise ValueError(f"Unknown VLM backend: {config.backend!r}")


def _make_openai_client(config: VlmConfig) -> VlmClient:
    """Backend that talks to any OpenAI-compatible server.

    Compatible with ``vllm serve``, ``transformers serve``,
    ``ktransformers serve``, and hosted endpoints. By default the server
    is expected to be already running. Set ``auto_serve=True`` to have
    this client spawn one (default: ``transformers serve``), wait until
    it's ready, and tear it down on process exit.

    Image blocks ``{"type":"image", "image":<PIL.Image>}`` are
    auto-converted to ``image_url`` data-URLs. Video blocks
    ``{"type":"video", "video":[<PIL>...]}`` are forwarded as
    multi-frame ``video_url`` items where supported.
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "openai package is required for backend='openai'. Install with `pip install openai`."
        ) from exc

    api_base = config.api_base
    api_key = config.api_key
    auto_serve = config.auto_serve
    api_bases: list[str] = [api_base]

    print(
        f"[lerobot-annotate] backend=openai model={config.model_id} "
        f"api_base={api_base} auto_serve={auto_serve}",
        flush=True,
    )
    if auto_serve:
        if config.parallel_servers > 1:
            print(
                f"[lerobot-annotate] spawning {config.parallel_servers} parallel servers",
                flush=True,
            )
            api_bases = _spawn_parallel_inference_servers(config)
        elif _server_is_up(api_base):
            print(f"[lerobot-annotate] reusing server already up at {api_base}", flush=True)
        else:
            print("[lerobot-annotate] no server reachable; spawning one", flush=True)
            api_base = _spawn_inference_server(config)
            api_bases = [api_base]
            print(f"[lerobot-annotate] server ready at {api_base}", flush=True)

    clients = [OpenAI(base_url=base, api_key=api_key) for base in api_bases]
    # round-robin counter for parallel mode
    rr_counter = {"i": 0}

    # ``mm_processor_kwargs`` is a vllm-specific extra; transformers serve
    # rejects it with HTTP 422. Send it only when explicitly opted in via
    # an env var (e.g. ``LEROBOT_OPENAI_SEND_MM_KWARGS=1`` for vllm).
    send_mm_kwargs = os.environ.get("LEROBOT_OPENAI_SEND_MM_KWARGS", "").lower() in {"1", "true", "yes"}

    rr_lock = threading.Lock()

    def _one_call(messages: Sequence[dict[str, Any]], max_tok: int, temp: float) -> str:
        api_messages, mm_kwargs = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": config.model_id,
            "messages": api_messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
        extra_body: dict[str, Any] = {}
        if send_mm_kwargs and mm_kwargs:
            extra_body["mm_processor_kwargs"] = {**mm_kwargs, "do_sample_frames": True}
        if config.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = config.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        with rr_lock:
            chosen = clients[rr_counter["i"] % len(clients)]
            rr_counter["i"] += 1
        response = chosen.chat.completions.create(**kwargs)
        # Some OpenAI-compatible servers can return a choice with no message
        # (safety filter, or a "thinking" model that spends the whole budget
        # before emitting content). Treat that as an empty reply so the
        # JSON-retry path handles it instead of crashing the run.
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        return (message.content if message is not None else None) or ""

    def _gen(batch: Sequence[Sequence[dict[str, Any]]], max_tok: int, temp: float) -> list[str]:
        if len(batch) <= 1 or config.client_concurrency <= 1:
            return [_one_call(messages, max_tok, temp) for messages in batch]
        # Parallel fan-out — vllm batches these on the server side.
        max_workers = min(config.client_concurrency, len(batch))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one_call, messages, max_tok, temp) for messages in batch]
            return [f.result() for f in futures]

    return _GenericTextClient(_gen, config)


def _bind_serve_port(cmd: str, port: int) -> str:
    """Bind a serve command to ``port``: substitute a ``{port}`` placeholder
    if present, else append ``--port`` when the command omits it (leaving an
    explicit ``--port`` untouched). Shared by the single- and parallel-server
    paths so a serve_command never reaches the server with a literal
    ``{port}``."""
    if "{port}" in cmd:
        return cmd.replace("{port}", str(port))
    if "--port" not in cmd:
        return f"{cmd} --port {port}"
    return cmd


def _spawn_parallel_inference_servers(config: VlmConfig) -> list[str]:
    """Spawn ``config.parallel_servers`` independent vllm replicas.

    Each replica:
    - is pinned to a single GPU via ``CUDA_VISIBLE_DEVICES``
    - listens on ``serve_port + i``
    - is shut down via the same atexit hook as the single-server path

    Returns the list of ``api_base`` URLs the client should round-robin
    across.
    """
    n = config.parallel_servers
    api_bases: list[str] = []
    procs: list[subprocess.Popen] = []
    ready_events: list[threading.Event] = []
    # Multiple readiness signals — uvicorn's own banner is suppressed at
    # ``--uvicorn-log-level warning``, so we also accept vllm's own
    # "Starting vLLM API server" line and the route-listing line. The
    # HTTP probe below is the ultimate fallback.
    ready_markers = (
        "Uvicorn running",
        "Application startup complete",
        "Starting vLLM API server",
        "Available routes are",
    )
    # Single lock for all server-stream threads so multibyte chars from
    # different servers don't interleave and tear UTF-8 sequences.
    print_lock = threading.Lock()

    base_cmd = config.serve_command or (
        f"vllm serve {shlex.quote(config.model_id)} "
        f"--tensor-parallel-size 1 "
        f"--max-model-len {config.max_model_len or 32768} "
        f"--uvicorn-log-level warning"
    )

    num_gpus = config.num_gpus if config.num_gpus > 0 else n
    for i in range(n):
        port = config.serve_port + i
        gpu = i % num_gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = _bind_serve_port(base_cmd, port)
        api_base = f"http://localhost:{port}/v1"
        api_bases.append(api_base)
        print(f"[server-{i}] launching on GPU {gpu} port {port}: {cmd}", flush=True)
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        procs.append(proc)
        ready = threading.Event()
        ready_events.append(ready)

        def _stream(idx: int, p: subprocess.Popen, ev: threading.Event) -> None:
            # Read whole lines and emit each line atomically under the
            # shared print_lock so output from N servers stays readable.
            assert p.stdout is not None
            for line in iter(p.stdout.readline, ""):
                with print_lock:
                    sys.stdout.write(f"[server-{idx}] {line}")
                    if not line.endswith(("\n", "\r")):
                        sys.stdout.write("\n")
                    sys.stdout.flush()
                if any(m in line for m in ready_markers):
                    ev.set()

        threading.Thread(target=_stream, args=(i, proc, ready), daemon=True).start()

        def _probe(idx: int, base: str, ev: threading.Event, p: subprocess.Popen) -> None:
            while not ev.is_set() and p.poll() is None:
                if _server_is_up(base):
                    print(f"[server-{idx}] ready (http probe)", flush=True)
                    ev.set()
                    return
                time.sleep(2)

        threading.Thread(target=_probe, args=(i, api_base, ready, proc), daemon=True).start()

    def _shutdown() -> None:
        for i, p in enumerate(procs):
            if p.poll() is None:
                print(f"[server-{i}] stopping pid={p.pid}", flush=True)
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)

    atexit.register(_shutdown)

    deadline = time.monotonic() + config.serve_ready_timeout_s
    while any(not ev.is_set() for ev in ready_events) and time.monotonic() < deadline:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                raise RuntimeError(
                    f"[server-{i}] inference server exited unexpectedly with rc={p.returncode}"
                )
        time.sleep(2)
    if any(not ev.is_set() for ev in ready_events):
        raise RuntimeError(f"[server] not all replicas became ready within {config.serve_ready_timeout_s}s")
    print(f"[lerobot-annotate] all {n} servers ready: {api_bases}", flush=True)
    return api_bases


def _server_is_up(api_base: str) -> bool:
    """Return True if ``api_base/models`` answers 200 within 2 seconds."""
    url = api_base.rstrip("/") + "/models"
    # ``api_base`` is the user-configured local-server URL we just spawned
    # or the user passed in via ``--vlm.api_base``; the bandit B310 warning
    # is for arbitrary user-controlled URLs with file:/ schemes which
    # cannot reach this code path.
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310  # nosec B310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_inference_server(config: VlmConfig) -> str:
    """Spawn ``transformers serve`` (or ``serve_command``), wait until it
    accepts ``/v1/models``, and register a shutdown hook.

    Streams the server's stdout/stderr to the parent terminal in
    real-time on a background thread so users can see model-load
    progress and errors as they happen.

    Returns the full ``api_base`` URL the OpenAI client should use.
    """
    cmd = config.serve_command
    if not cmd:
        cmd = (
            f"transformers serve {shlex.quote(config.model_id)} "
            f"--port {config.serve_port} --continuous-batching"
        )
    # Bind the single server to ``serve_port`` (what ``api_base`` below
    # targets): substitute a literal ``{port}`` placeholder, else append
    # ``--port``. Without this a serve_command carrying ``{port}`` would
    # reach the server unsubstituted and fail to parse.
    cmd = _bind_serve_port(cmd, config.serve_port)
    api_base = f"http://localhost:{config.serve_port}/v1"
    print(f"[server] launching: {cmd}", flush=True)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Watch the server output for the uvicorn readiness banner. This is
    # more reliable than polling /v1/models because transformers serve
    # rescans its cache on every model-list request, which can exceed
    # the urllib timeout and trigger an infinite probe loop.
    ready_event = threading.Event()
    # See _spawn_parallel_inference_servers for why we accept these.
    ready_markers = (
        "Uvicorn running",
        "Application startup complete",
        "Starting vLLM API server",
        "Available routes are",
    )

    def _probe() -> None:
        while not ready_event.is_set() and proc.poll() is None:
            if _server_is_up(api_base):
                print("[server] ready (http probe)", flush=True)
                ready_event.set()
                return
            time.sleep(2)

    threading.Thread(target=_probe, daemon=True).start()

    def _stream_output() -> None:
        # Read raw chunks instead of iterating lines so tqdm progress
        # bars (which overwrite using \r) flush in real time.
        assert proc.stdout is not None
        buf = ""
        prefix_started = False
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                # process exited; flush any tail
                if buf:
                    sys.stdout.write(buf)
                    sys.stdout.flush()
                return
            if not prefix_started:
                sys.stdout.write("[server] ")
                prefix_started = True
            sys.stdout.write(ch)
            sys.stdout.flush()
            buf += ch
            if ch in ("\n", "\r"):
                if any(marker in buf for marker in ready_markers):
                    ready_event.set()
                buf = ""
                prefix_started = False

    threading.Thread(target=_stream_output, daemon=True).start()

    def _shutdown() -> None:
        if proc.poll() is None:
            print(f"[server] stopping pid={proc.pid}", flush=True)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    atexit.register(_shutdown)

    deadline = time.monotonic() + config.serve_ready_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"[server] inference server exited unexpectedly with rc={proc.returncode}. "
                f"See [server] log lines above for the cause."
            )
        if ready_event.wait(timeout=2):
            return api_base
    proc.terminate()
    raise RuntimeError(f"[server] did not become ready within {config.serve_ready_timeout_s}s")


def _to_openai_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert internal messages to OpenAI chat format.

    Returns ``(api_messages, mm_kwargs)``. Multimodal-processor kwargs
    (``fps`` from ``video_url`` blocks) are extracted out so the caller
    can pass them via ``extra_body.mm_processor_kwargs`` rather than
    inside the content blocks (which transformers serve rejects).

    File-URL video blocks are inlined as base64 data URLs.
    """
    out_messages: list[dict[str, Any]] = []
    mm_kwargs: dict[str, Any] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out_messages.append({"role": message["role"], "content": content})
            continue
        out_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "text":
                out_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                out_blocks.append(
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(block["image"])}}
                )
            elif block_type == "video":
                frames = block.get("video", [])
                for img in frames:
                    out_blocks.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
            elif block_type == "video_url":
                video_url = dict(block["video_url"])
                url = video_url.get("url", "")
                if url.startswith("file://"):
                    video_url["url"] = _file_to_data_url(url[len("file://") :])
                out_blocks.append({"type": "video_url", "video_url": video_url})
                fps = block.get("fps")
                if fps is not None:
                    mm_kwargs["fps"] = fps
            else:
                out_blocks.append(block)
        out_messages.append({"role": message["role"], "content": out_blocks})
    return out_messages, mm_kwargs


def _file_to_data_url(path: str) -> str:
    """Read a local video file and return a base64 ``data:video/mp4`` URL."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _pil_to_data_url(image: Any) -> str:
    """Encode a PIL.Image as a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
