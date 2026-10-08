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
"""Unit tests for ``vlm_client`` helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.vlm_client import _bind_serve_port  # noqa: E402


def test_bind_serve_port_substitutes_placeholder() -> None:
    # The {port} placeholder is replaced everywhere it appears, regardless of
    # parallel vs single server — the bug was the single-server path passing
    # it through unsubstituted.
    cmd = "vllm serve M --max-model-len 32768 --port {port}"
    assert _bind_serve_port(cmd, 8000) == "vllm serve M --max-model-len 32768 --port 8000"


def test_bind_serve_port_appends_when_missing() -> None:
    assert _bind_serve_port("vllm serve M", 8001) == "vllm serve M --port 8001"


def test_bind_serve_port_leaves_explicit_port_untouched() -> None:
    cmd = "vllm serve M --port 9000"
    assert _bind_serve_port(cmd, 8000) == cmd


def test_model_revision_requires_matching_binding():
    from lerobot.annotations.steerable_pipeline.vlm_client import _bind_model_revision

    revision = "a" * 40
    assert _bind_model_revision("vllm serve M", revision).endswith(" --revision " + revision)
    assert _bind_model_revision("custom --sha {revision}", revision) == "custom --sha " + revision
    with pytest.raises(ValueError, match="requires a value"):
        _bind_model_revision("vllm serve M --revision", revision)
    with pytest.raises(ValueError, match="does not match"):
        _bind_model_revision("vllm serve M --revision other", revision)
    with pytest.raises(ValueError, match="Custom"):
        _bind_model_revision("custom M", revision)


def test_replicas_use_assigned_gpu_ids_and_pinned_revision(monkeypatch):
    import io

    from lerobot.annotations.steerable_pipeline import vlm_client
    from lerobot.annotations.steerable_pipeline.config import VlmConfig

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned-a,GPU-assigned-b")
    created = []

    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("Application startup complete\n")
            self.returncode = None
            self.pid = len(created)
            created.append((command, kwargs))

        def poll(self):
            return self.returncode

        def send_signal(self, signal):
            self.returncode = 0

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(vlm_client.subprocess, "Popen", Process)
    monkeypatch.setattr(vlm_client, "_server_is_up", lambda base: True)
    cfg = VlmConfig(parallel_servers=2, model_revision="a" * 40)
    shutdowns = []
    bases = vlm_client._spawn_parallel_inference_servers(cfg, shutdowns=shutdowns)
    assert bases == ["http://localhost:8000/v1", "http://localhost:8001/v1"]
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in created] == [
        "GPU-assigned-a",
        "GPU-assigned-b",
    ]
    assert all(command[command.index("--revision") + 1] == "a" * 40 for command, _ in created)
    shutdowns[0]()
    cfg.parallel_servers = 3
    with pytest.raises(ValueError, match="exceed"):
        vlm_client._spawn_parallel_inference_servers(cfg)


def test_nested_request_batches_have_one_shared_concurrency_limit(monkeypatch):
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from lerobot.annotations.steerable_pipeline.config import VlmConfig
    from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client

    count, maximum, closed = 0, 0, []
    lock = threading.Lock()

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            nonlocal count, maximum
            with lock:
                count += 1
                maximum = max(count, maximum)
            time.sleep(0.01)
            with lock:
                count -= 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])

        def close(self):
            closed.append(self)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    client = make_vlm_client(VlmConfig(auto_serve=False, client_concurrency=2))
    prompts = [[{"role": "user", "content": "test"}]] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: client.generate_json(prompts), range(4)))
    assert all(result == [{"ok": True}] * 4 for result in results)
    assert maximum <= 2 and count == 0
    client.close()
    client.close()
    assert len(closed) == 1
