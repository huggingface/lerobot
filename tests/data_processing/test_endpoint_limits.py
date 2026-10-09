# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Admission is shared across real processes, including fail-closed error paths."""

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pytest

from lerobot.data_processing.endpoint_limits import EndpointLimitServer, SharedEndpointLimit


@pytest.fixture
def coordinator(monkeypatch):
    monkeypatch.setenv("LEROBOT_ENDPOINT_LIMIT_TOKEN", "test-only-secret")
    with EndpointLimitServer(("127.0.0.1", 0), key="qwen", limit=2, token="test-only-secret") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _worker(url, start, count, maximum, lock):
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN", timeout=10)
    start.wait(timeout=10)
    for _ in range(2):
        with limit.permit():
            with lock:
                count.value += 1
                maximum.value = max(count.value, maximum.value)
            time.sleep(0.05)
            with lock:
                count.value -= 1


def test_limit_is_global_across_processes(coordinator):
    server, url = coordinator
    ctx = multiprocessing.get_context("spawn")
    start, lock = ctx.Event(), ctx.Lock()
    count, maximum = ctx.Value("i", 0), ctx.Value("i", 0)
    workers = [ctx.Process(target=_worker, args=(url, start, count, maximum, lock)) for _ in range(4)]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert maximum.value == 2 and count.value == 0
        assert not server.active
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def test_authentication_idempotency_and_ambiguous_inference(coordinator):
    import uuid

    server, url = coordinator
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
    request_id = str(uuid.uuid4())
    limit._post("acquire", request_id)
    limit._post("acquire", request_id)
    assert server.active == {request_id}
    limit._post("release", request_id)
    limit._post("release", request_id)
    assert not server.active
    with pytest.raises(HTTPError) as exc:
        limit._post("acquire", request_id)
    assert exc.value.code == 410
    limit.token = "wrong"
    with pytest.raises(HTTPError) as exc:
        limit._post("acquire", str(uuid.uuid4()))
    assert exc.value.code == 401 and not server.active
    limit.token = "test-only-secret"
    with pytest.raises(TimeoutError), limit.permit():
        raise TimeoutError("Inference may still be running")
    assert len(server.active) == 1  # No unsafe lease expiry/free on ambiguous timeout.
    with pytest.raises(ValueError), limit.permit():
        raise ValueError("Invalid schema after completed request")
    assert len(server.active) == 1  # The terminal error released only its own permit.


@pytest.mark.parametrize("status", [429, 503, 504])
@pytest.mark.parametrize("sdk_error", [False, True])
def test_gateway_timeout_preserves_inference_permit(coordinator, status, sdk_error):
    server, url = coordinator
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
    if sdk_error:
        # OpenAI-compatible SDKs expose the response status independently of urllib.
        class ServiceError(Exception):
            status_code = status

        error = ServiceError("Model service response")
    else:
        error = HTTPError("http://model.invalid", status, "Model service response", {}, None)
    with pytest.raises(type(error)), limit.permit():
        raise error
    # A proxy timeout does not confirm that upstream inference has stopped.
    assert len(server.active) == (1 if status == 504 else 0)


def test_vlm_clients_share_one_limit(coordinator, monkeypatch):
    pytest.importorskip("datasets")
    import sys
    from types import SimpleNamespace

    from lerobot.annotations.steerable_pipeline.config import VlmConfig
    from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client

    server, url = coordinator
    active, maximum = 0, 0
    lock = threading.Lock()

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    clients = [
        make_vlm_client(
            VlmConfig(
                auto_serve=False, client_concurrency=4, endpoint_limit_url=url, endpoint_limit_key="qwen"
            )
        )
        for _ in range(2)
    ]
    prompts = [[{"role": "user", "content": "test"}]] * 4
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(lambda client: client.generate_json(prompts), clients))
        assert outputs == [[{"ok": True}] * 4] * 2
        assert maximum == 2 and active == 0 and not server.active
    finally:
        for client in clients:
            client.close()
