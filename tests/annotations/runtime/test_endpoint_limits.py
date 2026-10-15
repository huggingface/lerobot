# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Admission is shared across real processes, including fail-closed error paths."""

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pytest

from lerobot.annotations.runtime.endpoint_limits import EndpointLimitServer, SharedEndpointLimit
from lerobot.annotations.runtime.errors import EndpointLimitError


@pytest.fixture
def coordinator(monkeypatch):
    monkeypatch.setenv("LEROBOT_ENDPOINT_LIMIT_TOKEN", "test-only-secret")
    with EndpointLimitServer(
        ("127.0.0.1", 0), key="qwen", limit=2, token="test-only-secret", admin_token="test-admin"
    ) as server:
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
    with pytest.raises(EndpointLimitError), limit.permit():
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
    with pytest.raises(EndpointLimitError if status == 504 else type(error)) as failure, limit.permit():
        raise error
    if status == 504:
        assert failure.value.__cause__ is error
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
    overlap = threading.Barrier(2)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                overlap.wait(timeout=10)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
                )
            finally:
                with lock:
                    active -= 1

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


@pytest.mark.parametrize("connect_failure", [False, True])
def test_request_retry_reacquires_shared_admission(coordinator, monkeypatch, connect_failure):
    pytest.importorskip("datasets")
    import sys
    from types import SimpleNamespace

    from lerobot.annotations.steerable_pipeline import vlm_client
    from lerobot.annotations.steerable_pipeline.config import VlmConfig

    server, url = coordinator
    operations, calls = [], []
    original = SharedEndpointLimit._post

    def post(self, operation, *args, **kwargs):
        operations.append(operation)
        return original(self, operation, *args, **kwargs)

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            assert len(server.active) == 1
            calls.append(kwargs)
            if len(calls) == 1:
                if connect_failure:
                    raise ConnectionRefusedError("inference was not sent")
                raise HTTPError("http://model.invalid", 503, "retry", {}, None)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    monkeypatch.setattr(SharedEndpointLimit, "_post", post)
    monkeypatch.setattr(vlm_client, "retry_delay", lambda _: 0)
    client = vlm_client.make_vlm_client(
        VlmConfig(auto_serve=False, endpoint_limit_url=url, endpoint_limit_key="qwen")
    )
    try:
        assert client.generate_json([[{"role": "user", "content": "test"}]]) == [{"ok": True}]
        assert operations == ["acquire", "release", "acquire", "release"]
        assert not server.active
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["read", "gateway", "admission", "release", "release_error"])
def test_unsafe_shared_requests_are_not_retried(coordinator, tmp_path, monkeypatch, caplog, failure):
    pytest.importorskip("datasets")
    import sys
    import uuid
    from types import SimpleNamespace

    from lerobot.annotations.runtime import worker
    from lerobot.annotations.runtime.artifacts import ArtifactStore
    from lerobot.annotations.runtime.errors import ErrorCategory, classify_error
    from lerobot.annotations.runtime.runtime import run_local
    from lerobot.annotations.steerable_pipeline import vlm_client
    from lerobot.annotations.steerable_pipeline.config import VlmConfig
    from tests.annotations.runtime._modules import Echo
    from tests.annotations.runtime.test_runtime import make_plan

    server, url = coordinator
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if failure == "read":
                raise TimeoutError("ambiguous model read")
            if failure == "gateway":
                raise HTTPError("http://model.invalid", 504, "ambiguous gateway timeout", {}, None)
            if failure == "release_error":
                raise HTTPError("http://model.invalid", 503, "failed release must not hide this", {}, None)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])

        def close(self):
            pass

    if failure == "admission":
        limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
        for _ in range(server.limit):
            limit._post("acquire", str(uuid.uuid4()))
    elif failure in {"release", "release_error"}:

        def release(*args):
            raise ConnectionError("coordinator unavailable after completion")

        monkeypatch.setattr(SharedEndpointLimit, "_release", release)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    monkeypatch.setattr(vlm_client, "retry_delay", lambda _: 0)
    monkeypatch.setattr(worker, "retry_delay", lambda _: 0)
    client = vlm_client.make_vlm_client(
        VlmConfig(
            auto_serve=False,
            endpoint_limit_url=url,
            endpoint_limit_key="qwen",
            endpoint_limit_timeout_s=0.05 if failure == "admission" else 2,
        )
    )
    prompts = [[{"role": "user", "content": "test"}]]
    try:
        if failure == "release":
            assert client.generate_json(prompts) == [{"ok": True}]
            assert "permit" in caplog.text and "retained" in caplog.text
        else:

            def process(*args):
                client.generate_json(prompts)
                pytest.fail("Uncertain request unexpectedly succeeded")

            monkeypatch.setattr(Echo, "process_batch", process)
            store = ArtifactStore(tmp_path / "run")
            plan = make_plan(store, size=1)
            with pytest.raises(RuntimeError, match="not retried") as exc:
                run_local(store, plan, max_retries=2)
            assert classify_error(exc.value) == ErrorCategory.PERMANENT
            assert exc.value.__cause__ is not None
            errors = store.list(f"{plan.prefix}/attempts/*/*/error.json")
            assert len(errors) == 1 and store.read_json(errors[0])["retryable"] is False
        assert len(calls) == (0 if failure == "admission" else 1)
        assert len(server.active) == (2 if failure == "admission" else 1)
    finally:
        client.close()


def test_interrupted_admission_clears_its_unsent_request(coordinator, monkeypatch):
    server, url = coordinator
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
    original = limit._post

    def post(operation, *args, **kwargs):
        result = original(operation, *args, **kwargs)
        if operation == "acquire":
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(limit, "_post", post)
    with pytest.raises(KeyboardInterrupt), limit.permit():
        pytest.fail("Interrupted admission must not send inference")
    assert not server.active


@pytest.mark.parametrize("kind", ["refused", "dns", "httpx_connect", "wrapped", "read"])
def test_only_known_connect_failures_release_permits(coordinator, kind):
    import socket

    import httpx

    server, url = coordinator
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
    errors = {
        "refused": ConnectionRefusedError(),
        "dns": socket.gaierror(),
        "httpx_connect": httpx.ConnectError("not connected"),
        "read": httpx.ReadTimeout("ambiguous"),
    }
    wrapped = RuntimeError("SDK wrapper")
    wrapped.__cause__ = errors["httpx_connect"]
    errors["wrapped"] = wrapped
    with pytest.raises(EndpointLimitError if kind == "read" else type(errors[kind])), limit.permit():
        raise errors[kind]
    assert len(server.active) == (1 if kind == "read" else 0)


def test_authenticated_status_and_drained_only_admin_reset(coordinator):
    import json
    from urllib.request import Request, urlopen

    server, url = coordinator
    limit = SharedEndpointLimit(url, "qwen", "LEROBOT_ENDPOINT_LIMIT_TOKEN")
    with pytest.raises(EndpointLimitError), limit.permit():
        raise TimeoutError()

    def post(path, token, **body):
        request = Request(
            url + path,
            data=json.dumps({"key": "qwen", **body}).encode(),
            headers={"Authorization": "Bearer " + token},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    assert len(post("/status", "test-only-secret")["active"]) == 1
    for token, body, status in [
        ("wrong", {}, 401),
        ("test-only-secret", {"drained": True}, 401),
        ("test-admin", {}, 400),
    ]:
        with pytest.raises(HTTPError) as exc:
            post("/reset", token, **body)
        assert exc.value.code == status and server.active
    assert post("/reset", "test-admin", drained=True)["active"] == []
    assert not server.active
