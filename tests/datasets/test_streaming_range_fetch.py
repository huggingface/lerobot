# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exercise signed-URL expiry through real HTTP response handling."""

import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from lerobot.streaming.range_fetch import NativeHTTPRangeFetcher


@pytest.fixture
def fetcher(monkeypatch):
    reader = NativeHTTPRangeFetcher("hf://datasets/test/data@revision", token=False, max_retries=0)
    reader.client.close()
    reader._source_urls["video.mp4"] = "https://huggingface.co/datasets/test/data/resolve/revision/video.mp4"
    monkeypatch.setattr(reader.api, "_build_hf_headers", lambda: {"authorization": "Bearer test-token"})
    yield reader
    reader.close()


@pytest.mark.parametrize("expired_status", [401, 403])
def test_expired_signed_url_refreshes_without_losing_range_or_leaking_token(fetcher, expired_status):
    requests = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "HEAD":
            assert request.headers["authorization"] == "Bearer test-token"
            # A cached redirect remains expired until the source lookup bypasses it.
            signature = "fresh" if request.url.query else "expired"
            return httpx.Response(302, headers={"Location": f"https://cdn.example/video?sig={signature}"})
        assert "authorization" not in request.headers
        assert request.headers["range"] == "bytes=3-5"
        if request.url.params["sig"] == "expired":
            return httpx.Response(expired_status)
        return httpx.Response(206, content=b"345")

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    assert fetcher.read_range("video.mp4", 3, 3) == b"345"
    assert fetcher.read_range("video.mp4", 3, 3) == b"345"
    assert [r.method for r in requests] == ["HEAD", "GET", "HEAD", "GET", "GET"]
    summary = fetcher.timing_summary()
    assert summary["range_jobs"] == 2
    assert summary["range_bytes"] == 6
    assert summary["range_url_refreshes"] == 1


@pytest.mark.parametrize("status", [401, 403])
def test_permanent_signed_url_denial_has_one_refresh(fetcher, status):
    methods = []

    def serve(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(302, headers={"Location": "https://cdn.example/video?sig=denied"})
        return httpx.Response(status)

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    with pytest.raises(PermissionError, match=f"{status} after URL refresh"):
        fetcher.read_range("video.mp4", 0, 1)
    assert methods == ["HEAD", "GET", "HEAD", "GET"]
    assert fetcher.timing_summary()["range_failed_requests"] == 1


def test_hub_authorization_denial_is_not_retried_as_signed_url_expiry(fetcher):
    requests = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401)

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    with pytest.raises(HfHubHTTPError):
        fetcher.read_range("video.mp4", 0, 1)
    assert len(requests) == 1


def test_concurrent_expired_ranges_preserve_each_payload(fetcher):
    barrier = threading.Barrier(4)
    fetcher._resolved_urls["video.mp4"] = "https://cdn.example/video?sig=expired"

    def serve(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            assert request.url.query
            return httpx.Response(302, headers={"Location": "https://cdn.example/video?sig=fresh"})
        assert "authorization" not in request.headers
        if request.url.params["sig"] == "expired":
            barrier.wait(timeout=5)
            return httpx.Response(401)
        start, stop = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=b"0123456789"[start : stop + 1])

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetcher.read_range, "video.mp4", i, 2) for i in range(4)]
        assert [future.result() for future in futures] == [b"01", b"12", b"23", b"34"]
