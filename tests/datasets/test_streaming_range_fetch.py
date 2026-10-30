# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exercise signed-URL expiry through real HTTP response handling."""

import gc
import threading
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from lerobot.streaming.range_fetch import NativeHTTPRangeFetcher


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeHTTPRangeFetcher]:
    reader = NativeHTTPRangeFetcher("hf://datasets/test/data@revision", token=False, max_retries=0)
    reader.client.close()
    reader._source_urls["video.mp4"] = "https://huggingface.co/datasets/test/data/resolve/revision/video.mp4"
    monkeypatch.setattr(reader.api, "_build_hf_headers", lambda: {"authorization": "Bearer test-token"})
    yield reader
    reader.close()


@pytest.mark.parametrize("linked_size", [None, "123"])
def test_redirect_content_length_is_not_the_video_size(
    fetcher: NativeHTTPRangeFetcher, linked_size: str | None
) -> None:
    """A redirect body's length must not poison MP4 header probe sizes."""
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "huggingface.co":
            headers = {"Location": "https://cdn.example/video", "Content-Length": "0"}
            if linked_size is not None:
                headers["X-Linked-Size"] = linked_size
            return httpx.Response(302, headers=headers)
        assert "authorization" not in request.headers
        return httpx.Response(200, headers={"Content-Length": "123"})

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    # Range fetching may resolve the URL before a later header probe asks for size.
    fetcher._resolve_url("video.mp4")
    assert fetcher.info_size("video.mp4") == 123
    assert fetcher.info_size("video.mp4") == 123
    assert len(requests) == (2 if linked_size is None else 1)


@pytest.mark.parametrize("redirect", [False, True])
def test_info_size_uses_advertised_object_size_without_duplicate_head(
    fetcher: NativeHTTPRangeFetcher, redirect: bool
) -> None:
    """Reuse object size learned during resolution, including direct small Hub files."""
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if redirect and request.url.host == "huggingface.co":
            return httpx.Response(
                302, headers={"Location": "https://cdn.example/video", "X-Linked-Size": "123"}
            )
        return httpx.Response(200, headers={"Content-Length": "123"})

    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    assert fetcher.info_size("video.mp4") == 123
    assert len(requests) == 1


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("exhaust", [False, True])
@pytest.mark.parametrize("operation", ["resolve", "refresh", "metadata"])
def test_head_status_retry_budget(
    fetcher: NativeHTTPRangeFetcher,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    exhaust: bool,
    operation: str,
) -> None:
    """Resolve, refresh and metadata HEADs share bounded retries and close responses."""
    fetcher.max_retries = 2
    responses: list[httpx.Response] = []
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    if operation == "metadata":
        fetcher._resolved_urls["video.mp4"] = "https://cdn.example/video"
    elif operation == "refresh":
        fetcher._resolved_urls["video.mp4"] = "https://cdn.example/video?sig=expired"

    def sleep(seconds: float) -> None:
        assert responses[-1].is_closed
        sleeps.append(seconds)

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if operation == "refresh" and request.method == "GET" and "expired" in str(request.url):
            return httpx.Response(403)
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(206, content=b"345")
        assert request.method == "HEAD"
        if operation == "metadata":
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == "Bearer test-token"
        if operation == "refresh":
            assert "_refresh" in request.url.params
        if exhaust or len(responses) < 2:
            response = httpx.Response(status)
        elif operation == "metadata":
            response = httpx.Response(200, headers={"Content-Length": "123"})
        else:
            response = httpx.Response(302, headers={"Location": "https://cdn.example/video?sig=fresh"})
        responses.append(response)
        return response

    monkeypatch.setattr("lerobot.streaming.range_fetch.time.sleep", sleep)
    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    if exhaust:
        with pytest.raises(HfHubHTTPError):
            if operation == "metadata":
                fetcher.info_size("video.mp4")
            else:
                fetcher.read_range("video.mp4", 3, 3)
    elif operation == "metadata":
        assert fetcher.info_size("video.mp4") == 123
    else:
        assert fetcher.read_range("video.mp4", 3, 3) == b"345"
    assert len(responses) == 3
    assert all(response.is_closed for response in responses)
    assert sleeps == [0.5, 1.0]
    assert sum(request.method == "GET" for request in requests) == (
        int(operation == "refresh") + int(not exhaust and operation != "metadata")
    )


@pytest.mark.parametrize("error_type", [httpx.ConnectTimeout, httpx.ReadError])
@pytest.mark.parametrize("exhaust", [False, True])
def test_head_transport_retries_are_bounded(
    fetcher: NativeHTTPRangeFetcher,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[httpx.TransportError],
    exhaust: bool,
) -> None:
    """Transport retries remain bounded and back off before each new attempt."""
    fetcher.max_retries = 2
    heads: list[httpx.Request] = []
    sleeps: list[float] = []

    def serve(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            heads.append(request)
            if exhaust or len(heads) < 3:
                raise error_type("transient HEAD failure", request=request)
            return httpx.Response(302, headers={"Location": "https://cdn.example/video"})
        return httpx.Response(206, content=b"345")

    monkeypatch.setattr("lerobot.streaming.range_fetch.time.sleep", sleeps.append)
    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    if exhaust:
        with pytest.raises(error_type, match="transient HEAD failure"):
            fetcher.read_range("video.mp4", 3, 3)
    else:
        assert fetcher.read_range("video.mp4", 3, 3) == b"345"
    assert len(heads) == 3
    assert sleeps == [0.5, 1.0]


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


@pytest.mark.parametrize("status", [401, 403])
def test_hub_authorization_denial_is_not_retried_as_signed_url_expiry(
    fetcher: NativeHTTPRangeFetcher, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Permanent Hub denials fail immediately even when retries are enabled."""
    fetcher.max_retries = 2
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    sleeps: list[float] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = httpx.Response(status)
        responses.append(response)
        return response

    monkeypatch.setattr("lerobot.streaming.range_fetch.time.sleep", sleeps.append)
    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    with pytest.raises(HfHubHTTPError):
        fetcher.read_range("video.mp4", 0, 1)
    assert len(requests) == 1
    assert responses[0].is_closed
    assert sleeps == []


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


@pytest.mark.parametrize("parts", [1, 4])
def test_finished_ranges_do_not_retain_video_bodies_until_gc(
    fetcher: NativeHTTPRangeFetcher, parts: int
) -> None:
    """Closed HTTP response cycles must not retain a second copy of fetched video bytes."""
    responses: list[weakref.ReferenceType[httpx.Response]] = []
    chunk_size = 1024 * 1024

    class VideoStream(httpx.SyncByteStream):
        def __init__(self, start: int, stop: int) -> None:
            self.start = start
            self.stop = stop

        def __iter__(self) -> Iterator[bytes]:
            first_size = max(0, min(self.stop + 1, chunk_size) - self.start)
            if first_size:
                yield b"a" * first_size
            second_size = self.stop + 1 - self.start - first_size
            if second_size:
                yield b"b" * second_size

    def serve(request: httpx.Request) -> httpx.Response:
        start, stop = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        response = httpx.Response(206, stream=VideoStream(start, stop), request=request)
        responses.append(weakref.ref(response))
        return response

    fetcher.subrange_parts = parts
    fetcher.subrange_min_bytes = 1
    if parts > 1:
        fetcher._subrange_pool = ThreadPoolExecutor(max_workers=parts)
    fetcher._resolved_urls["video.mp4"] = "https://cdn.example/video"
    fetcher.client = httpx.Client(transport=httpx.MockTransport(serve))
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(8):
            payload = fetcher.read_range("video.mp4", 0, 2 * chunk_size)
            assert len(payload) == 2 * chunk_size
            assert payload[:chunk_size] == b"a" * chunk_size
            assert payload[chunk_size:] == b"b" * chunk_size
            del payload
        # HTTPX response/stream cycles may survive until GC. They should retain
        # only response metadata, not megabytes of already delivered video.
        retained_body_bytes = 0
        for reference in responses:
            response = reference()
            if response is not None:
                assert response.is_closed
                with suppress(httpx.ResponseNotRead):
                    retained_body_bytes += len(response.content)
        assert retained_body_bytes == 0
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()
