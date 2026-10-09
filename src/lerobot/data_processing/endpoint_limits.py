# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Authenticated, cross-process/host admission for one shared model endpoint.

No expiring leases: losing a worker must not free a slot while inference might
still run. Drain/cancel inference before restarting this coordinator. Run behind
TLS or on a trusted private network; secrets are resolved from the environment.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .errors import ErrorCategory, TemporaryServiceError, classify_error


class EndpointLimitServer(ThreadingHTTPServer):
    """One authority per endpoint/model key, shared by all pipelines and workers."""

    daemon_threads = True

    def __init__(self, address, *, key: str, limit: int, token: str):
        if limit < 1 or not key or not token:
            raise ValueError("Endpoint coordinator requires a key, positive limit and secret token")
        self.key, self.limit, self.token = key, limit, token
        self.active: set[str] = set()
        self.released: deque[str] = deque(maxlen=10000)
        self.lock = threading.Lock()
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # Neither authorization headers nor request bodies go into logs.

    def do_POST(self):
        server = self.server
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + server.token):
            self.send_error(401)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid body size")
            value = json.loads(self.rfile.read(size))
            request_id = value["request_id"]
            uuid.UUID(request_id)
            if value["key"] != server.key or self.path not in {"/acquire", "/release"}:
                raise ValueError("Unknown endpoint key or operation")
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        with server.lock:
            if self.path == "/release":
                server.active.discard(request_id)
                server.released.append(request_id)
                status = 200
            elif request_id in server.released:
                status = 410
            elif request_id in server.active:
                status = 200  # Idempotent retry after a lost acquisition response.
            elif len(server.active) >= server.limit:
                status = 429
            else:
                server.active.add(request_id)
                status = 200
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()


class SharedEndpointLimit:
    def __init__(self, url: str, key: str, token_env: str, timeout: float = 300):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not key
            or timeout <= 0
        ):
            raise ValueError("Invalid shared endpoint coordinator configuration")
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.token = os.environ[token_env]
        if not self.token:
            raise ValueError("Shared endpoint coordinator token is empty")

    def _post(self, operation, request_id, timeout=10):
        request = Request(
            self.url + "/" + operation,
            data=json.dumps({"key": self.key, "request_id": request_id}).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        # URLs are restricted to HTTP(S) above; never forward authorization on
        # redirects to an unrelated host or another protocol.
        with build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            return response.status

    def _release(self, request_id):
        for attempt in range(3):
            try:
                self._post("release", request_id)
                return
            except (URLError, TimeoutError, ConnectionError) as exc:
                if isinstance(exc, HTTPError) and classify_error(exc) != ErrorCategory.SERVICE:
                    raise
                if attempt == 2:
                    raise TemporaryServiceError(
                        "Cannot release shared endpoint permit; admission remains closed"
                    ) from None
                time.sleep(0.1 * (2**attempt))

    @contextmanager
    def permit(self):
        request_id = str(uuid.uuid4())
        deadline = time.monotonic() + self.timeout
        acquired = False
        delay = 0.05
        try:
            while not acquired:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for shared endpoint admission")
                try:
                    self._post("acquire", request_id, timeout=min(10, remaining))
                    acquired = True
                except HTTPError as exc:
                    if exc.code != 429:
                        raise
                except (URLError, TimeoutError, ConnectionError):
                    pass  # Fail closed; never send inference without a permit.
                if not acquired:
                    # Avoid hundreds of waiting workers polling in lockstep.
                    jitter = 0.8 + int(request_id[:2], 16) / 640
                    time.sleep(min(delay * jitter, max(0, deadline - time.monotonic())))
                    delay = min(2, delay * 2)
        finally:
            if not acquired:
                self._release(request_id)  # No inference was sent; ambiguous acquisitions are safe to clear.
        try:
            yield
        except Exception as exc:
            if classify_error(exc) != ErrorCategory.NETWORK:
                self._release(request_id)
            # Network timeouts may leave inference running. Preserve that slot
            # until the operator has drained/cancelled the endpoint, not by TTL.
            raise
        else:
            self._release(request_id)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise HTTPError(request.full_url, code, "Coordinator redirects are not permitted", headers, fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--key", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--token-env", default="LEROBOT_ENDPOINT_LIMIT_TOKEN")
    args = parser.parse_args()
    with EndpointLimitServer(
        (args.host, args.port), key=args.key, limit=args.limit, token=os.environ[args.token_env]
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
