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
import logging
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

from .errors import EndpointLimitError, ErrorCategory, TemporaryServiceError, classify_error, request_not_sent

logger = logging.getLogger(__name__)


class EndpointLimitServer(ThreadingHTTPServer):
    """One authority per endpoint/model key, shared by all pipelines and workers."""

    daemon_threads = True

    def __init__(self, address, *, key: str, limit: int, token: str, admin_token: str | None = None):
        if limit < 1 or not key or not token:
            raise ValueError("Endpoint coordinator requires a key, positive limit and secret token")
        self.key, self.limit, self.token = key, limit, token
        if admin_token and hmac.compare_digest(admin_token, token):
            raise ValueError("Endpoint reset requires a separate administrator token")
        self.admin_token = admin_token
        self.active: set[str] = set()
        self.released: deque[str] = deque(maxlen=10000)
        self.lock = threading.Lock()
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # Neither authorization headers nor request bodies go into logs.

    def do_POST(self):
        server = self.server
        token = server.admin_token if self.path == "/reset" else server.token
        if not token or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
            self.send_error(401)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid body size")
            value = json.loads(self.rfile.read(size))
            if value["key"] != server.key or self.path not in {"/acquire", "/release", "/status", "/reset"}:
                raise ValueError("Unknown endpoint key or operation")
            if self.path in {"/acquire", "/release"}:
                request_id = value["request_id"]
                if not isinstance(request_id, str):
                    raise ValueError("Invalid request ID")
                uuid.UUID(request_id)
            elif self.path == "/reset" and value.get("drained") is not True:
                raise ValueError("Confirm inference is drained/cancelled before resetting")
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        with server.lock:
            if self.path in {"/status", "/reset"}:
                if self.path == "/reset":
                    server.released.extend(server.active)
                    server.active.clear()
                body = json.dumps(
                    {"key": server.key, "limit": server.limit, "active": sorted(server.active)}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            elif self.path == "/release":
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

    def _try_release(self, request_id):
        try:
            self._release(request_id)
            return True
        except Exception as exc:
            logger.warning(
                "Shared endpoint permit %s may be retained (%s); inspect status and drain/reset if needed",
                request_id,
                type(exc).__name__,
            )
            return False

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
        except BaseException as exc:
            self._try_release(request_id)  # No inference was sent; safe to clear ambiguous acquisitions.
            if not isinstance(exc, Exception):
                raise
            raise EndpointLimitError(
                "Shared endpoint admission failed; inference was not sent and is not retried"
            ) from exc
        try:
            yield
        except Exception as exc:
            status = exc.code if isinstance(exc, HTTPError) else getattr(exc, "status_code", None)
            safe_to_release = request_not_sent(exc) or (
                classify_error(exc) != ErrorCategory.NETWORK and status != 504
            )
            if safe_to_release and self._try_release(request_id):
                raise
            # Transport and gateway timeouts may leave upstream inference running.
            # Preserve the slot until the operator drains/cancels the endpoint, not by TTL.
            raise EndpointLimitError(
                "Shared endpoint permit retained; request is not retried. Drain/cancel inference before reset."
            ) from exc
        else:
            self._try_release(request_id)  # A lost release must not discard/resend a successful response.


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
    parser.add_argument(
        "--admin-token-env", help="Separate secret for drained-only /reset; omitted disables reset"
    )
    args = parser.parse_args()
    with EndpointLimitServer(
        (args.host, args.port),
        key=args.key,
        limit=args.limit,
        token=os.environ[args.token_env],
        admin_token=os.environ[args.admin_token_env] if args.admin_token_env else None,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
