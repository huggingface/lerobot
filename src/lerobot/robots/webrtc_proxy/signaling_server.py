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

"""WebSocket signaling relay: pairs a robot daemon with a cloud controller.

A session has two roles — ``robot`` (the robot daemon) and ``controller`` (the cloud
``WebRTCProxyRobot``). Each connects to ``GET /ws?session=<id>&role=<role>`` and the
relay forwards every message to the *other* role, buffering until that peer joins
(so the daemon may offer before a controller exists). When one side drops, the
relay sends ``{"kind":"bye"}`` to the survivor so the daemon can reset and loop.

This is the control-plane signaler for the **aiortc** (direct-UDP) backend. It carries
SDP plus, on connect, a small **ICE config** message (STUN urls) so peers can form a
server-reflexive candidate and connect directly across NAT — it never carries media.
With no STUN configured it hands out an empty list (host-candidate-only / LAN). aiortc
is direct-only; for a media relay (both ends behind restrictive NAT) use the LiveKit
backend instead (DESIGN §11.1). In production this runs as an ordinary (non-hostNetwork)
Deployment behind a normal Service/Ingress.

Run standalone:
    python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765 \
        --stun-url stun:stun.l.google.com:19302   # or a domestic STUN, or self-hosted
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import secrets
import socket
from collections import defaultdict

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

_OTHER = {"robot": "controller", "controller": "robot"}


def _split_env(name: str) -> list[str]:
    """Comma-separated env var -> list (for argparse defaults). Empty/unset -> []."""
    return [u.strip() for u in os.environ.get(name, "").split(",") if u.strip()]


class IceConfig:
    """Builds the STUN-server list the relay hands to each peer on connect.

    aiortc is the **direct-UDP** backend: host candidates connect on a LAN, and STUN adds
    a server-reflexive (public) candidate so peers can also connect directly across NAT
    when at least one side is reachable. STUN only *discovers* the public address — media
    still flows peer-to-peer, never through the relay. With no STUN configured this yields
    ``[]`` (host-candidate-only / LAN). For a genuine media **relay** (both ends behind
    restrictive NAT), use the LiveKit backend — we deliberately don't run a TURN relay
    under aiortc (see DESIGN §11.1).
    """

    def __init__(self, stun_urls: list[str] | None = None) -> None:
        self._stun = list(stun_urls or [])

    @property
    def enabled(self) -> bool:
        return bool(self._stun)

    def for_peer(self, name: str) -> list[dict]:
        return [{"urls": self._stun}] if self._stun else []


class SignalingRelay:
    """Forwards SDP between the two roles of each session, buffering for late joiners.

    A public relay must gate the door: with ``auth_token`` set, every WebSocket must
    present a matching token (``Authorization: Bearer <token>`` header, or ``?token=``
    query as a fallback) or it is rejected before pairing. NOTE: a single shared token
    only stops random scanners — it does NOT isolate sessions/tenants (anyone holding
    it can join any session_id). Multi-tenant deployments must move to per-session
    signed tokens validated here (DESIGN.md §12).
    """

    def __init__(self, auth_token: str | None = None, ice: IceConfig | None = None) -> None:
        self._auth_token = auth_token
        self._ice = ice or IceConfig()
        self._peers: dict[tuple[str, str], web.WebSocketResponse] = {}
        self._inbox: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def _authorized(self, request: web.Request) -> bool:
        if not self._auth_token:
            return True
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else request.query.get("token", "")
        return secrets.compare_digest(token, self._auth_token)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.Response(status=401, text="unauthorized")
        session = request.query.get("session", "default")
        role = request.query.get("role", "")
        if role not in _OTHER:
            return web.Response(status=400, text="role must be 'robot' or 'controller'")
        other = _OTHER[role]

        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._peers[(session, role)] = ws
        logger.info("signaling: %s joined session %s", role, session)

        # Hand out ICE config (STUN servers) FIRST, so the peer can build its
        # RTCPeerConnection with it before any SDP arrives. Always sent (empty list when no
        # STUN is configured) so the client protocol is uniform.
        await ws.send_json({"kind": "ice", "iceServers": self._ice.for_peer(f"{session}:{role}")})

        # Flush anything buffered for me before I joined.
        for msg in self._inbox.pop((session, role), []):
            await ws.send_json(msg)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = msg.json()
                    target = self._peers.get((session, other))
                    if target is not None and not target.closed:
                        await target.send_json(data)
                    else:
                        self._inbox[(session, other)].append(data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._peers.pop((session, role), None)
            peer = self._peers.get((session, other))
            if peer is not None and not peer.closed:
                await peer.send_json({"kind": "bye"})
            logger.info("signaling: %s left session %s", role, session)
        return ws


def make_app(auth_token: str | None = None, ice: IceConfig | None = None) -> web.Application:
    relay = SignalingRelay(auth_token=auth_token, ice=ice)
    app = web.Application()
    app.router.add_get("/ws", relay.handle)
    return app


async def start_relay(
    host: str = "127.0.0.1", port: int = 0, auth_token: str | None = None, ice: IceConfig | None = None
) -> tuple[web.AppRunner, int]:
    """Start the relay on ``host:port`` (port 0 => ephemeral). Returns (runner, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]
    runner = web.AppRunner(make_app(auth_token=auth_token, ice=ice))
    await runner.setup()
    await web.SockSite(runner, sock).start()
    logger.info(
        "signaling relay listening on ws://%s:%d/ws (auth=%s, ice=%s)",
        host,
        actual_port,
        bool(auth_token),
        (ice or IceConfig()).enabled,
    )
    return runner, actual_port


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTCProxyRobot WebSocket signaling relay")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 (server binds all ifaces by design)
    # FaaS web-app runtimes inject the listen port via $PORT; fall back to 8765 locally.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("SIGNALING_AUTH_TOKEN"),
        help="shared token every peer must present (Authorization: Bearer ...). Public relays should set it.",
    )
    parser.add_argument(
        "--stun-url",
        action="append",
        default=_split_env("STUN_URLS"),
        help="STUN url handed to peers for direct cross-NAT P2P (repeatable; or $STUN_URLS "
        "comma-separated). e.g. stun:stun.l.google.com:19302 (domestic: stun:stun.qq.com:3478). "
        "For a media relay across hard NATs, use the LiveKit backend instead.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ice = IceConfig(stun_urls=args.stun_url)

    async def _run() -> None:
        runner, _ = await start_relay(args.host, args.port, auth_token=args.auth_token, ice=ice)
        try:
            await asyncio.Event().wait()  # run forever
        finally:
            await runner.cleanup()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
