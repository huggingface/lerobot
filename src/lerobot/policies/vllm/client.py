"""Minimal OpenPI WebSocket client for a remote vLLM policy server.

Protocol (e.g. vllm-omni/examples/online_serving/gr00t):
  1. connect to ``ws://host:port/v1/realtime/robot/openpi``
  2. server sends msgpack-numpy ``PolicyServerConfig`` metadata
  3. client sends one msgpack-numpy observation dict
  4. server responds with ``dict[action_key -> ndarray(action_horizon, dim)]``
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import msgpack
import numpy as np

logger = logging.getLogger(__name__)


# --- Vendored OpenPI msgpack-numpy serializer ---------------------------------
# Wire-compatible with the server's `openpi_client.msgpack_numpy` (the
# `__ndarray__`/`__npgeneric__` envelope). NOTE: this is a DIFFERENT format from the
# standalone PyPI `msgpack-numpy` package (`nd`/`type`/`kind`/`data`), which the server
# cannot decode. Vendoring avoids depending on `openpi-client`, which pins numpy<2 and
# would conflict with lerobot (numpy>=2). Adapted from openpi-client (Apache-2.0).


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class _MsgpackNumpy:
    packb = staticmethod(functools.partial(msgpack.packb, default=_pack_array))
    unpackb = staticmethod(functools.partial(msgpack.unpackb, object_hook=_unpack_array))


def _import_msgpack_numpy():
    """Return the OpenPI-format serializer (prefer the real package if present)."""
    try:
        from openpi_client import msgpack_numpy  # type: ignore

        return msgpack_numpy
    except Exception:
        return _MsgpackNumpy


class OpenPIClient:
    """One websocket round-trip per inference call (stateless, reconnect each call).

    Reconnecting per call keeps the implementation simple and robust for batched eval
    where each env may be queried independently; the server tracks state by
    ``session_id`` carried in the observation.
    """

    def __init__(self, url: str, connect_timeout_s: float = 30.0, max_msg_bytes: int = 64 * 1024 * 1024):
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.max_msg_bytes = max_msg_bytes
        self._packer = _import_msgpack_numpy()
        try:
            import websockets.sync.client  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The vllm policy requires `websockets`. Install with `pip install websockets`."
            ) from exc

    def infer(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Send one raw observation; return ``(actions_dict, server_metadata)``."""
        import websockets.sync.client

        with websockets.sync.client.connect(
            self.url,
            open_timeout=self.connect_timeout_s,
            max_size=self.max_msg_bytes,
        ) as conn:
            metadata = self._packer.unpackb(conn.recv())  # server handshake metadata
            conn.send(self._packer.packb(observation))
            actions = self._packer.unpackb(conn.recv())

        if not isinstance(actions, dict):
            raise RuntimeError(
                f"vLLM OpenPI server must return a dict of ndarrays; got {type(actions).__name__}"
            )
        # The server returns {"type": "error", "message": ...} on failure.
        if actions.get("type") == "error":
            raise RuntimeError(f"vLLM server error: {actions.get('message', actions)}")
        return {k: np.asarray(v, dtype=np.float32) for k, v in actions.items()}, metadata

    def handshake(self) -> dict[str, Any]:
        """Connect and return just the server metadata (used to size images, etc.)."""
        import websockets.sync.client

        with websockets.sync.client.connect(
            self.url, open_timeout=self.connect_timeout_s, max_size=self.max_msg_bytes
        ) as conn:
            return self._packer.unpackb(conn.recv())
