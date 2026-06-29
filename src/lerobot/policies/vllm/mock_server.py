"""A minimal mock OpenPI WebSocket server for local (no-GPU) smoke testing.

It mimics the vllm-omni protocol:
  1. on connect, send msgpack-numpy ``PolicyServerConfig`` metadata,
  2. for each received observation, return a dict of dummy no-op action trajectories.

This lets you exercise the full lerobot-eval -> vllm -> LIBERO loop without a GPU
or the real model.

Run:
    python -m lerobot.policies.vllm.mock_server --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .client import _import_msgpack_numpy

logger = logging.getLogger("vllm-mock-server")

IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


LIBERO_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _make_metadata(action_horizon: int, image_hw: tuple[int, int], action_space: str) -> dict:
    keys = LIBERO_KEYS if action_space == "libero_7d" else ["eef_9d", "gripper_position", "joint_position"]
    return {
        "action_horizon": action_horizon,
        "action_keys": keys,
        "image_resolution": [int(image_hw[0]), int(image_hw[1])],
        "n_external_cameras": 1,
        "needs_wrist_camera": True,
        "needs_session_id": True,
    }


def _make_actions(obs: dict, action_horizon: int, joint_dim: int, action_space: str) -> dict[str, np.ndarray]:
    """Return plausible, finite no-op actions for smoke testing."""
    state = obs.get("state", {}) if isinstance(obs, dict) else {}
    if action_space == "libero_7d":
        # Zero pose deltas (no motion in relative mode); hold the observed gripper.
        grip = np.asarray(state.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
        g = float(grip[0]) if grip.size else 0.0
        out = {k: np.zeros((action_horizon, 1), dtype=np.float32) for k in LIBERO_KEYS}
        out["gripper"] = np.full((action_horizon, 1), g, dtype=np.float32)
        return out

    eef_9d = np.asarray(state.get("eef_9d", [0.0] * 9), dtype=np.float32).reshape(-1)
    if eef_9d.shape[0] < 9:
        eef_9d = np.concatenate([np.zeros(3, dtype=np.float32), IDENTITY_ROT6D])
    gripper = np.asarray(state.get("gripper_position", [0.5]), dtype=np.float32).reshape(-1)[:1]
    joints = np.asarray(state.get("joint_position", [0.0] * joint_dim), dtype=np.float32).reshape(-1)
    if joints.shape[0] < joint_dim:
        joints = np.zeros(joint_dim, dtype=np.float32)
    return {
        "eef_9d": np.tile(eef_9d[None, :9], (action_horizon, 1)).astype(np.float32),
        "gripper_position": np.tile(gripper[None, :1], (action_horizon, 1)).astype(np.float32),
        "joint_position": np.tile(joints[None, :joint_dim], (action_horizon, 1)).astype(np.float32),
    }


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    action_horizon: int = 16,
    joint_dim: int = 7,
    action_space: str = "libero_7d",
) -> None:
    from websockets.sync.server import serve as ws_serve

    packer = _import_msgpack_numpy()
    metadata = _make_metadata(action_horizon, (256, 256), action_space)

    def handler(conn):
        peer = getattr(conn, "remote_address", "?")
        logger.info("client connected: %s", peer)
        conn.send(packer.packb(metadata))
        for raw in conn:
            obs = packer.unpackb(raw)
            actions = _make_actions(obs, action_horizon, joint_dim, action_space)
            conn.send(packer.packb(actions))
        logger.info("client disconnected: %s", peer)

    logger.info("Mock vLLM OpenPI server listening on ws://%s:%d/v1/realtime/robot/openpi", host, port)
    with ws_serve(handler, host, port, max_size=64 * 1024 * 1024) as server:
        server.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--joint-dim", type=int, default=7)
    p.add_argument("--action-space", default="libero_7d", choices=["libero_7d", "eef_9d"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(args.host, args.port, args.action_horizon, args.joint_dim, args.action_space)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
