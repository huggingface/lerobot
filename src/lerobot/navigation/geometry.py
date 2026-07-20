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

"""Monocular geometry runners for the mapping pipeline.

A :class:`GeometryRunner` turns a stack of RGB views into per-pixel world
points, camera-frame points (depth), confidence, and camera-to-world
poses — the four arrays the voxel-map pipeline consumes.
:class:`LingBotMapRunner` wraps Ant Group's streaming LingBot-Map model
(feed-forward 3D reconstruction with persistent memory); the SDK import
is lazy so configs/tests/``--help`` don't pay the model cost.

Because LingBot-Map is monocular, its world frame has an unknown metric
scale. On a robot with wheel/leg odometry (the Unitree Go2 sport-mode
state), :func:`align_trajectory_to_odometry` fits a similarity transform
(scale + rotation + translation) from the model's camera trajectory to
the odometry trajectory, so the voxel map comes out metric and A* speeds
are real m/s. :class:`FakeGeometryRunner` produces deterministic planar
geometry for hardware-free tests.
"""

# ruff: noqa: N806  — R, U, S, Vt, D: conventional linear-algebra / array-dimension names
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

LOG = logging.getLogger(__name__)

DEFAULT_LINGBOT_CHECKPOINT = "robbyant/lingbot-map"


@dataclass(frozen=True)
class GeometryOutput:
    """Per-view geometry outputs (fp32, on CPU).

    The contract every :class:`GeometryRunner` emits and the voxel-map
    pipeline consumes.
    """

    points: np.ndarray  # (N, H, W, 3) world points
    local_points: np.ndarray  # (N, H, W, 3) camera-frame points; depth = [..., 2]
    conf: np.ndarray  # (N, H, W) in [0, 1]
    camera_poses: np.ndarray  # (N, 4, 4) camera-to-world, OpenCV convention


@runtime_checkable
class GeometryRunner(Protocol):
    """Turns ``(N, H, W, 3)`` uint8 RGB views into a :class:`GeometryOutput`."""

    def __call__(self, views_rgb_uint8: np.ndarray) -> GeometryOutput: ...


def _select_autocast(device: str) -> tuple[Any, str]:
    """Return (autocast context manager, label for logging)."""
    import torch

    if device != "cuda":
        return nullcontext(), "no-autocast"
    if not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype), f"cuda/{str(dtype).split('.')[-1]}"


class LingBotMapRunner:
    """Lazy-loaded LingBot-Map streaming reconstruction runner.

    Streaming feed-forward reconstruction with a persistent KV-cache keeps
    every view anchored to one consistent world frame — so, unlike
    window-based models, no cross-window pose stitching is needed. The
    model download/load is deferred to the first call.
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = DEFAULT_LINGBOT_CHECKPOINT,
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from lingbot_map import LingBotMap  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"lingbot-map is not importable ({exc}). Install it from "
                "github.com/robbyant/lingbot-map on the GPU host."
            ) from exc
        LOG.info("loading LingBot-Map (%s) on %s ...", self.checkpoint, self.device)
        self._model = LingBotMap.from_pretrained(self.checkpoint).to(self.device).eval()
        LOG.info("LingBot-Map loaded")

    def __call__(self, views_rgb_uint8: np.ndarray) -> GeometryOutput:
        import torch

        if views_rgb_uint8.ndim != 4 or views_rgb_uint8.shape[-1] != 3:
            raise ValueError(f"expected (N, H, W, 3), got {views_rgb_uint8.shape}")
        if views_rgb_uint8.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {views_rgb_uint8.dtype}")
        self._ensure_loaded()
        assert self._model is not None

        imgs = (
            torch.from_numpy(views_rgb_uint8)
            .to(self.device)
            .float()
            .div_(255.0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (N, 3, H, W)

        autocast_ctx, label = _select_autocast(self.device)
        LOG.info("LingBot-Map forward: N=%d, %s", views_rgb_uint8.shape[0], label)
        with torch.no_grad(), autocast_ctx:
            res = self._model(imgs[None])  # (1, N, ...)

        def _np(t) -> np.ndarray:
            return t.detach().float().cpu().numpy()

        points = _np(res["points"][0])
        local_points = _np(res["local_points"][0])
        conf = _np(res["conf"][0])
        if conf.ndim == 4:  # (N, H, W, 1) → (N, H, W)
            conf = conf[..., 0]
        camera_poses = _np(res["camera_poses"][0])
        return GeometryOutput(points, local_points, conf, camera_poses)


class FakeGeometryRunner:
    """Deterministic planar geometry for hardware-free tests.

    Emits a flat floor at ``depth`` metres in front of the camera with a
    pinhole model, unit confidence, and identity (or supplied) poses — no
    model required.
    """

    def __init__(self, depth: float = 3.0, focal_px: float = 100.0) -> None:
        self.depth = float(depth)
        self.focal_px = float(focal_px)

    def __call__(self, views_rgb_uint8: np.ndarray) -> GeometryOutput:
        if views_rgb_uint8.ndim != 4 or views_rgb_uint8.shape[-1] != 3:
            raise ValueError(f"expected (N, H, W, 3), got {views_rgb_uint8.shape}")
        n, h, w, _ = views_rgb_uint8.shape
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        x = (us - cx) * self.depth / self.focal_px
        y = (vs - cy) * self.depth / self.focal_px
        z = np.full_like(x, self.depth, dtype=np.float64)
        local = np.stack([x, y, z], axis=-1).astype(np.float32)  # (H, W, 3)
        local_points = np.broadcast_to(local, (n, h, w, 3)).copy()
        # Identity poses → world == camera frame.
        points = local_points.copy()
        conf = np.ones((n, h, w), dtype=np.float32)
        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (n, 4, 4)).copy()
        return GeometryOutput(points, local_points, conf, poses)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale s, rotation R, translation t) mapping
    ``src`` onto ``dst`` such that ``dst ≈ s · R @ src + t``.

    ``src``/``dst`` are ``(K, 3)``. Returns ``(s, R, t)``. Used to anchor a
    monocular trajectory to metric odometry.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be matching (K, 3); got {src.shape}, {dst.shape}")
    k = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    sc = src - mu_src
    dc = dst - mu_dst
    cov = (dc.T @ sc) / k
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (sc**2).sum() / k
    s = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    t = mu_dst - s * R @ mu_src
    return s, R, t


def align_trajectory_to_odometry(
    camera_positions: np.ndarray,
    odom_positions: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit the similarity transform from a monocular camera trajectory to a
    metric odometry trajectory (both ``(K, 3)``, time-aligned).

    Returns ``(scale, R, t)`` to apply to model world points/poses so the
    voxel map is metric. Needs at least 3 non-degenerate points.
    """
    if camera_positions.shape[0] < 3:
        raise ValueError("need at least 3 corresponding poses to fit a similarity")
    return umeyama_similarity(camera_positions, odom_positions)
