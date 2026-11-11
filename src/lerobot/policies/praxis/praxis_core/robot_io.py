# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/robot_io.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Robot input and output helpers of the single-arm rig. No neural networks.

The rig glue lives in :mod:`praxis.single_arm`. This module holds the parts
that do not depend on the rig layout:

* The chunk grid: 15 Hz rows, 24 rows per chunk, 4 raw frames per video
  latent. Chunk 0 of a context window owns the anchor latent and spans 21 raw
  frames. Every later chunk spans 24.
* Rotations and poses: wxyz quaternions, rot6d (the first two columns of the
  rotation matrix) and the 10-D row ``[xyz3, rot6d6, grip1]``.
* The normalization: q99 on the xyz and grip dims (clipped to [-1, 1]),
  mean/std on the proprio, and the inverse maps of the decode.
* The aspect crop of the tile canvas, the observed context stream and the
  action token packing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

FPS = 15.0
ACTION_CHUNK = 24
TEMPORAL_RATIO = 4
FRAMES_PER_TOKEN = 4

# Tiles whose aspect differs from the source by more than this get a centre
# crop to the tile aspect before the resize.
ASPECT_TOL = 0.03

# The gripper range of the rig (0 = open, 1 = closed).
GRIP_RANGE = (0.0, 1.0)


def latent_frames(raw_frames: int, temporal_ratio: int = TEMPORAL_RATIO) -> int:
    """Return the number of video latent frames for ``raw_frames`` raw frames."""
    return (int(raw_frames) - 1) // int(temporal_ratio) + 1


def contiguous_chunk_geometry(action_chunk: int, n_chunks: int) -> list[tuple[int, int]]:
    """Return ``(raw_offset, raw_len)`` per chunk of one contiguous window.

    Chunk 0 spans ``4 * L - 3`` raw frames because its first latent is the
    uncompressed anchor frame. Later chunks span ``4 * L`` raw frames. With
    ``L = 6`` and 4 chunks the window is 93 raw frames with chunk starts at
    0, 21, 45 and 69.
    """
    lat = (action_chunk - 1) // 4 + 1
    out: list[tuple[int, int]] = []
    off = 0
    for k in range(n_chunks):
        ln = 4 * lat - 3 if k == 0 else 4 * lat
        out.append((off, ln))
        off += ln
    return out


# ---------------------------------------------------------------------------
# Rotations and poses
# ---------------------------------------------------------------------------


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` matrices.

    The quaternion is normalized with a 1e-8 guard. This is the convention of
    the proprio and of the training targets.
    """
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def unit_quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert one wxyz quaternion to a float64 3x3 matrix.

    This is the convention of the command path. The quaternion is normalized
    exactly (no guard) and the matrix uses the ``x2 - y2 - z2 + w2`` diagonal
    form. The scalar operation order is fixed, so the result is bitwise
    reproducible.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = (float(v) for v in q)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Found zero norm quaternions in `quat`.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array(
        [
            [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ],
        dtype=np.float64,
    )


def rot6d(r: np.ndarray) -> np.ndarray:
    """Return the first two columns of ``(..., 3, 3)`` matrices as ``(..., 6)``."""
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Convert one rot6d ``[c0(3), c1(3)]`` to a float64 3x3 matrix.

    The two columns are made orthonormal with Gram-Schmidt. The third column
    is their cross product.
    """
    a0: np.ndarray = r6[:3].astype(np.float64)
    a1: np.ndarray = r6[3:6].astype(np.float64)
    c0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1 = a1 - np.dot(c0, a1) * c0
    c1 = a1 / (np.linalg.norm(a1) + 1e-8)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1)


def pose8_to_10d(pose8: np.ndarray, anchor8: np.ndarray | None = None) -> np.ndarray:
    """Convert ``[xyz3, quat_wxyz4, grip1]`` to ``[xyz3, rot6d6, grip1]`` (float32).

    With ``anchor8`` the position is relative to the anchor position and the
    rotation is ``R @ R_anchor^T``. The gripper stays absolute.
    """
    xyz = pose8[..., :3]
    r = quat_wxyz_to_matrix(pose8[..., 3:7])
    grip = pose8[..., 7:8]
    if anchor8 is not None:
        xyz = xyz - anchor8[..., :3]
        r_anchor = quat_wxyz_to_matrix(anchor8[..., 3:7])
        r = np.einsum("...ij,...kj->...ik", r, r_anchor)
    return np.concatenate([xyz, rot6d(r), grip], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormStats:
    """Normalization statistics of the actions and the proprio (float32)."""

    action_q01: np.ndarray
    action_q99: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path) -> NormStats:
        """Load the stats from a JSON file with the four keys of this class."""
        d = json.loads(Path(path).read_text())
        return cls(
            action_q01=np.asarray(d["action_q01"], dtype=np.float32),
            action_q99=np.asarray(d["action_q99"], dtype=np.float32),
            proprio_mean=np.asarray(d["proprio_mean"], dtype=np.float32),
            proprio_std=np.asarray(d["proprio_std"], dtype=np.float32),
        )


def q99_normalize(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """``clip(2 (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)`` on ``dims``."""
    out: np.ndarray = actions.astype(np.float32).copy()
    a: np.ndarray = actions[..., dims].astype(np.float32)
    lo = q01[dims].astype(np.float32)
    hi = q99[dims].astype(np.float32)
    out[..., dims] = np.clip(2.0 * (a - lo) / (hi - lo + np.float32(1e-8)) - 1.0, -1.0, 1.0)
    return out.astype(np.float32)


def q99_denormalize(
    normalized: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    dims: np.ndarray,
) -> np.ndarray:
    """Return ``(n + 1) / 2 * (q99 - q01) + q01`` on ``dims`` (the inverse map)."""
    out: np.ndarray = normalized.astype(np.float32).copy()
    n: np.ndarray = normalized[..., dims].astype(np.float32)
    lo = q01[dims].astype(np.float32)
    hi = q99[dims].astype(np.float32)
    a = (n + 1.0) / 2.0 * (hi - lo) + lo
    out[..., dims] = a
    return out.astype(np.float32)


def clip_normalized(rows: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Return a float32 copy of ``rows`` with ``dims`` clipped to [-1, 1].

    Training clips the q99-normalized action targets to [-1, 1]
    (:func:`q99_normalize`). The sampler can give values
    outside this range. A value in the range does not change.
    """
    out = np.asarray(rows, dtype=np.float32).copy()
    out[..., dims] = np.clip(out[..., dims], -1.0, 1.0)
    return out


def clip_grip(rel: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Return a float32 copy of ``rel`` with the gripper ``dims`` clipped to
    :data:`GRIP_RANGE`. A value in the range does not change."""
    out = np.asarray(rel, dtype=np.float32).copy()
    out[..., dims] = np.clip(out[..., dims], *GRIP_RANGE)
    return out


def denormalize_rows(rows: np.ndarray, stats: NormStats, dims: np.ndarray) -> np.ndarray:
    """Clip the normalized ``dims`` to [-1, 1], then apply the inverse q99 map.

    The result stays in ``[q01, q99]`` on ``dims``.
    """
    return q99_denormalize(clip_normalized(rows, dims), stats.action_q01, stats.action_q99, dims=dims)


def proprio_normalize(proprio: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return ``(p - mean) / max(std, 1e-6)`` in float32."""
    s = np.maximum(std.astype(np.float32), np.float32(1e-6))
    return ((proprio.astype(np.float32) - mean.astype(np.float32)) / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def aspect_crop_box(
    src_h: int, src_w: int, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> tuple[int, int, int, int]:
    """Return the centre crop ``(y0, y1, x0, x1)`` to the tile aspect.

    The box is the full frame when the two aspects agree within ``tol``.
    """
    src_ar, tile_ar = src_w / src_h, tile_w / tile_h
    if abs(tile_ar / src_ar - 1.0) <= tol:
        return 0, src_h, 0, src_w
    if src_ar > tile_ar:
        new_w = round(src_h * tile_ar)
        x0 = (src_w - new_w) // 2
        return 0, src_h, x0, x0 + new_w
    new_h = round(src_w / tile_ar)
    y0 = (src_h - new_h) // 2
    return y0, y0 + new_h, 0, src_w


def aspect_crop(fr: torch.Tensor, tile_h: int, tile_w: int, tol: float = ASPECT_TOL) -> torch.Tensor:
    """Apply :func:`aspect_crop_box` to a ``(..., H, W)`` frame stack."""
    y0, y1, x0, x1 = aspect_crop_box(fr.shape[-2], fr.shape[-1], tile_h, tile_w, tol)
    if (y0, y1, x0, x1) == (0, fr.shape[-2], 0, fr.shape[-1]):
        return fr
    return fr[..., y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Observed context stream
# ---------------------------------------------------------------------------


class ContiguousStreamContext:
    """Raw frame and pose streams since the start of the context window.

    The model trains on contiguous 93-frame windows that the video encoder
    encodes in one pass: one anchor latent, then 6-latent chunks. The serve
    side does the same. It appends the executed real frames to a stream and
    encodes the retained prefix again on each call. The clean context chunks
    are 6-latent slices of that encode. The causal video encoder makes this
    equal to one encode of the full window.

    Chunk 0 of a window spans 21 raw frames and later chunks span 24. The
    context holds at most ``max_context_chunks`` clean chunks (69 raws for
    3). When a new chunk overflows it, the stream drops the oldest 24 raws.
    The retained prefix then starts a new window with a new anchor frame.
    """

    def __init__(self, action_chunk: int = ACTION_CHUNK, max_context_chunks: int = 3):
        geom = contiguous_chunk_geometry(action_chunk, max_context_chunks + 1)
        self.first_len = geom[0][1]
        self.cont_len = geom[1][1] if len(geom) > 1 else 0
        self.lat_per_chunk = (action_chunk - 1) // 4 + 1
        self.max_context_chunks = int(max_context_chunks)
        self.max_raws = self.first_len + self.cont_len * (self.max_context_chunks - 1)
        self._clip: torch.Tensor | None = None
        self._poses: np.ndarray | None = None

    def reset(self) -> None:
        """Drop all retained frames and poses."""
        self._clip = None
        self._poses = None

    @property
    def n_raws(self) -> int:
        """Return the number of retained raw frames."""
        return 0 if self._clip is None else int(self._clip.shape[2])

    @property
    def depth(self) -> int:
        """Return the number of complete clean chunks in the stream."""
        t = self.n_raws
        if t < self.first_len:
            return 0
        return min(1 + (t - self.first_len) // self.cont_len, self.max_context_chunks)

    @property
    def expected_next_rows(self) -> int:
        """Return the raw count of the next executed chunk (21 first, then 24)."""
        return self.first_len if self.n_raws == 0 else self.cont_len

    def boundary_offsets(self) -> list[int]:
        """Return the window offsets of the retained clean chunk starts."""
        return [off for off, _ln in contiguous_chunk_geometry(self.cont_len, self.max_context_chunks)][
            : self.depth
        ]

    def append_executed(self, clip: torch.Tensor, poses: np.ndarray | None = None) -> None:
        """Append the executed chunk and slide the window when it overflows.

        ``clip`` is a prepared ``(1, 3, T, H, W)`` clip in [0, 255] with
        ``T = expected_next_rows``. ``poses`` holds the matching pose rows.
        The pose stream stays valid only while every call supplies poses.
        """
        t = int(clip.shape[2])
        if t != self.expected_next_rows:
            raise ValueError(
                f"executed chunk spans {t} raws, expected {self.expected_next_rows} "
                f"(window chunk {'0' if self.n_raws == 0 else 'k>=1'})"
            )
        was_empty = self._clip is None
        self._clip = clip if self._clip is None else torch.cat([self._clip, clip], dim=2)
        aligned = self._poses is not None or was_empty
        if poses is not None and len(poses) == t and aligned:
            p = np.asarray(poses, dtype=np.float32)
            self._poses = p if self._poses is None else np.concatenate([self._poses, p], axis=0)
        else:
            self._poses = None
        if self.n_raws > self.max_raws:
            drop = self.n_raws - self.max_raws
            self._clip = self._clip[:, :, drop:]
            if self._poses is not None:
                self._poses = self._poses[drop:]

    def cleans(self, encode: Callable[[torch.Tensor], torch.Tensor]) -> list[torch.Tensor]:
        """Encode the retained prefix once and return the clean chunks (oldest first)."""
        d = self.depth
        if d == 0:
            return []
        assert self._clip is not None
        n_raws = self.first_len + self.cont_len * (d - 1)
        lat = encode(self._clip[:, :, :n_raws])
        n = self.lat_per_chunk
        return [lat.narrow(-3, k * n, n).contiguous() for k in range(d)]

    def anchor_frame(self) -> torch.Tensor | None:
        """Return the window start frame ``(1, 3, 1, H, W)``, or ``None`` on cold start."""
        return None if self._clip is None else self._clip[:, :, :1]

    def latent_poses(self) -> np.ndarray | None:
        """Return the pose rows at the last raw of each retained clean latent."""
        d = self.depth
        if self._poses is None or d == 0 or len(self._poses) != self.n_raws:
            return None
        n_lat = d * self.lat_per_chunk
        offs = [0] + [4 * n for n in range(1, n_lat)]
        if offs[-1] >= len(self._poses):
            return None
        return np.stack([self._poses[o] for o in offs], axis=0)

    def boundary_poses(self) -> np.ndarray | None:
        """Return the pose rows at the retained clean chunk starts, or ``None``."""
        if self._poses is None or self.depth == 0:
            return None
        if len(self._poses) != self.n_raws:
            return None
        offs = self.boundary_offsets()
        if not offs or offs[-1] >= len(self._poses):
            return None
        return np.stack([self._poses[o] for o in offs], axis=0)


# ---------------------------------------------------------------------------
# Action tokens and rows
# ---------------------------------------------------------------------------


def action_rows(
    pred_action: torch.Tensor,
    depth: int,
    frames_per_token: int,
    action_dim: int,
) -> np.ndarray:
    """Unpack the predicted ``(1, L, F * D)`` tokens to the executed rows (float32).

    At ``depth == 0`` the chunk owns the window anchor latent, so token 0
    gives one row and the chunk gives ``1 + (L - 1) * F`` rows (21). At
    ``depth >= 1`` every token gives ``F`` rows (24).
    """
    a = pred_action.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[0]
    n_lat, packed = a.shape
    per = packed // frames_per_token
    first_tok_is_anchor = depth == 0
    rows: list[np.ndarray] = []
    for k in range(n_lat):
        tok = a[k].reshape(frames_per_token, per)
        for f in range(frames_per_token):
            if k == 0 and f > 0 and first_tok_is_anchor:
                continue
            rows.append(tok[f, :action_dim])
    return np.stack(rows, axis=0)


def pack_action_tokens(rows: np.ndarray, first_chunk: bool) -> np.ndarray:
    """Pack per-frame action rows of one chunk into tokens of 4 sub-frames.

    Chunk 0 owns the window anchor latent: token 0 repeats row 0 four times,
    and token j holds rows ``4j-3 .. 4j``. A later chunk holds 4 consecutive
    rows per token.
    """
    t, d = rows.shape
    if not first_chunk:
        return np.ascontiguousarray(rows, dtype=np.float32).reshape(t // 4, 4 * d)
    n_lat = (t - 1) // 4 + 1
    out = np.empty((n_lat, 4 * d), dtype=np.float32)
    out[0] = np.tile(rows[0], 4)
    for k in range(1, n_lat):
        out[k] = rows[4 * k - 3 : 4 * k + 1].reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Instruction text
# ---------------------------------------------------------------------------


def text_encoder_prompt(prompt: str) -> str:
    """Return the string for the text encoder. A blank prompt becomes one space."""
    return prompt if prompt.strip() else " "
