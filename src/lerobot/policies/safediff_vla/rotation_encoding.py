"""sin/cos Euler-angle representation for SafeDiff-VLA's `temporal_decoder` architectures.

Motivation (see `examples/safediff_vla/euler_branch_audit.py`'s dataset-wide audit): raw Euler
rotation actions (`action[..., 3:6]` = rx, ry, rz) wrap at +-pi, and rx in particular sits within
0.1 rad of that discontinuity for ~87% of training frames, with ~44% of all `action_horizon=50`
chunks containing at least one >pi consecutive-step jump. A plain MSE loss (or a Transformer
decoder head) has no way to know +pi and -pi are the same angle, so the raw representation
manufactures large, spurious "errors" exactly at the values the data visits most.

This module is the *only* place that representation conversion happens. Everything upstream
(dataset, `observation.state`/`action` MEAN_STD normalization) and downstream (the postprocessor,
`execution.ActionExecutor`, the VLABench env) is untouched and keeps talking in the original 7-D
`[x, y, z, rx, ry, rz, gripper]` layout, normalized exactly as before -- `SafeDiffVLAPolicy` just
un-normalizes the rotation slice back to raw radians, encodes/decodes it to/from sin/cos right at
its own input/output boundary, and re-normalizes on the way out, so nothing outside this policy's
`temporal_decoder`/`temporal_decoder_subgoal` forward/plan methods ever needs to know this exists.

Layout:
    raw / MEAN_STD-normalized 7-D: [x, y, z, rx, ry, rz, gripper]
    encoded 10-D:                  [x, y, z, sin(rx), cos(rx), sin(ry), cos(ry), sin(rz), cos(rz), gripper]

`x, y, z` and `gripper` pass through `encode`/`decode` completely unchanged (still whatever
MEAN_STD-normalized value the outer preprocessor/postprocessor already produces/expects) -- only
the rotation slice is touched.
"""

from __future__ import annotations

import torch
from torch import Tensor

RAW_DIM = 7
ENCODED_DIM = 10
ROTATION_SLICE_RAW = slice(3, 6)  # rx, ry, rz in the 7-D layout
ROTATION_SLICE_ENCODED = slice(3, 9)  # sin(rx),cos(rx),sin(ry),cos(ry),sin(rz),cos(rz) in the 10-D layout
GRIPPER_INDEX_RAW = 6
GRIPPER_INDEX_ENCODED = 9


def encode(
    x7_normalized: Tensor,
    rot_mean: Tensor,
    rot_std: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """`[..., 7]` (MEAN_STD-normalized, outer-preprocessor convention) -> `[..., 10]`.

    `rot_mean`/`rot_std` are the *same* per-dimension statistics the outer preprocessor used to
    normalize dims `[3:6]` -- un-normalizing with them recovers the exact original raw radian
    value (normalize/un-normalize are exact inverses regardless of how those particular stats
    were computed), which is what `sin`/`cos` need to be geometrically meaningful.
    """
    xyz = x7_normalized[..., :3]
    raw_rot = x7_normalized[..., ROTATION_SLICE_RAW] * (rot_std + eps) + rot_mean
    sincos = torch.stack((torch.sin(raw_rot), torch.cos(raw_rot)), dim=-1).flatten(-2)
    gripper = x7_normalized[..., GRIPPER_INDEX_RAW : GRIPPER_INDEX_RAW + 1]
    return torch.cat((xyz, sincos, gripper), dim=-1)


def decode(
    x10: Tensor,
    rot_mean: Tensor,
    rot_std: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """`[..., 10]` -> `[..., 7]` (MEAN_STD-normalized, outer-postprocessor convention).

    Each `(sin, cos)` pair is unit-normalized before `atan2` so an off-manifold prediction (the
    decoder has no hard constraint forcing `sin**2 + cos**2 == 1`) still decodes to a well-defined
    angle rather than a biased/degenerate one.
    """
    xyz = x10[..., :3]
    sincos = x10[..., ROTATION_SLICE_ENCODED].reshape(*x10.shape[:-1], 3, 2)
    sin_raw, cos_raw = sincos[..., 0], sincos[..., 1]
    norm = torch.sqrt(sin_raw.square() + cos_raw.square()).clamp_min(eps)
    raw_rot = torch.atan2(sin_raw / norm, cos_raw / norm)
    rot_normalized = (raw_rot - rot_mean) / (rot_std + eps)
    gripper = x10[..., GRIPPER_INDEX_ENCODED : GRIPPER_INDEX_ENCODED + 1]
    return torch.cat((xyz, rot_normalized, gripper), dim=-1)
