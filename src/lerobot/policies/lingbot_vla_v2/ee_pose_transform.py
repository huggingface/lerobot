import torch
import numpy as np
from torch import Tensor

__all__ = [
    "_is_quaternion_relative_type",
    "relative_pose_quaternion",
    "absolute_pose_quaternion",
]


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion in xyzw format. Shape: (..., 4)."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion sign so the scalar part keeps a stable non-negative sign."""
    sign = torch.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of quaternion in xyzw format. Shape: (..., 4)."""
    inv = quat_normalize(q).clone()
    inv[..., :3] = -inv[..., :3]  # negate xyz (imaginary part)
    return inv


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2 in xyzw format. Shape: (..., 4)"""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by quaternion(s) in xyzw format."""
    q = quat_normalize(q)
    v_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    return quat_multiply(quat_multiply(q, v_quat), quat_inverse(q))[..., :3]


def quat_rotate_inverse(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by the inverse of quaternion(s) in xyzw format."""
    return quat_rotate(quat_inverse(q), v)


def _resolve_quaternion_relative_type(relative_type: str | None) -> str:
    if relative_type in (None, "world", "quaternion", "quaternion_world"):
        return "world"
    if relative_type in ("local", "quaternion_local"):
        return "local"
    raise ValueError(f"Unsupported quaternion relative type: {relative_type}")


def _is_quaternion_relative_type(relative_type: str | None) -> bool:
    return relative_type in {"quaternion", "quaternion_world", "quaternion_local"}


def relative_pose_quaternion(
    action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int = 7,
    relative_type: str | None = "world",
) -> torch.Tensor:
    """Compute relative pose for xyz+quaternion(xyzw) representation.
    Handles concatenated multi-arm data by splitting into pose_dim chunks.
    action/state shape: (..., N*pose_dim) where N is number of arms.
    """
    relative_type = _resolve_quaternion_relative_type(relative_type)
    total_dim = action.shape[-1]
    assert total_dim % pose_dim == 0
    chunks = total_dim // pose_dim
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        a_xyz = action[..., s : s + 3]
        a_q = quat_normalize(action[..., s + 3 : s + pose_dim])
        s_xyz = state[..., s : s + 3]
        s_q = quat_normalize(state[..., s + 3 : s + pose_dim])
        if relative_type == "local":
            rel_xyz = quat_rotate_inverse(a_xyz - s_xyz, s_q)
        else:
            rel_xyz = a_xyz - s_xyz
        rel_q = quat_canonicalize(quat_normalize(quat_multiply(quat_inverse(s_q), a_q)))
        parts.append(torch.cat([rel_xyz, rel_q], dim=-1))
    return torch.cat(parts, dim=-1)


def absolute_pose_quaternion(
    rel_action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int = 7,
    relative_type: str | None = "world",
) -> torch.Tensor:
    """Recover absolute pose from relative pose (inverse of relative_pose_quaternion).
    rel_action/state shape: (..., N*pose_dim).
    """
    relative_type = _resolve_quaternion_relative_type(relative_type)
    total_dim = rel_action.shape[-1]
    assert total_dim % pose_dim == 0
    chunks = total_dim // pose_dim
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        r_xyz = rel_action[..., s : s + 3]
        r_q = quat_normalize(rel_action[..., s + 3 : s + pose_dim])
        s_xyz = state[..., s : s + 3]
        s_q = quat_normalize(state[..., s + 3 : s + pose_dim])
        if relative_type == "local":
            abs_xyz = s_xyz + quat_rotate(s_q, r_xyz)
        else:
            abs_xyz = s_xyz + r_xyz
        abs_q = quat_canonicalize(quat_normalize(quat_multiply(s_q, r_q)))
        parts.append(torch.cat([abs_xyz, abs_q], dim=-1))
    return torch.cat(parts, dim=-1)


def matrix_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix (..., 3, 3) to quaternion (..., 4) in xyzw format.

    Uses Shepperd's method for numerical stability.
    """
    batch_shape = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(*batch_shape, 4, dtype=R.dtype, device=R.device)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    q[..., 3] = 0.25 * s
    q[..., 0] = (m21 - m12) / s
    q[..., 1] = (m02 - m20) / s
    q[..., 2] = (m10 - m01) / s

    cond2 = (m00 > m11) & (m00 > m22) & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond2, 0.25 * s2, q[..., 0])
    q[..., 1] = torch.where(cond2, (m01 + m10) / s2, q[..., 1])
    q[..., 2] = torch.where(cond2, (m02 + m20) / s2, q[..., 2])
    q[..., 3] = torch.where(cond2, (m21 - m12) / s2, q[..., 3])

    cond3 = (~cond2) & (m11 > m22) & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond3, (m01 + m10) / s3, q[..., 0])
    q[..., 1] = torch.where(cond3, 0.25 * s3, q[..., 1])
    q[..., 2] = torch.where(cond3, (m12 + m21) / s3, q[..., 2])
    q[..., 3] = torch.where(cond3, (m02 - m20) / s3, q[..., 3])

    cond4 = (~cond2) & (~cond3) & (trace <= 0)
    s4 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond4, (m02 + m20) / s4, q[..., 0])
    q[..., 1] = torch.where(cond4, (m12 + m21) / s4, q[..., 1])
    q[..., 2] = torch.where(cond4, 0.25 * s4, q[..., 2])
    q[..., 3] = torch.where(cond4, (m10 - m01) / s4, q[..., 3])

    return quat_normalize(q)
