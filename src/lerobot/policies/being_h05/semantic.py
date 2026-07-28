"""Checkpoint-contract semantic packing for Being-H0.5.

Slot names and ranges are deliberately explicit. They are not generic padding.
"""

from __future__ import annotations

import torch

STATE_SLOTS = {
    "eef_position": (0, 3),
    "eef_rotation": (3, 6),
    "gripper_qpos": (44, 46),
    "base_position": (70, 73),
    "base_rotation": (73, 76),
}
ACTION_SLOTS = {
    "eef_position": (0, 3),
    "eef_rotation": (3, 6),
    "gripper_position": (18, 19),
    "base_motion": (70, 74),
    "control_mode": (74, 75),
}


def _quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(batch_dim + (9,)).unbind(-1)
    positive = torch.stack(
        [
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    safe = torch.where(positive > 0, positive, 1)
    q_abs = torch.where(positive > 0, safe.sqrt(), 0)
    quat_by_component = torch.stack(
        [
            torch.stack([q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()], dim=-1),
        ],
        dim=-2,
    )
    floor = q_abs.new_tensor(0.1)
    candidates = quat_by_component / (2 * q_abs[..., None].max(floor))
    index = q_abs.argmax(dim=-1, keepdim=True).unsqueeze(-1)
    quaternion = torch.gather(candidates, -2, index.expand(*batch_dim, 1, 4)).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector_norm = quaternion[..., 1:].norm(dim=-1, keepdim=True)
    half_angle = torch.atan2(vector_norm, quaternion[..., :1])
    sin_half_over_angle = 0.5 * torch.sinc(half_angle / torch.pi)
    return quaternion[..., 1:] / sin_half_over_angle


def _quat_xyzw_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    # This is SciPy Rotation.from_quat(q).as_rotvec() written directly so the
    # processor remains usable without importing an optional dependency.
    work = q.double()
    work = work / work.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(work.dtype).eps)
    scalar = work[..., 3:4]
    vector = work[..., :3]
    # A rotation at pi has scalar == 0, so q and -q are equally canonical.
    # PyTorch3D's matrix path selects the candidate whose dominant quaternion
    # component is positive; reproduce that tie-break to avoid a 2*pi jump.
    dominant = vector.gather(-1, vector.abs().argmax(dim=-1, keepdim=True))
    flip = (scalar < 0) | ((scalar.abs() <= 1e-12) & (dominant < 0))
    work = torch.where(flip, -work, work)
    vector = work[..., :3]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(vector_norm, work[..., 3:4])
    scale = torch.where(vector_norm > 1e-12, angle / vector_norm, 2 / work[..., 3:4])
    return (vector * scale).to(q.dtype)


def _author_base_quat_to_axis_angle(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Match the audited author's PyTorch3D conversion, including its ordering.

    RoboSuite emits xyzw, while the author passes the array directly to
    PyTorch3D's wxyz API. This ordering is therefore part of the released
    checkpoint contract even though it is not the physical base rotation.
    """
    q_as_xyzw = torch.cat([q_xyzw[..., 1:4], q_xyzw[..., 0:1]], dim=-1)
    return _matrix_to_axis_angle(_quat_xyzw_to_matrix(q_as_xyzw))


def atomic4_to_named(state: torch.Tensor, action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Map atomic_4 vectors to the semantics actually consumed by the checkpoint.

    Despite the author's intermediate column names ``world_abs_state`` and
    ``world_delta_action``, its released evaluator passes RoboSuite's
    base-relative EEF observation and base-frame controller command directly.
    This function preserves that empirically auditable contract.
    """
    if state.shape[-1] != 16:
        raise ValueError(f"atomic_4 state must be 16D, got {state.shape[-1]}")
    named = {
        "eef_position": state[..., 0:3],
        "eef_rotation": _quat_xyzw_to_axis_angle(state[..., 3:7]),
        "gripper_qpos": state[..., 14:16],
        "base_position": state[..., 7:10],
        "base_rotation": _author_base_quat_to_axis_angle(state[..., 10:14]),
    }
    if action is not None:
        if action.shape[-1] != 12:
            raise ValueError(f"atomic_4 action must be 12D, got {action.shape[-1]}")
        named.update(
            {
                "action.eef_position": action[..., 0:3],
                "action.eef_rotation": action[..., 3:6],
                "action.gripper_position": (1 - action[..., 6:7]) / 2,
                "action.base_motion": action[..., 7:11],
                "action.control_mode": (action[..., 11:12] + 1) / 2,
            }
        )
    return named


def pack_named(named: dict[str, torch.Tensor], slots: dict[str, tuple[int, int]], dim: int = 200):
    reference = next(iter(named.values()))
    leading = reference.shape[:-1]
    packed = torch.zeros(*leading, dim, dtype=reference.dtype, device=reference.device)
    valid = torch.zeros(*leading, dim, dtype=torch.bool, device=reference.device)
    for key, (start, end) in slots.items():
        value = named.get(key)
        if value is None:
            continue
        if value.shape[-1] != end - start:
            raise ValueError(f"{key} must be {end - start}D, got {value.shape[-1]}")
        packed[..., start:end] = value
        valid[..., start:end] = True
    return packed, valid


def normalize(value: torch.Tensor, mode: str, stats: dict[str, list[float]]) -> torch.Tensor:
    tensors = {
        key: torch.as_tensor(item, dtype=value.dtype, device=value.device) for key, item in stats.items()
    }
    if mode == "binary":
        return (value > 0.5).to(value.dtype)
    if mode == "q99":
        low, high = tensors["q01"], tensors["q99"]
    elif mode == "min_max":
        low, high = tensors["min"], tensors["max"]
    elif mode == "mean_std":
        mean, std = tensors["mean"], tensors["std"]
        return torch.where(std != 0, (value - mean) / torch.where(std != 0, std, 1), value)
    else:
        raise ValueError(f"Unknown Being-H normalization mode: {mode}")
    nonconstant = low != high
    scaled = 2 * (value - low) / torch.where(nonconstant, high - low, torch.ones_like(high)) - 1
    constant = value if mode == "q99" else torch.zeros_like(value)
    return torch.where(nonconstant, scaled, constant).clamp(-1, 1)


def inverse_normalize(value: torch.Tensor, mode: str, stats: dict[str, list[float]]) -> torch.Tensor:
    if mode == "binary":
        return (value > 0.5).to(value.dtype)
    tensors = {
        key: torch.as_tensor(item, dtype=value.dtype, device=value.device) for key, item in stats.items()
    }
    if mode == "q99":
        low, high = tensors["q01"], tensors["q99"]
        return (value + 1) / 2 * (high - low) + low
    if mode == "min_max":
        low, high = tensors["min"], tensors["max"]
        return (value + 1) / 2 * (high - low) + low
    if mode == "mean_std":
        return value * tensors["std"] + tensors["mean"]
    raise ValueError(f"Unknown Being-H normalization mode: {mode}")


def unpack_action(packed: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: packed[..., start:end] for key, (start, end) in ACTION_SLOTS.items()}


def named_to_atomic4_action(named: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            named["eef_position"],
            named["eef_rotation"],
            1 - 2 * (named["gripper_position"] > 0.5).to(named["gripper_position"].dtype),
            named["base_motion"],
            2 * (named["control_mode"] > 0.5).to(named["control_mode"].dtype) - 1,
        ],
        dim=-1,
    )
