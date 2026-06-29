"""Pure-numpy helpers to encode LIBERO observations into OpenPI requests and
decode the absolute action response back into LIBERO's 7-D action space.

LIBERO (via LeRobot's ``LiberoProcessorStep``) exposes a flat state vector
``[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`` and two RGB image streams.
The (legacy) DROID-style GR00T modality expects ``eef_9d = [xyz(3), rot6d(6)]``, a scalar
``gripper_position`` and a ``joint_position(7)`` vector.
"""

from __future__ import annotations

import numpy as np


def axisangle_to_matrix(axisangle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: axis-angle (3,) -> rotation matrix (3, 3)."""
    axisangle = np.asarray(axisangle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(axisangle))
    if theta < 1e-8:
        return np.eye(3)
    axis = axisangle / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_axisangle(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    cos_theta = np.clip((np.trace(mat) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # Near 180°: extract axis from the symmetric part.
        vals = np.clip((np.diag(mat) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(vals)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    axis = np.array(
        [mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]]
    ) / (2.0 * np.sin(theta))
    return axis * theta


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> 6D representation = first two columns, flattened."""
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float64)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D representation -> rotation matrix (3, 3) via Gram-Schmidt (matches GR00T)."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = rot6d[0:3], rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns


def eef_9d_from_pos_axisangle(eef_pos: np.ndarray, eef_axisangle: np.ndarray) -> np.ndarray:
    """[xyz(3), axisangle(3)] -> eef_9d = [xyz(3), rot6d(6)]."""
    R = axisangle_to_matrix(eef_axisangle)
    return np.concatenate([np.asarray(eef_pos, dtype=np.float64).reshape(3), matrix_to_rot6d(R)])


def gripper_qpos_to_position(gripper_qpos: np.ndarray, q_open: float, q_closed: float) -> float:
    """LIBERO gripper finger qpos (2,) -> normalized open fraction in [0, 1]."""
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    width = float(np.mean(np.abs(gripper_qpos)))
    denom = (q_open - q_closed) or 1.0
    return float(np.clip((width - q_closed) / denom, 0.0, 1.0))


def gripper_position_to_action(pos: float, action_open: float, action_close: float) -> float:
    """Normalized gripper open fraction in [0, 1] -> LIBERO gripper command.

    pos≈1 (open) -> action_open ; pos≈0 (closed) -> action_close.
    """
    pos = float(np.clip(pos, 0.0, 1.0))
    return float(action_close + pos * (action_open - action_close))


def chw01_to_hwc_uint8(img: np.ndarray, flip_180: bool = False) -> np.ndarray:
    """(C, H, W) float [0, 1] (LeRobot image) -> (H, W, 3) uint8 (server image)."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-3 else arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if flip_180:
        arr = arr[::-1, ::-1].copy()
    return arr
