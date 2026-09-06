from __future__ import annotations

import numpy as np
import torch

def to_rotvec_from_quat(q):
    """Convert quaternion [x,y,z,w] to rotation vector (axis-angle * angle).
    Supports numpy arrays or torch tensors. Batch-safe.
    """
    is_torch = isinstance(q, torch.Tensor)
    if is_torch:
        q_np = q.detach().cpu().numpy()
    else:
        q_np = np.asarray(q)

    # use scipy-like conversion without scipy dependency
    # q format: [x, y, z, w]
    x = q_np[..., 0]
    y = q_np[..., 1]
    z = q_np[..., 2]
    w = q_np[..., 3]
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w * w)
    small = s < 1e-8
    axis = np.zeros(q_np[..., :3].shape)
    axis[~small] = np.stack([x[~small] / s[~small], y[~small] / s[~small], z[~small] / s[~small]], axis=-1)
    # for very small angle, axis is arbitrary; keep zeros
    rotvec = axis * angle[..., None]

    if is_torch:
        return torch.as_tensor(rotvec, dtype=q.dtype, device=q.device)
    return rotvec
