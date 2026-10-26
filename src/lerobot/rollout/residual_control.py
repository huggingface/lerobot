import numpy as np

RESIDUAL_LIMITS = np.array(
    [0.5, 2.0, 1.0, 0.5, 0.3, 0.5],
    dtype=np.float32,
)

# 第1阶段严格设为0：
# PPO可以计算候选residual，但不会实际作用到机器人。
ACTIVE_SCALE = 0.0


def apply_residual(act_action, candidate_residual):
    act_action = np.asarray(act_action, dtype=np.float32)
    candidate_residual = np.asarray(candidate_residual, dtype=np.float32)

    if act_action.shape != (6,):
        raise ValueError(f"Expected ACT action shape (6,), got {act_action.shape}")

    if candidate_residual.shape != (6,):
        raise ValueError(
            f"Expected residual shape (6,), got {candidate_residual.shape}"
        )

    if not np.isfinite(act_action).all():
        raise ValueError("ACT action contains NaN or Inf")

    if not np.isfinite(candidate_residual).all():
        raise ValueError("PPO residual contains NaN or Inf")

    candidate_residual = np.clip(
        candidate_residual,
        -RESIDUAL_LIMITS,
        RESIDUAL_LIMITS,
    )

    active_residual = candidate_residual * ACTIVE_SCALE
    final_action = act_action + active_residual

    return final_action, active_residual
