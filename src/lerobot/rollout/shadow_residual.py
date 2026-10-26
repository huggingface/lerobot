import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from lerobot.rollout.residual_observation import build_residual_observation


PPO_PATH = '/home/pc-workzl/Downloads/lerobot061-src/outputs/residual_ppo_obs2054_pair_update8e_lr2e6_20261011'

LOG_PATH = Path(
    "/home/pc-workzl/Downloads/lerobot061-src/"
    "outputs/rl_shadow/shadow_actions.csv"
)

OBS2054_LOG_PATH = Path(
    "/home/pc-workzl/Downloads/lerobot061-src/"
    "outputs/rl_shadow/shadow_obs2054.f32"
)

ONPOLICY_LOG_PATH = Path(
    "/home/pc-workzl/Downloads/lerobot061-src/"
    "outputs/rl_shadow/shadow_onpolicy.f32"
)

RESIDUAL_SCALE = np.array(
    [0.5, 2.0, 1.0, 0.5, 0.3, 0.5],
    dtype=np.float32,
)

_MODEL = None


def _get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = PPO.load(
            PPO_PATH,
            device="cpu",
        )

    return _MODEL


def log_shadow_residual(obs_frame, action_dict):
    """
    Macro residual shadow policy.

    PPO samples one new action every 10 control frames.
    The clipped PPO action is held for the following 10 frames.

    Important:
    - per-frame CSV/obs logging is still kept for debugging;
    - macro PPO training logs are written only when PPO really samples;
    - this function itself does not decide whether residual reaches robot.
      residual_control.py ACTIVE_SCALE remains the final safety gate.
    """
    from lerobot.rollout.residual_macro import ResidualMacroHold

    state_keys = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    state = np.asarray(
        [obs_frame[k] for k in state_keys],
        dtype=np.float32,
    )

    if state.shape != (6,):
        raise ValueError(
            f"Expected state shape (6,), got {state.shape}"
        )

    if not np.isfinite(state).all():
        raise ValueError("State contains NaN or Inf")

    ppo_obs = build_residual_observation(obs_frame)

    if ppo_obs.shape != (2054,):
        raise ValueError(
            f"Expected PPO observation shape (2054,), got {ppo_obs.shape}"
        )

    if not np.isfinite(ppo_obs).all():
        raise ValueError(
            "PPO observation contains NaN or Inf"
        )

    # Lazy macro state. A fresh Python rollout process starts from frame 0.
    if not hasattr(log_shadow_residual, "_macro_hold"):
        log_shadow_residual._macro_hold = ResidualMacroHold(
            hold_frames=10
        )
        log_shadow_residual._frame_index = 0
        log_shadow_residual._raw_action = None
        log_shadow_residual._value = None
        log_shadow_residual._log_prob = None

    macro = log_shadow_residual._macro_hold
    frame_index = log_shadow_residual._frame_index

    sampled_now = macro.needs_sample

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    macro_obs_path = (
        LOG_PATH.parent /
        "shadow_macro_obs2054.f32"
    )

    macro_onpolicy_path = (
        LOG_PATH.parent /
        "shadow_macro_onpolicy.f32"
    )

    macro_csv_path = (
        LOG_PATH.parent /
        "shadow_macro_decisions.csv"
    )

    if sampled_now:
        model = _get_model()

        obs_tensor, _ = model.policy.obs_to_tensor(
            ppo_obs.reshape(1, -1)
        )

        with torch.no_grad():
            (
                raw_action_t,
                value_t,
                log_prob_t,
            ) = model.policy(
                obs_tensor,
                deterministic=False,
            )

        raw_action = (
            raw_action_t
            .cpu()
            .numpy()
            .reshape(6)
            .astype(np.float32)
        )

        value = np.float32(
            value_t
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        log_prob = np.float32(
            log_prob_t
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        normalized_action = np.clip(
            raw_action,
            -1.0,
            1.0,
        ).astype(np.float32)

        macro.set_new_action(
            normalized_action
        )

        log_shadow_residual._raw_action = (
            raw_action.copy()
        )
        log_shadow_residual._value = value
        log_shadow_residual._log_prob = log_prob

        # Only real PPO decision observations are stored here.
        with macro_obs_path.open("ab") as f:
            f.write(
                np.asarray(
                    ppo_obs,
                    dtype=np.float32,
                )
                .reshape(2054)
                .tobytes(order="C")
            )

        onpolicy_record = np.concatenate(
            [
                raw_action.astype(np.float32),
                np.asarray(
                    [value, log_prob],
                    dtype=np.float32,
                ),
            ]
        )

        with macro_onpolicy_path.open("ab") as f:
            f.write(
                onpolicy_record.tobytes(
                    order="C"
                )
            )

        candidate_residual = (
            normalized_action
            * RESIDUAL_SCALE
        ).astype(np.float32)

        macro_new_file = (
            not macro_csv_path.exists()
        )

        with macro_csv_path.open(
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)

            if macro_new_file:
                writer.writerow([
                    "timestamp",
                    "frame_index",
                    "decision_index",
                    *[
                        f"raw_action_{i}"
                        for i in range(6)
                    ],
                    "value",
                    "log_prob",
                    *[
                        f"ppo_normalized_{i}"
                        for i in range(6)
                    ],
                    *[
                        f"ppo_residual_{i}"
                        for i in range(6)
                    ],
                ])

            writer.writerow([
                time.time(),
                frame_index,
                macro.decision_count,
                *raw_action.tolist(),
                float(value),
                float(log_prob),
                *normalized_action.tolist(),
                *candidate_residual.tolist(),
            ])

    # One held PPO action is consumed on every control frame.
    normalized_action = macro.next_frame()

    residual = (
        normalized_action
        * RESIDUAL_SCALE
    ).astype(np.float32)

    act_action = np.asarray(
        list(action_dict.values()),
        dtype=np.float32,
    )

    if act_action.shape != (6,):
        raise ValueError(
            f"Expected ACT action shape (6,), got {act_action.shape}"
        )

    # Per-frame debug log.
    new_file = not LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        newline="",
    ) as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "timestamp",
                *[
                    f"state_{i}"
                    for i in range(6)
                ],
                *[
                    f"act_{i}"
                    for i in range(6)
                ],
                *[
                    f"ppo_normalized_{i}"
                    for i in range(6)
                ],
                *[
                    f"ppo_residual_{i}"
                    for i in range(6)
                ],
                "macro_sampled",
                "macro_decision_index",
            ])

        writer.writerow([
            time.time(),
            *state.tolist(),
            *act_action.tolist(),
            *normalized_action.tolist(),
            *residual.tolist(),
            int(sampled_now),
            macro.decision_count,
        ])

    # Keep the complete per-control-frame observation log for debugging.
    with OBS2054_LOG_PATH.open("ab") as obs_f:
        obs_f.write(
            np.asarray(
                ppo_obs,
                dtype=np.float32,
            )
            .reshape(2054)
            .tobytes(order="C")
        )

    log_shadow_residual._frame_index += 1

    return residual
