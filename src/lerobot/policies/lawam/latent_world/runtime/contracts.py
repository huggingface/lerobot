from __future__ import annotations

from typing import Any


def validate_policy_contract(config: Any, policy_cfg: Any) -> None:
    expected = int(policy_cfg.future_action_window_size + policy_cfg.past_action_window_size + 1)
    got = int(policy_cfg.action_horizon)
    if got != expected:
        raise ValueError(
            "Invalid LaWAM action window contract: expected "
            "`action_horizon = future_action_window_size + past_action_window_size + 1`, "
            f"got action_horizon={got}, future_action_window_size={policy_cfg.future_action_window_size}, "
            f"past_action_window_size={policy_cfg.past_action_window_size}."
        )

    horizon_sec = float(policy_cfg.flow_cfg.horizon_sec)
    if horizon_sec <= 0.0:
        raise ValueError(f"LaWAM `flow_cfg.horizon_sec` must be > 0, got {horizon_sec}.")
