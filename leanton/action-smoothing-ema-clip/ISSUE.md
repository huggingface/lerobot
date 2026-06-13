# Upstream Issue Draft — action-smoothing-ema-clip

**Title:** Post-hoc EMA smoothing and velocity clipping for rollout actions

## Motivation

SmolVLA produces per-frame prediction noise that causes high-frequency jitter at inference. This low-pass filter on commanded goal positions eliminates jitter without retraining. Actions are absolute goal positions (degrees for SO-101).

## Proposed solution

EMA smoothing + per-joint velocity clipping on commanded goal positions, applied just before `send_action` in the shared `send_next_action` dispatch function (covers all rollout strategies).

CLI-configurable via `RolloutConfig`:
- `--action_smoothing_enabled` (bool, default `True`)
- `--action_smoothing_alpha` (float, default `0.25`) — EMA blending factor
- `--action_smoothing_max_delta` (float, default `1.25`) — max per-frame change in action units
- `--action_smoothing_reset_threshold` (float, default `10.0`) — gap that skips EMA blending

Handles DAGGER correction boundaries correctly: `dagger.py` calls `reset_action_smoothing()` on correction resume, setting `_prev=None` so the first autonomous frame passes through cleanly (the policy just observed the current pose, so its output is near the physical position). No snap-back.

Three files changed:
- `rollout/configs.py` — smoothing fields on `RolloutConfig`
- `rollout/strategies/core.py` — `_smooth_action()` function + `send_next_action` wiring
- `rollout/strategies/dagger.py` — `reset_action_smoothing()` on correction resume + rollout start

## Implementation

Branch: `anikita/lerobot:feat/action-smoothing-ema-clip`
Patch: `leanton/action-smoothing-ema-clip/action-smoothing-ema-clip.diff` (209 lines, rebased on main@30790de1)

## Future work

- Constants could move from `RolloutConfig` dataclass defaults to a dedicated smoothing config group
- Periodic log (every 150 frames) could be `DEBUG` level instead of `INFO`

## Tested

On SO-101 hardware with foot pedal, SmolVLA policy, DAGGER strategy.
