# action-smoothing-ema-clip

**Target:** `src/lerobot/rollout/configs.py`, `src/lerobot/rollout/strategies/core.py`, `src/lerobot/rollout/strategies/dagger.py`
**Status:** `active`
**GitHub:** Not filed (post-hoc smoothing, not a bug fix)
**Bugfixes applied:** `a88983c0` (simplified reset + dagger.py state management)

## What

Post-hoc EMA smoothing + per-joint velocity clipping on absolute goal positions (degrees for SO-101).

**core.py:** `_smooth_action()` runs on every autonomous frame, just before `send_action`. Three paths:

| Condition | Behavior |
|---|---|
| `_prev == None` (first frame / after correction resume) | Seed `_prev` with `v`, pass through — arm stays put |
| `|v - _prev| > 10°` (EMA drift / direction change) | Skip EMA blending, clip delta at `MAX_JOINT_DELTA`, seed `_prev` with raw target |
| Normal | EMA blend + velocity clip |

**dagger.py:** Calls `reset_action_smoothing()` (sets `_prev = None`) on correction resume and rollout start. This, combined with the `dagger-rtc-fresh-obs-on-resume` patch, ensures the first autonomous frame after a correction passes through cleanly — `v` is near the current physical position because the policy just observed it.

CLI flags (all under `RolloutConfig`, defaults tuned for SO-101 at 15fps):

- `--action_smoothing_enabled` (bool, default `True`) — toggle smoothing on/off
- `--action_smoothing_alpha` (float, default `0.25`) — blending factor (25% new, 75% history)
- `--action_smoothing_max_delta` (float, default `1.25`) — max per-frame change in degrees (≈19°/s at 15fps)
- `--action_smoothing_reset_threshold` (float, default `10.0`) — gap that skips EMA blending (still clipped)

A periodic log line every 150 frames (~10s at 15fps):
`clip=True/False limit=X.XX°/frame alpha=X.XX frame=N [RESET]`

## Why

SmolVLA per-frame prediction noise causes high-frequency jitter at inference. This low-pass filter on commanded goal positions eliminates jitter without retraining. Actions are **absolute goal positions** (degrees for SO-101). The EMA smooths the trajectory; the clip limits per-frame velocity.

## How correction boundaries work

```
CORRECTION ENDS (F11 pressed)
  │
  ├─ dagger.py: reset_action_smoothing()   → _prev = None
  ├─ dagger.py: engine.reset() + notify_observation(fresh_obs)
  │
  ▼
First autonomous frame:
  _prev == None → seed with v, pass through  (v ≈ physical position)
  Arm stays put.  No snap-back.  ✅

Second frame:
  _prev = v (from frame 1), new v from policy
  |v - _prev| is small → normal EMA + clip
  Arm tracks smoothly.  ✅
```

## Validate

**User:** Run any rollout or teleop. The arm should move smoothly without visible jitter. Make a DAGGER correction with a large arm displacement — on resume (F10), the arm should continue smoothly from its post-correction position without a snap-back.

**Agent:**
```bash
grep -q "action_smoothing_alpha\|action_smoothing_max_delta\|action_smoothing_reset_threshold" ~/lerobot/src/lerobot/rollout/configs.py && echo "Config fields ✅" || echo "MISSING"
grep -q "_smooth_action(processed" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Injection ✅" || echo "MISSING"
grep -q "reset_action_smoothing()" ~/lerobot/src/lerobot/rollout/strategies/dagger.py && echo "Reset-on-resume ✅" || echo "MISSING"
```