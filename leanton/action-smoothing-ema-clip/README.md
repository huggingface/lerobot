# action-smoothing-ema-clip

**Target:** `src/lerobot/rollout/strategies/core.py`
**Status:** `active`
**GitHub:** Not filed (post-hoc smoothing, not a bug fix)

## What

Post-hoc EMA smoothing (alpha=0.3) + per-joint velocity clipping (2.0°/frame ≈ 20°/s at 10fps) on absolute goal positions. Auto-resets EMA state on position gaps >10° (DAGGER correction boundaries).

Constants (tune per-robot):

- `EMA_SMOOTHING_ALPHA = 0.3` — lower = smoother but more lag
- `MAX_JOINT_DELTA = 2.0` — max per-frame change in degrees
- `SMOOTHING_RESET_THRESHOLD = 10.0` — gap that triggers EMA reset

## Why

SmolVLA per-frame prediction noise causes high-frequency jitter at inference. This low-pass filter on commanded goal positions eliminates jitter without retraining. Actions are **absolute goal positions** (degrees for SO-101). The EMA smooths the commanded position trajectory; the clip limits the per-frame *change* (velocity), NOT the absolute position.

## Validate

**User:** Run any rollout or teleop. The arm should move smoothly without visible jitter. Make a DAGGER correction with a large arm displacement — on resume (F10), the arm should continue smoothly from its post-correction position without a snap-back.

**Agent:**
```bash
grep -q "EMA_SMOOTHING_ALPHA\|MAX_JOINT_DELTA\|SMOOTHING_RESET_THRESHOLD" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Constants ✅" || echo "MISSING"
grep -q "_smooth_action(processed)" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Injection ✅" || echo "MISSING"
```
