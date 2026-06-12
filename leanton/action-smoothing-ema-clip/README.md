# action-smoothing-ema-clip

**Target:** `src/lerobot/rollout/strategies/core.py`
**Status:** `active`
**GitHub:** Not filed (post-hoc smoothing, not a bug fix)
**Bugfixes applied:** `0bc80699` (reset-threshold bypassed velocity clip)

## What

Post-hoc EMA smoothing + per-joint velocity clipping on absolute goal positions. Auto-resets EMA state on position gaps >10° (DAGGER correction boundaries), but **still applies the velocity clip** even across the reset — the reset only skips EMA blending, it does not allow unlimited velocity.

Constants (tuned for SO-101 at 15fps):

- `EMA_SMOOTHING_ALPHA = 0.25` — lower = smoother but more lag (25% new, 75% history)
- `MAX_JOINT_DELTA = 1.25` — max per-frame change in degrees (≈19°/s at 15fps)
- `SMOOTHING_RESET_THRESHOLD = 10.0` — gap that triggers EMA reset

A periodic log line is emitted every 150 frames (~10s at 15fps):
`clip=True/False limit=X.XX°/frame alpha=X.XX frame=N [RESET]`
where `[RESET]` marks a correction-boundary frame.

## Why

SmolVLA per-frame prediction noise causes high-frequency jitter at inference. This low-pass filter on commanded goal positions eliminates jitter without retraining. Actions are **absolute goal positions** (degrees for SO-101). The EMA smooths the commanded position trajectory; the clip limits the per-frame *change* (velocity), NOT the absolute position.

## Bugfix (2026-06-12, `0bc80699`)

**Problem:** On DAGGER correction boundaries, the `SMOOTHING_RESET_THRESHOLD` path accepted the raw action **without velocity clipping** — the `continue` statement skipped the clip entirely. The arm received unlimited single-frame jumps (15-66° observed), causing sharp jerks on every correction resume.

**Fix:** The reset path now applies the velocity clip while still seeding the EMA state with the raw target. The arm ramps to the new position at `MAX_JOINT_DELTA` per frame instead of teleporting. The reset only means "don't EMA-blend across the correction gap" — velocity limiting is always active.

## Validate

**User:** Run any rollout or teleop. The arm should move smoothly without visible jitter. Make a DAGGER correction with a large arm displacement — on resume (F10), the arm should ramp smoothly from its post-correction position without a snap-back.

**Agent:**
```bash
grep -q "EMA_SMOOTHING_ALPHA\|MAX_JOINT_DELTA\|SMOOTHING_RESET_THRESHOLD" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Constants ✅" || echo "MISSING"
grep -q "_smooth_action(processed)" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Injection ✅" || echo "MISSING"
grep -q "max(-limit, min(limit, delta))" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Clip-in-reset ✅" || echo "MISSING"
```
