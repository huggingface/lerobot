# action-smoothing-ema-clip

**Target:** `src/lerobot/rollout/strategies/core.py`, `src/lerobot/rollout/strategies/dagger.py`
**Status:** `active`
**GitHub:** Not filed (post-hoc smoothing, not a bug fix)
**Bugfixes applied:** `2978b798` (snap-back on correction resume), `0bc80699` (reset-threshold bypassed velocity clip)

## What

Post-hoc EMA smoothing + per-joint velocity clipping on absolute goal positions. On DAGGER correction boundaries (>10° gap from stale state), the EMA state is reset — the raw policy output is accepted directly (it is near the current physical position — the policy just observed it) and `_prev` is re-seeded. On subsequent frames, normal EMA + clip resumes. `reset_action_smoothing()` is called on correction resume for a clean slate.

Constants (tuned for SO-101 at 15fps):

- `EMA_SMOOTHING_ALPHA = 0.25` — lower = smoother but more lag (25% new, 75% history)
- `MAX_JOINT_DELTA = 1.25` — max per-frame change in degrees (≈19°/s at 15fps)
- `SMOOTHING_RESET_THRESHOLD = 10.0` — gap that triggers EMA reset

A periodic log line is emitted every 150 frames (~10s at 15fps):
`clip=True/False limit=X.XX°/frame alpha=X.XX frame=N [RESET]`
where `[RESET]` marks a correction-boundary frame.

## Why

SmolVLA per-frame prediction noise causes high-frequency jitter at inference. This low-pass filter on commanded goal positions eliminates jitter without retraining. Actions are **absolute goal positions** (degrees for SO-101). The EMA smooths the commanded position trajectory; the clip limits the per-frame *change* (velocity), NOT the absolute position.

## Bugfixes

### Snap-back on correction resume (2026-06-12, `2978b798`)

**Problem:** The reset-threshold path was clipping the delta from the stale `_prev` (pre-correction smoothed position). Since `_prev` was frozen during teleop while the arm moved, the commanded position `prev_v + clip(±1.25°)` was still far from the current physical position — the arm snapped back toward the old position.

**Fix:** On reset, accept the policy output `v` directly (it IS near the current physical position — the policy just observed it). Seed `_prev` with `v` so subsequent frames blend normally from the correct baseline. Also added `reset_action_smoothing()` calls at correction-resume and rollout start for a clean slate.

### Reset-threshold bypassed velocity clip (2026-06-12, `0bc80699`)

**Problem:** The original code's `continue` in the reset-threshold branch skipped both EMA blending AND velocity clipping. On correction boundaries, the arm received unlimited single-frame jumps (15-66° observed).

**Fix:** (Obsoleted by `2978b798` — the reset now accepts `v` directly because `v` ≈ physical position, making per-frame clipping from stale `_prev` unnecessary.)

## Validate

**User:** Run any rollout or teleop. The arm should move smoothly without visible jitter. Make a DAGGER correction with a large arm displacement — on resume (F10), the arm should continue smoothly from its post-correction position without a snap-back.

**Agent:**
```bash
grep -q "EMA_SMOOTHING_ALPHA\|MAX_JOINT_DELTA\|SMOOTHING_RESET_THRESHOLD" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Constants ✅" || echo "MISSING"
grep -q "_smooth_action(processed)" ~/lerobot/src/lerobot/rollout/strategies/core.py && echo "Injection ✅" || echo "MISSING"
grep -q "reset_action_smoothing()" ~/lerobot/src/lerobot/rollout/strategies/dagger.py && echo "Reset-on-resume ✅" || echo "MISSING"
```