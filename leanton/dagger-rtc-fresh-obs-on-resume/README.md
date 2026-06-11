# dagger-rtc-fresh-obs-on-resume

**Target:** `src/lerobot/rollout/strategies/dagger.py`
**Status:** `active`
**GitHub:** [#3747](https://github.com/huggingface/lerobot/issues/3747)

## What

Feeds a fresh robot observation to the RTC engine after `engine.reset()` and before `engine.resume()` at the `PAUSED → AUTONOMOUS` transition in DAGGER.

## Why

During a DAGGER correction, `engine.notify_observation()` is never called (the correction loop bypasses `send_next_action`). The RTC engine's `_obs_holder` retains the last autonomous observation from *before* the pause. At `PAUSED → AUTONOMOUS`, `engine.reset()` clears the action queue but `_obs_holder` remains stale. The RTC thread produces its first action chunk from the pre-correction observation → snap-back to the pre-correction position.

The fix (4 lines in `dagger.py`, `_apply_transition`, `PAUSED → AUTONOMOUS` branch):
```python
engine.reset()
fresh_obs = robot.get_observation()
obs_processed = ctx.processors.robot_observation_processor(fresh_obs)
engine.notify_observation(obs_processed)
engine.resume()
```

Validated on SO-101. Correction → resume now continues smoothly from the post-correction position.

## Validate

**User:** Run a DAGGER rollout. Press F10 to pause, manually move the arm to a different position, press F10 to resume. The arm should continue smoothly from the new position without snapping back to where it was before the correction.

**Agent:**
```bash
grep -A4 "engine.reset()" ~/lerobot/src/lerobot/rollout/strategies/dagger.py | grep -q "notify_observation" && echo "Fresh obs feed ✅" || echo "MISSING"
```
