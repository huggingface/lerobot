# async-diffusion-crash

**Target:** `src/lerobot/policies/diffusion/modeling_diffusion.py`
**Status:** `active`
**GitHub:** [#3445](https://github.com/huggingface/lerobot/issues/3445)

## What

Adds three preprocessing steps to `DiffusionPolicy.predict_action_chunk()` that were previously only done by `select_action()` (bypassed by async inference).

## Why

The diffusion policy has two inference entry points:

| Method | Caller | Preprocessing |
|--------|--------|:-------------:|
| `select_action()` | Sync inference (`lerobot-record`) | ✅ Full |
| `predict_action_chunk()` | Async inference (`policy_server`) | ❌ None |

When the async server calls `predict_action_chunk()` directly:
- Per-camera image keys are never consolidated → image queue stays empty
- `ACTION` key remains in batch → empty action queue gets stacked → crash
- `populate_queues()` is never called → all observation queues stay empty

**Crash:** `RuntimeError: stack expects a non-empty TensorList`

## Fix

Adds three blocks before the existing `torch.stack` line:
1. Image key consolidation (copy from `select_action`)
2. Pop `ACTION` from batch
3. Call `populate_queues()`

## Validate

**User:** Run a rollout with a diffusion policy via async inference. The server should not crash.

**Agent:**
```bash
grep -q "populate_queues" ~/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py && echo "Queue populate ✅" || echo "MISSING"
grep -q "OBS_IMAGES\|image_features" ~/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py && echo "Image consolidate ✅" || echo "MISSING"
```
