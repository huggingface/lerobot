# Push Intermediate Checkpoints to Hub During Training

**Target:** `src/lerobot/scripts/lerobot_train.py`, `src/lerobot/policies/pretrained.py`, `src/lerobot/policies/factory.py`, `src/lerobot/configs/policies.py`
**Status:** `active`
**GitHub:** Not filed (planned upstream feature request)
**Diff basis:** `origin/main` @ `49755a3d` (clean upstream, 2026-06-12). Includes `revision` field on `PolicyConfig` as a bundled prerequisite — upstream doesn't have it yet.

## What

Adds a `push_checkpoints_to_hub` config flag (default `False`) that, when enabled, pushes model weights to the HuggingFace Hub at each `save_freq` interval as a named branch (`step-002500`, `step-005000`, etc.). V1 pushes model weights only — no optimizer/scheduler state. A `latest-checkpoint` branch auto-updates to always point at the most recent successful push, so resume is `--policy.revision=latest-checkpoint` (no step numbers needed).

## Why

`push_to_hub=true` only pushes the final model after training completes. On Colab or other ephemeral runtimes, a disconnection loses all local checkpoints. The user runs 6K-step chunks on L4 (~6 hours each) and has lost multiple runs to disconnects. This patch makes intermediate checkpoints survive runtime loss.

Four files changed:
- `configs/policies.py` — add `push_checkpoints_to_hub: bool = False`
- `policies/pretrained.py` — add `revision` param to `push_model_to_hub()` + `latest-checkpoint` pointer update after each successful step-branch push
- `policies/factory.py` — wire `revision` through `make_policy()` and `make_pre_post_processors()`
- `scripts/lerobot_train.py` — push model to Hub in checkpoint block when flag is set

## Validate

**User:**
1. Run training with `--policy.push_checkpoints_to_hub=true --save_freq=200 --steps=500`
2. Check Hub repo: branches `step-000200`, `step-000400` exist, and `latest-checkpoint` points at `step-000400`
3. Resume from latest: `--policy.path=<repo> --policy.revision=latest-checkpoint` — loads the most recent checkpoint automatically
4. Run training without `--policy.push_checkpoints_to_hub` — verify no mid-training pushes (regression)

**Agent:**
```bash
grep -q "push_checkpoints_to_hub" ~/lerobot/src/lerobot/configs/policies.py && echo "✅ config" || echo "MISSING config"
grep -q "revision: str | None = None" ~/lerobot/src/lerobot/policies/pretrained.py && echo "✅ pretrained" || echo "MISSING pretrained"
grep -q 'revision=revision' ~/lerobot/src/lerobot/policies/factory.py && echo "✅ factory" || echo "MISSING factory"
grep -q "Pushing checkpoint to Hub" ~/lerobot/src/lerobot/scripts/lerobot_train.py && echo "✅ train" || echo "MISSING train"
```
