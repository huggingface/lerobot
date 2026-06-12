# Push Intermediate Checkpoints to Hub During Training

**Target:** `src/lerobot/scripts/lerobot_train.py`, `src/lerobot/policies/pretrained.py`, `src/lerobot/policies/factory.py`, `src/lerobot/configs/policies.py`
**Status:** `active`
**GitHub:** Not filed (planned upstream feature request)
**Diff basis:** `origin/main` @ `49755a3d` (clean upstream, 2026-06-12). Includes `revision` field on `PolicyConfig` as a bundled prerequisite — upstream doesn't have it yet.
**Bugfixes applied:** `bed9786f` (config revision fallback), `42c2942e` (branch precreate)

## What

Adds a `push_checkpoints_to_hub` config flag (default `False`) that, when enabled, pushes model weights to the HuggingFace Hub at each `save_freq` interval as a named branch (`step-002500`, `step-005000`, etc.). V1 pushes model weights only — no optimizer/scheduler state. A `latest-checkpoint` branch auto-updates to always point at the most recent successful push, so resume is `--policy.revision=latest-checkpoint` (no step numbers needed). On first run (before any checkpoint exists), `from_pretrained()` silently falls back to `main` — the same command works for both fresh and resume training.

## Why

`push_to_hub=true` only pushes the final model after training completes. On Colab or other ephemeral runtimes, a disconnection loses all local checkpoints. The user runs 6K-step chunks on L4 (~6 hours each) and has lost multiple runs to disconnects. This patch makes intermediate checkpoints survive runtime loss.

Five files changed:
- `configs/policies.py` — add `push_checkpoints_to_hub: bool = False` + **bugfix:** generic revision fallback in `from_pretrained()` when a requested branch doesn't exist on Hub (rollout/config load was crashing on 404 before the fallback)
- `policies/pretrained.py` — add `revision` param to `push_model_to_hub()` + `latest-checkpoint` pointer update + `from_pretrained()` fallback to `main` + **bugfix:** `create_branch` before `upload_folder` (newer `huggingface_hub` requires the branch to exist before the preupload call)
- `policies/factory.py` — wire `revision` through `make_policy()` and `make_pre_post_processors()`
- `scripts/lerobot_train.py` — push model to Hub in checkpoint block when flag is set
- `processor/pipeline.py` — `from_pretrained()` fallback to `main` for processor loading

## Bugfixes (applied after initial patch)

### 1. Branch precreate (2026-06-12, `42c2942e`)
**Problem:** `upload_folder(revision="step-002500")` failed with 404 "Invalid rev id" — the Colab version of `huggingface_hub` no longer auto-creates branches via the preupload endpoint.
**Fix:** Call `api.create_branch(repo_id, branch=revision, exist_ok=True)` before `upload_folder` in `push_model_to_hub()`.

### 2. Config revision fallback (2026-06-12, `bed9786f`)
**Problem:** `lerobot-rollout` calls `PreTrainedConfig.from_pretrained(revision="latest-checkpoint")` directly, bypassing the `latest-checkpoint` fallback in `pretrained.py`. When the branch doesn't exist, it crashes with `FileNotFoundError` instead of falling back to `main`.
**Fix:** Generic revision fallback in `configs/policies.py` — if `hf_hub_download` fails and a revision was requested, retry with `revision=None` (main). Logs a warning on fallback.

## Validate

**User:**
1. Run training with `--policy.push_checkpoints_to_hub=true --save_freq=200 --steps=500`
2. Check Hub repo: branches `step-000200`, `step-000400` exist, and `latest-checkpoint` points at `step-000400`
3. Resume: `--policy.revision=latest-checkpoint` — loads `step-000400` automatically (no step number needed)
4. Start fresh training with `--policy.revision=latest-checkpoint` on a repo with no checkpoints — silently falls back to `main` (no crash)
5. Run training without `--policy.push_checkpoints_to_hub` — verify no mid-training pushes (regression)

**Agent:**
```bash
grep -q "push_checkpoints_to_hub" ~/lerobot/src/lerobot/configs/policies.py && echo "✅ config" || echo "MISSING config"
grep -q "create_branch" ~/lerobot/src/lerobot/policies/pretrained.py && echo "✅ branch precreate" || echo "MISSING branch precreate"
grep -q "revision: str | None = None" ~/lerobot/src/lerobot/policies/pretrained.py && echo "✅ pretrained" || echo "MISSING pretrained"
grep -q 'revision=revision' ~/lerobot/src/lerobot/policies/factory.py && echo "✅ factory" || echo "MISSING factory"
grep -q "Pushing checkpoint to Hub" ~/lerobot/src/lerobot/scripts/lerobot_train.py && echo "✅ train" || echo "MISSING train"
grep -q "revision is not None" ~/lerobot/src/lerobot/configs/policies.py && echo "✅ config fallback" || echo "MISSING config fallback"
```
