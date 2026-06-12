---
title: Plan — Mid-Training Hub Push for LeRobot
tags: [LeRobot, patch, training, checkpoint, HuggingFace-Hub]
created: 2026-06-06
status: plan
---

# Plan: Mid-Training Hub Push for LeRobot

> **Goal:** Push model checkpoints to HuggingFace Hub at each `save_freq` interval during training, so that Colab disconnects do not lose all progress. Each checkpoint is tagged with its step count for easy identification and invalidation.
> **Repo:** `huggingface/lerobot`
> **Patch name:** `mid-training-hub-push`
> **Diff basis:** `origin/main` (clean upstream). The `.diff` bundles the `revision` field on `PolicyConfig` as a prerequisite since upstream doesn't have it yet. Our `leanton` branch already has `revision` from the `rollout-policy-revision` patch — the source changes on `leanton` omit it, but the `.diff` file includes it for upstream compatibility.

---

## Why This Matters

### The Problem

Colab (and any ephemeral cloud GPU) can disconnect at any time. LeRobot currently:

1. Saves checkpoints **locally** at `save_freq` intervals — these are lost on disconnect
2. Pushes the model to Hub **only at the very end** of training (`lerobot_train.py:570-581`)
3. If the runtime dies at step 9,950 of a 20,000-step run: **everything is lost**

The user currently runs 6K-step chunks on L4 (~6 hours each) and has lost multiple runs to disconnects. Each lost run wastes compute time and delays the DAGGER iteration cycle.

### The Current Workaround (Segmented 5K Chunks)

The runbook (`7. SMOLVLA_FULL_EXPERT_TRAINING_RUNBOOK.md`) works around this by:
- Running short 5K-step chunks with `resume: False`
- Loading the previous checkpoint as `pretrained_path` (fresh optimizer/scheduler each time)
- Explicit LR scheduler flags to keep each chunk productive

This works but:
- Each chunk wastes ~100 steps on warmup
- The LR curve is a sawtooth, not a smooth cosine
- Requires manual intervention between chunks (re-launch Colab, re-download dataset, re-start training)

### What the Patch Enables

1. **Disconnect safety:** If Colab dies at step 9,950, you've lost at most `save_freq` steps (e.g., 2,500). Resume from the last Hub checkpoint.
2. **Longer chunks:** Run 10K or 20K steps in one go without fear. The 5K chunk pattern becomes optional, not mandatory.
3. **Resume from any checkpoint:** Each checkpoint is a named revision on Hub. `--policy.path=repo --policy.revision=step-07500` loads exactly that state.
4. **Invalidation:** Delete a bad checkpoint by removing its branch: `huggingface-cli repo delete <repo> --branch step-05000`

---

## ⚠️ Prerequisite: `--policy.revision` in Training — Mostly Done

The checkpoint push creates branches like `step-005000`. For training to load from them, `--policy.revision=step-N` must work end-to-end. **The infrastructure is ~80% there.** Only wiring is missing.

### Current State (Verified against codebase 2026-06-12)

| Component | Status | Detail |
|:---|:---|:---|
| `PolicyConfig` (`configs/policies.py:83`) | ✅ **Done** | `revision: str \| None = None` already exists |
| `PreTrainedPolicy.from_pretrained()` (`pretrained.py:87`) | ✅ **Done** | Accepts `revision`, passes to `hf_hub_download` and `Config.from_pretrained` |
| `PolicyProcessorPipeline.from_pretrained()` (`pipeline.py:623`) | ✅ **Done** | Accepts `revision`, passes through to hub download |
| `HubMixin.push_to_hub()` (`hub.py:137`) | ✅ **Done** | Accepts `branch=` param, passes to `upload_folder(revision=branch)` |
| `make_policy()` (`factory.py:559-560`) | ❌ **Missing** | Does NOT pass `cfg.revision` to `from_pretrained()` |
| `make_pre_post_processors()` (`factory.py:304,313`) | ❌ **Missing** | Does NOT pass `revision` to `PolicyProcessorPipeline.from_pretrained()` |
| Rollout (`lerobot-rollout`) | ❌ **Missing** | Tracked as issue #3549; separate patch `rollout-policy-revision` |

**Bottom line:** `--policy.revision=step-005000` is already a valid CLI flag — draccus picks it up from the dataclass field. `from_pretrained()` methods already accept and handle it. Only `make_policy()` and `make_pre_post_processors()` need one-line fixes to pass it through.

### What This Patch Must Also Add

| File | Change | Lines |
|:-----|:-------|:------|
| `policies/factory.py` | In `make_policy()`, pass `revision=cfg.revision` into `kwargs` before `from_pretrained()` | ~559 |
| `policies/factory.py` | In `make_pre_post_processors()`, pass `revision=` to `PolicyProcessorPipeline.from_pretrained()` | ~304, ~313 |

After this, `--policy.revision=step-005000` loads weights and processors from the named Hub branch. Combined with the main `push_checkpoints_to_hub` feature, intermediate checkpoints become fully loadable.

### Relationship to Issue #3549

Issue #3549 ("Missing `--policy.revision` flag in rollout") is the **rollout-side** equivalent. The local patch `rollout-policy-revision` fixes it there. This prerequisite is the **training-side** fix — same concept, same pattern, different script.

---

## How It Works

### Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ Training Loop (lerobot_train.py)                         │
│                                                          │
│  for step in range(cfg.steps):                           │
│    ... forward/backward/optimizer step ...               │
│                                                          │
│    if step % save_freq == 0:                             │
│      ┌──────────────────────────────────────┐            │
│      │ save_checkpoint(local)               │  EXISTING  │
│      │   ├── pretrained_model/              │            │
│      │   │   ├── config.json                │            │
│      │   │   ├── model.safetensors          │            │
│      │   │   ├── train_config.json          │            │
│      │   │   └── processor.pth              │            │
│      │   └── training_state/                │            │
│      │       ├── optimizer_state.safetensors│            │
│      │       ├── scheduler_state.json       │            │
│      │       └── training_step.json         │            │
│      └──────────────────────────────────────┘            │
│                                                          │
│      if cfg.push_checkpoints_to_hub:                     │
│      ┌──────────────────────────────────────┐  NEW       │
│      │ push_checkpoint_to_hub(              │            │
│      │   checkpoint_dir,                    │            │
│      │   step=step,                         │            │
│      │   revision=f"step-{step:06d}"        │            │
│      │ )                                    │            │
│      │   ├── model.safetensors              │            │
│      │   ├── config.json                    │            │
│      │   ├── train_config.json              │            │
│      │   ├── processor files                │            │
│      │   └── training_state/  (V2)          │            │
│      └──────────────────────────────────────┘            │
│                                                          │
│  # End of training: push final model to main (EXISTING)  │
│  policy.push_model_to_hub(cfg)                           │
└──────────────────────────────────────────────────────────┘
```

### Revision Naming Convention

Each checkpoint is pushed as a **Git branch** on the model repo:

```
anikitakis/vla_so101_pick_n_place_full_expert
├── main                          ← final model (existing behavior)
├── latest-checkpoint             ← ALWAYS points at the most recent checkpoint (auto-updating pointer)
├── step-002500                   ← checkpoint at step 2,500
├── step-005000                   ← checkpoint at step 5,000
├── step-007500
├── step-010000
├── step-012500
├── step-015000
├── step-017500
└── step-020000
```

**`latest-checkpoint` pointer:** After each successful checkpoint push, `push_model_to_hub` also updates a `latest-checkpoint` branch to point at the just-pushed step branch. This is a lightweight delete+create operation (no file transfer — just a ref update). The user can always resume with `--policy.revision=latest-checkpoint` without knowing the exact step number. If a push fails, `latest-checkpoint` stays at the previous valid checkpoint — it's only updated after a confirmed successful upload.

**Fallback on first run:** `from_pretrained()` has a built-in fallback for `latest-checkpoint`: if the branch doesn't exist yet (first training run before any checkpoint push), it silently falls back to `main`. This covers both the config download and model weight download paths. The user can set `--policy.revision=latest-checkpoint` unconditionally — it works on first run (falls back to `main`) and every subsequent run (uses the latest checkpoint). No manual revision switching.

**Why branches (not tags):**
- Branches can be loaded as `--policy.revision=<branch-name>` in LeRobot
- Deletable with standard Hub tools
- Git-native — no custom metadata layer needed
- The step count is immediately visible in the Hub UI (branches dropdown)

**Naming format:** `step-{step:06d}` — zero-padded to 6 digits. This ensures correct sort order in listings and matches LeRobot's existing `get_step_identifier()` format (line 42-43 of `train_utils.py`).

### V1 vs V2 Scope

| | V1 (Minimal) | V2 (Full Resume) |
|:---|:---|:---|
| **Model weights** | ✅ | ✅ |
| **Config files** | ✅ | ✅ |
| **Preprocessor/Postprocessor** | ✅ | ✅ |
| **Optimizer state** | ❌ | ✅ |
| **Scheduler state** | ❌ | ✅ |
| **RNG state** | ❌ | ✅ |
| **Resume method** | `--policy.path=repo --policy.revision=step-N` (loads weights via `pretrained_path` pattern) | `--resume True --policy.path=repo --policy.revision=step-N` (true resume with optimizer/scheduler restored) |
| **LR scheduler** | Fresh scheduler with explicit flags (current runbook pattern) | Scheduler continues from saved state |
| **Upload size** | ~2 GB (model only) | ~6 GB (model + optimizer adamw states) |
| **Upload time** | ~1-2 min | ~3-5 min |

**Recommendation: Ship V1 first.** V1 solves the core pain (losing all work on disconnect) with minimal code and upload overhead. V2 is a follow-up optimization that eliminates warmup waste on resume — but the current explicit scheduler flags already handle that adequately for 5K+ chunks.

---

## Files to Modify

### 0. `src/lerobot/configs/policies.py` — `revision` field

**On `leanton` branch:** `revision: str | None = None` is already present at line 83 (from `rollout-policy-revision` patch). No change needed in source.

**In `.diff` file (upstream-ready):** The `revision` field is included as a bundled prerequisite — upstream `main` doesn't have it yet. The diff will add it right after `pretrained_path`.

`--policy.revision=step-N` is a valid CLI flag via draccus once the field exists.

### 0b. `src/lerobot/policies/factory.py` — PREREQUISITE: Pass `revision` to `from_pretrained()`

**Line ~559** (in `make_policy()`, before `from_pretrained` call):

```python
# Before line 559 — add revision to kwargs:
if cfg.revision:
    kwargs["revision"] = cfg.revision

# Existing (line 559-560):
kwargs["pretrained_name_or_path"] = cfg.pretrained_path
policy = policy_cls.from_pretrained(**kwargs)
```

**Lines ~304, ~313** (in `make_pre_post_processors()`): `PolicyProcessorPipeline.from_pretrained()` already accepts `revision` (param at `pipeline.py:623`). Add `revision=revision_param` to both calls. The function needs to accept `revision: str | None = None` as a new kwarg, or receive it from the caller's config.

### 1. `src/lerobot/configs/policies.py` — Add `push_checkpoints_to_hub` field

**Lines ~70-71** (in `PolicyConfig`):

```python
# Existing:
push_to_hub: bool = True
repo_id: str | None = None

# Add after repo_id:
push_checkpoints_to_hub: bool = False
```

This makes `--policy.push_checkpoints_to_hub=true` available as a CLI flag. The default `False` means existing training runs are unaffected.

### 2. `src/lerobot/scripts/lerobot_train.py` — Add mid-training Hub push in checkpoint block

**Lines 495-513** (inside the `is_saving_step` block):

```python
# After update_last_checkpoint(checkpoint_dir), ~line 515:

if getattr(active_cfg, "push_checkpoints_to_hub", False):
    step_id = get_step_identifier(step, cfg.steps)
    logging.info(f"Pushing checkpoint to Hub at step {step}")
    try:
        unwrapped = accelerator.unwrap_model(policy)
        if not cfg.is_reward_model_training and cfg.policy.use_peft:
            unwrapped.push_model_to_hub(cfg, peft_model=unwrapped,
                                        revision=f"step-{step_id}")
        else:
            unwrapped.push_model_to_hub(cfg, revision=f"step-{step_id}")
        logging.info(f"Checkpoint pushed to {active_cfg.repo_id} @ step-{step_id}")
    except Exception as e:
        logging.warning(f"Hub push failed (local checkpoint safe): {e}")
```

**Key design decisions in this block:**

- **Model weights only (V1):** Processors don't change mid-training — pushing them on every checkpoint is unnecessary overhead. The processor pipeline is loaded via `make_policy`/`make_pre_post_processors` with `revision` on resume. Only model weights are pushed per checkpoint.
- **`try/except` wrapper:** If the Hub push fails (network blip, rate limit), the local checkpoint is already saved. Training continues. We log a warning rather than crashing.
- **Reuses `push_model_to_hub`:** No new Hub-logic code. Same method the final push uses.
- **`get_step_identifier()`:** Reuses existing zero-padding logic from `train_utils.py` (line 41-43). At 20K steps, step 5000 → `"005000"`, producing `revision="step-005000"`.
- **`active_cfg` not `cfg`:** `active_cfg = cfg.trainable_config` (line 293) — the policy config that has `repo_id`.

### 3. `src/lerobot/policies/pretrained.py` — Add `revision` parameter to `push_model_to_hub`

**Lines 207-247** (`push_model_to_hub` method):

```python
def push_model_to_hub(
    self,
    cfg: TrainPipelineConfig,
    peft_model=None,
    revision: str | None = None,  # NEW parameter
):
    api = HfApi()
    repo_id = api.create_repo(
        repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
    ).repo_id

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        saved_path = Path(tmp) / repo_id
        # ... (same save logic, lines 221-235) ...

        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=saved_path,
            revision=revision,                        # NEW: push to branch (None → main)
            commit_message=(
                f"Checkpoint at {revision}" if revision
                else "Upload policy weights, train config and readme"
            ),
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
            ignore_patterns=["*.tmp", "*.log"],
        )
        logging.info(f"Model pushed to {commit_info.repo_url.url}"
                     f"{' [revision=' + revision + ']' if revision else ''}")
```

**Key details:**
- `revision=None` preserves existing behavior (push to `main`)
- No `api.create_branch()` needed — `upload_folder(revision=...)` auto-creates the branch on first push. Redundant `create_branch` is dead code.
- The `upload_folder` call passes `revision` so it commits to the branch, not main.
- Commit message distinguishes checkpoints from final pushes for clarity in the Hub UI.

---

## Usage After Patch

### Training Command (Colab or Cloud GPU)

```bash
lerobot-train \
  --policy.path=anikitakis/vla_so101_pick_n_place_full_expert \
  --dataset.repo_id=anikitakis/pick_n_place_task_a_annotated_dagger \
  --policy.push_to_hub=true \
  --policy.push_checkpoints_to_hub=true \
  --policy.repo_id=anikitakis/vla_so101_pick_n_place_full_expert \
  --save_freq=2500 \
  --steps=20000 \
  ... (other flags unchanged) ...
```

The `--policy.push_checkpoints_to_hub=true` flag enables mid-training pushes. Without it, behavior is unchanged.

### Resume from a Hub Checkpoint

```bash
# Resume from the latest checkpoint (recommended — no step number needed)
lerobot-train \
  --policy.path=anikitakis/vla_so101_pick_n_place_full_expert \
  --policy.revision=latest-checkpoint \
  --policy.push_checkpoints_to_hub=true \
  ...

# Or from a specific step
lerobot-train \
  --policy.path=anikitakis/vla_so101_pick_n_place_full_expert \
  --policy.revision=step-005000 \
  --policy.push_checkpoints_to_hub=true \
  ...
```

Loads weights from the named branch. The scheduler starts fresh (V1 behavior). Use `latest-checkpoint` to always get the most recent checkpoint without tracking step numbers — the pointer auto-updates after every successful checkpoint push.

### Invalidating a Checkpoint

If a checkpoint is bad (e.g., training diverged at step 7500):

```bash
# Delete the branch from Hub
huggingface-cli repo delete anikitakis/vla_so101_pick_n_place_full_expert \
  --branch step-007500 --type model

# Or via Python:
from huggingface_hub import HfApi
api = HfApi()
api.delete_branch("anikitakis/vla_so101_pick_n_place_full_expert", branch="step-007500", repo_type="model")
```

The `main` branch and other checkpoints are untouched.

### Cleanup Checkpoints After Training

Once training completes successfully, old intermediate checkpoints can be cleaned up:

```bash
# Delete all checkpoint branches
for step in 002500 005000 007500 010000 012500 015000 017500; do
  huggingface-cli repo delete anikitakis/vla_so101_pick_n_place_full_expert \
    --branch step-$step --type model
done
```

Only `main` (the final model) remains. This keeps the Hub repo clean.

---

## Edge Cases & Risks

### 1. Push fails mid-training (network blip)
**Handled.** The `try/except` around the push ensures training continues. The local checkpoint is already saved. Next `save_freq` will retry the push.

### 2. Colab disconnects during Hub push
**Acceptable risk.** The upload is an atomic operation (single `upload_folder` call). If it fails mid-transfer, the branch either has the old content (from a prior push at the same step — unlikely) or is in an incomplete state. Incomplete state = load fails with corrupt weights → user skips to the previous checkpoint. The local filesystem is already gone (Colab disconnect), but the previous checkpoint branch is safe on Hub.

### 3. Re-pushing to same step (e.g., resume + re-run)
**Handled.** `upload_folder(revision=...)` auto-creates the branch on first push and creates a new commit on subsequent pushes. No error. No ambiguity. The latest commit wins.

### 4. Large model upload time blocking training
**Minimal impact.** For SmolVLA (~100M trainable params, ~2 GB on disk):
- L4 GPU + Colab internet: ~1-2 minutes per upload
- At `save_freq=2500` with 6K steps: 2 uploads per run (~3-4 minutes total overhead)
- At `save_freq=5000` with 20K steps: 4 uploads (~6-8 minutes total overhead)
- This is negligible compared to ~6 hours of training

For larger models (e.g., PI0.5, ~1B+ params): upload time grows. Consider increasing `save_freq` accordingly. But SmolVLA is the target for this patch.

### 5. Optimizer/scheduler state not pushed (V1 limitation)
**Acknowledged.** V1 pushes model weights only. When resuming from Hub checkpoint via `pretrained_path`, the optimizer and scheduler start fresh. This means:
- ~100 warmup steps wasted (acceptable with `--policy.scheduler_warmup_steps=100`)
- Slightly different loss trajectory vs true resume
- This is the same trade-off the current runbook already makes with segmented 5K chunks

### 6. Private repos
**Handled.** `api.create_repo()` already respects `self.config.private`. If the model repo is private, checkpoint branches are also private.

---

## Testing Plan

### Manual Test (Colab, L4 GPU)

1. **Setup:** Fork LeRobot with patch applied. Use a small test dataset (5 episodes from `pick_n_place_task_a_annotated`).
2. **Command:**
   ```bash
   lerobot-train \
     --policy.path=lerobot/smolvla_base \
     --dataset.repo_id=anikitakis/pick_n_place_task_a_annotated \
     --policy.push_checkpoints_to_hub=true \
     --policy.repo_id=anikitakis/test-checkpoint-push \
     --save_freq=200 \
     --steps=1000 \
     --batch_size=32
   ```
3. **Verify:**
   - Checkpoints appear on Hub at `step-000200`, `step-000400`, `step-000600`, `step-000800`, `step-001000`
   - Each branch is loadable: `--policy.path=anikitakis/test-checkpoint-push --policy.revision=step-000400`
   - `main` branch has the final model (step 1000)
   - Training log shows "Pushing checkpoint to Hub at step 200" etc.
4. **Test disconnect recovery:**
   - Run training, kill Colab runtime mid-training
   - Restart runtime, resume: `--policy.path=anikitakis/test-checkpoint-push --policy.revision=step-000400 --steps=1000`
   - Training continues from step 400 (with fresh optimizer, V1 behavior)
5. **Test invalidation:** Delete `step-000200` branch, verify it's gone from Hub UI and cannot be loaded.
6. **Cleanup:** Delete test repo `anikitakis/test-checkpoint-push`.

### Regression Test

- Run existing training command **without** `--policy.push_checkpoints_to_hub` flag. Verify behavior is unchanged (no mid-training pushes, final push to main works as before).

---

## Follow-Up: V2 (Optimizer/Scheduler State)

Once V1 is stable, V2 extends the checkpoint push to include `training_state/` (optimizer, scheduler, RNG state). This requires:

1. **`push_model_to_hub`** also uploads `training_state/*.safetensors` and `training_state/*.json`
2. **`load_training_state`** gains a Hub-aware variant that downloads from a revision
3. **`resume: True`** works from Hub checkpoints — not just local paths
4. **Upload size increases** ~3× (AdamW states are 2× model weights). Acceptable at `save_freq ≥ 5000`.

V2 is not blocking — V1 with explicit LR flags already works well for the DAGGER fine-tuning cycle.

---

## Related

- [[7. SMOLVLA_FULL_EXPERT_TRAINING_RUNBOOK]] — current runbook with segmented 5K chunks
- [[AGENT-WARMUP]] — session state and LR collapse analysis
- [[OPEN_UPSTREAM_ISSUES]] — "Push intermediate checkpoints to Hub" already listed under "Not Yet Filed"
- [[bugfix_rollout_policy_revision]] — existing patch for `--policy.revision` flag support (complementary — ensures revision loading works for checkpoints too)
