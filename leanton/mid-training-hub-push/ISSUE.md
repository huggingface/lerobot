# Upstream Issue Draft — mid-training-hub-push

**Title:** Push intermediate training checkpoints to Hub

## Motivation

`push_to_hub=true` only pushes the final model after training completes. On Colab (or any ephemeral runtime), a disconnection loses all local checkpoints. The user runs 6K-step training chunks on L4 GPUs (~6 hours) and has lost multiple runs to disconnects.

## Proposed solution

A `push_checkpoints_to_hub` config flag (default `False`) that, when enabled, pushes model weights to the HuggingFace Hub at each `save_freq` interval as a named branch (`step-002500`, `step-005000`, etc.). A `latest-checkpoint` branch auto-updates to always point at the most recent successful push, so resume is `--policy.revision=latest-checkpoint` — no step numbers to track. On first run (before any checkpoint exists), `from_pretrained()` silently falls back to `main`.

Five files changed:
- `configs/policies.py` — add `push_checkpoints_to_hub` field + generic revision fallback
- `policies/pretrained.py` — add `revision` param to `push_model_to_hub()` + `latest-checkpoint` pointer
- `policies/factory.py` — wire `revision` through `make_policy()` and `make_pre_post_processors()`
- `scripts/lerobot_train.py` — push model to Hub in checkpoint block when flag is set
- `processor/pipeline.py` — `from_pretrained()` fallback to `main` for processor loading

## Implementation

Branch: `anikita/lerobot:feat/mid-training-hub-push`
Patch: `leanton/mid-training-hub-push/mid-training-hub-push.diff` (320 lines, rebased on main@30790de1)

## Scope (V1)

- Model weights only (no optimizer/scheduler state) — documented as future work
- `latest-checkpoint` pointer uses delete+create (not atomic) — single-user training
- Requires recent `huggingface_hub` with branch support

## Tested

End-to-end on SO-101 + SmolVLA, both Colab and local Ubuntu.
