# PR: feat(policies): add delta action support for pi0, pi0.5, and pi0_fast

## Type / Scope

- **Type**: Feature
- **Scope**: policies/pi0, policies/pi05, policies/pi0_fast, processor, datasets

## Summary / Motivation

OpenPI trains pi0/pi0.5 models with **delta actions** — the model sees `action - current_state` rather than absolute joint positions. This makes learning easier (targets centered around zero) and can produce smoother action-chunk transitions. LeRobot's pi0 family was missing this transform, causing a distribution mismatch when fine-tuning from OpenPI pretrained weights and sometimes resulting in jittery motion during inference.

This PR adds an optional `use_delta_actions: bool` config flag to all three pi policies. When enabled, actions are converted to deltas (relative to current state) before the model during training, and converted back to absolute after prediction during inference. It also includes full support for Real-Time Chunking (RTC) with delta actions and a CLI command for precomputing delta action statistics.

## Related issues

- Related: https://github.com/huggingface/lerobot/pull/2891#issuecomment-3857478982

## What changed

### Processor pipeline

- **`delta_action_processor.py`**: Added `to_delta_actions()` / `to_absolute_actions()` functions, `DeltaActionsProcessorStep` (preprocessor) and `AbsoluteActionsProcessorStep` (postprocessor). Supports `delta_exclude_joints` to keep specific dimensions (e.g. gripper) in absolute space.
- **`normalize_processor.py`**: Added attribution comments for the `1e-6` epsilon, referencing OpenPI (`src/openpi/transforms.py`).
- **`processor/__init__.py`**: Updated exports for new processor steps.

### Policy configs & processors

- **`configuration_pi0.py` / `configuration_pi05.py` / `configuration_pi0_fast.py`**: Added `use_delta_actions`, `delta_exclude_joints`, and `action_feature_names` config fields.
- **`processor_pi0.py` / `processor_pi05.py` / `processor_pi0_fast.py`**: Added `DeltaActionsProcessorStep` to all three preprocessor pipelines and `AbsoluteActionsProcessorStep` to postprocessors.
- **`factory.py`**: Added `_reconnect_delta_absolute_steps()` to wire the postprocessor's `AbsoluteActionsProcessorStep` back to the preprocessor's cached state after deserialization. Stores `action_feature_names` from dataset metadata on the config.

### Dataset stats

- **`compute_stats.py`**: Added `compute_delta_action_stats()` — samples random action chunks, converts to delta space (relative to first state of each chunk), and computes statistics. This matches what the model sees during training.
- **`dataset_tools.py`**: Extended `recompute_stats()` with `delta_action`, `delta_exclude_joints`, and `chunk_size` parameters. When `delta_action=True`, delegates action stats to `compute_delta_action_stats`.

### CLI

- **`lerobot_edit_dataset.py`**: Added `recompute_stats` operation so users can precompute delta action stats from the command line:
  ```bash
  lerobot-edit-dataset \
      --repo_id your_dataset \
      --operation.type recompute_stats \
      --operation.delta_action true \
      --operation.chunk_size 50 \
      --operation.delta_exclude_joints "['gripper']"
  ```

### Training

- **`lerobot_train.py`**: Removed the on-the-fly delta action stats computation (~70 lines). Delta stats are now expected to be precomputed via `recompute_stats`. Uses `ACTION` / `OBS_STATE` constants instead of hardcoded strings.

### Documentation

- **`docs/source/pi0.mdx`** / **`docs/source/pi05.mdx`**: Added "Delta (Relative) Actions" sections explaining both the CLI and Python approaches for precomputing stats and enabling delta actions during training.
- **`policies/pi0/README.md`** / **`policies/pi05/README.md`**: Added delta action documentation.

### Tests

- **`tests/policies/test_delta_actions.py`** (new, 7 tests): Roundtrip, mutation, exclude_joints mask, and processor step tests.
- **`tests/policies/rtc/test_rtc_delta_actions.py`** (new, 24 tests): Full RTC + delta actions integration — verifies delta conversion through `ActionQueue`, inpainting with delta leftovers, non-delta policy compatibility, and numerical correctness.
- **`tests/policies/rtc/test_action_queue.py`**: Removed tests for the unused `get_processed_left_over()` method.

### Cleanup

- Removed unused `get_processed_left_over()` from `ActionQueue`.
- Replaced hardcoded `"action"` / `"observation.state"` strings with `ACTION` / `OBS_STATE` constants from `utils/constants.py`.

No breaking changes. Disabled by default.

## How was this tested (or how to run locally)

- Tests added: `tests/policies/test_delta_actions.py` and `tests/policies/rtc/test_rtc_delta_actions.py` (31 tests total)

  ```bash
  USE_TF=0 pytest tests/policies/test_delta_actions.py tests/policies/rtc/test_rtc_delta_actions.py -v
  ```

- To train with delta actions:

  ```bash
  # 1. Precompute delta stats
  lerobot-edit-dataset \
      --repo_id your_dataset \
      --operation.type recompute_stats \
      --operation.delta_action true \
      --operation.chunk_size 50

  # 2. Train
  lerobot-train \
      --dataset.repo_id=your_dataset \
      --policy.type=pi0 \
      --policy.use_delta_actions=true
  ```

## Checklist (required before merge)

- [x] Linting/formatting run (`pre-commit run -a`)
- [x] All tests pass locally (`pytest`)
- [x] Documentation updated
- [ ] CI is green

## Reviewer notes

- The delta conversion lives in the **processor pipeline** (`DeltaActionsProcessorStep` / `AbsoluteActionsProcessorStep`). The `AbsoluteActionsProcessorStep` reads cached state from its paired `DeltaActionsProcessorStep` — the wiring is done in `factory.py::_reconnect_delta_absolute_steps()` after deserialization.
- `delta_exclude_joints` allows keeping specific joints (e.g. gripper) in absolute space. The mask is built from `action_feature_names` on the config.
- Delta action stats must be precomputed before training (via `lerobot-edit-dataset --operation.type recompute_stats --operation.delta_action true`). The training script no longer computes them on the fly to avoid NCCL timeouts in multi-GPU setups.
- The `1e-6` epsilon in normalization matches OpenPI's implementation (`src/openpi/transforms.py`).
