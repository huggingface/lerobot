# pi0_fast: `make_policy` retains stale `input_features` from base model, breaking custom datasets

## Motivation

Training `pi0_fast` on a custom dataset fails with "All image features are missing from the batch" because `make_policy()` preserves the base model's `input_features` (LIBERO-style keys) instead of populating them from the actual dataset. Additionally, when `rename_map` is used, feature keys in the config don't match the post-rename keys in the batch.

This affects anyone fine-tuning `pi0_fast` (or any other policy with a pretrained base model that ships with `input_features`) on a custom dataset with different camera keys and/or `rename_map`.

## Proposed solution

Two changes in `src/lerobot/policies/factory.py`:

1. Always populate `cfg.input_features` from the dataset/env features, removing the `if not cfg.input_features` guard that preserves stale defaults from base model configs.

2. Apply `rename_map` to feature keys before populating `input_features`, so config keys match what `rename_observations_processor` produces at runtime.

## Implementation

Branch: `anikita/lerobot:leanton`
Diff: `leanton/pi0fast-input-features-from-dataset/pi0fast-input-features-from-dataset.diff`
Files changed: 1 (`src/lerobot/policies/factory.py`)

## Scope / limitations

- The rename_map application is safe because `output_features` (action keys) are set before the rename and won't be affected (rename_map only maps observation keys).
- This change also removes the need for policy-specific `input_features` defaults in base model configs — those become dead weight.

## Testing

- Trained π0_fast on a custom 3-camera pick-and-place dataset with `rename_map`. Training starts without the "missing image features" error.
- Verified that existing SmolVLA training flow is unaffected (SmolVLA checkpoints already have `input_features` populated from their dataset).
