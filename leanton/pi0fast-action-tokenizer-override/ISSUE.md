# pi0_fast: --policy.action_tokenizer_name has no effect (dead CLI flag)

## Motivation

The `--policy.action_tokenizer_name` flag is documented as a way to override the action tokenizer used by the FAST processor. However, the training script never passes this value to the preprocessor override dict, so the preprocessor always loads the tokenizer hardcoded in the base model's `policy_preprocessor.json`.

This is critical because `pi0fast-base` hardcodes `physical-intelligence/fast`, which is broken under transformers ≥ 5.x: its `bpe_tokenizer/` subfolder is missing, causing `AutoProcessor.from_pretrained` to fail. The fixed tokenizer (`lerobot/fast-action-tokenizer`) exists and has the correct structure, but there's no way to use it without this fix.

## Proposed solution

In `src/lerobot/scripts/lerobot_train.py`, when `active_cfg.action_tokenizer_name` is set, add it as an override to `preprocessor_overrides["action_tokenizer_processor"]`.

## Implementation

Branch: `anikita/lerobot:leanton`
Diff: `leanton/pi0fast-action-tokenizer-override/pi0fast-action-tokenizer-override.diff`
Files changed: 1 (`src/lerobot/scripts/lerobot_train.py`)

## Scope / limitations

- Only affects training; inference uses the tokenizer baked into the saved checkpoint.
- The override key `action_tokenizer_processor` matches the registry name in the preprocessor config JSON.

## Testing

- Trained π0_fast with `--policy.action_tokenizer_name=lerobot/fast-action-tokenizer`. The preprocessor loaded the correct tokenizer instead of the broken `physical-intelligence/fast`.
