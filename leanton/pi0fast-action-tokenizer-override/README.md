# pi0fast: flow --policy.action_tokenizer_name to preprocessor overrides

**Target:** `src/lerobot/scripts/lerobot_train.py`
**Status:** active
**GitHub:** [#3846](https://github.com/huggingface/lerobot/issues/3846)
**Diff basis:** `origin/main` @ `30790de1` (clean upstream, 2026-06-21)

## What

The training script now passes `policy.action_tokenizer_name` (when set) to the preprocessor override dict, so the CLI flag actually reaches the `action_tokenizer_processor` step.

## Why

`pi0fast-base`'s `policy_preprocessor.json` hardcodes `action_tokenizer_name: "physical-intelligence/fast"`. That repo is broken under transformers ≥ 5.x (missing `bpe_tokenizer/` subfolder required for non-standard tokenizer attribute names). The CLI flag `--policy.action_tokenizer_name=lerobot/fast-action-tokenizer` exists and is documented, but it only set a policy config attribute — it never flowed through to the preprocessor step config. The preprocessor always loaded the hardcoded (broken) tokenizer.

## Validate

**User:** Train π0_fast with `--policy.action_tokenizer_name=lerobot/fast-action-tokenizer`. The preprocessor loads the correct tokenizer.

**Agent:**
```bash
grep -q "action_tokenizer_name" ~/lerobot/src/lerobot/scripts/lerobot_train.py | grep "preprocessor_overrides" && echo "✅" || echo "MISSING"
```
