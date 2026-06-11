# rollout-policy-revision

**Target:** `src/lerobot/configs/policies.py`, `src/lerobot/rollout/configs.py`, `src/lerobot/rollout/context.py`
**Status:** `active`
**GitHub:** [#3549](https://github.com/huggingface/lerobot/issues/3549)

## What

Adds `--policy.revision` CLI flag to `lerobot-rollout`, allowing users to load a specific HuggingFace Hub model revision (commit hash, branch, or tag).

## Why

`lerobot-rollout` has no mechanism to load a specific model revision. Users who push multiple checkpoints during training and discover the latest is overfitted cannot roll back to an earlier commit. The underlying infrastructure (`PreTrainedConfig.from_pretrained()`, `PeftModel.from_pretrained()`) all support `revision` natively — it was just never wired to the CLI.

Three files needed changes:
1. **`configs/policies.py`** — add `revision` field to `PreTrainedConfig`, store after draccus parse
2. **`rollout/configs.py`** — extract `--policy.revision` from CLI and pass to `from_pretrained()`
3. **`rollout/context.py`** — pass `revision` to all model weight-loading calls (`PeftConfig`, `PeftModel`, `policy_class`)

## Validate

**User:** Run `lerobot-rollout --policy.revision=<commit_hash> ...` with a specific revision. The model should load from that commit. `lerobot-rollout --help` should show `--policy.revision`.

**Agent:**
```bash
grep -q "revision.*str.*None" ~/lerobot/src/lerobot/configs/policies.py && echo "Config field ✅" || echo "MISSING"
grep -q "revision" ~/lerobot/src/lerobot/rollout/configs.py && echo "Config wiring ✅" || echo "MISSING"
grep -q "revision" ~/lerobot/src/lerobot/rollout/context.py && echo "Context wiring ✅" || echo "MISSING"
```
