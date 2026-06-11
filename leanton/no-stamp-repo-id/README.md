# no-stamp-repo-id

**Target:** `src/lerobot/configs/dataset.py`
**Status:** `active`
**GitHub:** [#3722](https://github.com/huggingface/lerobot/issues/3722)

## What

Adds `no_stamp: bool = False` to `DatasetRecordConfig` and an early return in `stamp_repo_id()`. Use `--dataset.no_stamp true` to skip the timestamp suffix on `repo_id`.

## Why

`LeRobotDatasetConfig.stamp_repo_id()` unconditionally appends `_YYYYMMDD_HHMMSS` to the `repo_id` for every non-resume rollout. Users who supply their own versioned repo names (e.g., `rollout_pick_n_place_dagger_r1`, `r2`, etc.) are forced into timestamped repos with no way to opt out.

## Fix

```python
# In DatasetRecordConfig:
no_stamp: bool = False

# In stamp_repo_id():
if self.no_stamp:
    return
```

Usage: `--dataset.no_stamp true` in the rollout command.

## Validate

**User:** Run `lerobot-rollout --dataset.no_stamp true --dataset.repo_id=anikitakis/test ...`. The dataset on Hub should be named exactly `anikitakis/test` with no timestamp suffix. Without the flag, `anikitakis/test_20260609_...` (stamped).

**Agent:**
```bash
grep -q "no_stamp: bool = False" ~/lerobot/src/lerobot/configs/dataset.py && echo "Field ✅" || echo "MISSING"
grep -q "if self.no_stamp:" ~/lerobot/src/lerobot/configs/dataset.py && echo "Guard ✅" || echo "MISSING"
```
