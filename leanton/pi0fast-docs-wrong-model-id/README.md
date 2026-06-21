# pi0fast: fix wrong model ID in pi0fast.mdx

**Target:** `docs/source/pi0fast.mdx`
**Status:** active
**GitHub:** [#3847](https://github.com/huggingface/lerobot/issues/3847)
**Diff basis:** `origin/main` @ `30790de1` (clean upstream, 2026-06-21)

## What

Fix incorrect `pretrained_path` and `policy.path` values in the π0_fast documentation. `lerobot/pi0_fast_base` and `lerobot/pi0fast_base` (underscores) do not exist — the correct model ID is `lerobot/pi0fast-base` (hyphens).

## Why

Copy-paste of the training command from the docs fails with a 404 error when trying to download the nonexistent base model. The correct repo ID (`lerobot/pi0fast-base`) is used elsewhere in the same file and in the source code (`pretrained.py`), but two code blocks have the wrong ID.

## Validate

**User:** Copy the training command from the docs. Model downloads successfully.

**Agent:**
```bash
grep -c "pi0_fast_base\|pi0fast_base" ~/lerobot/docs/source/pi0fast.mdx | grep -q "^0$" && echo "✅" || echo "STILL WRONG"
```
