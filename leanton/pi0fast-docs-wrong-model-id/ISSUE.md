# pi0fast.mdx: wrong model ID in training and evaluation code blocks

## Motivation

The π0_fast documentation (`docs/source/pi0fast.mdx`) contains two incorrect model IDs:
- `--policy.pretrained_path=lerobot/pi0_fast_base` (line 99) — 404
- `--policy.path=lerobot/pi0fast_base` (line 190) — 404

The correct repo ID is `lerobot/pi0fast-base` (hyphens, not underscores). This is already used elsewhere in the same file (line 177) and in the source code (`src/lerobot/policies/pretrained.py:367`), confirming it's the intended ID.

## Proposed solution

Replace `pi0_fast_base` → `pi0fast-base` and `pi0fast_base` → `pi0fast-base` in the two affected code blocks.

## Implementation

Branch: `anikita/lerobot:leanton`
Diff: `leanton/pi0fast-docs-wrong-model-id/pi0fast-docs-wrong-model-id.diff`
Files changed: 1 (`docs/source/pi0fast.mdx`)

## Scope / limitations

- Docs-only change, no code impact.
- The correct model ID is already used in `pretrained.py`'s default mapping — this fix just aligns the documentation.
