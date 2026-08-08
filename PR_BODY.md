## Summary / Motivation

`multi_task_dit` currently hardcodes CLIP for both the vision and text encoder. This PR lets the
existing `vision_encoder_name` / `text_encoder_name` fields each accept three encoder families:
**CLIP** (default, unchanged), **DINOv3** (vision), and **SigLIP 2** (vision + text), chosen
automatically from each checkpoint's `model_type`. No schema change, just broader accepted values.

DINOv3 and SigLIP 2 are strong alternative encoders that can generalize better than CLIP on some
setups, while staying within this policy's lightweight, single-`AutoModel` design. The change is
**additive**: the CLIP path is behaviorally identical (same classes, same pooling, same attention
mask; the only addition is one `AutoConfig` model-type read at construction), and per-family handling
is a couple of `if` branches, with no base-class/subclass tree, consistent with the original PR
keeping the CLIP/LBM baseline intact.

**Validated in practice:** I used the DINOv3 path to train a real SO-101 towel-folding policy for the
ETH Zürich *Robot Learning* course cloth-folding project (including generalization to cloths unseen
during training). Full pipeline + training/eval recipe:
[LarsvanDorp/laundry_folding_robot](https://github.com/LarsvanDorp/laundry_folding_robot); deployed
model: [`larsvandorp/folding_dit`](https://huggingface.co/larsvandorp/folding_dit) (`multi_task_dit`
+ DINOv3 ViT-B/16). Swapping CLIP for DINOv3 / SigLIP 2 is an increasingly common move; both are
widely regarded as stronger general-purpose encoders than CLIP, and in our course most teams replaced
the CLIP encoder with DINOv3 or SigLIP 2.

## What changed

- `VisionEncoder` / `TextEncoder` dispatch on the checkpoint's `model_type` via `AutoModel` + small
  per-family handling:
  - **CLIP / DINOv3** use the CLS token (`last_hidden_state[:, 0]`); DINOv3's register tokens are skipped.
  - **SigLIP 2** uses the attention-pooled `pooler_output` (no CLS token). Fixed-resolution checkpoints
    (e.g. `siglip2-base-patch16-224`) only; NaFlex variants are rejected with a clear error.
- Text supports CLIP or SigLIP 2; vision-only encoders (DINO) are rejected. The tokenizer auto-selects
  from `text_encoder_name`.
- `hidden_size` is read dynamically for the projection. Config validation gives clear errors for:
  vision-only-as-text, unknown family, SigLIP text `>64` tokens, and NaFlex.
- Image normalization is unchanged (handled by the policy's preprocessing pipeline, same for every
  encoder, so the CLIP path stays byte-identical).
- Docs (`multi_task_dit.mdx` + policy README) and tests updated.
- **No breaking changes**: CLIP defaults preserved.

## How was this tested (or how to run locally)

Tests in `tests/policies/multi_task_dit/test_multi_task_dit.py` (local-only, matching the existing
module; CI skips it since it downloads gated/large weights) now exercise each family in both valid
slots plus the config guards. Verified locally with real weights (`clip-vit-base-patch16`,
`dinov3-vitb16`, `siglip2-base-patch16-224`):

```bash
pytest -q tests/policies/multi_task_dit/test_multi_task_dit.py   # 22 passed
pre-commit run -a                                                # ruff / typos / ... clean
```

Selecting an encoder:

```bash
# Vision
--policy.vision_encoder_name=openai/clip-vit-base-patch16              # default
--policy.vision_encoder_name=facebook/dinov3-vitb16-pretrain-lvd1689m
--policy.vision_encoder_name=google/siglip2-base-patch16-224

# Text (DINO is vision-only)
--policy.text_encoder_name=openai/clip-vit-base-patch16               # default
--policy.text_encoder_name=google/siglip2-base-patch16-224
--policy.tokenizer_max_length=64       # set with SigLIP 2 text (64-token context)
```

## Checklist (required before merge)

- [x] Linting/formatting run (`pre-commit run -a`)
- [x] All tests pass locally (the affected `multi_task_dit` tests; `pytest`)
- [x] Documentation updated
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #3695

## Reviewer notes

- CLIP is the default and unchanged, so the diff is additive. Validation is **family-level**, so other
  CLIP/DINO/SigLIP checkpoints (any size) work through the same paths; CLIP/DINOv3/SigLIP 2 are the
  tested-and-recommended set.
