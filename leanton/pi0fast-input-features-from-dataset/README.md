# pi0fast: populate input_features from dataset and apply rename_map

**Target:** `src/lerobot/policies/factory.py`
**Status:** active
**GitHub:** [#3845](https://github.com/huggingface/lerobot/issues/3845)
**Diff basis:** `origin/main` @ `30790de1` (clean upstream, 2026-06-21)

## What

`make_policy()` now always populates `cfg.input_features` from the dataset (or env) features, overriding any generic defaults baked into a base model's `config.json`. When `rename_map` is provided, feature keys are renamed before assignment so config keys match what `rename_observations_processor` produces at runtime.

## Why

π0_fast's base model (`lerobot/pi0fast-base`) ships with LIBERO-style input features:
```
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

The old code (`if not cfg.input_features`) preserved these stale defaults because `input_features` was non-empty. The policy's `prepare_images()` then looked for `base_0_rgb` in the batch — which never exists for any dataset except LIBERO. This breaks training with OOM on a misleading error about missing image features.

Additionally, when `rename_map` is set (e.g., `observation.images.annotated → observation.images.camera1`), the config's `input_features` were populated with pre-rename dataset keys, but the batch arriving at `policy.forward()` has post-rename keys (renamed by the preprocessor). The mismatch caused "All image features are missing from the batch" for any dataset using `rename_map`.

## Validate

**User:** Train π0_fast on a non-LIBERO dataset with `rename_map`. Training starts and loss decreases.

**Agent:**
```bash
grep -q "rename_map.get(k, k)" ~/lerobot/src/lerobot/policies/factory.py && echo "✅" || echo "MISSING"
```
