# Rename Map and Empty Cameras

When you train, evaluate, or record with a robot policy, your **dataset** or **environment** provides observations under one set of keys (e.g. `observation.images.front`, `observation.images.eagle`), while your **policy** expects another (e.g. `observation.images.image`, `observation.images.image2`). The **rename map** bridges that gap without changing the policy or data source.

> **Scope:** The rename map only renames **observation** keys (images and state). Action keys are not affected.

## Why observation keys don't always match

Policies have a fixed set of **input feature names** baked into their pretrained config. For example:

- [pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero) expects `observation.images.base_0_rgb` and `observation.images.left_wrist_0_rgb`.
- [xvla-base](https://huggingface.co/lerobot/xvla-base) expects `observation.images.image`, `observation.images.image2`, and `observation.images.image3`.

Your dataset might use different names entirely (e.g. `observation.images.front`, `observation.images.eagle`, `observation.images.glove`), and your eval environment might use yet another set. Rather than editing the policy config or renaming columns in the dataset, you pass a **rename map**: a JSON dictionary that maps source keys to the keys the policy expects. Renaming happens inside the preprocessor pipeline, so the policy always sees its expected keys.

## Using the rename map

Pass the mapping as a JSON string on the command line. The convention is always:

```
--rename_map='{"source_key": "policy_key", ...}'
```

where **source_key** is what the dataset or environment provides, and **policy_key** is what the policy expects.

Only listed keys are renamed; everything else passes through unchanged. Order of entries doesn't matter.

Supported policies: **PI0**, **PI05**, **PI0Fast**, **SmolVLA**, and **XVLA**.

### Training

Suppose you fine-tune [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) on a dataset with images under `observation.images.front`, `observation.images.eagle`, and `observation.images.glove`. XVLA expects `observation.images.image`, `observation.images.image2`, and `observation.images.image3`:

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=./outputs/xvla_training \
  --job_name=xvla_training \
  --policy.path="lerobot/xvla-base" \
  --policy.repo_id="HF_USER/xvla-your-robot" \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --steps=20000 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --rename_map='{"observation.images.front": "observation.images.image", "observation.images.eagle": "observation.images.image2", "observation.images.glove": "observation.images.image3"}'
```

### Evaluation

A policy that expects `observation.images.base_0_rgb` and `observation.images.left_wrist_0_rgb` (e.g. [pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero)), but the LIBERO environment returns `observation.images.image` and `observation.images.image2`:

```bash
lerobot-eval \
  --policy.path=lerobot/pi0fast-libero \
  --env.type=libero \
  ... \
  --rename_map='{"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.left_wrist_0_rgb"}'
```

## Alternative: edit the policy config directly

If you always use the same dataset or environment, you can **edit the policy's `config.json`** so its observation keys match your data source. Then no rename map is needed.

The tradeoff: modifying the policy config ties it to one data source. A rename map keeps one policy usable across many datasets and environments.

## Empty cameras: fewer views than the policy expects

Some policies are built for a fixed number of image inputs. If your dataset has fewer cameras, you can set **`empty_cameras`** in the policy config instead of modifying the model architecture.

### How it works

Setting `empty_cameras=N` adds N placeholder image features to the policy config, named:

```
observation.images.empty_camera_0
observation.images.empty_camera_1
...
```

At runtime, these keys have no corresponding data in the batch. The policy fills them with masked dummy tensors (padded with `-1` for SigLIP-based vision encoders, with a zero attention mask), so the extra image slots are effectively ignored during training and inference.

### Example

XVLA-base has three visual inputs and `empty_cameras=0` by default. Your dataset only has two cameras:

1. Set `--policy.empty_cameras=1`.
2. The config adds a third key: `observation.images.empty_camera_0`.
3. Use the rename map for your two real cameras as usual.
4. The third slot is masked out — no fake images needed in your dataset.

## Quick reference

| Goal                                    | What to do                                                                  |
| --------------------------------------- | --------------------------------------------------------------------------- |
| Dataset keys ≠ policy keys              | `--rename_map='{"dataset_key": "policy_key", ...}'`                         |
| Env keys ≠ policy keys (eval)           | `--rename_map='{"env_key": "policy_key", ...}'`                             |
| Rollout with different keys (inference) | `--rename_map='{"source_key": "policy_key", ...}'`.                         |
| Fewer cameras than policy expects       | `--policy.empty_cameras=N` (supported by PI0, PI05, PI0Fast, SmolVLA, XVLA) |
| Avoid passing a rename map              | Edit the policy's `config.json` so its keys match your data source          |
