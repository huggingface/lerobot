# FLUX 3 Action PEFT example

[`lora.json`](lora.json) is a shared task-adaptation example for existing SO-101 and
DROID checkpoints. It contains training settings only. Pass `--policy.path` to load
the checkpoint's model configuration and saved processors; no example config is
needed for inference. The same `lora.json` is included in each Hub package.

See the [FLUX 3 Action guide](../../docs/source/flux3.mdx) for installation, checkpoint
downloads, dataset requirements, camera mapping and larger training budgets.
The policy and shared encoders load automatically by repo ID:

```bash
# SO-101
lerobot-train \
  --config_path=examples/flux3/lora.json \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --dataset.repo_id=YOUR_ORG/YOUR_SO101_DATASET \
  --output_dir=outputs/so101_lora

# DROID: retain the checkpoint's execution and inference settings.
lerobot-train \
  --config_path=examples/flux3/lora.json \
  --policy.path=black-forest-labs/flux-3-action-droid \
  --dataset.repo_id=YOUR_ORG/YOUR_DROID_TASK_DATASET \
  --output_dir=outputs/droid_lora
```

The default is one GPU, effective batch 8 and 2,500 optimizer updates, with rank/alpha
32 LoRA and EMA decay 0.999. Outputs stay local unless Hub uploads are enabled.

SO-101 retains its saved inference defaults: 32 executed actions, guidance 3/3 and no
compilation. For guidance 4/1 and compilation, explicitly add
`--policy.guidance_scale=4 --policy.guidance_scale_action=1 --policy.compile_model=true`.

**DROID validation status:** Configuration loading has been checked, but end-to-end
DROID PEFT training and task performance have not yet been validated.

For training a new robot embodiment from the base model or full finetuning, use our
standalone [flux-action](https://github.com/black-forest-labs/flux-action) repository.
