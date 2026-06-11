# Feature Request: Push Intermediate Checkpoints to Hub During Training

## Problem

`push_to_hub=true` only pushes the **final model** after all training steps complete (`lerobot_train.py:583-591`). Intermediate checkpoints saved via `save_freq` are written to local disk only (`lerobot_train.py:505-521`). On Colab or other ephemeral runtimes, a disconnection loses all local checkpoints — the user must restart training from scratch.

## Current Behavior

```
save_freq=2000  →  checkpoint saved locally at step 2000, 4000, 6000...
push_to_hub=true →  final model pushed ONCE after the loop ends
```

If the runtime dies at step 4500, the user loses everything — no checkpoint exists on the Hub.

## Proposed Behavior

Add a config flag (e.g., `push_checkpoints_to_hub: bool = False`) that, when enabled, pushes each intermediate checkpoint to the Hub as it's saved. This gives:

- **Colab resilience**: runtime disconnect → resume from last pushed checkpoint on Hub
- **Training monitoring**: inspect model at intermediate steps without downloading the full output directory
- **Experiment safety net**: long training runs (20K+ steps) don't need to restart from zero on any failure

## Suggested Implementation

In `lerobot_train.py`, inside the `is_saving_step` block (line ~505), add:

```python
if cfg.push_checkpoints_to_hub:
    unwrapped_model.push_model_to_hub(cfg, commit_message=f"Checkpoint step {step}")
    preprocessor.push_to_hub(cfg.repo_id)
    postprocessor.push_to_hub(cfg.repo_id)
```

Add the config field in `src/lerobot/configs/train.py` or `policies.py`:

```python
push_checkpoints_to_hub: bool = False
```

## Workaround (Current)

A background thread using `huggingface_hub.HfApi.upload_folder` to watch the local checkpoints directory and push new ones as they appear. Functional but fragile — needs to run in a separate Colab cell, can't survive its own disconnection.

## Related

- Colab runtime disconnection risk (12h cap, idle timeout)
- The `save_freq` config already creates full checkpoint directories locally — the infrastructure exists, only the upload call is missing
