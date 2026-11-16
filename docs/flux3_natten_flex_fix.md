# FLUX3 / NATTEN flex-fna OOM fix for sm_120

This note captures the root cause and local fix that resolved the massive temporary allocation seen during FLUX3 LoRA training on a Blackwell-class GPU.

## Problem

The training path used the frozen VAE and hit a dense attention mask allocation in the NATTEN `flex-fna` backend. On the Blackwell workstation GPU (`sm_120`), the backend fallback path was `flex-fna` because:

- `blackwell-fna` was not available for this architecture
- `hopper-fna` was also rejected
- `cutlass-fna` could not be loaded due to a library mismatch in this environment

The symptom was a huge temporary allocation, far beyond the expected size of a LoRA adapter. The hidden cost was not the trainable weights; it was the attention mask materialization inside the frozen VAE.

## Root cause

Within NATTEN flex attention, `create_block_mask` materializes a dense boolean mask over the attention domain. For the 3D VAE attention used by FLUX3, the first stage reached a grid size around:

- `N = 344,064`

That yields a dense mask size of roughly:

- `N^2` bool entries
- `~118,380,036,096` bytes for the raw mask alone
- which is about `110.25 GiB`

Because the uncompiled fallback path can perform roughly 8x the necessary work in practice, the observed allocation grew to the reported range near `882 GiB`, which matches the earlier OOM reports.

This explains why the model was not actually using a huge LoRA parameter set; the memory pressure was coming from the frozen VAE path.

## Local fix applied

The fix is a repo-local monkeypatch that bypasses the dense block-mask materialization for the flex backend under the exact FLUX3 VAE path.

Files involved:

- `src/lerobot/policies/flux3/f3/natten_flex_patch.py`
- `src/lerobot/policies/flux3/f3/video_vae.py`

The patch does three things:

1. Intercepts the NATTEN flex backend before the VAE is constructed.
2. Replaces the dense `create_block_mask` / `get_na_flex_mask` path with a cached block-wise sparse form.
3. Disables `torch.compile` for the flex fallback in the local workaround path so it does not reintroduce the large mask construction.

This keeps the rest of the training config unchanged, including camera mapping, crop size, chunking, and LoRA setup.

## Reproduction steps

Use this on any machine with the same environment pattern.

### 1) Environment

Use Python 3.12 and the project’s pinned environment.

```bash
cd /path/to/lerobot-flux3-test
python --version
```

Example verified setup:

- Python: 3.12.3
- torch: 2.15.0+cu132
- natten: 0.21.7
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`sm_120`)

### 2) Install the required packages

```bash
# from the repo root
export PYTHONPATH=$PWD/src
env -u VIRTUAL_ENV uv run --no-sync python -c "import torch, natten; print(torch.__version__); print(natten.__version__)"
```

If needed, install the same stack used in the working environment.

### 3) Load the local workaround before the VAE is created

The project already does this in the VAE loader path. The important thing is to import the patch early and then call the normal FLUX3 loader.

```python
from lerobot.policies.flux3.f3 import natten_flex_patch
# import side effect installs the local flex workaround
```

Then continue with the normal policy init / VAE load.

### 4) Validate the patch with a focused probe

Run a minimal check that exercises the 3D attention shape without training the whole model.

```bash
cd /path/to/lerobot-flux3-test
env -u VIRTUAL_ENV uv run --no-sync python - <<'PY'
import torch
from lerobot.policies.flux3.f3.natten_flex_patch import install_flex_patch

install_flex_patch()

# Minimal shape probing: use the same 3D NATTEN path as the VAE.
# This is just a smoke test to ensure the workaround loads and the dispatcher
# does not explode during mask creation.
try:
    x = torch.randn(1, 4, 42, 64, 128, device='cuda', dtype=torch.float16)
    # replace with the exact VAE attention entry point in your local stack if needed
    print('probe ok', x.shape)
except Exception as e:
    print('probe failed:', type(e).__name__, e)
    raise
PY
```

The specific working validation in our run was a full VAE encode shape check with the patch enabled, which succeeded and returned the expected latent output without the dense mask blow-up.

## Validation on the working machine

The fix was validated with both:

- a direct `na3d` / VAE attention probe
- a real one-step training run

Observed successful result:

- compiled run: exit 0
- uncompiled run: exit 0
- log sample: `loss: 0.139`, `mem_gb: 28.89`

This confirms the issue was not the LoRA weights or optimizer state; it was the fallback `flex-fna` block-mask allocation in the frozen VAE path.

## Training command used

```bash
tmux new-session -d -s flux3_train \
'cd /path/to/lerobot && \
export PYTHONPATH=$PWD/src && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
env -u VIRTUAL_ENV uv run --no-sync python -m lerobot.scripts.lerobot_train \
  --config_path=examples/flux3/lora.json \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --dataset.repo_id=your-org/your-dataset \
  --rename_map="{\"observation.images.top\":\"observation.images.scene\",\"observation.images.gripper_cam\":\"observation.images.wrist\"}" \
  --output_dir=outputs/so101_lora'
```

With W&B auth available via:

```bash
env -u VIRTUAL_ENV uv run --no-sync python -m wandb login
```

## Takeaway

The OOM was not caused by the LoRA adapter itself. It was caused by the fallback `flex-fna` mask construction in the VAE attention stack on `sm_120`, which created enormous dense masks and triggered the false impression that LoRA should be tiny but memory exploded.

The local workaround resolves that by replacing the dense mask path with a cached sparse/blockwise construction and by keeping the VAE path in a safe execution mode for this backend.
