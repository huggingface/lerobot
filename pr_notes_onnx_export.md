# PR Notes: ONNX/TensorRT Export for LeRobot Policies (Issue #3146)

## Summary

Adds a `lerobot-export` CLI that exports trained LeRobot policies to ONNX (and optionally TensorRT) for edge deployment on devices such as Jetson Orin Nano/NX.

## Motivation

Users running trained policies on Jetson-class devices need an optimized inference format. ONNX + TensorRT FP16 provides significant speedup over PyTorch on SM_87 GPUs without requiring Python training dependencies at inference time.

---

## Architecture

### Registry-based dispatch

```python
# core.py
WRAPPER_REGISTRY: dict[str, WrapperFactory] = {
    "act":       _make_act_wrapper,       # registered via setdefault on first lookup
    "diffusion": _make_diffusion_wrapper,
}

def make_export_wrapper(policy, cfg) -> (nn.Module, ExportSpec):
    factory = WRAPPER_REGISTRY.get(policy.config.type)
    if factory is None:
        return _make_generic_wrapper(policy, cfg)  # ACT-clone fallback only
    return factory(policy, cfg)

# Plugin API for third-party / follow-up PRs:
@register_export_wrapper("vqbet")
def make_vqbet_wrapper(policy, cfg):
    return VQBeTStatelessWrapper(policy), ExportSpec(...)
```

### ACTInferenceWrapper (`wrappers.py`)

- **Exports**: ResNet backbone + Transformer encoder/decoder + action head.
- **Does NOT export**: VAE encoder (uses `latent = zeros` at inference), action queue, temporal ensembler.
- Inputs: `(robot_state, cam_0, cam_1, ...)` — each camera is a separate named ONNX input.
- Reconstructs `OBS_IMAGES` as a Python list inside `forward()`.
- **Dynamic batch_size with `exporter=dynamo`**: legacy tracing bakes `torch.zeros([batch_size, latent_dim])` as a B=1 constant. The dynamo path (`torch.export` + `Dim`) symbolicizes it. With `exporter=legacy`, batch_size is fixed at 1.

### DiffusionUNetWrapper — default mode (`wrappers.py`)

- Exports only `DiffusionConditionalUnet1d` — a single denoising step.
- Inputs: `(sample, timestep, global_cond)` → dynamic batch_size supported on both exporter paths.
- Caller runs the DDPM/DDIM loop in Python and calls this model at each step.
- Caller computes `global_cond` via `policy.diffusion._prepare_global_conditioning(batch)`.

### DiffusionDDIMWrapper — full loop mode (`wrappers.py`)

- Activated with `--diffusion-mode=ddim-N` where N is the number of steps.
- DDIM step is pure tensor math → traceable at export time.
- Loop is **unrolled** N times in `forward()` so the full denoising trajectory is a single ONNX graph.
- DDIM schedule (alpha/beta buffers) pre-computed and registered as ONNX constants.
- Restriction: deterministic DDIM only (eta=0). Number of steps is **fixed** at export time.
- Requires `noise_scheduler_type='DDIM'` in the Diffusion config.

### GenericPolicyWrapper — strict fallback (`wrappers.py`)

The generic fallback supports only ACT-clone style policies (`hasattr(policy, "model")` AND `n_obs_steps == 1` AND callable `policy.model`). For known-incompatible types it raises `NotImplementedError` with a clear pointer to the registration API:

| Policy | Reason rejected |
|---|---|
| SAC | No `.model` attribute (uses `actor` / `critic`) |
| VQBET | `model.forward(batch, rollout: bool)` signature mismatch |
| TDMPC | Model decomposed into `encode/pi/dynamics`; planning loop not traceable |
| PI0, PI0_Fast, SmoLVLA | `forward()` takes 7+ positional tensors, not a dict |

For these types, register a custom factory:

```python
from lerobot.export import register_export_wrapper

@register_export_wrapper("vqbet")
def make_vqbet_wrapper(policy, cfg):
    return VQBeTStatelessWrapper(policy), ExportSpec(...)
```

### ONNX exporter backends (`onnx_export.py`)

| Backend | Detail |
|---|---|
| `legacy` | `torch.onnx.export()` with TorchScript tracer. Stable but bakes Python control flow and `torch.zeros([batch_size, ...])` calls as constants. |
| `dynamo` | `torch.onnx.export(..., dynamo=True)` using `torch.export`. Supports symbolic `Dim` for batch_size. Requires `onnxscript`. |
| `auto` (default) | Tries `dynamo` first, falls back to `legacy` with a warning if dynamo fails. |

`ExportSpec` carries both `dynamic_axes` (legacy) and `dynamic_shapes` (dynamo, tuple form matching positional sample inputs) so each backend gets the right input signal.

### Normalization handling

The `from_pretrained()` method does NOT load the preprocessor pipeline — it loads only model weights. The CLI explicitly loads it via `make_pre_post_processors(policy_cfg, pretrained_path)` and:

1. **Always** saves the canonical processor artifacts (`policy_preprocessor.json`, `policy_postprocessor.json`, plus their safetensors state files) into the output directory. Edge clients can use `lerobot.processor.hotswap_stats()` to reuse them.
2. **By default** also writes a flat `normalization_stats.json` for clients that don't want the full pipeline:

```json
{
  "observation.state": {"mean": [...], "std": [...]},
  "action": {"mean": [...], "std": [...]}
}
```

3. With `--fold-normalization`, a `NormalizedWrapper` bakes `(x - mean) / std` and `y * std + mean` into the ONNX graph as constant nodes; clients send/receive raw tensors.

Stats are read via the **public** `step.stats` attribute on `NormalizerProcessorStep` / `UnnormalizerProcessorStep`.

### TensorRT (`tensorrt_export.py`)

`trtexec` subprocess + engine cache keyed by `{stem}_{precision}_sm{sm}_{onnx_md5[:8]}.engine`. Hardware guards:

- `precision='fp8'` → always raises (no SM has FP8 in this lineup).
- `precision='int8'` on SM_87 → raises with a pointer to FP16.
- `precision='int8'` elsewhere → requires `--calibration-data`.
- `precision='fp16'` on SM<80 → warning only.

---

## Validation

After every ONNX export, `validate_onnx()` compares PyTorch vs ONNX Runtime outputs:
- **max_abs_error** — maximum absolute difference
- **cos_sim** — cosine similarity (1.0 = identical)
- **allclose** — `torch.allclose(rtol=1e-3, atol=1e-5)` pass/fail

---

## File Structure

```
src/lerobot/
  scripts/
    lerobot_export.py          # CLI entry point (ExportConfig + export_policy())
  export/
    __init__.py                # Public API
    core.py                    # ExportSpec (dynamic_axes + dynamic_shapes), registry
    wrappers.py                # ACTInferenceWrapper, DiffusionUNetWrapper,
                               # DiffusionDDIMWrapper, GenericPolicyWrapper
    sample_inputs.py           # make_sample_inputs() — policy-aware zero tensors
    onnx_export.py             # auto / dynamo / legacy branching
    tensorrt_export.py         # export_to_tensorrt() via trtexec subprocess
    validation.py              # validate_onnx() — cos_sim + max_abs_err
    normalization.py           # save_normalization_stats(), NormalizedWrapper

tests/
  test_export.py               # Unit tests (CPU, toy configs, no real backbone weights)
```

## Modified Files

| File | Change |
|---|---|
| `pyproject.toml` | Added `lerobot-export` script + `export` optional dep group (`onnx`, `onnxruntime`, `onnxscript`) |
| `uv.lock` | Regenerated with new dependencies |
| `src/lerobot/utils/import_utils.py` | Added `_onnx_available`, `_onnxruntime_available` |

---

## ExportConfig Fields

| Field | Default | Notes |
|---|---|---|
| `policy` | (required) | Hub repo or local path; loaded via `--policy.path=` |
| `output_path` | `Path("exported_model")` | |
| `format` | `"onnx"` | or `"tensorrt"` |
| `precision` | `"fp32"` | `"fp32"` / `"fp16"` / `"int8"` |
| `opset_version` | `18` | Native LayerNorm + improved attention coverage |
| `exporter` | `"auto"` | `"auto"` / `"dynamo"` / `"legacy"` |
| `validate` | `True` | Run PyTorch vs ORT parity check |
| `rtol` / `atol` | `1e-3` / `1e-5` | Numerical tolerances |
| `device` | `"cpu"` | Tracing device |
| `batch_size` | `1` | Sample-input batch size |
| `trt_workspace_gb` | `4` | TRT workspace |
| `force_rebuild` | `False` | Skip TRT engine cache |
| `calibration_data` | `None` | Required for INT8 TRT |
| `fold_normalization` | `False` | Bake normalization into ONNX |
| `diffusion_mode` | `"unet-only"` | or `"ddim-N"` |

---

## CLI Usage

```bash
# Install export extras
uv sync --locked --extra export

# ACT — ONNX FP32 (default exporter=auto, opset=18)
lerobot-export --policy.path=lerobot/act_pusht_image --output-path=./exports/act

# ACT with dynamic batch_size via dynamo (required for batch>1)
lerobot-export --policy.path=lerobot/act_pusht_image --exporter=dynamo

# ACT — force legacy tracer (batch_size=1 only)
lerobot-export --policy.path=lerobot/act_pusht_image --exporter=legacy

# Diffusion — UNet only (single step, scheduler in Python)
lerobot-export --policy.path=lerobot/diffusion_pusht --output-path=./exports/diffusion

# Diffusion — full DDIM loop, 10 steps baked in
lerobot-export --policy.path=lerobot/diffusion_pusht --diffusion-mode=ddim-10

# FP16 ONNX
lerobot-export --policy.path=lerobot/act_pusht_image --precision=fp16

# TensorRT FP16 (requires CUDA + trtexec on PATH)
lerobot-export \
    --policy.path=lerobot/act_pusht_image \
    --format=tensorrt \
    --precision=fp16 \
    --device=cuda

# INT8 TensorRT (requires calibration data, not supported on SM_87)
lerobot-export \
    --policy.path=lerobot/act_pusht_image \
    --format=tensorrt \
    --precision=int8 \
    --calibration-data=./calib_data

# Fold normalization into the ONNX graph
lerobot-export --policy.path=lerobot/act_pusht_image --fold-normalization

# Run unit tests
uv run pytest tests/test_export.py -svv
```

---

## Output Layout

```
exports/act/
├── act_fp32.onnx                                         # exported model
├── normalization_stats.json                              # flat stats (default)
├── policy_preprocessor.json                              # canonical preprocessor
├── policy_preprocessor_step_0_normalizer_processor.safetensors
├── policy_postprocessor.json                             # canonical postprocessor
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

(For TensorRT, additionally `.trt_cache/<stem>_<precision>_sm<sm>_<md5>.engine`.)

---

## Key Technical Constraints

| Constraint | Detail |
|---|---|
| ACT + `exporter=legacy` | `torch.zeros([B, latent_dim])` inside `ACT.forward` is traced as a B=1 constant. Use `--exporter=dynamo` for dynamic batch. |
| SM_87 (Jetson Orin Nano/NX) | No FP8 or INT8 tensor cores. Only FP16 is reliably accelerated. TRT builder raises an error for INT8/FP8 on SM_87. |
| INT8 elsewhere | Requires calibration data via `--calibration-data`. |
| FP16 on SM<80 | Warning only; may have limited acceleration. |
| Diffusion full DDIM | Requires `noise_scheduler_type='DDIM'`. DDPM scheduler is not traceable due to conditional branching. |
| VQBET / TDMPC / SAC / PI0 / SmoLVLA | Generic fallback raises `NotImplementedError` (see table above). Register a custom factory via `register_export_wrapper(...)`. |

---

## Test Coverage (`tests/test_export.py`)

- `ExportSpec` defaults
- `register_export_wrapper` decorator
- ACT wrapper forward shape
- Diffusion UNet wrapper forward shape
- ONNX export + numerical parity (ACT, Diffusion UNet)
- TensorRT no-CUDA error path
- `make_sample_inputs` for ACT / Diffusion UNet / unknown mode
- Generic wrapper forward (ACT-clone)
- **NEW**: Normalization stats round-trip via `make_act_pre_post_processors`
- **NEW**: Generic wrapper raises `NotImplementedError` for SAC/VQBET/TDMPC/PI0/SmoLVLA
- **NEW**: Generic wrapper raises `NotImplementedError` for `n_obs_steps > 1`
- **NEW**: `auto` exporter falls back to legacy with WARNING when dynamo raises
- **NEW**: `legacy` exporter explicit smoke test
- **NEW**: Invalid `exporter=` raises `ValueError`
- **NEW**: ACT `ExportSpec.dynamic_shapes` is populated and structurally matches `sample_inputs`

---

## Known Limitations and Follow-up Opportunities

| Limitation | Status |
|---|---|
| ACT batch_size=1 under legacy tracing | **Resolved**: use `--exporter=dynamo` (default `auto`). |
| TDMPC | Out of scope; encoder-only export possible via `register_export_wrapper`. |
| VLA models (PI0, SmoLVLA) | Out of scope; reference reflex-vla for guidance, register custom factory. |
| VQBET / SAC | Out of scope; register custom factory bypassing queues / actor-critic. |
| Diffusion INT8 on SM_87 | Blocked by hardware; raised as an error. |
| Stats injected at runtime (e.g., per-robot calibration) | Use `lerobot.processor.hotswap_stats()` on the saved `policy_preprocessor.json`. |
