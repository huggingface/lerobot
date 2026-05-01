# PR Notes: ONNX/TensorRT Export for LeRobot Policies (Issue #3146)

## Summary

Adds a `lerobot-export` CLI that exports trained LeRobot policies to ONNX (and optionally TensorRT) for edge deployment on devices such as Jetson Orin Nano/NX.

## Motivation

Users running trained policies on Jetson-class devices need an optimized inference format. ONNX + TensorRT FP16 provides significant speedup over PyTorch on SM_87 GPUs without requiring Python training dependencies at inference time.

---

## Architecture

### Per-policy co-location with auto-discovery

ONNX export adapters live next to each policy module under
``src/lerobot/policies/<type>/export_<type>.py`` — same convention as
``modeling_<type>.py`` / ``configuration_<type>.py`` / ``processor_<type>.py``.
``make_export_wrapper`` resolves them by naming convention + dynamic import.

```python
# core.py
def make_export_wrapper(policy, cfg) -> (nn.Module, ExportSpec):
    policy_type = policy.config.type

    # 1. Explicit registration (third-party / runtime overrides) wins.
    if policy_type in WRAPPER_REGISTRY:
        return WRAPPER_REGISTRY[policy_type](policy, cfg)

    # 2. Auto-discover via convention.
    factory = _try_load_builtin_factory(policy_type)
    if factory is not None:
        return factory(policy, cfg)

    # 3. Otherwise raise NotImplementedError with a clear extension guide.
    raise NotImplementedError(...)
```

Plugin API for third-party packages or runtime overrides:

```python
from lerobot.export import register_export_wrapper

@register_export_wrapper("vqbet")
def make_vqbet_wrapper(policy, cfg):
    return VQBeTStatelessWrapper(policy), ExportSpec(...)
```

### ACTInferenceWrapper (`policies/act/export_act.py`)

- **Exports**: ResNet backbone + Transformer encoder/decoder + action head.
- **Does NOT export**: VAE encoder (uses `latent = zeros` at inference), action queue, temporal ensembler.
- Inputs: `(robot_state, cam_0, cam_1, ...)` — each camera is a separate named ONNX input.
- Reconstructs `OBS_IMAGES` as a Python list inside `forward()`.
- **batch_size in the resulting ONNX**: the legacy tracer bakes
  `torch.zeros([batch_size, latent_dim])` as a B=1 constant. The dynamo
  exporter accepts a symbolic `Dim`, but `ACT.forward` still uses concrete
  `batch_size` values for intermediate allocations, so the exporter ends up
  specializing to 1 anyway. Both backends therefore produce a fixed-batch
  ONNX in practice for ACT — see "Verification on real Hub models" for the
  measured numbers. Removing the model-side batch-dependent allocations is a
  follow-up.

### DiffusionUNetWrapper — default mode (`policies/diffusion/export_diffusion.py`)

- Exports only `DiffusionConditionalUnet1d` — a single denoising step.
- Inputs: `(sample, timestep, global_cond)` → dynamic batch_size supported on both exporter paths.
- Caller runs the DDPM/DDIM loop in Python and calls this model at each step.
- Caller computes `global_cond` via `policy.diffusion._prepare_global_conditioning(batch)`.

### DiffusionDDIMWrapper — full loop mode (`policies/diffusion/export_diffusion.py`)

- Activated with `--diffusion-mode=ddim-N` where N is the number of steps.
- DDIM step is pure tensor math → traceable at export time.
- Loop is **unrolled** N times in `forward()` so the full denoising trajectory is a single ONNX graph.
- DDIM schedule (alpha/beta buffers) pre-computed and registered as ONNX constants.
- Restriction: deterministic DDIM only (eta=0). Number of steps is **fixed** at export time.
- Requires `noise_scheduler_type='DDIM'` in the Diffusion config.

### Unsupported policies — explicit `NotImplementedError`

For policies without a corresponding ``export_<type>.py``, ``make_export_wrapper``
raises ``NotImplementedError`` with concrete instructions on how to add support.
There is no silent fallback: the generic ACT-clone wrapper has been removed in
favor of an honest "unsupported" signal, because:

| Policy | Why a generic wrapper does not work |
|---|---|
| SAC | No `.model` attribute (uses `actor` / `critic`) |
| VQBET | `model.forward(batch, rollout: bool)` signature mismatch |
| TDMPC | Model decomposed into `encode/pi/dynamics`; planning loop not traceable |
| PI0, PI0_Fast, SmoLVLA | `forward()` takes 7+ positional tensors, not a dict |

For these (and any other) types, see the next section.

### How to add export support for a new policy

Adding ONNX export for a new policy type is a **localized change** — no edits
inside ``lerobot/export/`` are required. The steps:

1. Create ``src/lerobot/policies/<type>/export_<type>.py``.
2. Define a function with the canonical name:
   ```python
   def make_<type>_export_wrapper(policy, cfg) -> tuple[nn.Module, ExportSpec]:
       ...
   ```
3. **Pick the closest adapter pattern** (see below) instead of writing a
   bespoke ``nn.Module`` wrapper class.
4. Add tests at ``tests/policies/test_export_<type>.py``.

The reusable adapters live in ``src/lerobot/export/adapters/``.

#### Pattern A: DictBatch — `forward(batch: dict, **kwargs) -> Tensor`

Use for: ACT, VQ-BeT (with ``rollout=True``), SAC actor (with thin pre-wrapping),
TDMPC encoder, MultiTaskDiT observation_encoder, GR00T (with adapter).

```python
from lerobot.export.adapters import DictBatchAdapter, DictBatchSpec
from lerobot.export.core import ExportSpec

def make_my_export_wrapper(policy, cfg):
    spec = DictBatchSpec(
        input_feature_keys=[OBS_STATE, "observation.images.cam"],
        image_keys=["observation.images.cam"],
        image_convention="stacked",   # or "list" / "single"
        image_stack_dim=2,             # for VQ-BeT-style (B, n_obs_steps, n_cams, ...)
        extra_kwargs={"rollout": True},
        output_index=None,             # or int / dict-key
    )
    wrapper = DictBatchAdapter(policy.<network_attr>, spec)
    sample_inputs = ...   # build zero tensors matching spec.input_feature_keys
    return wrapper, ExportSpec(...)
```

Reference implementations:
- ``src/lerobot/policies/act/export_act.py`` — single-step list-of-images
- ``src/lerobot/policies/vqbet/export_vqbet.py`` — multi-step stacked images, `rollout=True`

#### Pattern B: Iterative denoising — N-step unrolled loop

Use for: Diffusion (DDIM), MultiTaskDiT, GR00T, flow-matching VLAs (pi0/pi05/smolvla/wall_x).

```python
from lerobot.export.adapters import IterativeDenoisingAdapter

class MyDenoisingAdapter(IterativeDenoisingAdapter):
    def __init__(self, model, num_steps):
        super().__init__(num_steps=num_steps)
        self.model = model
        # Register schedule constants (alphas, sigmas, ...) as buffers here.
        self.register_buffer("alpha_t_buf", ...)

    def _call_model(self, sample, step_idx, *cond):
        # One model forward at this step.
        ...

    def _step(self, sample, model_output, step_idx):
        # Pure-tensor scheduler update.
        ...
```

Reference implementation: ``src/lerobot/policies/diffusion/export_diffusion.py:DiffusionDDIMWrapper``.

#### Pattern C: Bespoke single forward

Use for: a single-step submodule whose forward is simple (e.g. Diffusion's
UNet exposed independently of the scheduler). Subclass ``nn.Module`` directly.

Reference implementation: ``src/lerobot/policies/diffusion/export_diffusion.py:DiffusionUNetWrapper``.

#### Not exportable as a single ONNX module

- **TDMPC**: CEM planning loop is iterative optimization, not a deterministic
  forward. Export the encoder only as a partial deployment.
- **pi0_fast**: autoregressive token decoding with KV cache; needs a custom
  adapter (separate design).
- **wall_x**: Mixture-of-Experts; ONNX MoE op coverage is limited.

#### Third-party policies (outside lerobot)

Skip the per-policy file convention and register at runtime:

```python
from lerobot.export import register_export_wrapper

@register_export_wrapper("my_external_policy")
def make_my_external_policy_export_wrapper(policy, cfg):
    return MyWrapper(policy), ExportSpec(...)
```

Explicit registration takes precedence over auto-discovery.

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

By default a single comparison runs on the baseline sample inputs (zeros). Pass
`--validation-trials=N` (or `--validation_trials=N` in draccus underscore form)
to run N additional comparisons with random Gaussian inputs (same shapes/dtypes
as the baseline). The aggregated result reports the worst max_abs_error and the
minimum cos_sim across all trials.

---

## File Structure

```
src/lerobot/
  scripts/
    lerobot_export.py          # CLI entry point (ExportConfig + export_policy())
  export/
    __init__.py                # Public API
    core.py                    # ExportSpec, registry, make_export_wrapper (auto-discovery),
                               # make_batch_dynamic_axes_and_shapes helper
    adapters/                  # Reusable adapter primitives
      __init__.py
      dict_batch.py            # DictBatchAdapter + DictBatchSpec (Pattern A)
      iterative.py             # IterativeDenoisingAdapter ABC (Pattern B)
    sample_inputs.py           # make_zero_inputs_from_features (generic helper)
    onnx_export.py             # auto / dynamo / legacy branching
    tensorrt_export.py         # export_to_tensorrt() via trtexec subprocess
    validation.py              # validate_onnx() — cos_sim + max_abs_err + multi-trial
    normalization.py           # save_normalization_stats(), NormalizedWrapper
  policies/
    act/
      export_act.py            # ACT factory using DictBatchAdapter (Pattern A)
    diffusion/
      export_diffusion.py      # DiffusionUNetWrapper (C) + DiffusionDDIMWrapper (B)
    vqbet/
      export_vqbet.py          # VQ-BeT factory using DictBatchAdapter (Pattern A)

tests/
  test_export.py               # Framework-level tests (registry, helpers, normalization, TRT)
  policies/
    test_export_act.py         # Per-policy ACT export tests
    test_export_diffusion.py   # Per-policy Diffusion export tests
    test_export_vqbet.py       # Per-policy VQ-BeT export tests
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

## Verification on real Hub models

End-to-end runs against pretrained checkpoints downloaded from the HuggingFace
Hub, on macOS / CPU (Python 3.12, PyTorch 2.7+, opset 18). Both runs use the
new `--validation_trials=5` flag (baseline + 5 random-input parity checks).

### `lerobot/act_aloha_sim_insertion_human` (ACT, ~207 MB)

```bash
uv run lerobot-export \
  --policy.path=lerobot/act_aloha_sim_insertion_human \
  --output_path=./outputs/verify_act \
  --device=cpu \
  --exporter=legacy \
  --validation_trials=5
```

Result:
```
Validation PASSED (baseline + 5 random trial(s)):
  worst max_abs_error = 2.09e-06
  min cos_sim         = 1.000000
  allclose(rtol=1e-3, atol=1e-5) = True
```

ONNX I/O contract (verified independently with `onnxruntime`):
- Inputs: `observation_state: (1, 14)`, `observation_images_top: (1, 3, 480, 640)`
- Output: `action_chunk: (1, 100, 14)`
- Shapes are fixed at batch_size=1.

**Note on `--exporter=dynamo` for ACT**: `--exporter=dynamo` also runs to
completion on this checkpoint (validation likewise PASSED, max_abs_error
2.09e-06), but the resulting ONNX still has `batch_size=1` baked in. The
specialization happens because `ACT.forward` allocates
`torch.zeros([batch_size, latent_dim])` and a few other batch-dependent
intermediates inside the model code, which causes `torch.export` to bind the
symbolic `Dim` to its concrete trace value. Genuine batch>1 export for ACT
needs an upstream change to `ACT.forward` (e.g. preallocating those tensors
as buffers); that is out of scope for this PR.

### `lerobot/diffusion_pusht` (Diffusion UNet, ~1.0 GB)

```bash
uv run lerobot-export \
  --policy.path=lerobot/diffusion_pusht \
  --output_path=./outputs/verify_diffusion \
  --device=cpu \
  --validation_trials=5
```

Result:
```
Validation PASSED (baseline + 5 random trial(s)):
  worst max_abs_error = 9.06e-06
  min cos_sim         = 1.000000
  allclose(rtol=1e-3, atol=1e-5) = True
```

ONNX I/O contract (verified independently with `onnxruntime`, batch_size=4):
- Inputs: `sample: (batch_size, 16, 2)`, `timestep: (batch_size,) int64`, `global_cond: (batch_size, 132)`
- Output: `denoised: (batch_size, 16, 2)`
- All inputs / outputs are dynamic over `batch_size`.

### Notes

- These older Hub checkpoints do not ship the new `policy_preprocessor.json`
  format; the CLI logs a warning and falls back to processors built from the
  config alone, so `normalization_stats.json` ends up empty (`{}`). Newer
  models trained with the current pipeline (e.g. `lerobot/smolvla_base`) ship
  the canonical artifacts and produce real stats.
- TensorRT engine builds are out of scope for this verification (no CUDA on the
  host). The ONNX → TRT path is exercised by `tests/test_export.py` via the
  hardware-guard error path only; full GPU verification will land in a CI job.

---

## Known Limitations and Follow-up Opportunities

| Limitation | Status |
|---|---|
| ACT batch_size > 1 | **Not resolved by exporter switch alone**: even `--exporter=dynamo` produces a fixed batch_size=1 ONNX because of `torch.zeros([batch_size, ...])` calls inside `ACT.forward`. Requires modifying `ACT.forward` to remove the batch-dependent intermediates (preallocate as buffers). |
| TDMPC | Out of scope; encoder-only export possible via `register_export_wrapper`. |
| VLA models (PI0, SmoLVLA) | Out of scope; reference reflex-vla for guidance, register custom factory. |
| VQBET / SAC | Out of scope; register custom factory bypassing queues / actor-critic. |
| Diffusion INT8 on SM_87 | Blocked by hardware; raised as an error. |
| Stats injected at runtime (e.g., per-robot calibration) | Use `lerobot.processor.hotswap_stats()` on the saved `policy_preprocessor.json`. |
