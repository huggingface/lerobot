# pi0 → ONNX (decomposed) + verification

Full pi0 `select_action` cannot be exported as a single ONNX graph: it contains a
10-step Python flow-matching loop, a transformers KV-cache passed between passes, custom
attention, and mixed precision. Instead we split inference into two ONNX graphs and drive
the Euler loop on the host, which is the standard way to deploy KV-cache flow/diffusion
models:

- **`prefix.onnx`** — SigLIP vision tower + gemma_2b prefix. Runs **once** per observation.
  Inputs: preprocessed images + masks + language tokens/mask. Outputs: per-layer K/V cache
  tensors (`keys`, `values`) + `prefix_pad_masks`.
- **`denoise.onnx`** — the gemma_300m action expert. Runs **once per Euler step** (10×),
  consuming the cached K/V. Inputs: `state, x_t, timestep, keys, values, prefix_pad_masks`.
  Output: velocity `v_t`.

Everything runs in **float32 on CPU** (onnxruntime's CPU kernels don't support bf16, and
this machine has no CUDA). The graphs call the exact same methods as `PI0Pytorch.sample_actions`
/ `denoise_step`, so the decomposition is numerically identical to PyTorch (verified below).

## Requirements

- Free disk: **~22–25 GB** (weights 8.9 GB + `prefix.onnx` ~9 GB + `denoise.onnx` ~1.3 GB + headroom).
- **RAM: ~24 GB recommended.** Exporting the fp32 prefix graph holds a ~13 GB model plus
  the ONNX graph. On an 18 GB machine this thrashes on memory compression and effectively
  stalls, so run steps 1–2 on a box with more RAM (or a cloud GPU box). `denoise.onnx` and
  the verification are light and run anywhere. The loaders already use a memory-frugal
  per-tensor copy (`load_policy_fp32_lowmem`) to minimize peak.
- Deps (once):

```bash
uv pip install onnx onnxruntime onnxscript accelerate
```

## Validate the logic with no download (optional, instant)

Runs the entire pipeline on a shrunk random pi0 — proves the decomposition + export are correct.

```bash
uv run python -m pi0_onnx.validate_tiny
```

## Full conversion + verification (real checkpoint)

Run in order from the repo root. Artifacts land in `pi0_onnx/artifacts/`.

```bash
# 1. PyTorch reference (downloads ~8.9GB weights; writes reference.npz)
uv run python -m pi0_onnx.reference       --model-id alexzai/pi0_your_task

# 2. Export the prefix/VLM encoder graph (heavy; external-data ONNX)
uv run python -m pi0_onnx.export_prefix   --model-id alexzai/pi0_your_task

# 3. Export the denoise-step graph (small; frees paligemma first)
uv run python -m pi0_onnx.export_denoise  --model-id alexzai/pi0_your_task

# 4. Run both ONNX graphs + verify vs the PyTorch reference
uv run python -m pi0_onnx.run_onnx
```

Step 4 prints the prefix K/V diff and the final action-chunk `max abs diff` vs PyTorch,
and exits non-zero if it exceeds `--atol` (default 2e-3).

## Notes / knobs

- `--steps N` on `reference.py` overrides `num_inference_steps` (default 10 from the config).
  The verification uses whatever was baked into `reference.npz`.
- Image preprocessing (resize + normalize to [-1, 1]) and action un-normalization are simple,
  identical host-side steps and are applied outside the graphs; feed `prefix.onnx` preprocessed
  images. Language tokens use a fixed deterministic sequence for the numeric check — for real
  deployment, tokenize the actual task string with the PaliGemma tokenizer.
- Performance: `denoise.onnx` is small and fast; `prefix.onnx` (2.3B params, fp32, CPU) is the
  cost. This is a correctness-first decomposition — a production build would quantize (int8/fp16)
  and/or target a GPU/NPU execution provider.
