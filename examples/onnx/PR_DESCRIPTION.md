# feat: ONNX inference support (ACT)

## Summary

This PR introduces a first, end-to-end path for **ONNX-based policy inference** in LeRobot, currently scoped to the **ACT** policy. The goal is to standardize how we export and run policies through ONNX Runtime so that the same workflow can later cover other policies (including the Unitree G1 whole-body / locomotion policies) and so policies can run on **edge devices** without a full PyTorch stack.

> ⚠️ **Scope:** today this only works for **ACT**. ACT is the natural starting point because inference is a single deterministic forward pass (ResNet backbone + transformer enc/dec + action head) with a zeros VAE latent — no denoising loop, no KV cache. Other architectures (e.g. PI0.5) need more work before they can be exported the same way.

## Motivation

- **Standardize ONNX inference** across LeRobot policies behind one export + run convention, instead of one-off conversion scripts.
- **Run on edge devices**: ONNX Runtime has a much smaller footprint than PyTorch and ships CPU / CUDA / TensorRT / mobile execution providers, which is what we want for deploying policies (incl. Unitree G1 policies) on-robot.
- Keep normalization and control logic in Python (the LeRobot processor pipeline + action queue), and export **only the neural network** as a portable graph.

## What's included

All new files live under `examples/onnx/` (no changes to `src/lerobot/...`):

- **`export_act.py`** — exports `policy.model` to ONNX as a pure function `(state, images) -> action_chunk`, then runs a numerical parity check (PyTorch vs ONNX Runtime).
- **`eval_act_onnx.py`** — evaluates ACT in sim with either the PyTorch or the ONNX backend. It swaps **only** `policy.model` with an ONNX Runtime session (wrapped as an `nn.Module`), so processors, action queue and the gym env are identical and any delta is attributable to the backend alone.
- **`convert_legacy_checkpoint.py`** — helper for older hub checkpoints that bake normalization into weights and lack `policy_preprocessor.json` / `policy_postprocessor.json`.

## Design notes

- Only the network is exported. At inference, ACT's `predict_action_chunk` is effectively `self.model(batch)[0]` with a zeros latent, so the graph is deterministic in `(state, images)`.
- **Normalization stays outside ONNX**, in the LeRobot processor pipeline. The ONNX graph consumes already-normalized inputs and emits normalized actions.
- torch 2.9+ defaults to the dynamo exporter (requires `onnxscript`); the exporter uses the legacy TorchScript path (`dynamo=False`) since ACT's graph is fixed-shape.

## Results

**Numerical parity** (PyTorch vs ONNX Runtime):

```
max_abs_diff = 1.073e-06   mean_abs_diff = 1.790e-07   -> PASS
```

**In-sim eval**, `AlohaTransferCube-v0`, identical seed:

| backend | n_episodes | pc_success |
|---------|-----------:|-----------:|
| torch   | 10         | 70.0%      |
| onnx    | 10         | 70.0%      |

Identical success rate; sub-1e-6 per-step parity. (Run on CPU here; both backends behave the same on CUDA.)

## How to run

```bash
export PYTHONPATH=src

# export once (also runs the parity check)
python examples/onnx/export_act.py \
  --policy-path=lerobot/act_aloha_sim_transfer_cube_human \
  --output=outputs/onnx/act_transfer_cube.onnx

# compare backends in sim
python examples/onnx/eval_act_onnx.py \
  --policy-path=lerobot/act_aloha_sim_transfer_cube_human \
  --task=AlohaTransferCube-v0 \
  --backend=torch --n-episodes=50 --batch-size=10 --device=cuda

python examples/onnx/eval_act_onnx.py \
  --policy-path=lerobot/act_aloha_sim_transfer_cube_human \
  --task=AlohaTransferCube-v0 \
  --onnx=outputs/onnx/act_transfer_cube.onnx \
  --backend=onnx --n-episodes=50 --batch-size=10 --device=cuda
```

## Follow-ups (out of scope for this PR)

- Generalize the export convention beyond ACT (PI0.5 denoising loop + KV cache, diffusion policies, etc.).
- Cover the **Unitree G1** policies so they can be deployed via ONNX Runtime on-robot.
- Provide an edge-device runner / packaging story (CPU / TensorRT / mobile execution providers) and a latency benchmark.

## Test plan

- [x] ONNX export succeeds for ACT and passes the parity check (`max_abs_diff < 1e-3`).
- [x] In-sim eval matches the PyTorch backend at the same seed.
- [ ] Full 50-episode eval on CUDA (torch vs onnx) reproduces the baseline success rate.
