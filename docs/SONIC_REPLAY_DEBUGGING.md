# SONIC replay instability — root cause & prevention

This documents the multi-day debugging of "SONIC motion replay is unstable / jitters /
lags / dies on the floor", so we don't chase the same ghosts again.

## TL;DR

There were **two independent problems**, and they masked each other:

1. **Wrong conda environment (the "lag"/jitter).** The debugging env `lerobot_sonic`
   had a **CUDA-13** stack that the machine's GPU driver cannot run, so ONNX Runtime
   silently fell back to CPU and oversubscribed threads. The known-good env
   `lerobot312` has a **CUDA-12** stack matching the driver, so the encoder/decoder/
   planner run on the GPU (~12–20 ms planner inference) and the control loop holds
   ~48–50 Hz.
2. **SMPL root-motion feeding (the NaN/`unstable` crash).** Passing the per-frame SMPL
   root quaternion into the mode-2 anchor produced a root-acceleration spike
   (`Nan, Inf or huge value in QACC at DOF 0`) mid-episode. Disabling it gives clean
   tracking.

Neither is an algorithmic bug in the ported SONIC pipeline. A lot of earlier "fixes"
(ORT thread caps, `MAX_DELTA_PER_STEP` clamp, planner-disable toggle, resampling)
were chasing symptom #1 in the wrong environment and were reverted.

## Environment: what "good" looks like

Run the replay in `lerobot312` (CUDA-12), pointing at the current sonic checkout:

```bash
conda activate lerobot312
PYTHONPATH=/home/yope/Documents/sonic/lerobot/src \
  lerobot-replay \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --dataset.repo_id=lerobot/SMPL_samples --dataset.episode=12
```

Known-good versions (`lerobot312`):

| package         | good (`lerobot312`) | broken (`lerobot_sonic`) |
|-----------------|---------------------|--------------------------|
| GPU driver      | CUDA 12.8 (`12080`) | same (unchanged)         |
| torch           | `2.10.0+cu128`      | `2.11.0+cu130`           |
| onnxruntime     | `onnxruntime-gpu 1.26.0` | CPU `1.27.0` / cu13 mismatch |
| cudnn           | cu12 (bundled)      | `nvidia-cudnn-cu13 9.19` |
| mujoco          | `3.8.1`             | `3.10.0`                 |

### How to verify the GPU path is actually live (do this FIRST)

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"          # must be True
python -c "import onnxruntime as ort; print(ort.get_available_providers())" # must list CUDAExecutionProvider
```

If `torch.cuda.is_available()` is `False` or `CUDAExecutionProvider` is missing, STOP —
you are in the wrong/broken env. Do not "optimize" anything else until this passes.

### Why CUDA 13 was fatal here

The GPU driver supports up to **CUDA 12.8**. A CUDA-13 build of torch/onnxruntime
cannot initialize on it:
- `torch.cuda.is_available()` returns `False` (silent CPU fallback).
- `onnxruntime-gpu` (a CUDA-12 build) can't find a matching cuDNN because only the
  CUDA-13 cuDNN is installed → `CUDNN failure 1001: CUDNN_STATUS_NOT_INITIALIZED`.

Installing `onnxruntime` and `onnxruntime-gpu` **together** also breaks: they share the
`onnxruntime` namespace and whichever installs last clobbers the other's shared libs.
Keep only `onnxruntime-gpu` in a GPU env.

## Root motion: the NaN/unstable crash

Symptom:
```
WARNING: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 4.196.
```
`DOF 0` is the floating base/root. Feeding the per-frame SMPL root quaternion
(`root.*` action keys) into `controller.smpl_root_quat` injected a discontinuity in the
reference root trajectory (frame-to-frame jump and/or 30 Hz→50 Hz timing mismatch) that
the tracker converted into an exploding base acceleration.

Current mitigation (in `sonic_whole_body.py`, `run_step`): the per-frame root quaternion
is ignored (`self.controller.smpl_root_quat = None`) so the anchor stays self-driven.
Result: clean tracking, no NaN.

Proper fix (follow-up, not yet done): smooth/slerp-filter the reference root trajectory
(or resample to the control rate) before feeding it to the anchor, then re-enable.

## Prevention checklist

- **Always confirm the env before debugging behavior.** Run the two verification
  commands above. Most of the "instability" was environment, not code.
- **Pin the GPU stack** to match the driver (CUDA 12.8): `torch ==2.10.0+cu128`,
  `onnxruntime-gpu ==1.26.0`, `mujoco ==3.8.1`. Do not let `lerobot_sonic` drift to a
  CUDA-13 stack.
- **Never install `onnxruntime` and `onnxruntime-gpu` side by side.**
- **Don't add band-aid clamps/thread-caps/resampling to hide a CPU-fallback**; fix the
  env instead. Those changes were reverted.
- **Root trajectory must be continuous / rate-matched** before it drives the anchor.
