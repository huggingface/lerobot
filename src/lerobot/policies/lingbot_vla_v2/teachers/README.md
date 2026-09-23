# Distillation teachers: first-party runtimes

This directory is reserved for **LeRobot-maintained, weight-compatible teacher
implementations**. It contains no copied upstream source tree and does not
import or require an upstream repository checkout at runtime.

## Dependency contract

| Teacher | Runtime source | Developer supplies |
|---|---|---|
| MoGe depth | `native_depth_models.py` first-party LeRobot runtime | `model.pt` weight |
| LingBot-Depth / MoRGBD | `morgbd_teacher.py` first-party LeRobot runtime | `depth/model.pt` weight |
| DINO-video | `dino_video/` first-party LeRobot runtime | `teacher_step_10000.pth` + `config.yaml` |

`depth_teachers.py` owns only the common teacher lifecycle and target-extraction
contract: frozen modules outside the policy module tree; current/future depth
and video target tensors passed to the policy during training only.

## Current status

The previous bundled MoGe/MoRGBD runtimes and DINO external-provider path were
removed. They violated the package boundary by introducing third-party source
or requiring developers to clone an upstream repository.

**Depth teachers are now first-party and verified**:
`native_depth_models.py` (MoGe v2) and `morgbd_teacher.py` (LingBot-Depth /
MoRGBD) restore the published checkpoints locally and reproduce the upstream
numbers — measured against the upstream runtime on real weights: MoGe depth
maps match to ~2e-7 relative; MoRGBD features/class token match bit-exactly
once the RNG-dependent depth-embed quirk (below) is synced.

The MoRGBD runtime reproduces one verified upstream load quirk on purpose: the
published checkpoint stores the 1-channel depth patch embedding under
`depth_mask_patch_embed.*` while the runtime module is `depth_patch_embed.*`,
so upstream's `strict=False` load leaves that conv at construction
initialization (PyTorch default init) and drops 40 checkpoint tensors (the
same two embed weights plus a `normal_head` the upstream wrapper never
constructs). The first-party runtime loads with the same coverage, warns about
exactly that key set, and rejects any other mismatch. Unlike upstream, its
initialization is seeded, so runs are reproducible.

**DINO-video is now first-party and verified** under [`dino_video/`](./dino_video/):
public facade `teacher.py`, strict weight loader `checkpoint.py` (345 backbone
tensors; only the 14 distillation-head tensors may be unused), ViT/token layout
`backbone.py`, 3D RoPE `rope.py`, and frame-block-causal attention `attention.py`.
Measured against the upstream teacher on real weights it is bit-exact on the
upstream SDPA reference (0.0 diff, 100% bf16-exact) across T=2/3, warmup
`current_index=1`, fps None/1.0/30/49, `cls_pool=mean/last`, B=1/2. The default
attention backend is SDPA; `attention_backend="flex"` is an optional accelerator
gated on a randomized SDPA parity probe (upstream's own flex eager path is
numerically lossy on some GPUs, so it is never silently selected). Repository /
checkout keys in `align_params.video` (e.g. `upstream_root`) are rejected.

## DINO-video licensing boundary

The upstream DINO-video teacher runtime is Meta DINOv3-licensed; its source is
neither vendored nor copied here. The shipped runtime is an independent,
weight-compatible first-party implementation (approved 2026-09-01), and the
teacher weights remain separate developer downloads under their own terms.
