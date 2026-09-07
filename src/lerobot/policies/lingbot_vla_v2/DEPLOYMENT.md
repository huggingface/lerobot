# LingBot-VLA 2.0 — Real-Robot Inference Guide

An operator guide for deploying a fine-tuned LingBot-VLA 2.0 policy on a physical robot
with this fork: loading a converted checkpoint, choosing an inference mode, and validating
the serving configuration before actuation. It is deliberately self-contained for the
deployment step and links out for everything upstream (training, conversion, hardware
bring-up).

> **What this is not:** a training guide, a checkpoint-conversion guide, or a benchmark
> report. See the navigation table below. It also documents only what this snapshot
> ships — `lerobot-rollout` has exactly two inference backends, `sync` and `rtc`; there
> is no remote-inference backend in this tree.

## Contents

1. [Safety](#1-safety)
2. [Prerequisites and where the other docs live](#2-prerequisites-and-where-the-other-docs-live)
3. [Checkpoint preflight — the baked recipe](#3-checkpoint-preflight--the-baked-recipe)
4. [Action-chunk semantics](#4-action-chunk-semantics)
5. [Choosing an inference mode](#5-choosing-an-inference-mode)
6. [Commands](#6-commands)
7. [Recommended starting points](#7-recommended-starting-points)
8. [Validation before actuation](#8-validation-before-actuation)
9. [Monitoring and troubleshooting](#9-monitoring-and-troubleshooting)

## 1. Safety

A 6B-parameter VLA will move the arm with full torque on the first chunk. Before the
first run on any new checkpoint or configuration:

- **Calibrate and home the robot first.** Uncalibrated encoders mean the policy's
  proprioception is wrong from tick zero; the arm may lunge to what it believes is the
  current pose. Follow the calibration procedure in your robot's guide (reBot B601:
  `docs/source/rebot_b601.mdx`).
- **Clear the workspace and keep hands out for the first run.** Run short
  (`--duration` measured in tens of seconds) before trusting longer episodes.
- **Have the e-stop / power cut reachable.** Do not rely on Ctrl-C arriving before the
  current action chunk finishes.
- **Expect the first chunk to be slow.** With compile + CUDA-graph options baked in, the
  first `predict_action_chunk` pays inductor compilation (a cold cache can take minutes)
  plus graph warm-up and capture. A multi-second gap before the first motion is expected
  behavior, not a hang.
- **Do not debug with your hands.** Use `--display_data=true` (Rerun) or the async
  client's queue visualization to observe behavior; use teleoperation to reset the scene.

## 2. Prerequisites and where the other docs live

- CUDA GPU (the CUDA-graph options are CUDA-only; see §9). Roughly 16 GB of device
  memory for the 6B policy in bfloat16 with the fast recipe.
- This repo installed with the LingBot extras:
  `uv sync --locked --extra lingbot_vla2` — add `--extra async` for the policy-server /
  robot-client mode (or `uv run pip install -e ".[lingbot_vla2,async]"`).
- A **converted, self-contained LeRobot checkpoint**: its `config.json` carries the
  per-embodiment `robot_config` and `norm_stats` embedded. Conversion from the upstream
  checkpoint format is documented in the policy README (navigation below) — do not point
  rollout at a raw upstream checkpoint.
- A calibrated robot with working cameras whose observation keys match the checkpoint's
  camera mapping (`robot_config` → `images` → `origin_keys`).

| Topic | Document |
| --- | --- |
| Policy training, fine-tuning, checkpoint conversion | `docs/source/policy_lingbot_vla_v2_README.md` |
| Full policy walkthrough (data format, new embodiments) | `docs/source/lingbot_vla_v2.mdx` |
| Generic `lerobot-rollout` flags and strategies | `docs/source/inference.mdx` |
| RTC background and theory | `docs/source/inference.mdx` §Real-Time Chunking |
| Async inference (policy server / robot client) | `docs/source/async.mdx` |
| reBot B601 hardware: ports, calibration, teleop | `docs/source/rebot_b601.mdx` |
| Depth / DINO-video distillation fine-tuning | `docs/source/lingbot_vla_v2_depth_dino_README.md` |
| Inference-optimization engineering notes | `docs/lingbot-inference-opt-plan.md` |

## 3. Checkpoint preflight — the baked recipe

Every inference-acceleration switch lives in the checkpoint's `config.json`. A converted
checkpoint that carries the fast recipe needs **zero performance CLI flags** at rollout
time — loading it *is* the configuration step. Before the first run, inspect what your
checkpoint actually baked in:

```bash
python3 - <<'EOF'
import json
cfg = json.load(open("/path/to/ckpt/config.json"))
keys = [
    "moe_implementation", "moe_backend", "attention_implementation", "dtype",
    "compile_predict_velocity", "compile_predict_velocity_mode", "compile_prefix",
    "use_cudagraph_denoise", "use_cudagraph_prefix", "use_cudagraph_prefix_full",
    "precompute_grid_thw", "preprocess_device",
    "num_steps", "chunk_size", "n_action_steps",
]
for k in keys:
    print(f"{k:38s} {cfg.get(k, '<absent → repo default>')}")
EOF
```

Interpret the output against this table. Repo defaults come from
`src/lerobot/policies/lingbot_vla_v2/configuration_lingbot_vla_v2.py`.

| Key | Repo default | Fast recipe (measured on RTX 4090) | Notes |
| --- | --- | --- | --- |
| `moe_implementation` | `fused` | **keep `fused`** | Weight **layout** metadata, not a speed knob. It must match the on-disk expert-weight layout of the checkpoint; changing it breaks weight loading. Never edit. |
| `moe_backend` | `sparse_static` | `sparse_static` | Runtime MoE execution path. The static padded-`bmm` implementation; alternatives in the config docstring (`auto`/`dense`/`sparse`/`sparse_static_gmm`/`eager`). |
| `attention_implementation` | `sdpa` | `eager` | With the custom 2D joint-attention mask, `eager` measured 8–17 ms faster than `sdpa` (which falls back to the math path) on 4090. |
| `dtype` | `bfloat16` | `bfloat16` | |
| `compile_predict_velocity` | `false` | `true` | Compiles the per-step denoiser. First call pays inductor compilation. |
| `compile_predict_velocity_mode` | `default` | `max-autotune-no-cudagraphs` | Autotuning is not bitwise-stable across cold caches — see §8. |
| `compile_prefix` | `false` | `true` | Requires `compile_predict_velocity=true`. Superseded by the prefix CUDA graph when that is on. |
| `use_cudagraph_denoise` | `false` | `true` | Whole denoise loop as one CUDA graph replay. CUDA only. |
| `use_cudagraph_prefix` | `false` | `true` | 36-layer prefix KV-fill as a graph; KV aliased into the denoise graph. |
| `use_cudagraph_prefix_full` | `false` | `true` | Additionally captures the vision tower as a graph (embed glue stays eager). |
| `precompute_grid_thw` | `false` | `true` | Caches the Qwen3-VL grid metadata; required for stable ViT capture. |
| `preprocess_device` | `null` (CPU) | `cuda` | Batched on-GPU image preprocessing; bit-exact vs CPU. Set explicitly to `"cpu"` to opt out. |
| `num_steps` | `10` | `10` eval / `7` real-robot | Flow-matching denoise steps — a **runtime variable**, see §7. |
| `chunk_size` | `50` | `50` | Prediction horizon, fixed by training. Do not edit at deploy time. |
| `n_action_steps` | `50` | real-robot: `5` | Actions consumed per chunk — a **runtime variable**, see §4/§7. |

**Editing the recipe, if you must:** copy `config.json` aside, change one key, re-run
preflight, and re-run the §8 validation. Treat it as a configuration change requiring
revalidation — not a flag flip. If the checkpoint came from training rather than
conversion, prefer re-saving the checkpoint with the training config rather than
hand-editing.

## 4. Action-chunk semantics

Four different "lengths" are in play; conflating them is the most common deployment
mistake. With `chunk_size = 50`:

| Parameter | Where | Meaning |
| --- | --- | --- |
| `chunk_size` | policy config | How many actions one model call predicts. Fixed by training. |
| `n_action_steps` | policy config | How many of those are queued and executed before a fresh model call (`select_action` refills the queue when it empties). This is the LeRobot equivalent of upstream LingBot's `use_length`. |
| `execution_horizon` | RTC config | How much of the leftover chunk RTC still expects to execute when it re-anchors guidance. Not an execution cadence. |
| `actions_per_chunk` | async robot client | How many actions the policy server returns per request (server truncates the chunk to this). |

Small `n_action_steps` (short execution, frequent replanning) trades compute for
reactivity: the model runs every `n_action_steps / fps` seconds, so the budget is
`n_action_steps × (1/fps)` for preprocess + one sample. Large `n_action_steps` amortizes
compute but compounds open-loop prediction error.

## 5. Choosing an inference mode

| Mode | How it runs | Use when | Caveats |
| --- | --- | --- | --- |
| **sync** (default) | `lerobot-rollout`, one blocking model call whenever the action queue empties | Local machine, stable setup, first bring-up | Robot holds position while a chunk computes; reaction latency = chunk latency |
| **rtc** | `lerobot-rollout --inference.type=rtc`, chunk production in a background thread with guidance against the leftover chunk | Latency or jitter you want to hide; smoother reaction under load | Guided steps **bypass the denoise CUDA graph** (by design — guidance needs autograd), so the baked graph speedup does not apply while RTC guidance is active |
| **async** (policy server / robot client) | Two processes: `lerobot.async_inference.policy_server` (GPU box) + `robot_client` (robot box), gRPC | Policy must run on a different machine than the robot | A separate program, **not** a `lerobot-rollout` backend. Validate the topology end-to-end before real runs; measured to oscillate under heavy latency jitter |

## 6. Commands

All commands assume `uv run` from this repo and placeholders `<ckpt>` (path to the
converted checkpoint directory), `<robot>`/`<port>` (your robot type and port), and a
task string matching the checkpoint's training language.

### Sync (default)

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=<ckpt> \
    --robot.type=<robot> --robot.port=<port> \
    --robot.cameras='{"camera_top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --task="pick up the cube" \
    --fps=5 \
    --duration=150 \
    --play_sounds=false
```

- Keep `--duration` ≥ 150 s for the first run: the first chunk pays compile + capture
  warm-up (§1).
- Camera keys **must** match the checkpoint's `robot_config` mapping — e.g. a checkpoint
  mapped `front → camera_top`, `wrist → camera_wrist_left` expects exactly those keys;
  unmapped canonical views are zero-filled, and a missing mapped camera is a hard error.
- A slower `--fps` (5) gives the policy headroom on first bring-up; raise it only after
  watching queue behavior.

### RTC

Same command, plus the RTC backend flags (defaults shown are the code defaults):

```bash
lerobot-rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --inference.rtc.prefix_attention_schedule=LINEAR \
    --inference.queue_threshold=30 \
    --policy.path=<ckpt> \
    --robot.type=<robot> --robot.port=<port> \
    --task="pick up the cube" \
    --duration=60 --device=cuda
```

`queue_threshold` (in action steps) sizes the buffer against production latency: if you
see the queue starving (robot holds between motions), raise it before touching anything
else.

### Async (policy server + robot client)

On the GPU box:

```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

On the robot box:

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<gpu-host>:8080 \
    --robot.type=<robot> --robot.port=<port> --robot.id=<robot_id> \
    --robot.cameras='{...}' \
    --task="pick up the cube" \
    --policy_type=lingbot_vla_v2 \
    --pretrained_name_or_path=<ckpt> \
    --policy_device=cuda \
    --actions_per_chunk=5 \
    --chunk_size_threshold=0.5
```

- `actions_per_chunk` is **required** (no default). Keep it `≤ n_action_steps` and
  aligned with your intended execution horizon (§7).
- `chunk_size_threshold` code default is `0.5` (the generic docs table says 0.7 —
  stale; the code is authoritative). Lower ≈ more synchronous; higher ≈ more
  overlapping chunks and more inference load.
- Start `--debug_visualize_queue_size=true` and watch the queue plot before trusting
  the loop.

### reBot B601 handoff

Ports (`lerobot-find-port`, udev permissions), calibration, and teleoperation are in
`docs/source/rebot_b601.mdx` — do those first. Registered types on this fork:
`rebot_b601_follower`, `bi_rebot_b601_follower`, `rebot_102_leader`,
`bi_rebot_102_leader`. Then substitute into the sync/RTC commands above.

## 7. Recommended starting points

These are starting points, not universal values. Label discipline: **defaults** are what
the code ships; **field-tuned** values were productive on one measured setup and must be
re-validated on yours.

| Scenario | `num_steps` | execution length | Mode | Notes |
| --- | --- | --- | --- | --- |
| RoboTwin / comparable benchmark eval | checkpoint default (10) | checkpoint default (`n_action_steps=50`) | sync | Retain defaults for comparability with reported scores; change only inside a documented speed/quality study. |
| Real robot, first bring-up | 10 | 50 | sync | Zero surprises: full graph path, no extra flags. |
| Real robot, short-horizon closed loop *(field-tuned)* | 7 | `n_action_steps=5` | sync or rtc | Field result from a 25 fps dual-arm setup: going from `use_length=10` to `5` (here: `n_action_steps=5`) changed hovering-no-contact into real grasp attempts. Budget check: 5 × 40 ms = 200 ms per replan. |
| RTC under high link latency *(field-tuned)* | 7 | `n_action_steps=5` + `execution_horizon=24` + `queue_threshold=30` | rtc | Tuned on a ~600–800 ms RTT cloud link; the horizon must exceed the link's delay (in steps) or guidance anchoring degrades. Measure your own delay before adopting. |

## 8. Validation before actuation

Before the first actuating run on a new checkpoint or after any `config.json` edit:

1. **Preflight the config** — §3: confirm the recipe keys and that `moe_implementation`
   is untouched.
2. **One robotless sample** (any machine with the checkpoint):

   ```bash
   python3 - <<'EOF'
   import torch
   from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
   policy = LingbotVLAV2Policy.from_pretrained("<ckpt>").to("cuda").eval()
   # ... assemble one observation batch via the policy's preprocessor ...
   # (see docs/source/policy_lingbot_vla_v2_README.md §Load A Policy)
   EOF
   ```

   Watch the log lines: each enabled feature reports capture/compile status
   (`use_cudagraph_denoise: captured ...`, `use_cudagraph_prefix: captured ...`,
   `use_cudagraph_prefix_full: captured the vision tower ...`). A *fallback warning*
   here means the fast path silently degraded — resolve it before actuating.

3. **GPU-side parity/graph checks** (portable via env vars; run on the deployment GPU):

   ```bash
   GRAPH_ROOT=$PWD GRAPH_CKPT=<ckpt> GRAPH_OUT=/tmp/graph_integ \
       python3 bench/graph_integ.py t7     # vision+prefix+denoise graphs, bitwise
   GRAPH_ROOT=$PWD GRAPH_CKPT=<ckpt> GRAPH_OUT=/tmp/graph_integ \
       python3 bench/graph_integ.py t7b    # shape change → re-capture, bitwise
   python3 bench/check_gpu_preprocess.py --ckpt <ckpt>   # CPU vs GPU preprocess parity
   ```

   `t7c` is the forced prefix-capture-failure regression: it verifies the
   eager fallback is bitwise against eager reference (4090 validation: PASS).

**Numerical discipline:** any behavior-affecting change must be compared bitwise with
compile **off** (eager), fixed seed and fixed noise, per the template in
`bench/graph_integ.py`. Under `max-autotune-no-cudagraphs` a cold inductor cache
re-tunes kernels and legitimately shifts bf16 outputs by ~0.1 max-abs for byte-identical
code — eager kernels are the only deterministic reference.

## 9. Monitoring and troubleshooting

| Symptom | Cause / remedy |
| --- | --- |
| First chunk takes seconds to minutes | Compile + graph capture warm-up. Expected. Keep `--duration` ≥ 150 s on first runs. |
| Log: `use_cudagraph_*: capture failed ... using the eager/plain ...` | Capture failed and the feature fell back (safe but slower). Read the stated reason (host sync during capture, recompile, OOM). Fix the cause — common triggers: probes/wrappers inside the compiled region, mismatched dtype, non-CUDA device. |
| Log: `... shapes changed; re-capturing ...` | Observation shape changed (camera resolution, batch). Expected on the first call after a change; the graph re-captures once per new shape. Avoid shape flicker in steady state. |
| Robot holds between motions (sync) | Chunk latency exceeds the execution budget — lower `n_action_steps` (or `fps`), or move to RTC. |
| Robot holds between motions (RTC) | Queue starvation: raise `--inference.queue_threshold` (buffer depth in steps) before anything else; check `execution_horizon` exceeds your link delay in steps. |
| RTC smoothness worse than sync despite lower latency | Guidance steps bypass the denoise CUDA graph — per-step cost rises. Compare with `max_guidance_weight` effectively off before tuning further. |
| Async queue oscillates / empties | Watch `--debug_visualize_queue_size=true`; adjust `chunk_size_threshold` (0.5–0.6 typical) and `actions_per_chunk`; reduce `fps` if the server can't keep up. gRPC async is known to oscillate under heavy latency jitter — prefer RTC on bad links. |
| CUDA OOM on load | ~16 GB needed with the full recipe. Free other processes on the device, or drop `use_cudagraph_prefix_full`/`use_cudagraph_prefix` first (they cost extra capture pools), then compile flags. |
| Latency tracker shows ever-growing max | Known pitfall: `LatencyTracker.max()` is monotonic; read a windowed/p95 statistic instead of the running max. |
| Actions look normalized / wild magnitudes | The checkpoint lacks a matching `norm_stats`/`robot_config`, or the un-apply step was skipped. Do not run the arm. Re-check conversion; the policy raises rather than silently emitting normalized values. |

**Known boundaries of this snapshot:** CUDA graphs and compile are CUDA-only (CPU/other
backends fall back to eager gracefully); RTC guidance bypasses the denoise graph; there
is no remote/serve-your-own backend in `lerobot-rollout` — use the async policy server
for cross-machine serving.
