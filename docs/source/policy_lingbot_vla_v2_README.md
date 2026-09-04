# LingBot-VLA 2.0

`lingbot_vla_v2` is the LeRobot policy wrapper for LingBot-VLA 2.0. It combines a
Qwen3-VL vision-language backbone with a sparse MoE Qwen2 action expert and flow-matching
continuous action generation over the canonical 55-D robot state/action space.

Use this policy through the standard LeRobot interfaces: `lerobot-train`,
`make_policy_config`, `make_policy`, `from_pretrained`, and `predict_action_chunk` /
`select_action`.

## Install

Install LeRobot with the optional LingBot-VLA v2 dependencies:

```bash
pip install -e ".[training,lingbot_vla2]"
```

The default config expects a Qwen3-VL processor/tokenizer and a LingBot-VLA v2 checkpoint:

```text
Qwen/Qwen3-VL-4B-Instruct
robbyant/lingbot-vla-v2-6b
```

Use local paths with `--policy.processor_path`, `--policy.tokenizer_path`, or
`--policy.path` when running offline.

## Train

First convert the raw upstream checkpoint into the self-contained LeRobot format:

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-lerobot \
  --robot-config-path <robot_config.yaml> \
  --norm-stats-path <norm_stats.json>
```

Then use the normal LeRobot training CLI, initializing from the converted checkpoint with
`--policy.path`. Do not also pass `--policy.type`; LeRobot infers `lingbot_vla_v2` from the
checkpoint's `config.json`.

```bash
lerobot-train \
  --dataset.repo_id=<repo_id> \
  --dataset.root=<dataset_root> \
  --policy.path=<converted-lingbot-vla-v2-6b> \
  --policy.robot_config_path=<robot_config.yaml> \
  --policy.norm_stats_path=<norm_stats.json> \
  --policy.processor_path=<qwen3_vl_processor_or_model_path> \
  --policy.tokenizer_path=<qwen3_vl_processor_or_model_path> \
  --policy.image_max_pixels=262144 \
  --policy.image_min_pixels=131072 \
  --policy.device=cuda \
  --batch_size=1 \
  --steps=5000 \
  --save_freq=2500 \
  --output_dir=outputs/train/lingbot_vla_v2
```

For an offline smoke test, add `--policy.push_to_hub=false`, `--save_checkpoint=false`,
and `--num_workers=0`, then use local paths for the processor, tokenizer, and checkpoint.

Required data assets:

- `robot_config_path`: maps dataset state/action/image keys into LingBot-VLA v2 canonical slots.
- `norm_stats_path`: stores per-slot normalization stats used by the LingBot feature transform.
- `processor_path` / `tokenizer_path`: Qwen3-VL processor/tokenizer path or Hub id. Use local
  paths when the training node cannot reach the Hugging Face Hub.

The robot config maps dataset keys into the canonical LingBot slots. The norm-stats JSON is
used by the LingBot feature transform, so the saved LeRobot processor pipeline does not use
the generic LeRobot normalizer/unnormalizer steps.

### Native-depth / DINO-video fine-tuning

The action-only conversion above intentionally drops predictive-distillation heads. To retain
the official native-depth checkpoint heads and train with the frozen MoGe/MoRGBD/DINO-video
teachers, convert with `--include-depth-heads` and follow the weight-download-only recipe in
[`lingbot_vla_v2_depth_dino_README.md`](./lingbot_vla_v2_depth_dino_README.md). All three
teacher runtimes are first-party LeRobot implementations verified against the upstream
teachers on real weights; no upstream checkout is ever required.

## Training on consumer GPUs (24GB-class, measured)

### Single 24GB card

Both flags are mandatory:

- `--policy.train_expert_only=true`
- `--policy.gradient_checkpointing=true`

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended. Measured: 22.0GB peak
memory, batch size 1, ~0.7s/step.

Full single-card command (validated end to end):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=local/rebot_shakehands --dataset.root=/root/datasets/rebot_shakehands \
  --dataset.streaming=false --policy.type=lingbot_vla_v2 \
  --policy.pretrained_path=<converted_ckpt> \
  --policy.robot_config_path=<robot_config.yaml> --policy.norm_stats_path=<norm_stats.json> \
  --policy.dtype=bfloat16 --policy.optimizer_fused=true --policy.push_to_hub=false \
  --policy.train_expert_only=true --policy.gradient_checkpointing=true \
  --batch_size=1 --steps=10 --log_freq=1 --num_workers=4 \
  --save_checkpoint=false --output_dir=<out> --job_name=smoke --wandb.enable=false
```

### 2x24GB with FSDP2

Four requirements — missing any one of them fails the run:

1. `train_expert_only=true`.
2. The accelerate YAML must `auto_wrap` three layer classes, given as a comma-separated
   string: `Qwen3VLVisionBlock,Qwen3VLTextDecoderLayer,Qwen2DecoderLayer`.
3. `fsdp_offload_params=true`.
4. The `policy()` call bug fixed on this branch (present upstream).

Measured: ~23s/step (the offload cost). `cpu_ram_efficient_loading` must be turned off
(accelerate bug). Full fine-tuning on 2x24GB is physically infeasible — the optimizer
states alone are 96GB across 2 cards.

### Rollout

```bash
lerobot-rollout --strategy.type=base --policy.path=<ckpt> --robot.type=<robot> \
  --task="..." --fps=5 --duration=150 --play_sounds=false
```

The first chunk pays the compile/capture warmup (a cold inductor cache can take minutes),
so keep `--duration` at 150s or above. Without a real robot, a third-party mock robot can
be registered through the `register_third_party_plugins` mechanism; see the reference
implementation in `bench/rollout_plugin/`.

## Adapting to a New Embodiment

Fine-tuning on a robot the checkpoint was not converted for only requires two new assets —
a robot-config YAML and a norm-stats JSON — passed as `--policy.robot_config_path` /
`--policy.norm_stats_path`. Explicit paths take precedence over the assets embedded in the
checkpoint (a warning is logged when they differ), and checkpoints saved during fine-tuning
embed the new assets so they remain self-contained.

Single-arm example (7-DoF: 6 arm joints + gripper, absolute joint angles, `front` + `wrist`
cameras) — the filled dims are packed from position 0 of each canonical slot, unfilled slots
are zero-padded and masked out of the loss:

```yaml
# rebot.yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state: { start: 0, end: 6 }
  - observation.state.effector.position:
      origin_keys:
        - observation.state: { start: 6, end: 7 }
actions:
  - action.arm.position:
      origin_keys:
        - action: { start: 0, end: 6 }
      subtract_state: false # absolute actions; true only for state-relative deltas
  - action.effector.position:
      origin_keys:
        - action: { start: 6, end: 7 }
      subtract_state: false
images: # unmapped canonical views (camera_wrist_right) are zero-filled
  - observation.images.camera_top:
      origin_keys: observation.images.front
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.wrist
norm_stats: rebot_norm_stats.json
```

The norm-stats JSON holds per-slot `mean`/`std` over the filled dims (the default
`canonical_norm_type` is `meanstd`), e.g. `{"norm_stats": {"action.arm.position": {"mean":
[...6], "std": [...6]}, ...}}` — derivable from a LeRobot dataset's `meta/stats.json`.

Deploy the fine-tuned checkpoint on the robot with `lerobot-rollout`
(`--strategy.type=episodic` to record evaluation episodes); camera names must match the
`origin_keys` above. `select_action` returns actions in the robot's raw action space.

See `docs/source/lingbot_vla_v2.mdx` for the full walkthrough.

## Resume

Resume from a saved LeRobot checkpoint by passing the checkpoint's `train_config.json`:

```bash
lerobot-train \
  --resume=true \
  --config_path=outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model/train_config.json \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/lingbot_vla_v2_resume
```

## Load A Policy

```python
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

policy = LingbotVLAV2Policy.from_pretrained(
    "outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model"
)
policy.to("cuda").eval()
```

For batched LeRobot observations, use `select_action`. For open-loop action chunks, use
`predict_action_chunk`; it returns a `(batch, chunk_size, action_dim)` tensor after the
policy postprocessing path has mapped canonical actions back to raw dataset action keys.

## Inference acceleration

Final config, measured on an RTX 4090. Every switch is baked into the converted
checkpoint's `config.json`, so loading the checkpoint is enough — no CLI flags needed:

| Config key                 | Value      | Notes                                                                                                                                                                                                 |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attention_implementation` | `eager`    | Joint attention in bf16. 8-17ms faster than `sdpa`: with the custom 2D mask, `sdpa` falls back to the math path, while `eager`'s pointwise ops get fused by inductor. ViT internals still use `sdpa`. |
| `dtype`                    | `bfloat16` |                                                                                                                                                                                                       |
| `compile_predict_velocity` | `true`     | `mode="max-autotune-no-cudagraphs"`, per-step scope.                                                                                                                                                  |
| `compile_prefix`           | `true`     |                                                                                                                                                                                                       |
| `preprocess_device`        | `cuda`     |                                                                                                                                                                                                       |
| `precompute_grid_thw`      | `true`     |                                                                                                                                                                                                       |
| `use_cudagraph_denoise`    | `true`     | New switch, default `false` — see below.                                                                                                                                                              |
| `num_steps`                | `7`        |                                                                                                                                                                                                       |

Steps vs. latency, measured with `cuda.Event` around policy-level `sample_actions` on an
RTX 4090 with the config above:

| `num_steps` | CUDA graph on | CUDA graph off |  Gain |
| ----------: | ------------: | -------------: | ----: |
|           4 |       108.5ms |        135.5ms | -27ms |
|           7 |       130.9ms |        175.9ms | -45ms |
|          10 |       153.0ms |        227.0ms | -74ms |

Note: the 7-step 130.9ms matches the official README's 130ms claim numerically, but that
claim is not reproducible from the public repo — it equals the official stack's pure GPU
busy time (113.7ms) and requires an internal CUDA graph build.

### `use_cudagraph_denoise`

Captures the entire denoise loop into a single CUDA graph replay.

- Zero numerical change — bitwise verified (`max|delta|=0`) across varying observations,
  first call, `sdpa`/`eager`, and with/without compile.
- The first call costs two extra warm-up + capture passes.
- Observation shape changes drop the stale graph and re-capture (a warning is logged
  once per new shape); a warm-up/capture failure disables the graph for the instance
  and falls back to the plain loop.
- CUDA only.

Numerical validation discipline: any performance change must pass the bitwise-comparison
template with fixed seed and fixed noise in `bench/` (the `graph_integ.py` pattern).
Bitwise comparisons must run with compile **off** (eager): under
`mode="max-autotune-no-cudagraphs"` a cold or invalidated inductor cache re-tunes kernel
choices, and the resulting bf16 reassociation makes outputs differ by ~0.1 (max abs) even
for byte-identical code — measured 0.06-0.13 on the same checkpoint. Eager kernels are
deterministic, so only an eager-mode bitwise diff of exactly 0 proves a refactor
numerically neutral.

### RTC (Real-Time Chunking)

RTC under `policies/rtc/` works on this branch. Measured equivalent reaction latency
(30fps simulation): sync ~1.0-1.2s, async ~700ms, RTC ~580ms. Guidance overhead is only
+9.1% — LeRobot's `RTCProcessor` gradient path is an identity mapping, so with frozen
parameters there is no extra backward. Two pitfalls: `LatencyTracker.max()` is
monotonically increasing and should be replaced with a p95 window; gRPC async is not
recommended (it oscillates under latency jitter).

## Tests

Run the lightweight registration and config tests:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_lingbot_vla_v2.py
```

Run the feature-transform tests when a local Qwen3-VL processor is available:

```bash
export LINGBOT_VLA_V2_QWEN3VL=/path/to/Qwen3-VL-4B-Instruct
pytest -q tests/policies/lingbot_vla_v2/test_feature_transform.py
```

Run the optional Triton grouped-MoE parity test on a CUDA machine with Triton installed:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_moe_eager.py
```

## Citation

If you use this policy, please cite the upstream LingBot-VLA 2.0 project and LeRobot.
The upstream project is available at:

```text
https://github.com/Robbyant/lingbot-vla-v2
```

## Implementation Notes

The Qwen3-VL backbone adaptation and sparse-MoE action expert are vendored from the upstream
LingBot-VLA 2.0 implementation and Hugging Face Transformers, with Apache-2.0 license headers
retained. FlashAttention is optional. The default attention implementation is `sdpa` in the
model dtype (bf16); `eager`, `fa2`, `flex`, and `flex_cached` can be selected through the policy
config, and `attention_fp32=true` restores the original fp32-attention parity path.

For MoE inference, the fused expert path tries the optional upstream Triton kernel first, then
the in-tree Triton grouped-GEMM backend, and finally the grouped-by-expert eager fallback. The
training path uses the eager fallback for autograd stability; the eager fallback groups routes with
a single argsort (one host sync per layer) instead of per-expert nonzero scans.
