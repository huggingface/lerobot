# LingBot-VLA 2.0

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Robbyant/lingbot-vla-v2/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![LeRobot](https://img.shields.io/badge/%F0%9F%A4%97-LeRobot%20Policy-yellow)](https://github.com/huggingface/lerobot)

</div>

**LingBot-VLA 2.0** is a LeRobot policy that combines a Qwen3-VL vision-language backbone with a sparse-MoE Qwen2 action expert and flow-matching continuous action generation over the canonical 55-D robot state/action space.

🤗 Qwen3-VL vision-language backbone with native multi-view image support.

🤗 Sparse-MoE Qwen2 action expert with flow-matching continuous action heads.

🤗 Real-robot fine-tuning fits on a single 24GB consumer GPU via expert-only training + gradient checkpointing (LoRA optional), with a validated FSDP2 path for 2×24GB.

## Quick Start

```bash
pip install "lerobot[lingbot_vla2]"
lerobot-info
```

> Requires Python ≥3.12. `flash-attn` is optional (the sdpa/eager attention
> fallbacks are used when it is absent); the feature-transform tests expect
> the Qwen3-VL processor files locally (see [Model Weights](#model-weights)).

## Model Weights

All weights are hosted on [Hugging Face](https://huggingface.co) and, where marked, mirrored on [ModelScope](https://modelscope.cn). The base checkpoint is gated on HF — request access first; the ModelScope mirror is an alternative.

| Asset                            | Hub id                                                                                                                                                 | Size   | Needed for                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | -------------------------- |
| Base VLA checkpoint (pretrained) | `robbyant/lingbot-vla-v2-6b` — [HF](https://huggingface.co/robbyant/lingbot-vla-v2-6b) · [MS](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b) | ~26 GB | all training and inference |
| Qwen3-VL processor / tokenizer   | `Qwen/Qwen3-VL-4B-Instruct` — [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [MS](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)    | ~13 GB | always                     |

```bash
# ModelScope (China-friendly mirror of the base checkpoint)
modelscope download --model Robbyant/lingbot-vla-v2-6b
```

Use local paths with `--policy.tokenizer_path` or `--policy.pretrained_path` when running offline.

## Dataset

LingBot-VLA 2.0 trains on standard **LeRobotDataset** — no custom format.

```bash
lerobot-record \
  --robot.type=<robot_name> \
  --teleop.type=<teleoperator_name> \
  --dataset.repo_id=${HF_USER}/my-dataset
```

To map your robot into the canonical LingBot slots you need two small assets (a robot-config YAML and a norm-stats JSON). See [Adapting to a New Embodiment](#adapting-to-a-new-embodiment) for a filled-in example.

## Train

Two training profiles are supported, chosen at checkpoint-conversion time. They differ in loss type, normalization, and hardware budget:

| Profile    | Loss    | Normalization                                    | Intended for                                                | Hardware               |
| ---------- | ------- | ------------------------------------------------ | ----------------------------------------------------------- | ---------------------- |
| `robotwin` | `L1_fm` | `bounds_99_woclip` (q01/q99 bounds, no clipping) | RoboTwin / sim benchmarks — maximize success rate           | datacenter GPUs        |
| `real`     | `fm`    | `meanstd`                                        | Real-robot fine-tuning — fast iteration on limited hardware | single 24GB card works |

The two paths below share the converter and the `lerobot-train` CLI but differ in data prep, profile, and hardware requirements — follow the one that matches your target.

### Path A — RoboTwin Training & Evaluation (`--profile robotwin`): high fidelity, heavy compute

Target: the RoboTwin simulation benchmark. Success rate is the metric; compute is assumed cheap (datacenter GPUs, larger batches, no memory-saving flags).

```
RoboTwin HDF5 ──robotwin_to_lerobot.py──▶ lerobot dataset ──lerobot-train──▶ lerobot ckpt
RoboTwin sim  ◀──official eval client──websocket── lingbot_vla_v2_policy_lerobot.py ◀┘
```

Everything runs inside the lerobot stack — no upstream LingBot training code.

**1. Convert data** (14-dim dual-arm, 3 cameras, teacher-shifted actions):

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.robotwin_to_lerobot \
  --input-dir /path/to/robotwin_task_episodes \
  --repo-id my_robotwin_task \
  --fps 15 --mode video
```

**2. Convert + train** (bakes `L1_fm` loss + `bounds_99_woclip` normalization; norm stats are derived from the dataset automatically):

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-robotwin-6b \
  --profile robotwin
```

```bash
lerobot-train \
  --dataset.repo_id=my_robotwin_task --dataset.root=/path/to/lerobot_dataset \
  --dataset.streaming=false --dataset.video_backend=pyav \
  --policy.path=./lingbot-robotwin-6b --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.loss_type=L1_fm --policy.freeze_vision_encoder=false --policy.vlm_causal=true \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=5e-5 \
  --policy.scheduler_warmup_steps=0 --policy.scheduler_decay_steps=50000 \
  --policy.gradient_checkpointing=true --policy.moe_backend=sparse_static \
  --policy.use_moe_expert_lr=true \
  --batch_size=16 --steps=50000 --save_freq=5000 --num_workers=8 --seed=42 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/lingbot_vla_v2_robotwin
```

> The converted checkpoint embeds optimizer/scheduler values that override the code defaults — pass the training-hyperparameter flags explicitly; do not omit them.

**Multi-GPU** (verified on 8×A100-80GB) — data-parallel via torchrun:

```bash
torchrun --nproc_per_node=8 -m lerobot.scripts.lerobot_train <same args>
```

FSDP2 — add the sharding degree and mixed precision (the policy declares `_fsdp_wrap_modules`, so no manual wrap-modules flag needed):

```bash
torchrun --nproc_per_node=8 -m lerobot.scripts.lerobot_train <same args> \
  --parallelism.dp_shard=-1 \
  --accelerator.mixed_precision=bf16
```

Smoke-test with 2 processes × 20 steps before scaling up.

**4. Evaluate** through LeRobot's native RoboTwin env:

```bash
lerobot-eval \
  --policy.path=./lingbot-robotwin-6b \
  --env.type=robotwin \
  --env.task=beat_block_hammer \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --rename_map='{"observation.images.head_camera": "observation.images.cam_high", "observation.images.left_camera": "observation.images.cam_left_wrist", "observation.images.right_camera": "observation.images.cam_right_wrist"}'
```

> The `--rename_map` maps the env's camera keys (`head_camera`/`left_camera`/`right_camera`) onto the checkpoint's expected keys (`cam_high`/`cam_left_wrist`/`cam_right_wrist`).

### Path B — Real-Robot Fine-Tuning (`--profile real`): fast, light

Target: a physical robot. Iteration speed matters more than the last point of accuracy; a single 24GB consumer card is enough.

**1. Convert** (format-only: `fm` loss + identity processor; the embodiment is provided at training time):

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-real \
  --profile real
```

**2. Train (single 24GB card)** — `--policy.train_expert_only=true` and `--policy.gradient_checkpointing=true` are mandatory at this memory budget; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended. Measured: 22.0GB peak, batch size 1, ~0.7s/step. The slot mappings describe how your robot's raw joints map onto the canonical 55-D space; normalization stats are derived from the dataset automatically:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --policy.path=./lingbot-vla-v2-6b-real \
  --dataset.repo_id=${HF_USER}/my-dataset \
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 6}}]}}' \
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 6}}], "subtract_state": false}}' \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --batch_size=1 --steps=30000 --save_freq=5000 --num_workers=4 \
  --output_dir=outputs/train/lingbot_vla_v2
```

**Multi-GPU (A100-class)** — drop the two memory-saving flags (gradient checkpointing off is ~45% faster when memory allows), raise `--batch_size` to 2-4 per process, and launch via torchrun (FSDP2: add the flags from the RoboTwin path above):

```bash
torchrun --nproc_per_node=8 -m lerobot.scripts.lerobot_train <same args, --batch_size=2>
```

**LoRA (optional).** Add two flags to either command to train expert-only LoRA adapters (~0.2B parameters instead of 6B):

```bash
lerobot-train <same args> --peft.use_peft=true --peft.r=32
```

Merge adapter checkpoints back into the base weights with `scripts/export_merged.py` before deployment.

**Worked example (verified).** Fine-tuning from the base checkpoint on a 6-DoF single-arm dataset (5 arm joints + 1 gripper, cameras `top` + `wrist`):

```bash
lerobot-train \
  --dataset.repo_id=maximellerbach/omx_multicubes \
  --policy.path=miracle-techlink/lingbot-vla-v2-6b-lerobot \
  --policy.tokenizer_path=Qwen/Qwen3-VL-4B-Instruct \
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}], "subtract_state": false}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}], "subtract_state": false}}' \
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --batch_size=1 --steps=30000 --save_freq=5000
```

The slot mappings are typed dict fields passed as JSON on the CLI (same convention as `--policy.normalization_mapping` on pi05). This runs end-to-end: the preprocessor maps the 6-D raw state/action onto the canonical 55-D slots, training produces checkpoints with the slot mapping + dataset stats embedded, and `lerobot-rollout` / `lerobot-eval` on the saved checkpoint map back to the robot's 6-D action space.

**3. Deploy** — see [Inference & Deployment](#inference--deployment) below (`lerobot-rollout` on the robot).

A validated 2×24GB FSDP2 path also exists (Accelerate `fully_shard` with a CPU-offloaded optimizer, gradient checkpointing, validated robot config, and embedded norm stats).

### Optimizer

The default recipe uses AdamW and is fully supported. A Muon-based optimizer
(`--policy.optimizer_type=muon`) matching the upstream training recipe is provided by
the standalone Muon PR — it implements the 3D-MoE / FSDP2-distributed Muon that
`torch.optim.Muon` does not (batched Newton–Schulz over expert stacks, sharded-parameter
mega-batching). The benchmark numbers in this README were produced with the default
AdamW recipe unless a checkpoint notes otherwise.

## Inference & Deployment

**Sync (default)** — start here for first bring-up:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=<ckpt> \
  --robot.type=<robot> --robot.port=<port> \
  --task="pick up the cube" \
  --fps=5 --duration=150
```

- **First-run warmup is slow by design.** The first chunk triggers `torch.compile` and CUDA-graph capture; with a cold inductor cache this can take minutes. Keep `--duration` ≥ 150s on the first run.
- **Camera keys are part of the checkpoint contract.** They must match the `robot_config` mapping baked into the checkpoint; unmapped canonical views are zero-filled.

**RTC (recommended for real-robot closed loop)** — chunk production runs in a background thread with guidance against the leftover chunk, hiding chunk latency from the control loop:

```bash
lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --inference.queue_threshold=30 \
  --policy.path=<ckpt> \
  --robot.type=<robot> --robot.port=<port> \
  --task="pick up the cube" \
  --fps=25 --duration=150
```

For a real-robot closed loop, set `num_steps=7` and `n_action_steps=5` in the checkpoint's `config.json`; keep the checkpoint defaults (10/50, sync) for first bring-up and benchmark eval.

Inference speed is baked into the checkpoint's `config.json` — sparse MoE routing, a hand-written grouped-`bmm` MoE kernel, CUDA graphs, and `torch.compile` — delivering ~170ms per 7-step denoise chunk on an RTX 4090 (model-only latency; camera capture and observation assembly are extra).

## Adapting to a New Embodiment

Fine-tuning on a robot the checkpoint was not converted for requires only the slot mappings — passed as `--policy.state_slots` / `--policy.action_slots` (typed dict fields). The norm stats are derived from the dataset automatically (LeRobot's `dataset_stats` mechanism), and checkpoints saved during fine-tuning embed the slot mappings so they remain self-contained.

The canonical slot vocabulary (the 55-D layout, per-slot normalization modes), and how the norm stats are derived are covered in the full walkthrough: [`lingbot_vla_v2.mdx`](./lingbot_vla_v2.mdx).

## Citation

If you use this policy, please cite the upstream LingBot-VLA 2.0 project and LeRobot:

```bibtex
@misc{lingbotvla2025,
    title = {LingBot-VLA 2.0},
    author = {Robbyant Team},
    howpublished = {\url{https://github.com/Robbyant/lingbot-vla-v2}},
    year = {2025}
}
```

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = {\url{https://github.com/huggingface/lerobot}},
    year = {2024}
}
```
