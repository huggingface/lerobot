# LingBot-VLA 2.0

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Robbyant/lingbot-vla-v2/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![LeRobot](https://img.shields.io/badge/%F0%9F%A4%97-LeRobot%20Policy-yellow)](https://github.com/huggingface/lerobot)

</div>

**LingBot-VLA 2.0** is a LeRobot policy that combines a Qwen3-VL vision-language backbone with a sparse-MoE Qwen2 action expert and flow-matching continuous action generation over the canonical 55-D robot state/action space.

🤗 Qwen3-VL vision-language backbone with native multi-view image support.

🤗 Sparse-MoE Qwen2 action expert with flow-matching continuous action heads.

🤗 Optional predictive-distillation heads (native-depth / DINO-video) with frozen, first-party teacher implementations — no upstream checkout required.

🤗 Real-robot fine-tuning fits on a single 24GB consumer GPU via expert-only training + gradient checkpointing (LoRA optional), with a validated FSDP2 path for 2×24GB.

## Quick Start

```bash
pip install "lerobot[lingbot_vla2]"
lerobot-info
```

## Model & Teacher Weights

All weights are hosted on [Hugging Face](https://huggingface.co) and, where marked, mirrored on [ModelScope](https://modelscope.cn). The base checkpoint is gated on HF — request access first; the ModelScope mirror is an alternative.

| Asset | Hub id | Size | Needed for |
| --- | --- | --- | --- |
| Base VLA checkpoint (pretrained) | `robbyant/lingbot-vla-v2-6b` — [HF](https://huggingface.co/robbyant/lingbot-vla-v2-6b) · [MS](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b) | ~26 GB | all training and inference |
| Qwen3-VL processor / tokenizer | `Qwen/Qwen3-VL-4B-Instruct` — [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [MS](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) | ~13 GB | always |
| MoGe-v2 depth teacher | `Ruicheng/moge-2-vitb-normal` — [HF](https://huggingface.co/Ruicheng/moge-2-vitb-normal) ([`model.pt`](https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/main/model.pt)) | 419 MB | `--include-depth-heads` |
| MoRGBD depth teacher | `robbyant/lingbot-vla-v2-6b` → [`depth/model.pt`](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/depth) | 1.32 GB | `--include-depth-heads` |
| DINO-video teacher | `robbyant/lingbot-vla-v2-6b` → [`dino_video/`](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/dino_video) (`teacher_step_10000.pth` + `config.yaml`) | 1.40 GB | `--include-depth-heads` |

One-shot download for the distillation recipe:

```bash
# Hugging Face
hf download Ruicheng/moge-2-vitb-normal model.pt
hf download robbyant/lingbot-vla-v2-6b --include "depth/*" "dino_video/*"

# ModelScope (China-friendly mirror of the base checkpoint)
modelscope download --model Robbyant/lingbot-vla-v2-6b
```

Note: the MoGe teacher has its own repo, but the MoRGBD and DINO-video teachers have no standalone repository — they are files inside the gated `robbyant/lingbot-vla-v2-6b` checkpoint with no per-file link. Either use the `hf download --include` command above, or open the linked `depth/` / `dino_video/` folders in a browser and download the files manually after gaining access to the base checkpoint. The full distillation recipe is in [`lingbot_vla_v2_depth_dino_README.md`](../../../../../docs/source/lingbot_vla_v2_depth_dino_README.md).

Use local paths with `--policy.processor_path`, `--policy.tokenizer_path`, or `--policy.pretrained_path` when running offline.

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

| Profile | Loss | Normalization | Intended for | Hardware |
| --- | --- | --- | --- | --- |
| `robotwin` | `L1_fm` | `bounds_99_woclip` (q01/q99 bounds, no clipping) | RoboTwin / sim benchmarks — maximize success rate | datacenter GPUs |
| `real` | `fm` | `meanstd` | Real-robot fine-tuning — fast iteration on limited hardware | single 24GB card works |

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
python scripts/robotwin_to_lerobot.py \
  --input-dir /path/to/robotwin_task_episodes \
  --repo-id my_robotwin_task \
  --fps 15 --mode video
```

**2. Norm stats** (`bounds_99_woclip` needs the `q01/q99` quantiles):

```bash
python gen_rebot_norm_stats.py \
  --dataset-root /path/to/lerobot_dataset \
  --quantiles --out norm_stats.robotwin.json
```

**3. Convert + train** (bakes `L1_fm` loss + `bounds_99_woclip` normalization):

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-robotwin-6b \
  --robot-config-path robotwin.yaml \
  --norm-stats-path norm_stats.robotwin.json \
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

FSDP2 — add the sharding degree and the wrap-unit classes, with `--policy.dtype=float32` (the old `accelerate launch --config_file` flow is superseded on this branch; mixed precision under sharding supports `no`/`bf16` only):

```bash
torchrun --nproc_per_node=8 -m lerobot.scripts.lerobot_train <same args> \
  --parallelism.dp_shard=-1 \
  --accelerator.fsdp.wrap_modules='["Qwen3VLTextDecoderLayer","Qwen3VLVisionBlock","Qwen2DecoderLayer"]'
```

Smoke-test with 2 processes × 20 steps before scaling up.

**4. Evaluate** on the official benchmark: copy `scripts/lingbot_vla_v2_policy_lerobot.py` into the RoboTwin checkout's `deploy/`, then point the **unchanged** official launcher at it:

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path ./lingbot-robotwin-6b \
  --eval_workdir /path/to/RoboTwin \
  --inference_script deploy/lingbot_vla_v2_policy_lerobot.py \
  --conda_sh /path/to/miniconda3/etc/profile.d/conda.sh \
  --inference_env lerobot --sim_env RoboTwin \
  --num_tasks 1 --num_gpus 1 --num_per_gpu 1   # smoke; --num_tasks 50 for the full benchmark
```

Full walkthrough with download links and the sim/training environment split: [`ROBOTWIN_GUIDE.md`](./ROBOTWIN_GUIDE.md).

### Path B — Real-Robot Fine-Tuning (`--profile real`): fast, light

Target: a physical robot. Iteration speed matters more than the last point of accuracy; a single 24GB consumer card is enough.

**1. Convert** (bakes `fm` loss + `meanstd` normalization into the checkpoint):

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-real \
  --robot-config-path <robot_config.yaml> \
  --norm-stats-path <norm_stats.json> \
  --profile real
```

**2. Train (single 24GB card)** — `--policy.train_expert_only=true` and `--policy.gradient_checkpointing=true` are mandatory at this memory budget; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended. Measured: 22.0GB peak, batch size 1, ~0.7s/step.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --policy.path=./lingbot-vla-v2-6b-real \
  --dataset.repo_id=${HF_USER}/my-dataset \
  --policy.robot_config_path=<robot_config.yaml> --policy.norm_stats_path=<norm_stats.json> \
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
lerobot-train <same args> --policy.use_peft=true --policy.peft.r=32
```

Merge adapter checkpoints back into the base weights with `scripts/export_merged.py` before deployment.

**Distillation teachers (optional, A100-class).** Convert with `--include-depth-heads` (loads the official depth/DINO heads and embeds the teacher `align_params`), then enable the frozen MoGe/MoRGBD/DINO-video teachers at train time — verified with FSDP2 on 8×A100:

```bash
torchrun --nproc_per_node=N -m lerobot.scripts.lerobot_train <same args> \
  --policy.use_depth=true --policy.dataset_fps=<dataset fps>
```

Acceptance: `depth_loss`, `future_depth_loss`, and `future_video_loss` all appear in the training logs. Teacher weights are listed in [Model & Teacher Weights](#model--teacher-weights); the full recipe is in [`lingbot_vla_v2_depth_dino_README.md`](../../../../../docs/source/lingbot_vla_v2_depth_dino_README.md).

**3. Deploy** — see [Inference & Deployment](#inference--deployment) below (`lerobot-rollout` on the robot).

A validated 2×24GB FSDP2 path also exists — see [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the accelerate YAML and the four hard requirements.

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

Cross-machine serving (gRPC policy server), the safety checklist, per-key config, and RTC tuning: [`DEPLOYMENT.md`](./DEPLOYMENT.md).

## Adapting to a New Embodiment

Fine-tuning on a robot the checkpoint was not converted for requires only two new assets — a robot-config YAML and a norm-stats JSON — passed as `--policy.robot_config_path` / `--policy.norm_stats_path`. Explicit paths take precedence over the assets embedded in the checkpoint (a warning is logged when they differ), and checkpoints saved during fine-tuning embed the new assets so they remain self-contained.

A filled-in single-arm example (7-DoF, absolute joint angles, `front` + `wrist` cameras) and the canonical-slot semantics are in the full walkthrough: [`docs/source/lingbot_vla_v2.mdx`](../../../../../docs/source/lingbot_vla_v2.mdx).

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

