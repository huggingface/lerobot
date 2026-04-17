# LeRobot LIBERO Training Benchmark

Train and evaluate all LeRobot policies on [LIBERO](https://libero-project.github.io/) and publish results as a HuggingFace leaderboard dataset.

## Policies

| Policy         | Base Model           | GPUs | LR     | Chunk | Notes                                 |
| -------------- | -------------------- | ---- | ------ | ----- | ------------------------------------- |
| pi0            | lerobot/pi0_base     | 8    | 2.5e-5 | 30    | PaliGemma + Gemma flow matching       |
| pi0_fast       | lerobot/pi0fast-base | 8    | 2.5e-5 | 30    | Requires tokenizer pre-training       |
| pi05           | lerobot/pi05_base    | 8    | 2.5e-5 | 30    | Quantiles normalization               |
| groot          | nvidia/GR00T-N1.5-3B | 8    | 1e-4   | 30    | bf16, diffusion head + projector only |
| act            | From scratch         | 1    | 1e-5   | 30    | ResNet-18, lightweight                |
| diffusion      | From scratch         | 1    | 1e-4   | 32\*  | U-Net, horizon must be divisible by 8 |
| smolvla        | lerobot/smolvla_base | 8    | 1e-4   | 30    | SmolVLM2-500M                         |
| xvla           | lerobot/xvla-widowx  | 4    | 1e-4   | 32\*  | Florence2 + CLIP                      |
| multi_task_dit | From scratch         | 1    | 2e-5   | 32\*  | CLIP + DiT                            |

\* These policies use `horizon` rather than `chunk_size`. Set to 32 (nearest valid value to 30).

## Training spec

- **Steps**: 5,000 per policy
- **Batch size**: 32 per GPU (effective BS = 256 for multi-GPU)
- **Dataset**: `lerobot/libero` (libero_spatial)
- **Evaluation**: 20 episodes after training
- **LR**: each policy's default optimizer/scheduler preset
- **Results**: each SLURM job publishes its own row to the HF leaderboard dataset automatically

## Quick start

### 1. Generate SLURM scripts

```bash
python benchmarks/libero/run_benchmark.py \
    --output_dir /scratch/lerobot-benchmark \
    --hub_org lerobot
```

### 2. Submit jobs

```bash
# If using pi0_fast, submit tokenizer first:
sbatch /scratch/lerobot-benchmark/slurm_scripts/00_tokenizer.sh
# Wait, then submit pi0_fast

# All other policies can run in parallel:
for script in /scratch/lerobot-benchmark/slurm_scripts/[0-9][0-9]_*.sh; do
    [[ "$script" == *pi0_fast* ]] && continue
    sbatch "$script"
done
```

Each job publishes its result to `lerobot/benchmark-libero` on the Hub when it finishes.

## Prerequisites

- SLURM cluster with CUDA GPUs (A100 80GB recommended for VLM policies)
- `pip install lerobot[pi,smolvla,groot,xvla,multi_task_dit,libero] datasets`
- `huggingface-cli login`
