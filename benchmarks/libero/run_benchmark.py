#!/usr/bin/env python
"""Generate SLURM sbatch scripts for training all LeRobot policies on LIBERO.

Each generated script trains one policy, evaluates it, and publishes its
results row to a HuggingFace leaderboard dataset — no separate collection
step needed.

Usage:
    # Generate scripts for all policies:
    python benchmarks/libero/run_benchmark.py \\
        --output_dir /scratch/lerobot-benchmark --hub_org lerobot

    # Generate for a subset:
    python benchmarks/libero/run_benchmark.py \\
        --policies pi0 smolvla act \\
        --output_dir /scratch/lerobot-benchmark --hub_org lerobot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Policy benchmark configs
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PolicyBenchmarkConfig:
    """Training configuration for a single policy on a benchmark."""

    policy_type: str
    policy_path: str | None = None
    num_gpus: int = 1
    chunk_size: int | None = None  # Set on policies that use chunk_size (not horizon)
    extra_policy_args: dict[str, str] = field(default_factory=dict)
    needs_tokenizer: bool = False
    tokenizer_args: dict[str, str] = field(default_factory=dict)


COMMON_TRAINING_ARGS: dict[str, str] = {
    "dataset.repo_id": "lerobot/libero",
    "dataset.use_imagenet_stats": "false",
    "env.type": "libero",
    "env.task": "libero_spatial",
    "steps": "5000",
    "batch_size": "32",
    "eval_freq": "0",
    "save_freq": "5000",
    "save_checkpoint": "true",
    "log_freq": "100",
    "wandb.enable": "true",
    "policy.push_to_hub": "true",
    "rename_map": (
        '{"observation.images.image":"observation.images.camera1",'
        '"observation.images.image2":"observation.images.camera2"}'
    ),
}

EVAL_ARGS: dict[str, str] = {
    "env.type": "libero",
    "env.task": "libero_spatial",
    "eval.n_episodes": "20",
    "eval.batch_size": "10",
}

POLICY_CONFIGS: dict[str, PolicyBenchmarkConfig] = {
    "pi0": PolicyBenchmarkConfig(
        policy_type="pi0",
        policy_path="lerobot/pi0_base",
        num_gpus=8,
        chunk_size=30,
        extra_policy_args={
            "policy.n_action_steps": "30",
            "policy.scheduler_decay_steps": "5000",
        },
    ),
    "pi0_fast": PolicyBenchmarkConfig(
        policy_type="pi0_fast",
        policy_path="lerobot/pi0fast-base",
        num_gpus=8,
        chunk_size=30,
        extra_policy_args={
            "policy.n_action_steps": "30",
            "policy.scheduler_decay_steps": "5000",
        },
        needs_tokenizer=True,
        tokenizer_args={
            "repo_id": "lerobot/libero",
            "action_horizon": "30",
            "encoded_dims": "0:7",
            "normalization_mode": "QUANTILES",
            "vocab_size": "1024",
            "scale": "10.0",
            "push_to_hub": "true",
        },
    ),
    "pi05": PolicyBenchmarkConfig(
        policy_type="pi05",
        policy_path="lerobot/pi05_base",
        num_gpus=8,
        chunk_size=30,
        extra_policy_args={
            "policy.n_action_steps": "30",
            "policy.scheduler_decay_steps": "5000",
        },
    ),
    "groot": PolicyBenchmarkConfig(
        policy_type="groot",
        policy_path=None,
        num_gpus=8,
        chunk_size=30,
        extra_policy_args={
            "policy.n_action_steps": "30",
            "policy.base_model_path": "nvidia/GR00T-N1.5-3B",
            "policy.tune_diffusion_model": "true",
            "policy.tune_projector": "true",
            "policy.tune_llm": "false",
            "policy.tune_visual": "false",
            "policy.use_bf16": "true",
        },
    ),
    "act": PolicyBenchmarkConfig(
        policy_type="act",
        policy_path=None,
        num_gpus=1,
        chunk_size=30,
        extra_policy_args={"policy.n_action_steps": "30"},
    ),
    "diffusion": PolicyBenchmarkConfig(
        policy_type="diffusion",
        policy_path=None,
        num_gpus=1,
        chunk_size=None,
        extra_policy_args={
            "policy.horizon": "32",
            "policy.n_action_steps": "30",
            "policy.n_obs_steps": "2",
        },
    ),
    "smolvla": PolicyBenchmarkConfig(
        policy_type="smolvla",
        policy_path="lerobot/smolvla_base",
        num_gpus=8,
        chunk_size=30,
        extra_policy_args={
            "policy.n_action_steps": "30",
            "policy.load_vlm_weights": "true",
            "policy.freeze_vision_encoder": "false",
            "policy.train_expert_only": "false",
            "policy.scheduler_decay_steps": "5000",
        },
    ),
    "xvla": PolicyBenchmarkConfig(
        policy_type="xvla",
        policy_path="lerobot/xvla-widowx",
        num_gpus=4,
        chunk_size=32,
        extra_policy_args={
            "policy.n_action_steps": "32",
            "policy.scheduler_decay_steps": "5000",
        },
    ),
    "multi_task_dit": PolicyBenchmarkConfig(
        policy_type="multi_task_dit",
        policy_path=None,
        num_gpus=1,
        chunk_size=None,
        extra_policy_args={
            "policy.horizon": "32",
            "policy.n_action_steps": "30",
        },
    ),
}

ALL_POLICY_NAMES = list(POLICY_CONFIGS.keys())

# GPU memory estimates (GB) for SLURM --mem allocation
GPU_MEM_ESTIMATES: dict[str, int] = {
    "pi0": 320,
    "pi0_fast": 320,
    "pi05": 280,
    "groot": 320,
    "act": 64,
    "diffusion": 64,
    "smolvla": 160,
    "xvla": 160,
    "multi_task_dit": 64,
}


# ──────────────────────────────────────────────────────────────────────
# SLURM script generation
# ──────────────────────────────────────────────────────────────────────


def _cli_args(args: dict[str, str]) -> str:
    """Build a backslash-continued CLI arg string with proper shell quoting."""
    lines = []
    for key, value in args.items():
        if any(c in str(value) for c in ["{", "}", " ", '"', "'"]):
            lines.append(f"    --{key}='{value}'")
        else:
            lines.append(f"    --{key}={value}")
    return " \\\n".join(lines)


def _training_cli_args(
    policy_name: str,
    output_dir: Path,
    hub_org: str,
    benchmark_uuid: str,
) -> str:
    cfg = POLICY_CONFIGS[policy_name]
    args: dict[str, str] = {}
    args.update(COMMON_TRAINING_ARGS)
    args["policy.type"] = cfg.policy_type
    if cfg.policy_path:
        args["policy.path"] = cfg.policy_path
    if cfg.chunk_size is not None:
        args["policy.chunk_size"] = str(cfg.chunk_size)
    args.update(cfg.extra_policy_args)
    args["output_dir"] = str(output_dir / "train" / policy_name)
    args["policy.repo_id"] = f"{hub_org}/{policy_name}_libero"
    args["wandb.project"] = "lerobot-libero-benchmark"
    args["wandb.run_name"] = f"{policy_name}_{benchmark_uuid[:8]}"
    return _cli_args(args)


def _publish_snippet(
    policy_name: str,
    output_dir: Path,
    hub_org: str,
    benchmark_uuid: str,
    hub_dataset: str,
) -> str:
    """Inline Python that each SLURM job runs to publish its own result row."""
    cfg = POLICY_CONFIGS[policy_name]
    steps = int(COMMON_TRAINING_ARGS["steps"])
    bs = int(COMMON_TRAINING_ARGS["batch_size"])
    eff_bs = bs * cfg.num_gpus
    train_dir = output_dir / "train" / policy_name

    return textwrap.dedent(f"""\
        python3 -c "
        import json, os, re, sys
        from pathlib import Path
        from datetime import datetime, timezone

        timing = {{}}
        tp = Path('{output_dir}/logs/{policy_name}_timing.txt')
        if tp.exists():
            for ln in tp.read_text().splitlines():
                if '=' in ln:
                    k, _, v = ln.partition('=')
                    timing[k.strip()] = v.strip()

        # Parse eval results
        eval_sr, eval_per_task, eval_n = None, '{{}}', 0
        eval_dir = Path('{train_dir}/eval_results')
        if eval_dir.exists():
            for jf in eval_dir.glob('**/*.json'):
                try:
                    d = json.loads(jf.read_text())
                except Exception:
                    continue
                if 'avg_success_rate' in d:
                    eval_sr = d['avg_success_rate']
                elif 'eval_info' in d and 'avg_success_rate' in d.get('eval_info', {{}}):
                    eval_sr = d['eval_info']['avg_success_rate']
                pt = {{k: v for k, v in d.items() if 'success_rate' in k and k != 'avg_success_rate'}}
                if pt:
                    eval_per_task = json.dumps(pt)
                if 'n_episodes' in d:
                    eval_n = d['n_episodes']

        # Parse final loss from SLURM stdout
        final_loss = None
        for lf in sorted(Path('{output_dir}/logs').glob('{policy_name}_*.out'), reverse=True):
            losses = re.findall(r'\\\"loss\\\"\\s*:\\s*([\\d.e+-]+)', lf.read_text())
            if losses:
                final_loss = float(losses[-1])
                break

        # Parse peak GPU mem
        peak_mem = 0.0
        csv_p = Path('{output_dir}/logs/{policy_name}_gpu_mem.csv')
        if csv_p.exists():
            for ln in csv_p.read_text().splitlines():
                parts = ln.strip().split(',')
                if len(parts) >= 2:
                    try:
                        peak_mem = max(peak_mem, float(parts[1].strip()))
                    except ValueError:
                        pass

        # Parse train config for optimizer details
        lr, opt_wd, sched_type, sched_warmup, sched_decay = 0.0, 0.0, '', 0, 0
        freeze_ve, train_eo, grad_ckpt = False, False, False
        cfg_path = Path('{train_dir}/checkpoints/{steps:06d}/pretrained_model/train_config.json')
        if cfg_path.exists():
            tc = json.loads(cfg_path.read_text())
            o = tc.get('optimizer', {{}})
            lr = o.get('lr', 0.0)
            opt_wd = o.get('weight_decay', 0.0)
            s = tc.get('scheduler', {{}})
            sched_type = s.get('type', '')
            sched_warmup = s.get('num_warmup_steps', 0)
            sched_decay = s.get('num_decay_steps', 0)
            p = tc.get('policy', {{}})
            freeze_ve = p.get('freeze_vision_encoder', False)
            train_eo = p.get('train_expert_only', False)
            grad_ckpt = p.get('gradient_checkpointing', False)

        row = {{
            'benchmark_uuid': '{benchmark_uuid}',
            'policy_type': '{policy_name}',
            'policy_repo_id': '{hub_org}/{policy_name}_libero',
            'base_model_repo_id': '{cfg.policy_path or ""}',
            'dataset_repo_id': '{COMMON_TRAINING_ARGS["dataset.repo_id"]}',
            'env_type': '{COMMON_TRAINING_ARGS["env.type"]}',
            'env_task': '{COMMON_TRAINING_ARGS["env.task"]}',
            'steps': {steps},
            'batch_size_per_gpu': {bs},
            'num_gpus': {cfg.num_gpus},
            'effective_batch_size': {eff_bs},
            'total_samples_seen': {steps * eff_bs},
            'chunk_size': {cfg.chunk_size or 0},
            'learning_rate': lr,
            'optimizer_type': 'AdamW',
            'optimizer_weight_decay': opt_wd,
            'scheduler_type': sched_type,
            'scheduler_warmup_steps': sched_warmup,
            'scheduler_decay_steps': sched_decay,
            'freeze_vision_encoder': freeze_ve,
            'train_expert_only': train_eo,
            'gradient_checkpointing': grad_ckpt,
            'eval_success_rate': eval_sr,
            'eval_success_rate_per_task': eval_per_task,
            'eval_n_episodes': eval_n,
            'final_train_loss': final_loss,
            'training_time_s': float(timing.get('TRAINING_TIME_S', 0)),
            'peak_gpu_memory_mb': peak_mem or float(timing.get('MAX_GPU_MEM_MB', 0)),
            'gpu_type': timing.get('GPU_TYPE', 'unknown'),
            'lerobot_commit': timing.get('LEROBOT_COMMIT', 'unknown'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }}

        # Save locally
        Path('{train_dir}/benchmark_result.json').write_text(json.dumps(row, indent=2, default=str))

        # Push to HF dataset
        try:
            from datasets import Dataset, load_dataset
            try:
                existing = load_dataset('{hub_dataset}', split='train')
                rows = existing.to_list() + [row]
            except Exception:
                rows = [row]
            Dataset.from_list(rows).push_to_hub('{hub_dataset}', split='train')
            print('Published result to {hub_dataset}')
        except ImportError:
            print('datasets library not installed — result saved locally only')
        except Exception as e:
            print(f'Failed to push to hub: {{e}} — result saved locally')
        "
    """)


def _generate_sbatch_script(
    policy_name: str,
    output_dir: Path,
    hub_org: str,
    benchmark_uuid: str,
    hub_dataset: str,
    lerobot_commit: str,
) -> str:
    cfg = POLICY_CONFIGS[policy_name]
    steps = int(COMMON_TRAINING_ARGS["steps"])
    log_dir = output_dir / "logs"
    train_dir = output_dir / "train" / policy_name
    checkpoint_path = train_dir / f"checkpoints/{steps:06d}/pretrained_model"

    training_args = _training_cli_args(policy_name, output_dir, hub_org, benchmark_uuid)
    eval_args = _cli_args(EVAL_ARGS)
    publish = _publish_snippet(policy_name, output_dir, hub_org, benchmark_uuid, hub_dataset)

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=bench_{policy_name}
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --gres=gpu:{cfg.num_gpus}
        #SBATCH --cpus-per-task={cfg.num_gpus * 8}
        #SBATCH --mem={GPU_MEM_ESTIMATES.get(policy_name, 128)}G
        #SBATCH --time=06:00:00
        #SBATCH --output={log_dir}/{policy_name}_%j.out
        #SBATCH --error={log_dir}/{policy_name}_%j.err

        set -euo pipefail

        echo "=========================================="
        echo "LeRobot LIBERO Benchmark — {policy_name}"
        echo "UUID: {benchmark_uuid}"
        echo "Start: $(date -Iseconds)"
        echo "Host: $(hostname) | GPUs: {cfg.num_gpus}"
        echo "=========================================="

        START_TIME=$(date +%s)

        # GPU memory monitoring (every 30s)
        nvidia-smi --query-gpu=index,memory.used,memory.total,gpu_name \\
            --format=csv,noheader,nounits -l 30 \\
            > "{log_dir}/{policy_name}_gpu_mem.csv" &
        GPU_MONITOR_PID=$!

        # ── Training ──────────────────────────────────────────────────
        echo "[$(date -Iseconds)] Starting training..."
        accelerate launch --num_processes={cfg.num_gpus} \\
            $(which lerobot-train) \\
        {training_args}
        TRAIN_EXIT=$?
        TRAIN_END=$(date +%s)
        echo "[$(date -Iseconds)] Training exit code: $TRAIN_EXIT"

        # ── Evaluation ────────────────────────────────────────────────
        EVAL_EXIT=1
        if [ $TRAIN_EXIT -eq 0 ]; then
            echo "[$(date -Iseconds)] Starting evaluation..."
            lerobot-eval \\
                --policy.path="{checkpoint_path}" \\
            {eval_args} \\
                --output_dir="{train_dir}/eval_results"
            EVAL_EXIT=$?
            echo "[$(date -Iseconds)] Eval exit code: $EVAL_EXIT"
        else
            echo "[$(date -Iseconds)] Skipping eval — training failed."
        fi

        # ── Timing ────────────────────────────────────────────────────
        END_TIME=$(date +%s)
        kill $GPU_MONITOR_PID 2>/dev/null || true

        cat > "{log_dir}/{policy_name}_timing.txt" <<TIMING_EOF
        BENCHMARK_UUID={benchmark_uuid}
        POLICY_TYPE={policy_name}
        TRAINING_TIME_S=$((TRAIN_END - START_TIME))
        TOTAL_TIME_S=$((END_TIME - START_TIME))
        TRAIN_EXIT=$TRAIN_EXIT
        EVAL_EXIT=$EVAL_EXIT
        MAX_GPU_MEM_MB=$(awk -F',' '{{print $2}}' "{log_dir}/{policy_name}_gpu_mem.csv" 2>/dev/null | sort -n | tail -1)
        GPU_TYPE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1 | xargs)
        LEROBOT_COMMIT={lerobot_commit}
        TIMING_EOF

        # ── Publish result to HF dataset ──────────────────────────────
        echo "[$(date -Iseconds)] Publishing result..."
        {publish}

        echo "=========================================="
        echo "Done: $(date -Iseconds)"
        echo "Training: $((TRAIN_END - START_TIME))s | Total: $((END_TIME - START_TIME))s"
        echo "=========================================="
    """)


def _generate_tokenizer_script(
    output_dir: Path,
    hub_org: str,
    benchmark_uuid: str,
) -> str:
    cfg = POLICY_CONFIGS["pi0_fast"]
    log_dir = output_dir / "logs"
    tokenizer_hub_repo = f"{hub_org}/fast-tokenizer-libero"

    tok_args = dict(cfg.tokenizer_args)
    tok_args["hub_repo_id"] = tokenizer_hub_repo

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=bench_tokenizer
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=8
        #SBATCH --mem=64G
        #SBATCH --time=01:00:00
        #SBATCH --output={log_dir}/tokenizer_%j.out
        #SBATCH --error={log_dir}/tokenizer_%j.err

        set -euo pipefail
        echo "LeRobot — FAST Tokenizer | UUID: {benchmark_uuid}"

        lerobot-train-tokenizer \\
        {_cli_args(tok_args)}

        echo "Tokenizer pushed to: {tokenizer_hub_repo}"
    """)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for LeRobot LIBERO benchmark.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=ALL_POLICY_NAMES,
        choices=ALL_POLICY_NAMES,
        help="Policies to benchmark (default: all).",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Root output directory.")
    parser.add_argument("--hub_org", type=str, default="lerobot", help="HuggingFace org.")
    parser.add_argument("--hub_dataset", type=str, default=None, help="HF dataset repo for results.")
    parser.add_argument("--uuid", type=str, default=None, help="Override benchmark UUID.")
    args = parser.parse_args()

    benchmark_uuid = args.uuid or str(uuid.uuid4())
    output_dir: Path = args.output_dir.resolve()
    policies: list[str] = args.policies
    hub_org: str = args.hub_org
    hub_dataset: str = args.hub_dataset or f"{hub_org}/benchmark-libero"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    scripts_dir = output_dir / "slurm_scripts"
    log_dir = output_dir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for p in policies:
        (output_dir / "train" / p).mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    # Tokenizer job for pi0_fast
    tokenizer_path = None
    if "pi0_fast" in policies:
        script = _generate_tokenizer_script(output_dir, hub_org, benchmark_uuid)
        tokenizer_path = scripts_dir / "00_tokenizer.sh"
        tokenizer_path.write_text(script)
        tokenizer_path.chmod(0o755)
        generated["tokenizer"] = tokenizer_path
        tokenizer_hub_repo = f"{hub_org}/fast-tokenizer-libero"
        POLICY_CONFIGS["pi0_fast"].extra_policy_args["policy.action_tokenizer_name"] = tokenizer_hub_repo

    # Per-policy scripts
    for i, name in enumerate(sorted(policies), start=1):
        script = _generate_sbatch_script(name, output_dir, hub_org, benchmark_uuid, hub_dataset, commit)
        path = scripts_dir / f"{i:02d}_{name}.sh"
        path.write_text(script)
        path.chmod(0o755)
        generated[name] = path

    # Manifest
    manifest = {
        "benchmark_uuid": benchmark_uuid,
        "timestamp": datetime.now(UTC).isoformat(),
        "lerobot_commit": commit,
        "hub_org": hub_org,
        "hub_dataset": hub_dataset,
        "policies": policies,
        "output_dir": str(output_dir),
        "scripts": {k: str(v) for k, v in generated.items()},
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Instructions
    print("=" * 60)
    print("LeRobot LIBERO Benchmark — Scripts Generated")
    print(f"UUID: {benchmark_uuid}")
    print(f"Output: {output_dir}")
    print(f"Results dataset: {hub_dataset}")
    print("=" * 60)
    print()
    for _name, path in sorted(generated.items()):
        print(f"  {path}")
    print()

    if tokenizer_path:
        print("IMPORTANT: pi0_fast requires tokenizer training FIRST.")
        print(f"  1. sbatch {tokenizer_path}")
        print("  2. Wait for completion")
        print(f"  3. sbatch {generated.get('pi0_fast', 'N/A')}")
        print("  4. All other policies can run in parallel")
    else:
        print("All scripts can be submitted in parallel.")
    print()
    print("Each job publishes its result to the HF dataset automatically.")


if __name__ == "__main__":
    main()
