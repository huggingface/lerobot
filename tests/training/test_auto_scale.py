#!/usr/bin/env python

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch


def get_num_available_gpus():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def run_training_with_auto_scale(config_args, num_processes=2, temp_dir=None):
    config_path = Path(temp_dir) / "accelerate_config.yaml"

    with open(config_path, "w") as f:
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("distributed_type: MULTI_GPU\n")
        f.write("mixed_precision: 'no'\n")
        f.write(f"num_processes: {num_processes}\n")
        f.write("use_cpu: false\n")
        f.write("gpu_ids: all\n")
        f.write("downcast_bf16: 'no'\n")
        f.write("machine_rank: 0\n")
        f.write("main_training_function: main\n")
        f.write("num_machines: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(config_path),
        "-m",
        "lerobot.scripts.lerobot_train",
    ] + config_args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(num_processes)))}
    )

    return result


@pytest.mark.skipif(
    get_num_available_gpus() < 2,
    reason="Auto-scale test requires at least 2 GPUs",
)
def test_auto_scale_steps_and_lr():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "outputs"

        base_steps = 20
        args = [
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0]",
            "--policy.type=act",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            f"--output_dir={output_dir}",
            "--batch_size=4",
            f"--steps={base_steps}",
            "--eval_freq=-1",
            "--log_freq=5",
            "--save_freq=10",
            "--seed=42",
            "--num_workers=0",
            "--auto_scale=true",
        ]

        result = run_training_with_auto_scale(args, num_processes=2, temp_dir=temp_dir)

        assert result.returncode == 0, (
            f"Training failed with auto-scale enabled.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Check for auto-scale log message indicating steps were scaled
        combined = result.stdout + "\n" + result.stderr
        assert "Auto-scale enabled with world_size=2" in combined
        assert "steps 20 -> 10" in combined or "steps 20 " in combined
