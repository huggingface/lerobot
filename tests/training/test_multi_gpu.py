#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-GPU Training Tests

This module tests multi-GPU training functionality with accelerate.
These tests are designed to run on machines with 2+ GPUs and are executed
in the nightly CI workflow.

The tests automatically generate accelerate configs and launch training
with subprocess to properly test the distributed training environment.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_num_available_gpus():
    """Returns the number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def download_dataset(repo_id, episodes):
    """
    Pre-download dataset to avoid race conditions in multi-GPU training.

    Args:
        repo_id: HuggingFace dataset repository ID
        episodes: List of episode indices to download
    """
    # Simply instantiating the dataset will download it
    _ = LeRobotDataset(repo_id, episodes=episodes)
    print(f"Dataset {repo_id} downloaded successfully")


def run_accelerate_training(config_args, num_processes=4, temp_dir=None):
    """
    Helper function to run training with accelerate launch.

    Args:
        config_args: List of config arguments to pass to lerobot_train.py
        num_processes: Number of processes (GPUs) to use
        temp_dir: Temporary directory for outputs

    Returns:
        subprocess.CompletedProcess result
    """

    config_path = Path(temp_dir) / "accelerate_config.yaml"

    # Write YAML config
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
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(num_processes)))},
    )

    return result


@pytest.mark.skipif(
    get_num_available_gpus() < 2,
    reason="Multi-GPU tests require at least 2 GPUs",
)
class TestMultiGPUTraining:
    """Test suite for multi-GPU training functionality."""

    def test_basic_multi_gpu_training(self):
        """
        Test that basic multi-GPU training runs successfully.
        Verifies that the training completes without errors.
        """
        # Pre-download dataset to avoid race conditions
        download_dataset("lerobot/pusht", episodes=[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"

            config_args = [
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0]",
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--output_dir={output_dir}",
                "--batch_size=4",
                "--steps=10",
                "--eval_freq=-1",
                "--log_freq=5",
                "--save_freq=10",
                "--seed=42",
                "--num_workers=0",
            ]

            result = run_accelerate_training(config_args, num_processes=4, temp_dir=temp_dir)

            # Check that training completed successfully
            assert result.returncode == 0, (
                f"Multi-GPU training failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

            # Verify checkpoint was saved
            checkpoints_dir = output_dir / "checkpoints"
            assert checkpoints_dir.exists(), "Checkpoints directory was not created"

            # Verify that training completed
            assert "End of training" in result.stdout or "End of training" in result.stderr

    def test_checkpoint_saving_multi_gpu(self):
        """
        Test that checkpoints are correctly saved during multi-GPU training.
        Only the main process (rank 0) should save checkpoints.
        """
        # Pre-download dataset to avoid race conditions
        download_dataset("lerobot/pusht", episodes=[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"

            config_args = [
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0]",
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--output_dir={output_dir}",
                "--batch_size=4",
                "--steps=20",
                "--eval_freq=-1",
                "--log_freq=5",
                "--save_freq=10",
                "--seed=42",
                "--num_workers=0",
            ]

            result = run_accelerate_training(config_args, num_processes=2, temp_dir=temp_dir)

            assert result.returncode == 0, (
                f"Training failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

            # Verify checkpoint directory exists
            checkpoints_dir = output_dir / "checkpoints"
            assert checkpoints_dir.exists(), "Checkpoints directory not created"

            # Count checkpoint directories (should have checkpoint at step 10 and 20)
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
            assert len(checkpoint_dirs) >= 1, f"Expected at least 1 checkpoint, found {len(checkpoint_dirs)}"

            # Verify checkpoint contents
            for checkpoint_dir in checkpoint_dirs:
                # Check for model files
                model_files = list(checkpoint_dir.rglob("*.safetensors"))
                assert len(model_files) > 0, f"No model files in checkpoint {checkpoint_dir}"

                # Check for training state
                training_state_dir = checkpoint_dir / "training_state"
                assert training_state_dir.exists(), f"No training state in checkpoint {checkpoint_dir}"

                # Verify optimizer state exists
                optimizer_state = training_state_dir / "optimizer_state.safetensors"
                assert optimizer_state.exists(), f"No optimizer state in checkpoint {checkpoint_dir}"
