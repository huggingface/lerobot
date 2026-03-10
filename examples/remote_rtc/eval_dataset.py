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
Evaluate Real-Time Chunking (RTC) on dataset samples using remote inference.

This script evaluates RTC performance on dataset samples by communicating
with a remote RTC policy server. It compares action predictions with and
without RTC, measuring consistency and ground truth alignment.

The server runs the heavy policy inference on a powerful machine (e.g., with GPU),
while this client can run on a lightweight computer.

Usage:
    # First, start the server on a powerful machine:
    python examples/remote_rtc/rtc_policy_server.py \
        --host=0.0.0.0 \
        --port=8080

    # Then, run this evaluation script:
    python examples/remote_rtc/eval_dataset.py \
        --server_address=192.168.1.100:8080 \
        --policy_type=smolvla \
        --pretrained_name_or_path=helper2424/smolvla_check_rtc_last3 \
        --policy_device=cuda \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --seed=10

    # With Pi0.5 policy:
    python examples/remote_rtc/eval_dataset.py \
        --server_address=192.168.1.100:8080 \
        --policy_type=pi05 \
        --pretrained_name_or_path=lerobot/pi05_libero_finetuned \
        --policy_device=cuda \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --rtc.execution_horizon=10 \
        --seed=10
"""

import logging
import os
import pickle  # nosec
import random
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import draccus
import grpc
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.profiling import RTCProfiler, RTCProfilingRecord
from lerobot.policies.rtc.remote import RTCActionData, RTCObservationData, RTCRemotePolicyConfig
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class RTCEvalConfig:
    """Configuration for remote RTC dataset evaluation."""

    # Policy configuration (required fields first)
    policy_type: str = field(metadata={"help": "Type of policy (smolvla, pi0, pi05)"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Policy device
    policy_device: str = field(default="cuda", metadata={"help": "Device for policy inference on server"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address"})

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=20,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Evaluation parameters
    seed: int = field(default=42, metadata={"help": "Random seed"})
    inference_delay: int = field(default=4, metadata={"help": "Simulated inference delay"})
    output_dir: str = field(default="rtc_remote_eval_output", metadata={"help": "Output directory"})
    enable_profiling: bool = field(
        default=False,
        metadata={"help": "Collect per-request timing and save profiling artifacts"},
    )
    profiling_run_name: str = field(
        default="remote_rtc_dataset",
        metadata={"help": "Filename prefix for profiling artifacts"},
    )
    verbose_request_logging: bool = field(
        default=False,
        metadata={"help": "Enable per-request timing logs"},
    )
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Enable torch.compile on the server policy"},
    )
    torch_compile_mode: str = field(
        default="reduce-overhead",
        metadata={"help": "torch.compile mode (reduce-overhead, max-autotune, default)"},
    )

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")


class RTCEvaluator:
    """Evaluator for RTC on dataset samples using remote inference."""

    def __init__(self, cfg: RTCEvalConfig):
        self.cfg = cfg
        self.request_idx = 0
        self.sim_queue_size = 0
        self.sim_action_index = 0

        # Load dataset
        logger.info(f"Loading dataset: {cfg.dataset.repo_id}")

        # Get metadata for delta_timestamps calculation
        logger.debug("Getting dataset metadata...")
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)

        # Create a temporary policy config to resolve delta_timestamps
        logger.debug("Loading policy config...")
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_name_or_path)
        delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

        logger.debug("Creating LeRobotDataset...")
        self.dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            delta_timestamps=delta_timestamps,
        )
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")

        # Note: Preprocessing is done on server side, not client
        # Initialize gRPC connection
        logger.debug(f"Creating gRPC channel to {cfg.server_address}...")
        self.channel = grpc.insecure_channel(
            cfg.server_address,
            grpc_channel_options(initial_backoff="0.1s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Create lerobot features from dataset
        self.lerobot_features = {}
        self.profiler = RTCProfiler(cfg.enable_profiling, cfg.output_dir, cfg.profiling_run_name)

        logger.info(f"Ready to connect to server at {cfg.server_address}")

    def connect(self) -> bool:
        """Connect to server and send policy instructions."""
        try:
            logger.debug("Sending Ready signal to server...")
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            logger.info(f"Connected to server in {time.perf_counter() - start_time:.4f}s")

            # Send policy configuration
            logger.debug("Sending policy instructions...")
            policy_config = RTCRemotePolicyConfig(
                policy_type=self.cfg.policy_type,
                pretrained_name_or_path=self.cfg.pretrained_name_or_path,
                lerobot_features=self.lerobot_features,
                rtc_config=self.cfg.rtc,
                device=self.cfg.policy_device,
                use_torch_compile=self.cfg.use_torch_compile,
                torch_compile_mode=self.cfg.torch_compile_mode,
            )

            policy_config_bytes = pickle.dumps(policy_config)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_config_bytes))

            logger.info(f"Policy instructions sent | Type: {self.cfg.policy_type}")
            return True

        except grpc.RpcError as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def _request_actions(
        self,
        observation: dict[str, Any],
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
        execution_horizon: int,
        label: str,
    ) -> RTCActionData:
        """Send observation and get actions from remote server."""
        logger.debug(f"Preparing observation (delay={inference_delay}, horizon={execution_horizon})...")

        t_start = time.perf_counter()
        queue_size_before = self.sim_queue_size
        action_index_before = self.sim_action_index

        rtc_obs = RTCObservationData(
            observation=observation,
            timestamp=time.time(),
            timestep=action_index_before,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

        obs_bytes = pickle.dumps(rtc_obs)
        t_pickle = time.perf_counter()
        pickle_ms = (t_pickle - t_start) * 1000

        logger.debug(f"Sending observation ({len(obs_bytes)} bytes, pickle: {pickle_ms:.1f}ms)...")
        obs_iterator = send_bytes_in_chunks(
            obs_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(obs_iterator)
        t_send = time.perf_counter()
        send_ms = (t_send - t_pickle) * 1000

        # Get actions
        logger.debug("Waiting for actions from server...")
        actions_response = self.stub.GetActions(services_pb2.Empty())
        t_response = time.perf_counter()
        roundtrip_ms = (t_response - t_send) * 1000

        if len(actions_response.data) == 0:
            raise RuntimeError("Empty response from server")

        action_data = pickle.loads(actions_response.data)  # nosec
        t_unpickle = time.perf_counter()
        unpickle_ms = (t_unpickle - t_response) * 1000

        total_ms = (t_unpickle - t_start) * 1000
        chunk_size = int(action_data.actions.shape[0])
        realized_delay = max(int(inference_delay), 0)
        queue_size_after = max(chunk_size - realized_delay, 0)
        self.sim_queue_size = queue_size_after
        self.sim_action_index = 0
        server_timing = getattr(action_data, "timing", None)

        self.profiler.add(
            RTCProfilingRecord(
                request_idx=self.request_idx,
                timestamp=time.time(),
                label=label,
                payload_bytes=len(obs_bytes),
                queue_size_before=queue_size_before,
                queue_size_after=queue_size_after,
                action_index_before=action_index_before,
                inference_delay_requested=inference_delay,
                realized_delay=realized_delay,
                client_pickle_ms=pickle_ms,
                client_send_ms=send_ms,
                client_get_actions_ms=roundtrip_ms,
                client_unpickle_ms=unpickle_ms,
                client_total_ms=total_ms,
                server_queue_wait_ms=(
                    server_timing.queue_wait_ms if server_timing is not None else None
                ),
                server_preprocess_ms=(
                    server_timing.preprocess_ms if server_timing is not None else None
                ),
                server_inference_ms=(
                    server_timing.inference_ms if server_timing is not None else None
                ),
                server_postprocess_ms=(
                    server_timing.postprocess_ms if server_timing is not None else None
                ),
                server_pickle_ms=server_timing.pickle_ms if server_timing is not None else None,
                server_total_ms=server_timing.total_ms if server_timing is not None else None,
            )
        )
        self.request_idx += 1

        if self.cfg.verbose_request_logging:
            logger.info(
                f"Actions received | "
                f"pickle: {pickle_ms:.1f}ms | "
                f"send: {send_ms:.1f}ms | "
                f"roundtrip: {roundtrip_ms:.1f}ms | "
                f"unpickle: {unpickle_ms:.1f}ms | "
                f"total: {total_ms:.1f}ms"
            )
        return action_data

    def run_evaluation(self):
        """Run evaluation comparing RTC and non-RTC on dataset samples."""
        logger.info("Starting evaluation...")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.cfg.output_dir}")

        if not self.connect():
            logger.error("Failed to connect to server")
            return

        logger.info("=" * 60)
        logger.info("Starting RTC evaluation on dataset samples")
        logger.info(f"Inference delay: {self.cfg.inference_delay}")
        logger.info(f"Execution horizon: {self.cfg.rtc.execution_horizon}")
        logger.info("=" * 60)

        # Load two random samples (send raw to server for preprocessing)
        logger.debug("Loading samples from dataset...")
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        loader_iter = iter(data_loader)
        first_sample = next(loader_iter)
        second_sample = next(loader_iter)
        logger.debug("Samples loaded (sending raw to server)")

        # Step 1: Generate previous chunk (without RTC)
        logger.info("=" * 60)
        logger.info("Step 1: Generating previous chunk (baseline)")
        logger.info("=" * 60)

        set_seed(self.cfg.seed)

        prev_chunk_response = self._request_actions(
            observation=first_sample,  # Send raw sample
            inference_delay=0,
            prev_chunk_left_over=None,
            execution_horizon=0,
            label="prev_chunk_baseline",
        )
        prev_chunk_left_over = prev_chunk_response.original_actions[:25]
        logger.info(f"Previous chunk shape: {prev_chunk_left_over.shape}")

        # Step 2: Generate actions WITHOUT RTC
        logger.info("=" * 60)
        logger.info("Step 2: Generating actions WITHOUT RTC")
        logger.info("=" * 60)

        set_seed(self.cfg.seed)

        no_rtc_response = self._request_actions(
            observation=second_sample,  # Send raw sample
            inference_delay=0,
            prev_chunk_left_over=None,
            execution_horizon=0,
            label="no_rtc",
        )
        no_rtc_actions = no_rtc_response.original_actions
        logger.info(f"No-RTC actions shape: {no_rtc_actions.shape}")

        # Step 3: Generate actions WITH RTC
        logger.info("=" * 60)
        logger.info("Step 3: Generating actions WITH RTC")
        logger.info("=" * 60)

        set_seed(self.cfg.seed)

        rtc_response = self._request_actions(
            observation=second_sample,  # Send raw sample
            inference_delay=self.cfg.inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=self.cfg.rtc.execution_horizon,
            label="rtc",
        )
        rtc_actions = rtc_response.original_actions
        logger.info(f"RTC actions shape: {rtc_actions.shape}")

        # Plot comparison
        logger.info("=" * 80)
        logger.info("Plotting results...")
        self._plot_comparison(rtc_actions, no_rtc_actions, prev_chunk_left_over)

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully")

        profiling_artifacts = self.profiler.finalize()
        if profiling_artifacts:
            logger.info("Saved profiling artifacts:")
            for name, path in profiling_artifacts.items():
                logger.info(f"  - {name}: {path}")

        # Cleanup
        self.channel.close()

    def _plot_comparison(
        self,
        rtc_actions: torch.Tensor,
        no_rtc_actions: torch.Tensor,
        prev_chunk: torch.Tensor,
    ):
        """Plot comparison of RTC vs non-RTC actions."""
        rtc_plot = rtc_actions.cpu().numpy()
        no_rtc_plot = no_rtc_actions.cpu().numpy()
        prev_chunk_plot = prev_chunk.cpu().numpy()

        num_dims = min(rtc_plot.shape[-1], 6)

        fig, axes = plt.subplots(num_dims, 1, figsize=(16, 12))
        fig.suptitle("Remote RTC Evaluation: Action Comparison", fontsize=16)

        for dim_idx in range(num_dims):
            ax = axes[dim_idx] if num_dims > 1 else axes

            # Plot previous chunk (ground truth)
            ax.plot(
                range(len(prev_chunk_plot)),
                prev_chunk_plot[:, dim_idx],
                color="red",
                linewidth=2.5,
                alpha=0.8,
                label="Previous Chunk (Ground Truth)" if dim_idx == 0 else None,
            )

            # Plot no-RTC actions
            ax.plot(
                range(len(no_rtc_plot)),
                no_rtc_plot[:, dim_idx],
                color="blue",
                linewidth=2,
                alpha=0.7,
                label="No RTC" if dim_idx == 0 else None,
            )

            # Plot RTC actions
            ax.plot(
                range(len(rtc_plot)),
                rtc_plot[:, dim_idx],
                color="green",
                linewidth=2,
                alpha=0.7,
                label="RTC" if dim_idx == 0 else None,
            )

            # Add vertical lines for inference delay and execution horizon
            if self.cfg.inference_delay > 0:
                ax.axvline(
                    x=self.cfg.inference_delay - 1,
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Inference Delay ({self.cfg.inference_delay})" if dim_idx == 0 else None,
                )

            if self.cfg.rtc.execution_horizon > 0:
                ax.axvline(
                    x=self.cfg.rtc.execution_horizon,
                    color="purple",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Execution Horizon ({self.cfg.rtc.execution_horizon})" if dim_idx == 0 else None,
                )

            ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step", fontsize=10) if num_dims > 1 else axes.set_xlabel("Step", fontsize=10)

        # Add legend
        handles, labels = (axes[0] if num_dims > 1 else axes).get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center right",
            fontsize=9,
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0.9,
        )

        output_path = os.path.join(self.cfg.output_dir, "remote_rtc_comparison.png")
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_path}")
        plt.close(fig)


@draccus.wrap()
def main(cfg: RTCEvalConfig):
    """Main entry point for remote RTC dataset evaluation."""
    set_seed(cfg.seed)

    logger.info("Configuration:\n%s", pformat(asdict(cfg)))
    logger.info("=" * 80)
    logger.info("Remote RTC Dataset Evaluation")
    logger.info("=" * 80)

    evaluator = RTCEvaluator(cfg)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
