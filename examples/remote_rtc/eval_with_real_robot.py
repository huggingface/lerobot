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
Evaluate Real-Time Chunking (RTC) with a real robot using remote inference.

This script controls a robot and communicates with a remote RTC policy server,
managing action queues with RTC-specific merging for smooth action execution.

The server runs the heavy policy inference on a powerful machine (e.g., with GPU),
while this client runs on a lightweight computer connected to the robot.

Usage:
    # First, start the server on a powerful machine:
    python examples/remote_rtc/rtc_policy_server.py \
        --host=0.0.0.0 \
        --port=8080

    # Then, run this client on the robot's computer:

    # Run with SO100 robot and SmolVLA policy
    python examples/remote_rtc/eval_with_real_robot.py \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --server_address=192.168.1.100:8080 \
        --policy_type=smolvla \
        --pretrained_name_or_path=helper2424/smolvla_check_rtc_last3 \
        --policy_device=cuda \
        --task="Move the object" \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --duration=120

    # Run with Pi0.5 policy
    python examples/remote_rtc/eval_with_real_robot.py \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --server_address=192.168.1.100:8080 \
        --policy_type=pi05 \
        --pretrained_name_or_path=lerobot/pi05_libero_finetuned \
        --policy_device=cuda \
        --task="Pick up the cube" \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --duration=120
"""

import logging
import math
import pickle  # nosec
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.profiling import RTCProfiler, RTCProfilingRecord
from lerobot.policies.rtc.remote import RTCActionData, RTCObservationData, RTCRemotePolicyConfig
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RobotClientConfig:
    """Configuration for RTC Robot Client."""

    # Robot configuration
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy (smolvla, pi0, pi05)"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    policy_device: str = field(default="cuda", metadata={"help": "Device for policy inference on server"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address"})

    # Task configuration
    task: str = field(default="", metadata={"help": "Task instruction"})

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=20,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Control configuration
    fps: float = field(default=10.0, metadata={"help": "Action execution frequency (Hz)"})
    duration: float = field(default=60.0, metadata={"help": "Duration to run (seconds)"})

    # Action queue threshold - when queue size drops below this, request new actions
    action_queue_threshold: int = field(
        default=30,
        metadata={"help": "Request new actions when queue size drops below this value"},
    )
    enable_profiling: bool = field(
        default=False,
        metadata={"help": "Collect per-request timings and queue metrics"},
    )
    profiling_output_dir: str = field(
        default="rtc_remote_profile_output",
        metadata={"help": "Directory for profiling artifacts"},
    )
    profiling_run_name: str = field(
        default="remote_rtc_robot",
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
    compile_warmup_delay: list[int] = field(
        default_factory=lambda: [0, 4],
        metadata={"help": "Warmup inference delays per call, e.g. [0,4,5,6]. Empty list disables warmup."},
    )

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if any(delay < 0 for delay in self.compile_warmup_delay):
            raise ValueError("All compile_warmup_delay values must be >= 0")

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps


class RobotWrapper:
    """Thread-safe wrapper for robot access."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = threading.Lock()

    def get_observation(self) -> dict[str, Any]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Any):
        with self.lock:
            return self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


class RobotClient:
    """Robot client with RTC action queue management."""

    def __init__(self, config: RobotClientConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.request_idx = 0

        # Initialize robot
        logger.info(f"Initializing robot: {config.robot.type}")
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        self.robot_wrapper = RobotWrapper(self.robot)

        # Create lerobot features mapping
        self.lerobot_features = hw_to_dataset_features(
            self.robot.observation_features, "observation"
        )

        # Initialize gRPC connection
        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Initialize RTC action queue
        self.action_queue = ActionQueue(config.rtc)

        # Latency tracking for inference delay calculation
        self.latency_tracker = LatencyTracker()
        self.profiler = RTCProfiler(
            config.enable_profiling,
            config.profiling_output_dir,
            config.profiling_run_name,
        )

        # Robot processors
        self.robot_observation_processor = make_default_robot_observation_processor()
        self.robot_action_processor = make_default_robot_action_processor()

        logger.info(f"RobotClient initialized, connecting to {config.server_address}")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        """Connect to server and send policy instructions."""
        try:
            # Handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            logger.info(f"Connected to server in {time.perf_counter() - start_time:.4f}s")

            # Send policy configuration
            policy_config = RTCRemotePolicyConfig(
                policy_type=self.config.policy_type,
                pretrained_name_or_path=self.config.pretrained_name_or_path,
                lerobot_features=self.lerobot_features,
                rtc_config=self.config.rtc,
                device=self.config.policy_device,
                use_torch_compile=self.config.use_torch_compile,
                torch_compile_mode=self.config.torch_compile_mode,
            )

            policy_config_bytes = pickle.dumps(policy_config)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_config_bytes))

            logger.info(
                f"Policy instructions sent | "
                f"Type: {self.config.policy_type} | "
                f"Device: {self.config.policy_device} | "
                f"Compile: {self.config.use_torch_compile} ({self.config.torch_compile_mode})"
            )

            return True

        except grpc.RpcError as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def stop(self):
        """Stop the client and cleanup."""
        self.shutdown_event.set()
        self.robot.disconnect()
        self.channel.close()
        logger.info("Client stopped")

    def save_profiling_artifacts(self) -> dict[str, str]:
        artifacts = self.profiler.finalize()
        if artifacts:
            logger.info("Saved profiling artifacts:")
            for name, path in artifacts.items():
                logger.info(f"  - {name}: {path}")
        return artifacts

    def _prepare_observation(self, task: str) -> dict[str, Any]:
        """Capture and prepare observation for sending to server."""
        raw_obs = self.robot_wrapper.get_observation()

        # Apply robot observation processor
        obs_processed = self.robot_observation_processor(raw_obs)

        # Build dataset frame with proper keys
        obs_with_features = build_dataset_frame(
            self.lerobot_features, obs_processed, prefix="observation"
        )

        # Convert to tensors and prepare for policy
        for name in obs_with_features:
            obs_with_features[name] = torch.from_numpy(obs_with_features[name])
            if "image" in name:
                obs_with_features[name] = obs_with_features[name].type(torch.float32) / 255
                obs_with_features[name] = obs_with_features[name].permute(2, 0, 1).contiguous()
            obs_with_features[name] = obs_with_features[name].unsqueeze(0)

        obs_with_features["task"] = [task]
        obs_with_features["robot_type"] = (
            self.robot.name if hasattr(self.robot, "name") else ""
        )

        return obs_with_features

    def _run_remote_request(
        self,
        observation: dict[str, Any],
        *,
        action_index_before: int,
        queue_size_before: int,
        inference_delay: int,
        prev_actions: torch.Tensor | None,
        execution_horizon: int,
        label: str,
        merge_actions: bool,
        observation_ms: float,
    ) -> tuple[RTCActionData, float, int]:
        request_idx = self.request_idx
        request_start = time.perf_counter()

        rtc_obs = RTCObservationData(
            observation=observation,
            timestamp=time.time(),
            timestep=action_index_before,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_actions,
            execution_horizon=execution_horizon,
        )

        obs_bytes = pickle.dumps(rtc_obs)
        pickle_done = time.perf_counter()
        client_pickle_ms = (pickle_done - request_start) * 1000

        obs_iterator = send_bytes_in_chunks(
            obs_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(obs_iterator)
        send_done = time.perf_counter()
        client_send_ms = (send_done - pickle_done) * 1000

        actions_response = self.stub.GetActions(services_pb2.Empty())
        response_done = time.perf_counter()
        client_get_actions_ms = (response_done - send_done) * 1000

        if len(actions_response.data) == 0:
            raise RuntimeError("Empty response from server")

        rtc_action_data: RTCActionData = pickle.loads(actions_response.data)  # nosec
        unpickle_done = time.perf_counter()
        client_unpickle_ms = (unpickle_done - response_done) * 1000

        new_latency = unpickle_done - request_start
        client_total_ms = new_latency * 1000
        time_per_step = 1.0 / self.config.fps
        new_delay = math.ceil(new_latency / time_per_step)
        applied_delay = new_delay

        if merge_actions:
            applied_delay = self.action_queue.merge(
                rtc_action_data.original_actions,
                rtc_action_data.actions,
                new_delay,
                action_index_before,
            )
            queue_size_after = self.action_queue.qsize()
        else:
            queue_size_after = queue_size_before

        server_timing = getattr(rtc_action_data, "timing", None)
        self.profiler.add(
            RTCProfilingRecord(
                request_idx=request_idx,
                timestamp=time.time(),
                label=label,
                payload_bytes=len(obs_bytes),
                queue_size_before=queue_size_before,
                queue_size_after=queue_size_after,
                action_index_before=action_index_before,
                inference_delay_requested=inference_delay,
                realized_delay=applied_delay,
                client_observation_ms=observation_ms,
                client_pickle_ms=client_pickle_ms,
                client_send_ms=client_send_ms,
                client_get_actions_ms=client_get_actions_ms,
                client_unpickle_ms=client_unpickle_ms,
                client_total_ms=client_total_ms,
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
                server_pickle_ms=(
                    server_timing.pickle_ms if server_timing is not None else None
                ),
                server_total_ms=server_timing.total_ms if server_timing is not None else None,
            )
        )
        self.request_idx += 1

        if self.config.verbose_request_logging:
            logger.info(
                f"[GET_ACTIONS] {label} | "
                f"observation: {observation_ms:.1f}ms | "
                f"total: {client_total_ms:.1f}ms | "
                f"delay: {applied_delay} | "
                f"queue: {queue_size_after}"
            )

        return rtc_action_data, new_latency, applied_delay

    def warmup_compiled_policy(self) -> None:
        warmup_delays = list(self.config.compile_warmup_delay)
        if len(warmup_delays) == 0:
            return

        logger.info(
            "Running remote warmup requests: %d, delays=%s",
            len(warmup_delays),
            warmup_delays,
        )
        prev_actions = None

        for warmup_idx, delay in enumerate(warmup_delays):
            observation_start = time.perf_counter()
            observation = self._prepare_observation(self.config.task)
            observation_ms = (time.perf_counter() - observation_start) * 1000

            try:
                rtc_action_data, _, _ = self._run_remote_request(
                    observation,
                    action_index_before=0,
                    queue_size_before=self.action_queue.qsize(),
                    inference_delay=delay,
                    prev_actions=prev_actions,
                    execution_horizon=self.config.rtc.execution_horizon,
                    label="warmup",
                    merge_actions=False,
                    observation_ms=observation_ms,
                )
            except RuntimeError:
                logger.warning("Warmup request returned empty response, stopping warmup early")
                break

            if warmup_idx < len(warmup_delays) - 1:
                chunk_size = int(rtc_action_data.original_actions.shape[0])
                next_delay = warmup_delays[warmup_idx + 1]
                if next_delay < chunk_size:
                    prev_actions = rtc_action_data.original_actions[next_delay:].clone()
                else:
                    prev_actions = None

        self.latency_tracker = LatencyTracker()
        logger.info("Remote warmup finished")

    def get_actions_thread(self):
        """Thread function to request action chunks from remote server."""
        try:
            logger.info("[GET_ACTIONS] Starting get actions thread")

            threshold = self.config.action_queue_threshold

            if not self.config.rtc.enabled:
                threshold = 0

            while self.running:
                if self.action_queue.qsize() <= threshold:
                    queue_size_before = self.action_queue.qsize()
                    action_index_before = self.action_queue.get_action_index()
                    prev_actions = self.action_queue.get_left_over()

                    # Calculate inference delay from latency
                    time_per_step = 1.0 / self.config.fps
                    inference_delay = math.ceil(self.latency_tracker.max() / time_per_step)

                    # Prepare observation
                    observation_start = time.perf_counter()
                    observation = self._prepare_observation(self.config.task)
                    observation_ms = (time.perf_counter() - observation_start) * 1000

                    try:
                        _, new_latency, new_delay = self._run_remote_request(
                            observation,
                            action_index_before=action_index_before,
                            queue_size_before=queue_size_before,
                            inference_delay=inference_delay,
                            prev_actions=prev_actions,
                            execution_horizon=self.config.rtc.execution_horizon,
                            label="robot_live",
                            merge_actions=True,
                            observation_ms=observation_ms,
                        )
                    except RuntimeError:
                        logger.warning("[GET_ACTIONS] Empty response from server")
                        continue
                    self.latency_tracker.add(new_latency)

                    # Warn if threshold is too small
                    if self.config.action_queue_threshold < self.config.rtc.execution_horizon + new_delay:
                        logger.warning(
                            "[GET_ACTIONS] action_queue_threshold too small. "
                            f"Should be > execution_horizon + delay = "
                            f"{self.config.rtc.execution_horizon + new_delay}"
                        )

                else:
                    time.sleep(0.01)

            logger.info("[GET_ACTIONS] Thread shutting down")

        except Exception as e:
            logger.error(f"[GET_ACTIONS] Fatal error: {e}")
            traceback.print_exc()
            sys.exit(1)

    def actor_thread(self):
        """Thread function to execute actions on the robot."""
        try:
            logger.info("[ACTOR] Starting actor thread")

            action_count = 0
            action_interval = 1.0 / self.config.fps

            while self.running:
                start_time = time.perf_counter()

                action = self.action_queue.get()

                if action is not None:
                    action = action.cpu()
                    action_dict = {
                        key: action[i].item()
                        for i, key in enumerate(self.robot_wrapper.action_features())
                    }
                    action_processed = self.robot_action_processor((action_dict, None))
                    self.robot_wrapper.send_action(action_processed)
                    action_count += 1

                dt = time.perf_counter() - start_time
                time.sleep(max(0, action_interval - dt - 0.001))

            logger.info(f"[ACTOR] Thread shutting down. Total actions: {action_count}")

        except Exception as e:
            logger.error(f"[ACTOR] Fatal error: {e}")
            traceback.print_exc()
            sys.exit(1)


@draccus.wrap()
def main(cfg: RobotClientConfig):
    """Main entry point for RTC Robot Client."""
    logger.info(pformat(asdict(cfg)))

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    client = RobotClient(cfg)

    if not client.start():
        logger.error("Failed to connect to server")
        return

    client.warmup_compiled_policy()

    # Start threads
    get_actions_thread = threading.Thread(
        target=client.get_actions_thread,
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()

    actor_thread = threading.Thread(
        target=client.actor_thread,
        daemon=True,
        name="Actor",
    )
    actor_thread.start()

    logger.info(f"Running for {cfg.duration} seconds...")
    start_time = time.time()

    try:
        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            time.sleep(1.0)

            if int(time.time() - start_time) % 5 == 0:
                logger.info(f"[MAIN] Queue size: {client.action_queue.qsize()}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        logger.info("Shutting down...")
        client.shutdown_event.set()

        get_actions_thread.join(timeout=5)
        actor_thread.join(timeout=5)

        client.save_profiling_artifacts()
        client.stop()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
