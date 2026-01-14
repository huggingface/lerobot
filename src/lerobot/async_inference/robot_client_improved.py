import os as _os
import sys as _sys
import time as _time

_IMPORT_TIMING_ENABLED = _os.getenv("LEROBOT_IMPORT_TIMING", "0") == "1"
_IMPORT_T0 = _time.perf_counter() if _IMPORT_TIMING_ENABLED else 0.0

import logging
import pickle  # nosec
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import cv2  # type: ignore
import numpy as np
import grpc

from lerobot.robots.utils import make_robot_from_config
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs_improved import RobotClientImprovedConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    FPSTracker,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from .utils.diagnostics import DiagnosticsQueue
from .utils.latency_estimation import make_latency_estimator
from .utils.trajectory_viz import TrajectoryVizClient
from .utils.metrics import ExperimentMetricsWriter
from .utils.simulation import DropSimulator, MockRobot

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


# Sentinel value for shutdown signaling (must not conflict with valid action steps)
_SHUTDOWN_SENTINEL = -999999


# =============================================================================
# Action Schedule (OrderedDict-based with merge logic)
# =============================================================================


@dataclass
class ScheduledAction:
    """An action scheduled for execution at a specific step.

    Attributes:
        action: The action tensor/array to execute.
        source_step: The action step at which the source observation was captured.
    """

    action: np.ndarray
    source_step: int


@dataclass
class MergeStats:
    """Statistics from merging an action chunk into the schedule.

    Used for tracking action discontinuity (L2 distance between old and new
    actions at overlapping timesteps) to assess RTC smoothness.

    Attributes:
        overlap_count: Number of overlapping non-frozen actions compared.
        mean_l2: Mean L2 distance across overlapping actions (0.0 if no overlap).
        max_l2: Maximum L2 distance across overlapping actions (0.0 if no overlap).
    """

    overlap_count: int
    mean_l2: float
    max_l2: float


class ActionSchedule:
    def __init__(self):
        self._schedule: OrderedDict[int, ScheduledAction] = OrderedDict()

    def __len__(self) -> int:
        return len(self._schedule)

    def pop_front(self) -> tuple[int, np.ndarray, int] | None:
        """Pop and return the first (lowest action step) scheduled action.

        Returns:
            Tuple of (action, source_step) or None if empty.
        """
        if not self._schedule:
            return None
        # OrderedDict maintains insertion order; pop first item
        step, scheduled = self._schedule.popitem(last=False)
        return step, scheduled.action, scheduled.source_step

    def get_exec_prefix(self, *, current_step: int, max_len: int) -> np.ndarray | None:
        """Get up to `max_len` executable actions immediately after `current_step`.

        This is used to build the frozen-prefix payload for server-side RTC.
        """
        if max_len <= 0:
            return None
        out: list[np.ndarray] = []
        for step, scheduled in self._schedule.items():
            if step <= current_step:
                continue
            out.append(scheduled.action.astype(np.float32, copy=False))
            if len(out) >= max_len:
                break
        if not out:
            return None
        return np.asarray(out, dtype=np.float32, order="C")

    def get_frozen_chunks_info(
        self, *, current_step: int, max_len: int
    ) -> list[tuple[int, int, int]] | None:
        """Get list of (src_step, start_idx, end_idx) spans for the frozen prefix.

        This returns information needed to look up raw actions in the server's cache.
        Handles frozen prefixes that span multiple source chunks due to merging.

        Args:
            current_step: The current action step being executed.
            max_len: Maximum number of actions to include in the frozen prefix.

        Returns:
            List of (src_step, start_idx, end_idx) tuples in execution order, or None if empty.
            Each tuple specifies a contiguous slice from a cached chunk on the server.
        """
        if max_len <= 0:
            return None

        chunks: list[tuple[int, int, int]] = []
        current_src_step: int | None = None
        current_start: int | None = None
        current_end: int = 0
        count = 0

        for step, scheduled in self._schedule.items():
            if step <= current_step:
                continue

            # Index of this action within its source chunk
            chunk_idx = step - scheduled.source_step

            if current_src_step is None:
                # First action in prefix
                current_src_step = scheduled.source_step
                current_start = chunk_idx
                current_end = chunk_idx + 1
            elif scheduled.source_step == current_src_step and chunk_idx == current_end:
                # Contiguous with current span (same source, consecutive index)
                current_end = chunk_idx + 1
            else:
                # New span - save current and start new
                if current_start is not None:
                    chunks.append((current_src_step, current_start, current_end))
                current_src_step = scheduled.source_step
                current_start = chunk_idx
                current_end = chunk_idx + 1

            count += 1
            if count >= max_len:
                break

        # Save final span
        if current_src_step is not None and current_start is not None:
            chunks.append((current_src_step, current_start, current_end))

        return chunks if chunks else None

    def get_size(self) -> int:
        """Get the current schedule size."""
        return len(self._schedule)

    def is_empty(self) -> bool:
        """Check if schedule is empty."""
        return len(self._schedule) == 0

    def merge(
        self,
        incoming_actions: list[TimedAction],
        source_step: int,
        current_action_step: int,
        latency_steps: int,
        logger: logging.Logger | None = None,
    ) -> MergeStats:
        """Merge incoming actions using freshest-observation-wins strategy.

        Respects the frozen action invariant: actions within latency_steps of
        the current execution point cannot be modified.

        Args:
            incoming_actions: List of TimedAction from the server.
            source_step: The action step at which the source observation was captured.
            current_action_step: The most recently executed action step (n*).
            latency_steps: Current latency estimate in action steps (ℓ̂_Δ).
            logger: Optional logger for debug output.

        Returns:
            MergeStats with L2 discrepancy metrics for overlapping actions.
        """
        # Use counters instead of per-action logging to avoid ~1ms per log call
        stale_count = 0
        frozen_count = 0
        inserted_count = 0
        updated_count = 0

        # Track L2 discrepancy for overlapping actions (non-frozen)
        l2_distances: list[float] = []

        # Track if we need to re-sort (only if inserting out of order)
        max_existing_step = max(self._schedule.keys()) if self._schedule else -1
        needs_sort = False

        for timed_action in incoming_actions:
            step = timed_action.get_timestep()
            action = timed_action.get_action()

            # Skip stale actions (already executed)
            if step <= current_action_step:
                stale_count += 1
                continue

            existing = self._schedule.get(step)

            if existing is None:
                # New action step: always schedule it, even if it's in the frozen window.
                # The frozen-action invariant only prevents *modifying* already-scheduled actions.
                self._schedule[step] = ScheduledAction(action=action, source_step=source_step)
                inserted_count += 1
                # Check if this insertion is out of order
                if step < max_existing_step:
                    needs_sort = True
                else:
                    max_existing_step = step
                continue

            # Compute L2 discrepancy for ALL overlapping actions (for analysis metrics)
            # This includes frozen actions - we want to measure what the discrepancy would be
            old_arr = np.asarray(existing.action, dtype=np.float32).reshape(-1)
            new_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if old_arr.shape == new_arr.shape and old_arr.size > 0:
                l2 = float(np.linalg.norm(new_arr - old_arr))
                l2_distances.append(l2)

            # Existing action: do not modify if frozen.
            if self._is_frozen(step, current_action_step, latency_steps):
                frozen_count += 1
                continue

            if source_step > existing.source_step:
                # Fresher observation wins (only for non-frozen actions)
                self._schedule[step] = ScheduledAction(action=action, source_step=source_step)
                updated_count += 1

        # Re-sort only if we inserted out of order
        if needs_sort:
            sorted_items = sorted(self._schedule.items(), key=lambda x: x[0])
            self._schedule = OrderedDict(sorted_items)

        # Single summary log instead of per-action logs (saves ~20ms for 23 log calls)
        if logger and (stale_count or frozen_count):
            logger.debug(
                f"Merge stats: {stale_count} stale, {frozen_count} frozen, "
                f"{inserted_count} inserted, {updated_count} updated, resort={needs_sort}"
            )

        # Compute aggregate L2 stats
        overlap_count = len(l2_distances)
        if overlap_count > 0:
            mean_l2 = float(np.mean(l2_distances))
            max_l2 = float(np.max(l2_distances))
        else:
            mean_l2 = 0.0
            max_l2 = 0.0

        return MergeStats(overlap_count=overlap_count, mean_l2=mean_l2, max_l2=max_l2)

    @staticmethod
    def _is_frozen(action_step: int, current_step: int, latency_steps: int) -> bool:
        """Check if an action step is frozen (cannot be modified).

        Frozen(j, t) ≡ j ≤ n*(t) + ℓ̂_Δ

        Actions within the frozen window will be executed before any new
        inference result can possibly arrive and be merged.
        """
        return action_step <= current_step + latency_steps

    def get_step_range(self) -> tuple[int, int] | None:
        """Get the range of action steps in the schedule.

        Returns:
            Tuple of (min_step, max_step) or None if empty.
        """
        if not self._schedule:
            return None
        steps = list(self._schedule.keys())
        return min(steps), max(steps)

    def clear(self) -> None:
        """Clear all scheduled actions."""
        self._schedule.clear()


# =============================================================================
# Observation Request (for SPSC mailbox)
# =============================================================================


@dataclass
class ObservationRequest:
    """Request for an observation capture, sent from main thread to obs sender.

    Attributes:
        action_step: The current action step when the request was made.
        task: The task description string.
    """

    action_step: int
    task: str
    rtc_meta: dict[str, Any] | None = None


# =============================================================================
# Action Chunk (received from server)
# =============================================================================


@dataclass
class ReceivedActionChunk:
    """Action chunk received from the server with metadata.

    Attributes:
        actions: List of TimedAction from the server.
        source_step: The action step at which the source observation was captured.
        measured_latency: Measured round-trip time for this chunk.
    """

    actions: list[TimedAction]
    source_step: int
    measured_latency: float


# =============================================================================
# Improved Robot Client
# =============================================================================


class RobotClientImproved:
    """Latency-adaptive asynchronous inference robot client.

    This implementation follows the latency-adaptive async inference algorithm with:
    - 3-thread architecture (main, observation sender, action receiver)
    - SPSC one-slot mailboxes for thread communication
    - Jacobson-Karels latency estimation
    - Cool-down mechanism for inference triggering
    - Frozen action invariant
    - Freshest-observation-wins merging strategy
    """

    prefix = "robot_client_improved"
    logger = get_logger(prefix)

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    def __init__(self, config: RobotClientImprovedConfig):
        """Initialize the improved robot client.

        Args:
            config: Configuration for the robot client.
        """
        self.config = config

        # Use mock robot in simulation mode, real robot otherwise
        if config.simulation_mode:
            self.robot = MockRobot()
            self.robot.connect()
            # Mock features for simulation
            lerobot_features = {
                "observation.state": list(self.robot.state_features),
                "action": list(self.robot.action_features),
            }
            self.logger.info("Simulation mode: using MockRobot")
        else:
            self.robot = make_robot_from_config(config.robot)
            self.robot.connect()
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Drop simulators for experiments
        self._obs_drop_sim = DropSimulator(config=config.drop_obs_config)
        self._action_drop_sim = DropSimulator(config=config.drop_action_config)

        self.server_address = config.server_address
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rtc_enabled=config.rtc_enabled,
            rtc_execution_horizon=config.rtc_execution_horizon,
            rtc_max_guidance_weight=config.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
            rtc_sigma_d=config.rtc_sigma_d,
            rtc_full_trajectory_alignment=config.rtc_full_trajectory_alignment,
            num_flow_matching_steps=config.num_flow_matching_steps,
            # Spike injection (passed to server for experiments)
            spikes=config.spikes,
        )

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        # Shutdown coordination
        self.shutdown_event = threading.Event()

        # Action state: n(t), initialized to -1 per algorithm.
        # Note: Only the main control loop thread reads/writes action_step.
        self.action_step: int = -1

        # Latency estimation (configurable: JK or max_last_10)
        # Upper bound is enforced per RTC paper constraint: d <= H - s_min
        self.latency_estimator = make_latency_estimator(
            kind=config.latency_estimator_type,
            fps=config.fps,
            alpha=config.latency_alpha,
            beta=config.latency_beta,
            k=config.latency_k,
            action_chunk_size=config.actions_per_chunk,
            execution_horizon=config.rtc_execution_horizon,
        )

        # Action schedule (replaces Queue with OrderedDict)
        self.action_schedule = ActionSchedule()

        # Cool-down counter O^c(t).
        # Note: Only the main control loop thread reads/writes obs_cooldown.
        self.obs_cooldown: int = 0

        # SPSC Mailboxes (one-slot queues)
        # Observation request mailbox: main thread -> observation sender
        self._obs_request_mailbox: Queue[ObservationRequest] = Queue(maxsize=1)

        # Action mailbox: action receiver -> main thread
        self._action_mailbox: Queue[ReceivedActionChunk] = Queue(maxsize=1)

        # Synchronization barrier for thread startup
        self.start_barrier = threading.Barrier(3)  # 3 threads: main, obs sender, action receiver

        # Debug tracking
        self.action_queue_sizes: list[int] = []

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Optional diagnostics (queue-based, lock-free for producers)
        self._diagnostics: DiagnosticsQueue | None = None
        if config.diagnostics_enabled:
            self._diagnostics = DiagnosticsQueue(config, self.shutdown_event)
            self._diagnostics.start_consumer(self.logger)

        # Trajectory visualization: send chunks to policy server via gRPC
        # Uses a queue + background thread to avoid blocking the control loop
        self._trajectory_chunk_queue: Queue[services_pb2.TrajectoryChunk] = Queue(maxsize=10)
        self._trajectory_sender_thread: threading.Thread | None = None
        self._trajectory_viz_client: TrajectoryVizClient | None = None
        if config.trajectory_viz_enabled:
            self._trajectory_sender_thread = threading.Thread(
                target=self._trajectory_chunk_sender,
                name="trajectory_chunk_sender",
                daemon=True,
            )
            self._trajectory_sender_thread.start()
            self.logger.info("Trajectory visualization enabled (sending to policy server)")

            # WebSocket client for sending executed actions directly to viz server
            self._trajectory_viz_client = TrajectoryVizClient(ws_url=config.trajectory_viz_ws_url)
            self._trajectory_viz_client.start()
            # Wire up executed action callback if diagnostics is enabled
            if self._diagnostics is not None:
                self._diagnostics.set_executed_action_callback(self._trajectory_viz_client.on_executed_action)

        # Experiment metrics collector (CSV export)
        self._experiment_metrics: ExperimentMetricsWriter | None = None
        if config.experiment_metrics_path:
            self._experiment_metrics = ExperimentMetricsWriter()
            self.logger.info(f"Experiment metrics enabled, will write to: {config.experiment_metrics_path}")

        self.logger.info("Robot connected and ready")

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def current_action_step(self) -> int:
        """Get the most recently executed action step n*(t).

        Note: Only the main control loop thread should access this property.
        """
        return max(self.action_step, -1)

    def start(self) -> bool:
        """Start the robot client and connect to the policy server."""
        try:
            t_total_start = time.perf_counter()

            # Server handshake
            t_ready_start = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            t_ready_done = time.perf_counter()
            self.logger.debug(
                "Connected to policy server (Ready RPC) in %.2fms",
                self._ms(t_ready_done - t_ready_start),
            )

            # Send policy configuration
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            t_policy_rpc_start = time.perf_counter()
            self.stub.SendPolicyInstructions(policy_setup)
            t_policy_rpc_done = time.perf_counter()

            self.shutdown_event.clear()

            # Prime latency estimator with initial RTT measurements
            t_prime_start = time.perf_counter()
            self._prime_latency_estimator()
            t_prime_done = time.perf_counter()

            self.logger.info(
                "Client init complete | Ready: %.2fms | Policy RPC: %.2fms | Priming: %.2fms | Total: %.2fms",
                self._ms(t_ready_done - t_ready_start),
                self._ms(t_policy_rpc_done - t_policy_rpc_start),
                self._ms(t_prime_done - t_prime_start),
                self._ms(time.perf_counter() - t_total_start),
            )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self) -> None:
        """Stop the robot client."""
        self.shutdown_event.set()

        # Unblock any waiting threads with sentinel values
        with suppress(Full):
            self._obs_request_mailbox.put_nowait(
                ObservationRequest(action_step=_SHUTDOWN_SENTINEL, task="")
            )
        with suppress(Full):
            self._action_mailbox.put_nowait(
                ReceivedActionChunk(actions=[], source_step=_SHUTDOWN_SENTINEL, measured_latency=0.0)
            )

        # Flush experiment metrics if enabled
        if self._experiment_metrics is not None and self.config.experiment_metrics_path:
            self._experiment_metrics.flush(self.config.experiment_metrics_path)
            summary = self._experiment_metrics.get_summary()
            self.logger.info(
                f"Experiment metrics written to {self.config.experiment_metrics_path}: {summary}"
            )

        # Stop trajectory viz client if enabled
        if self._trajectory_viz_client is not None:
            self._trajectory_viz_client.stop()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    # -------------------------------------------------------------------------
    # Latency Priming
    # -------------------------------------------------------------------------

    def _prime_latency_estimator(self) -> bool:
        """Prime the latency estimator with initial RTT measurements.

        Sends a configurable number of probe observations and measures RTT
        from the action responses. This reduces uncertainty at startup.

        Returns:
            True if priming succeeded, False otherwise.
        """
        prime_count = self.config.latency_prime_count
        if prime_count <= 0:
            self.logger.debug("Latency priming disabled (prime_count=0)")
            return True

        self.logger.info(f"Starting latency priming with {prime_count} rounds...")
        samples: list[float] = []
        timeout_s = self.config.latency_prime_timeout_s

        try:
            # Open the streaming RPC for receiving actions
            stream = self.stub.StreamActionsDense(services_pb2.Empty())

            for i in range(prime_count):
                try:
                    # Capture observation from robot
                    t_start = time.time()
                    raw_observation = self.robot.get_observation()
                    raw_observation["task"] = self.config.task

                    # Encode images for transport
                    encoded_observation, _ = _encode_images_for_transport(
                        raw_observation, jpeg_quality=60
                    )

                    # Create timed observation (use negative timestep to mark as priming)
                    timed_obs = TimedObservation(
                        timestamp=t_start,
                        observation=encoded_observation,
                        timestep=-(i + 1),  # Negative to distinguish from real observations
                    )

                    # Send observation to server
                    if not self._send_observation(timed_obs):
                        self.logger.warning(f"Priming round {i + 1}: failed to send observation")
                        continue

                    # Wait for action response with timeout
                    # Note: We use a simple iteration with a deadline
                    deadline = time.time() + timeout_s
                    action_received = False

                    for dense in stream:
                        t_receive = time.time()
                        if dense.t > 0 and dense.a > 0:
                            # Calculate RTT from the observation timestamp
                            t0 = float(dense.t0)
                            rtt = t_receive - t0
                            samples.append(rtt)
                            action_received = True
                            self.logger.debug(
                                f"Priming round {i + 1}/{prime_count}: RTT = {rtt * 1000:.2f}ms"
                            )
                            break

                        if time.time() > deadline:
                            self.logger.warning(
                                f"Priming round {i + 1}: timeout waiting for action response"
                            )
                            break

                    if not action_received:
                        self.logger.warning(f"Priming round {i + 1}: no valid action received")

                except Exception as e:
                    self.logger.warning(f"Priming round {i + 1} failed: {e}")
                    continue

            # Cancel the stream
            stream.cancel()

        except grpc.RpcError as e:
            self.logger.warning(f"Latency priming failed (gRPC error): {e}")
            return False

        # Prime the estimator with collected samples
        if samples:
            self.latency_estimator.prime(samples)
            mean_rtt_ms = sum(samples) / len(samples) * 1000
            self.logger.info(
                f"Latency priming complete: {len(samples)}/{prime_count} samples, "
                f"mean RTT = {mean_rtt_ms:.2f}ms, estimate = {self.latency_estimator.estimate_steps} steps"
            )
            return True
        else:
            self.logger.warning("Latency priming failed: no valid samples collected")
            return False

    # -------------------------------------------------------------------------
    # Observation Sender Thread
    # -------------------------------------------------------------------------

    def observation_sender(self) -> None:
        """Captures, encodes, and sends observations to the policy server."""
        self.start_barrier.wait()
        self.logger.info("Observation sender thread starting")

        last_good_observation: RawObservation | None = None
        last_good_observation_time: float | None = None
        consecutive_capture_failures = 0

        while self.running:
            try:
                # Wait for an observation request from the main thread
                t_wait_start = time.perf_counter()
                try:
                    request = self._obs_request_mailbox.get(timeout=0.1)
                except Empty:
                    continue
                t_wait_end = time.perf_counter()

                # Emit wait time (how long obs sender was idle waiting for work)
                if self._diagnostics is not None:
                    self._diagnostics.emit_obs_wait(self._ms(t_wait_end - t_wait_start))

                # Sentinel value to unblock on shutdown
                if request.action_step == _SHUTDOWN_SENTINEL:
                    continue

                t_capture_start = time.perf_counter()

                # Capture observation from robot
                used_fallback = False
                try:
                    raw_observation = self.robot.get_observation()
                    last_good_observation = raw_observation
                    last_good_observation_time = time.time()
                    consecutive_capture_failures = 0
                except Exception as e:
                    consecutive_capture_failures += 1
                    if (
                        self.config.obs_fallback_on_failure
                        and last_good_observation is not None
                        and last_good_observation_time is not None
                        and (time.time() - last_good_observation_time) <= self.config.obs_fallback_max_age_s
                    ):
                        used_fallback = True
                        raw_observation = last_good_observation
                        self.logger.warning(
                            "Observation capture failed (%s). Reusing last good observation (age=%.2fs, consecutive_failures=%s).",
                            e,
                            time.time() - last_good_observation_time,
                            consecutive_capture_failures,
                        )
                    else:
                        self.logger.error(
                            "Observation capture failed (%s). No usable fallback (consecutive_failures=%s).",
                            e,
                            consecutive_capture_failures,
                        )
                        continue

                # Avoid mutating cached observation dict if we are reusing it.
                if used_fallback:
                    raw_observation = dict(raw_observation)
                raw_observation["task"] = request.task
                if request.rtc_meta is not None:
                    raw_observation["__rtc__"] = request.rtc_meta

                t_capture_done = time.perf_counter()

                # Encode images for transport
                t_encode_start = time.perf_counter()
                encoded_observation, encode_stats = _encode_images_for_transport(
                    raw_observation, jpeg_quality=60
                )
                t_encode_done = time.perf_counter()

                if encode_stats["images_encoded"] > 0:
                    self.logger.debug(
                        "Encoded %s images in %.2fms | raw=%s -> encoded=%s",
                        encode_stats["images_encoded"],
                        self._ms(t_encode_done - t_encode_start),
                        encode_stats["raw_bytes_total"],
                        encode_stats["encoded_bytes_total"],
                    )

                # Create timed observation
                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    observation=encoded_observation,
                    timestep=request.action_step
                )

                # Check if observation should be dropped (simulation/experiments)
                if self._obs_drop_sim.should_drop():
                    self.logger.debug("Dropping observation #%s (simulated drop)", request.action_step)
                    continue

                # Send to server
                t_send_start = time.perf_counter()
                self._send_observation(timed_obs)
                t_send_done = time.perf_counter()

                if self._diagnostics is not None:
                    self._diagnostics.emit_observation_sender(
                        capture_ms=self._ms(t_capture_done - t_capture_start),
                        encode_ms=self._ms(t_encode_done - t_encode_start),
                        send_ms=self._ms(t_send_done - t_send_start),
                    )

                self.logger.debug(
                    "Observation #%s sent | capture: %.2fms | encode: %.2fms | send: %.2fms",
                    request.action_step,
                    self._ms(t_capture_done - t_capture_start),
                    self._ms(t_encode_done - t_encode_start),
                    self._ms(t_send_done - t_send_start),
                )

            except Exception as e:
                self.logger.error(f"Error in observation sender: {e}")

    def _send_observation(self, obs: TimedObservation) -> bool:
        """Send a timed observation to the policy server via gRPC."""
        try:
            observation_bytes = pickle.dumps(obs)
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation: {e}")
            return False

    # -------------------------------------------------------------------------
    # Trajectory Chunk Sender Thread
    # -------------------------------------------------------------------------

    def _trajectory_chunk_sender(self) -> None:
        """Background thread that sends trajectory chunks to the policy server."""
        while self.running:
            try:
                # Wait for a chunk to send (with timeout to check shutdown)
                try:
                    chunk = self._trajectory_chunk_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Send to server (best-effort, don't block on errors)
                try:
                    self.stub.SendTrajectoryChunk(chunk)
                except grpc.RpcError as e:
                    self.logger.debug(f"Failed to send trajectory chunk: {e}")

            except Exception as e:
                self.logger.debug(f"Error in trajectory chunk sender: {e}")

    def _queue_trajectory_chunk(
        self,
        source_step: int,
        actions: list[np.ndarray],
        frozen_len: int,
    ) -> None:
        """Queue a trajectory chunk for sending to the policy server (non-blocking)."""
        if not actions:
            return

        # Convert actions to packed float32 bytes
        action_dim = actions[0].shape[0] if len(actions) > 0 else 0
        actions_array = np.stack([a.astype(np.float32) for a in actions], axis=0)
        actions_bytes = actions_array.tobytes()

        chunk = services_pb2.TrajectoryChunk(
            source_step=source_step,
            num_actions=len(actions),
            action_dim=action_dim,
            actions_f32=actions_bytes,
            frozen_len=frozen_len,
            timestamp=time.time(),
        )

        # Non-blocking put: drop if queue is full
        try:
            self._trajectory_chunk_queue.put_nowait(chunk)
        except Full:
            # Drop oldest and add new
            with suppress(Empty):
                self._trajectory_chunk_queue.get_nowait()
            with suppress(Full):
                self._trajectory_chunk_queue.put_nowait(chunk)

    # -------------------------------------------------------------------------
    # Action Receiver Thread
    # -------------------------------------------------------------------------

    def action_receiver(self) -> None:
        """Receives actions from the server via streaming."""
        self.start_barrier.wait()
        self.logger.info("Action receiver thread starting")
        last_chunk_time: float | None = None
        while self.running:
            try:
                t_rpc_start = time.perf_counter()
                stream = self.stub.StreamActionsDense(services_pb2.Empty())
                t_rpc_done = time.perf_counter()
                if self._diagnostics is not None:
                    self._diagnostics.emit_action_receiver(
                        rpc_ms=self._ms(t_rpc_done - t_rpc_start),
                        deser_ms=0.0,
                        latency_ms=0.0,
                    )

                for dense in stream:
                    if not self.running:
                        break
                    t_chunk_received = time.perf_counter()
                    # Emit chunk gap timing (time since last chunk)
                    if last_chunk_time is not None and self._diagnostics is not None:
                        self._diagnostics.emit_chunk_gap(self._ms(t_chunk_received - last_chunk_time))
                    last_chunk_time = t_chunk_received
                    self._handle_actions_dense(dense, rpc_ms=0.0)

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    self.logger.error(
                        "Server does not implement StreamActionsDense. "
                        "This client is streaming-only for actions; please update the server."
                    )
                    self.stop()
                    return
                self.logger.error(f"Error in StreamActionsDense: {e}")
                time.sleep(0.1)

    def _handle_actions_dense(self, dense: services_pb2.ActionsDense, rpc_ms: float) -> None:
        """Decode a dense action chunk into TimedAction list and publish to main thread."""
        receive_time = time.time()

        t = int(dense.t)
        a = int(dense.a)
        if t <= 0 or a <= 0:
            return

        t_deser_start = time.perf_counter()
        actions = np.frombuffer(dense.actions_f32, dtype=np.float32)
        if actions.size != t * a:
            raise ValueError(f"ActionsDense buffer size mismatch: {actions.size} != {t*a}")
        actions = actions.reshape(t, a)
        t_deser_done = time.perf_counter()

        t0 = float(dense.t0)
        i0 = int(dense.i0)
        dt = float(dense.dt)

        measured_latency = receive_time - t0
        timed_actions = [
            TimedAction(timestamp=t0 + i * dt, timestep=i0 + i, action=actions[i]) for i in range(t)
        ]

        self.logger.debug(
            "Received %s dense actions for step #%s | RPC: %.2fms | decode: %.2fms | latency: %.2fms",
            t,
            i0,
            rpc_ms,
            self._ms(t_deser_done - t_deser_start),
            self._ms(measured_latency),
        )

        if self._diagnostics is not None:
            self._diagnostics.emit_action_receiver(
                rpc_ms=rpc_ms,
                deser_ms=self._ms(t_deser_done - t_deser_start),
                latency_ms=self._ms(measured_latency),
            )

        # Check if action chunk should be dropped (simulation/experiments)
        if self._action_drop_sim.should_drop():
            self.logger.debug("Dropping action chunk for step #%s (simulated drop)", i0)
            return

        self._publish_received_actions(
            timed_actions=timed_actions,
            source_step=i0,
            measured_latency=measured_latency,
        )

    def _publish_received_actions(
        self,
        *,
        timed_actions: list[TimedAction],
        source_step: int,
        measured_latency: float,
    ) -> None:
        # Put in action mailbox (overwrite if full - one-slot mailbox)
        chunk = ReceivedActionChunk(
            actions=timed_actions,
            source_step=source_step,
            measured_latency=measured_latency,
        )

        if self._action_mailbox.full():
            with suppress(Empty):
                _ = self._action_mailbox.get_nowait()

        with suppress(Full):
            self._action_mailbox.put_nowait(chunk)

    # -------------------------------------------------------------------------
    # Main Thread: Control Loop
    # -------------------------------------------------------------------------

    def control_loop(self, task: str | None = None) -> None:
        """Main control loop following Algorithm 1 from the paper.

        This loop:
        1. Executes actions if available
        2. Checks inference trigger condition and requests observations
        3. Processes incoming action chunks
        4. Maintains control frequency

        Args:
            task: Optional task override (uses config.task if not provided).
        """
        self.start_barrier.wait()
        self.logger.info("Control loop starting")

        task = task or self.config.task

        prev_loop_start: float | None = None
        prev_action: np.ndarray | None = None
        next_tick: float | None = time.perf_counter() if self.config.control_use_deadline_clock else None

        while self.running:
            t_loop_start = time.perf_counter()
            if prev_loop_start is not None and self._diagnostics is not None:
                self._diagnostics.emit_loop_dt_ms(self._ms(t_loop_start - prev_loop_start))
            prev_loop_start = t_loop_start

            # Experiment metrics tracking for this tick
            _tick_obs_sent = False
            _tick_action_received = False
            _tick_measured_latency_ms: float | None = None
            _tick_chunk_overlap_count: int | None = None
            _tick_chunk_mean_l2: float | None = None
            _tick_chunk_max_l2: float | None = None

            # Phase timing tracking
            _phase_exec_ms = 0.0
            _phase_trigger_ms = 0.0
            _phase_merge_ms = 0.0

            # ---------------------------------------------------------------------
            # Step 1: Execute action if available
            # ---------------------------------------------------------------------
            t_phase1_start = time.perf_counter()
            if not self.action_schedule.is_empty():
                result = self.action_schedule.pop_front()
                if result is not None:
                    step, action, _ = result
                    t_send_start = time.perf_counter()
                    self.robot.send_action(self._action_array_to_dict(action))
                    t_send_done = time.perf_counter()

                    # Keep action_step aligned with the schedule's action-step keys.
                    # Only the main control loop thread writes this.
                    self.action_step = step

                    self.logger.debug(
                        "Executed action #%s | send_action: %.2fms",
                        self.action_step,
                        self._ms(t_send_done - t_send_start),
                    )

                    if self._diagnostics is not None:
                        self._diagnostics.emit_send_action(
                            send_ms=self._ms(t_send_done - t_send_start),
                            send_t=t_send_done,
                        )
                        if prev_action is not None:
                            self._diagnostics.emit_action_delta(prev_action, action)
                        prev_action = action
                        # Emit executed action for trajectory visualization
                        self._diagnostics.emit_executed_action(
                            step=step,
                            action=action.tolist(),
                        )

                    # Record executed action for experiment trajectory visualization
                    if self._experiment_metrics is not None:
                        self._experiment_metrics.record_executed_action(
                            step=step,
                            action=action,
                        )

            t_phase1_end = time.perf_counter()
            _phase_exec_ms = self._ms(t_phase1_end - t_phase1_start)

            # Track queue size for debugging and starvation detection
            schedule_size = self.action_schedule.get_size()
            self.action_queue_sizes.append(schedule_size)
            is_starved = schedule_size == 0
            if self._diagnostics is not None:
                self._diagnostics.emit_starvation(is_starved)

            # ---------------------------------------------------------------------
            # Step 2: Check inference trigger condition
            # ---------------------------------------------------------------------
            t_phase2_start = time.perf_counter()
            latency_steps = self.latency_estimator.estimate_steps
            epsilon = self.config.epsilon

            # Inference condition: |ψ(t)| ≤ ℓ̂_Δ + ε AND O^c(t) = 0 (if cooldown enabled)
            trigger_threshold = latency_steps + epsilon
            if self.config.cooldown_enabled:
                should_trigger = schedule_size <= trigger_threshold and self.obs_cooldown == 0
            else:
                # Classic async baseline: always trigger when schedule is low
                should_trigger = schedule_size <= trigger_threshold

            if should_trigger:
                current_step = self.current_action_step

                # Put observation request in mailbox
                # Clamp to 0 so the server produces chunks starting at 0 on startup (consistent with the
                # original async inference implementation that uses max(latest_action, 0)).
                rtc_meta: dict[str, Any] | None = None
                if self.config.rtc_enabled:
                    t_rtc_start = time.perf_counter()
                    frozen_len = max(
                        0, min(int(latency_steps), int(self.config.actions_per_chunk))
                    )
                    # Get list of (src_step, start, end) spans for server-side cache lookup
                    # Handles frozen prefix spanning multiple source chunks
                    frozen_chunks = self.action_schedule.get_frozen_chunks_info(
                        current_step=current_step, max_len=frozen_len
                    )
                    self.logger.debug(
                        "RTC: current_step=%s, frozen_len=%s, frozen_chunks=%s, schedule_size=%s",
                        current_step,
                        frozen_len,
                        frozen_chunks,
                        self.action_schedule.get_size(),
                    )
                    rtc_meta = {
                        "enabled": True,
                        "latency_steps": int(latency_steps),
                        "frozen_chunks": frozen_chunks,  # List of (src_step, start, end) or None
                    }
                    t_rtc_end = time.perf_counter()
                    if self._diagnostics is not None:
                        self._diagnostics.emit_rtc_build(self._ms(t_rtc_end - t_rtc_start))

                request = ObservationRequest(
                    action_step=max(current_step, 0),
                    task=task,
                    rtc_meta=rtc_meta,
                )

                # Always reset cooldown when trigger fires (before attempting put)
                # This must be outside suppress(Full) to ensure it always executes
                if self.config.cooldown_enabled:
                    self.obs_cooldown = trigger_threshold

                # Empty the mailbox
                if self._obs_request_mailbox.full():
                    with suppress(Empty):
                        _ = self._obs_request_mailbox.get_nowait()

                # Put the request in the mailbox
                with suppress(Full):
                    self._obs_request_mailbox.put_nowait(request)

                _tick_obs_sent = True
                self.logger.info(
                    "Triggered inference | step: %s | schedule: %s | latency_steps: %s | cooldown set to: %s",
                    current_step,
                    schedule_size,
                    latency_steps,
                    self.obs_cooldown,
                )
            else:
                # Decrement cooldown: O^c(t+1) = max(O^c(t) - 1, 0)
                # Only decrement in 'cooldown' mode (default behavior for drop recovery)
                # In 'merge_reset' mode, cooldown is only reset when actions are merged
                if self.config.cooldown_enabled and self.config.inference_reset_mode == "cooldown":
                    self.obs_cooldown = max(self.obs_cooldown - 1, 0)

            t_phase2_end = time.perf_counter()
            _phase_trigger_ms = self._ms(t_phase2_end - t_phase2_start)

            # ---------------------------------------------------------------------
            # Step 3: Check for incoming action chunks
            # ---------------------------------------------------------------------
            t_phase3_start = time.perf_counter()
            try:
                chunk = self._action_mailbox.get_nowait()
                if chunk.source_step != _SHUTDOWN_SENTINEL:  # Not a sentinel
                    current_step = self.current_action_step
                    latency_steps = self.latency_estimator.estimate_steps

                    # Update latency estimate
                    self.latency_estimator.update(chunk.measured_latency)

                    # Merge actions into schedule
                    t_merge_start = time.perf_counter()
                    merge_stats = self.action_schedule.merge(
                        incoming_actions=chunk.actions,
                        source_step=chunk.source_step,
                        current_action_step=current_step,
                        latency_steps=latency_steps,
                        logger=self.logger,
                    )
                    t_merge_done = time.perf_counter()

                    new_estimate = self.latency_estimator.estimate_steps
                    _tick_action_received = True
                    _tick_measured_latency_ms = self._ms(chunk.measured_latency)
                    # Track discrepancy stats from the merge
                    _tick_chunk_overlap_count = merge_stats.overlap_count
                    _tick_chunk_mean_l2 = merge_stats.mean_l2
                    _tick_chunk_max_l2 = merge_stats.max_l2
                    self.logger.info(
                        "Merged %s actions from step #%s | latency: %.2fms | new estimate: %s steps | "
                        "schedule size: %s | merge time: %.2fms | overlap: %s, mean_l2: %.4f, max_l2: %.4f",
                        len(chunk.actions),
                        chunk.source_step,
                        self._ms(chunk.measured_latency),
                        new_estimate,
                        self.action_schedule.get_size(),
                        self._ms(t_merge_done - t_merge_start),
                        merge_stats.overlap_count,
                        merge_stats.mean_l2,
                        merge_stats.max_l2,
                    )

                    # In merge_reset mode, reset cooldown when actions are merged
                    # This mimics RTC-style behavior where inference readiness is gated
                    # by action arrival rather than time-based cooldown
                    if self.config.inference_reset_mode == "merge_reset":
                        self.obs_cooldown = 0

                    # Send action chunk to policy server for trajectory visualization
                    if self.config.trajectory_viz_enabled and chunk.actions:
                        # Extract action arrays from TimedAction list
                        actions_arrays = [ta.action for ta in chunk.actions]
                        self._queue_trajectory_chunk(
                            source_step=chunk.source_step,
                            actions=actions_arrays,
                            frozen_len=latency_steps,
                        )

                    # Record chunk for experiment trajectory visualization
                    if self._experiment_metrics is not None and chunk.actions:
                        actions_arrays = [ta.action for ta in chunk.actions]
                        self._experiment_metrics.record_chunk(
                            source_step=chunk.source_step,
                            actions=actions_arrays,
                            frozen_len=int(latency_steps),
                        )

            except Empty:
                pass

            t_phase3_end = time.perf_counter()
            _phase_merge_ms = self._ms(t_phase3_end - t_phase3_start)

            # Emit phase timings
            if self._diagnostics is not None:
                self._diagnostics.emit_loop_phases(
                    exec_ms=_phase_exec_ms,
                    trigger_ms=_phase_trigger_ms,
                    merge_ms=_phase_merge_ms,
                )

            # ---------------------------------------------------------------------
            # Step 4: Maintain control frequency
            # ---------------------------------------------------------------------
            elapsed = time.perf_counter() - t_loop_start
            if next_tick is None:
                sleep_s = max(0.0, self.config.environment_dt - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    if self._diagnostics is not None:
                        self._diagnostics.emit_overrun()
                    self.logger.debug(
                        "Control loop overran | elapsed: %.2fms | target: %.2fms",
                        self._ms(elapsed),
                        self._ms(self.config.environment_dt),
                    )
            else:
                # Deadline-based clock: reduces drift and jitter when occasional overruns happen.
                next_tick += self.config.environment_dt
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # If we're behind, count an overrun and re-anchor to now to avoid runaway lag.
                    if self._diagnostics is not None:
                        self._diagnostics.emit_overrun()
                    self.logger.debug(
                        "Control loop overran (deadline clock) | late_by: %.2fms | target: %.2fms",
                        self._ms(-sleep_s),
                        self._ms(self.config.environment_dt),
                    )
                    next_tick = now

            if self._diagnostics is not None:
                self._diagnostics.emit_log_context(
                    step=self.current_action_step,
                    schedule_size=self.action_schedule.get_size(),
                    latency_steps=self.latency_estimator.estimate_steps,
                    cooldown=self.obs_cooldown,
                )

            # Record experiment metrics for this tick
            if self._experiment_metrics is not None:
                self._experiment_metrics.record_tick(
                    step=self.current_action_step,
                    schedule_size=self.action_schedule.get_size(),
                    latency_estimate_steps=self.latency_estimator.estimate_steps,
                    cooldown=self.obs_cooldown,
                    obs_sent=_tick_obs_sent,
                    action_received=_tick_action_received,
                    measured_latency_ms=_tick_measured_latency_ms,
                    chunk_overlap_count=_tick_chunk_overlap_count,
                    chunk_mean_l2=_tick_chunk_mean_l2,
                    chunk_max_l2=_tick_chunk_max_l2,
                )

    def _action_array_to_dict(self, action_array: np.ndarray) -> dict[str, float]:
        """Convert action array to dictionary keyed by robot action features."""
        return {key: action_array[i].item() for i, key in enumerate(self.robot.action_features)}


# =============================================================================
# Image Encoding for Transport
# =============================================================================


def _is_uint8_hwc3_image(x: Any) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.dtype != np.uint8:
        return False
    if x.ndim != 3:
        return False
    h, w, c = x.shape
    if h <= 0 or w <= 0:
        return False
    return c == 3


def _encode_images_for_transport(
    observation: Any,
    jpeg_quality: int,
) -> tuple[Any, dict[str, int]]:
    """Recursively JPEG-encode uint8 HWC3 images inside an observation structure."""
    stats = {"images_encoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _encode_any(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _encode_any(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_encode_any(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_encode_any(v) for v in x)

        if not _is_uint8_hwc3_image(x):
            return x

        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to JPEG-encode image for transport")

        payload = bytes(buf)
        stats["images_encoded"] += 1
        stats["raw_bytes_total"] += int(x.nbytes)
        stats["encoded_bytes_total"] += len(payload)
        return {"__lerobot_image_encoding__": "jpeg", "quality": int(jpeg_quality), "data": payload}

    return _encode_any(observation), stats


# =============================================================================
# Entry Point
# =============================================================================


def async_client_improved(cfg: RobotClientImprovedConfig) -> None:
    """Run the improved async inference client."""
    logging.info(pformat(cfg.__dict__))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClientImproved(cfg)

    if client.start():
        # Start observation sender thread
        obs_sender_thread = threading.Thread(
            target=client.observation_sender,
            name="observation_sender",
            daemon=True,
        )

        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.action_receiver,
            name="action_receiver",
            daemon=True,
        )

        obs_sender_thread.start()
        action_receiver_thread.start()

        try:
            # Main thread runs the control loop
            client.control_loop()

        finally:
            client.stop()
            obs_sender_thread.join(timeout=2.0)
            action_receiver_thread.join(timeout=2.0)

            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_sizes)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    import draccus

    draccus.wrap()(async_client_improved)()
