import logging
import pickle  # nosec
import threading
import time
from collections import deque
from sortedcontainers import SortedDict
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

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
from .utils.action_filter import (
    ActionFilter,
    AdaptiveLowpassFilter,
    ButterworthFilter,
    FilterContext,
    HoldStableFilter,
    MedianFilter,
    NoFilter,
)
from .utils.latency_estimation import make_latency_estimator
from .utils.metrics import DiagnosticMetrics, EvExecutedAction, ExperimentMetricsWriter, Metrics
from .utils.simulation import DropSimulator, DuplicateSimulator, MockRobot, ReorderSimulator
from .utils.trajectory_viz import TrajectoryVizClient
from .utils.compression import encode_images_for_transport
from .lww_register import LWWReader, LWWRegister

# ---------------------------------------------------------------------------
# CAUSALITY MODEL — TWO CLOCKS
# ---------------------------------------------------------------------------
# The system uses two logical clocks:
#
#   control_step (t ∈ ℕ):
#       Monotone counter incremented every tick of the control loop.
#       Used as the LWW register key and watermark, ensuring that dropped
#       messages never stall the system (t always advances).
#
#   action_step (j ∈ ℤ, aka n(t)):
#       Execution index incremented when an action is executed on the robot.
#       Used to index into the action schedule and to compute offsets within
#       action chunks (chunk_idx = j - chunk_start_step).
#
# An observation captured at control_step t with chunk_start_step n_k
# produces actions for execution steps [n_k, n_k + H).
# Each scheduled action carries src_control_step (for freshness) and
# chunk_start_step (for RTC slice-offset computation).
#
# Merge rule: For execution step j, accept incoming action iff:
#   1. j > current_action_step (not yet executed), AND
#   2. src_control_step > existing.src_control_step (fresher observation wins)
#
# LWW registers provide transport monotonicity: only strictly newer messages
# (by control_step) can update the register, ensuring causality is preserved
# across thread boundaries.
# ---------------------------------------------------------------------------


@dataclass
class ScheduledAction:
    """An action scheduled for execution at a specific step.

    Attributes:
        action: The action tensor/array to execute.
        src_control_step: The control-loop tick t that produced this action (freshness key).
        chunk_start_step: The action step n_k where the source chunk starts (for RTC offset math).
    """

    action: np.ndarray
    src_control_step: int
    chunk_start_step: int


@dataclass
class MergeStats:
    """Statistics from merging an action chunk into the schedule.

    Used for tracking action discontinuity (L2 distance between old and new
    actions at overlapping timesteps) to assess RTC smoothness.

    Attributes:
        overlap_count: Number of overlapping non-hard-masked actions compared.
        mean_l2: Mean L2 distance across overlapping actions (0.0 if no overlap).
        max_l2: Maximum L2 distance across overlapping actions (0.0 if no overlap).
    """

    overlap_count: int
    mean_l2: float
    max_l2: float


class ActionSchedule:
    def __init__(self):
        self._schedule: SortedDict[int, ScheduledAction] = SortedDict()

    def __len__(self) -> int:
        return len(self._schedule)

    def pop_front(self) -> tuple[int, np.ndarray, int] | None:
        """Pop and return the first (lowest action step) scheduled action.

        Returns:
            Tuple of (step, action, src_control_step) or None if empty.
        """
        if not self._schedule:
            return None
        # SortedDict maintains sorted key order; pop first (lowest key) item
        step, scheduled = self._schedule.popitem(0)
        return step, scheduled.action, scheduled.src_control_step

    def get_masking_chunk_spans(
        self, *, current_step: int, max_len: int
    ) -> list[tuple[int, int, int]] | None:
        """Get list of (src_control_step, start_idx, end_idx) spans for RTC masking prefix.

        This returns information needed to look up raw actions in the server's cache
        (keyed by src_control_step).  The offset within a cached chunk is computed as
        ``step - scheduled.chunk_start_step``.

        The prefix covers both hard mask and soft mask regions.
        Handles prefixes that span multiple source chunks due to merging.

        Args:
            current_step: The current action step being executed.
            max_len: Total number of actions to include (d + epsilon).

        Returns:
            List of (src_control_step, start_idx, end_idx) tuples in execution order,
            or None if empty.  Each tuple specifies a contiguous slice from a cached
            chunk on the server.
        """
        if max_len <= 0:
            return None

        chunks: list[tuple[int, int, int]] = []
        current_src_control_step: int | None = None
        current_start: int | None = None
        current_end: int = 0
        count = 0

        for step, scheduled in self._schedule.items():
            if step <= current_step:
                continue

            # Index of this action within its source chunk (offset by chunk_start_step)
            chunk_idx = step - scheduled.chunk_start_step

            if current_src_control_step is None:
                # First action in prefix
                current_src_control_step = scheduled.src_control_step
                current_start = chunk_idx
                current_end = chunk_idx + 1
            elif scheduled.src_control_step == current_src_control_step and chunk_idx == current_end:
                # Contiguous with current span (same source, consecutive index)
                current_end = chunk_idx + 1
            else:
                # New span - save current and start new
                if current_start is not None:
                    chunks.append((current_src_control_step, current_start, current_end))
                current_src_control_step = scheduled.src_control_step
                current_start = chunk_idx
                current_end = chunk_idx + 1

            count += 1
            if count >= max_len:
                break

        # Save final span
        if current_src_control_step is not None and current_start is not None:
            chunks.append((current_src_control_step, current_start, current_end))

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
        src_control_step: int,
        chunk_start_step: int,
        current_action_step: int,
        logger: logging.Logger | None = None,
    ) -> MergeStats:
        """Merge incoming actions using freshest-observation-wins strategy.

        Args:
            incoming_actions: List of TimedAction from the server.
            src_control_step: The control-loop tick t that produced this chunk (freshness key).
            chunk_start_step: The action step n_k where this chunk starts.
            current_action_step: The most recently executed action step (n*).
            logger: Optional logger for debug output.

        Returns:
            MergeStats with L2 discrepancy metrics for overlapping actions.
        """
        # Use counters instead of per-action logging to avoid ~1ms per log call
        stale_count = 0
        frozen_count = 0
        inserted_count = 0
        updated_count = 0

        # Track L2 discrepancy for overlapping actions (non-hard-masked)
        l2_distances: list[float] = []

        for timed_action in incoming_actions:
            step = timed_action.get_action_step()
            action = timed_action.get_action()

            # Skip stale actions (already executed)
            if step <= current_action_step:
                stale_count += 1
                continue

            existing = self._schedule.get(step)

            if existing is None:
                # New action step: always schedule it, even if it's in the hard mask window.
                # The hard-mask invariant only prevents *modifying* already-scheduled actions.
                self._schedule[step] = ScheduledAction(
                    action=action, src_control_step=src_control_step, chunk_start_step=chunk_start_step
                )
                inserted_count += 1
                continue

            # Compute L2 discrepancy for ALL overlapping actions (for analysis metrics)
            # This includes hard-masked actions - we want to measure what the discrepancy would be
            old_arr = np.asarray(existing.action, dtype=np.float32).reshape(-1)
            new_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if old_arr.shape == new_arr.shape and old_arr.size > 0:
                l2 = float(np.linalg.norm(new_arr - old_arr))
                l2_distances.append(l2)

            if src_control_step > existing.src_control_step:
                # Fresher observation wins (only for non-hard-masked actions)
                self._schedule[step] = ScheduledAction(
                    action=action, src_control_step=src_control_step, chunk_start_step=chunk_start_step
                )
                updated_count += 1

        # Single summary log instead of per-action logs (saves ~20ms for 23 log calls)
        if logger and (stale_count or frozen_count):
            logger.debug(
                f"Merge stats: {stale_count} stale, {frozen_count} hard-masked, "
                f"{inserted_count} inserted, {updated_count} updated"
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

    def clear(self) -> None:
        """Clear all scheduled actions."""
        self._schedule.clear()

@dataclass
class ObservationRequest:
    """Request for an observation capture, sent from main thread to obs sender.

    Attributes:
        control_step: The control-loop tick t when this request was made (LWW key).
        chunk_start_step: The action step n_k where the resulting chunk should start.
        task: The task description string.
    """

    control_step: int
    chunk_start_step: int
    task: str
    rtc_meta: dict[str, Any] | None = None


@dataclass
class ReceivedActionChunk:
    """Action chunk received from the server with metadata.

    Attributes:
        actions: List of TimedAction from the server.
        src_control_step: The control-loop tick t that produced this chunk.
        chunk_start_step: The action step n_k where this chunk starts.
        measured_latency: Measured round-trip time for this chunk.
    """

    actions: list[TimedAction]
    src_control_step: int
    chunk_start_step: int
    measured_latency: float

class RobotClientImproved:
    """Latency-adaptive asynchronous inference robot client.

    This implementation follows the latency-adaptive async inference algorithm with:
    - 3-thread architecture (main, observation sender, action receiver)
    - SPSC last-write-wins registers for thread communication
    - Jacobson-Karels latency estimation
    - Cool-down mechanism for inference triggering
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

        # Use mock robot when no physical robot is available
        if config.use_mock_robot:
            self.robot = MockRobot()
            self.robot.connect()
            # Mock features for simulation
            lerobot_features = {
                "observation.state": list(self.robot.state_features),
                "action": list(self.robot.action_features),
            }
        else:
            self.robot = make_robot_from_config(config.robot)
            self.robot.connect()
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Drop simulators for experiments
        self._obs_drop_sim = DropSimulator(config=config.drop_obs_config)
        self._action_drop_sim = DropSimulator(config=config.drop_action_config)

        # Duplicate simulators for experiments
        self._obs_dup_sim = DuplicateSimulator(config=config.dup_obs_config)
        self._action_dup_sim = DuplicateSimulator(config=config.dup_action_config)

        # Reorder simulators for experiments (hold-and-swap)
        self._obs_reorder_sim = ReorderSimulator(config=config.reorder_obs_config)
        self._action_reorder_sim = ReorderSimulator(config=config.reorder_action_config)

        self.server_address = config.server_address
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rtc_enabled=config.rtc_enabled,
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

        # Shutdown coordination
        self.shutdown_event = threading.Event()
        self._active_action_stream: grpc.Future | None = None  # Cancel on stop to unblock action_receiver

        # Action state: n(t), initialized to -1 per algorithm.
        # Note: Only the main control loop thread reads/writes action_step.
        self.action_step: int = -1

        # Control-loop tick counter t ∈ ℕ (monotone, incremented every tick).
        # Used as the LWW logical clock so that dropped messages never stall watermarks.
        self.control_step: int = 0

        # Latency estimation (configurable: JK or max_last_10)
        # Upper bound: d <= H/2 per RTC constraint (with s = d, d <= H - s becomes d <= H/2)
        self.latency_estimator = make_latency_estimator(
            kind=config.latency_estimator_type,
            fps=config.fps,
            alpha=config.latency_alpha,
            beta=config.latency_beta,
            k=config.latency_k,
            action_chunk_size=config.actions_per_chunk,
        )

        # Action schedule (replaces Queue with OrderedDict)
        self.action_schedule = ActionSchedule()

        # Cool-down counter O^c(t).
        # Note: Only the main control loop thread reads/writes obs_cooldown.
        self.obs_cooldown: int = 0

        # SPSC Mailboxes (one-slot queues)
        # Observation request register: main thread -> observation sender
        self._obs_request_reg: LWWRegister[ObservationRequest | None] = LWWRegister(
            initial_control_step=-1, initial_value=None
        )
        self._obs_request_reader: LWWReader[ObservationRequest | None] = self._obs_request_reg.reader(
            initial_watermark=-1
        )

        # Action register: action receiver -> main thread
        self._action_reg: LWWRegister[ReceivedActionChunk | None] = LWWRegister(
            initial_control_step=-1, initial_value=None
        )
        self._action_reader: LWWReader[ReceivedActionChunk | None] = self._action_reg.reader()

        # Synchronization barrier for thread startup
        self.start_barrier = threading.Barrier(3)  # 3 threads: main, obs sender, action receiver

        # Debug tracking (bounded to ~10 min at control rate to prevent unbounded growth)
        _max_queue_history = self.config.fps * 600  # 10 minutes
        self.action_queue_sizes: deque[int] = deque(maxlen=_max_queue_history)

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Metrics (two categories):
        # - experiment: written to disk (CSV + trajectory JSON) when metrics_path is set
        # - diagnostic: periodic console output (avg/max timings) when enabled
        diag = DiagnosticMetrics(
            fps=config.fps,
            window_s=config.metrics_diagnostic_window_s,
            interval_s=config.metrics_diagnostic_interval_s,
            enabled=config.metrics_diagnostic_enabled,
            verbose=config.metrics_diagnostic_verbose,
            prefix="DIAG",
        )
        diag.start()

        exp: ExperimentMetricsWriter | None = None
        if config.metrics_path:
            exp = ExperimentMetricsWriter(path=config.metrics_path)

        self._metrics = Metrics(experiment=exp, diagnostic=diag)

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

            # WebSocket client for sending executed actions directly to viz server
            self._trajectory_viz_client = TrajectoryVizClient(ws_url=config.trajectory_viz_ws_url)
            self._trajectory_viz_client.start()

        # Action filter state (for adaptive_lowpass and hold_stable modes)
        self._filter_prev_action: np.ndarray | None = None

        # Butterworth filter state (second-order sections for numerical stability)
        self._butter_sos: np.ndarray | None = None
        self._butter_zi: np.ndarray | None = None  # Filter state per joint

        # Action filter (class-based, with optional hard-mask lookahead)
        self._action_filter: ActionFilter = self._create_action_filter()

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def current_action_step(self) -> int:
        """Get the most recently executed action step n*(t).

        Note: Only the main control loop thread should access this property.
        """
        return max(self.action_step, -1)

    def _create_action_filter(self) -> ActionFilter:
        """Create the action filter based on configuration.

        Returns:
            Configured ActionFilter instance.
        """
        cfg = self.config
        mode = cfg.action_filter_mode

        if mode == "none":
            return NoFilter()
        elif mode == "adaptive_lowpass":
            return AdaptiveLowpassFilter(
                alpha_min=cfg.action_filter_alpha_min,
                alpha_max=cfg.action_filter_alpha_max,
                deadband=cfg.action_filter_deadband,
            )
        elif mode == "hold_stable":
            return HoldStableFilter(deadband=cfg.action_filter_deadband)
        elif mode == "butterworth":
            return ButterworthFilter(
                cutoff=cfg.action_filter_butterworth_cutoff,
                order=cfg.action_filter_butterworth_order,
                fps=cfg.fps,
                gain=cfg.action_filter_gain,
                use_lookahead=cfg.action_filter_use_frozen_lookahead,
                past_buffer_size=cfg.action_filter_past_buffer_size,
                lookahead_blend=cfg.action_filter_lookahead_blend,
            )
        elif mode == "median":
            return MedianFilter(
                past_buffer_size=cfg.action_filter_past_buffer_size,
                use_lookahead=cfg.action_filter_use_frozen_lookahead,
            )
        else:
            return NoFilter()

    def _peek_frozen_actions(self) -> list[np.ndarray]:
        """Peek at hard-masked scheduled actions without consuming them.

        Returns actions scheduled between current_step+1 and current_step+latency_steps.
        These are guaranteed not to be overwritten by new inference results.

        Returns:
            List of hard-masked action arrays.
        """
        result = []
        current = self.action_step
        frozen_limit = current + self.latency_estimator.estimate_steps

        for step, scheduled in self.action_schedule._schedule.items():
            if step > current and step <= frozen_limit:
                result.append(scheduled.action)

        return result

    def start(self) -> bool:
        """Start the robot client and connect to the policy server."""
        try:
            t_total_start = time.perf_counter()

            # Server handshake
            t_ready_start = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            t_ready_done = time.perf_counter()
            self._metrics.diagnostic.timing_s("ready_rpc_ms", t_ready_done - t_ready_start)

            # Send policy configuration
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            t_policy_rpc_start = time.perf_counter()
            self.stub.SendPolicyInstructions(policy_setup)
            t_policy_rpc_done = time.perf_counter()
            self._metrics.diagnostic.timing_s("policy_rpc_ms", t_policy_rpc_done - t_policy_rpc_start)

            self.shutdown_event.clear()

            # Initialize latency estimate to a fixed starting value.
            # We intentionally avoid latency "priming" RPCs (which add startup latency and
            # interact poorly with monotone mailbox semantics).
            self.latency_estimator.update(0.5)
            self._metrics.diagnostic.timing_s("client_init_total_ms", time.perf_counter() - t_total_start)

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self) -> None:
        """Stop the robot client."""
        self.shutdown_event.set()

        # Flush experiment metrics if enabled (disk output; behavior unchanged)
        if self._metrics.experiment is not None and self.config.metrics_path:
            self._metrics.experiment.flush(self.config.metrics_path)

        # Stop trajectory viz client if enabled
        if self._trajectory_viz_client is not None:
            self._trajectory_viz_client.stop()

        self.robot.disconnect()

        self.channel.close()
        self._metrics.diagnostic.stop()

    def signal_stop(self) -> None:
        """Signal the client to stop without disconnecting the robot.
        
        Use this when you want to stop the control loop but keep the robot
        and server connection alive for subsequent experiments.
        """
        self.shutdown_event.set()

        # Cancel active gRPC action stream so action_receiver unblocks promptly
        stream = self._active_action_stream
        if stream is not None:
            with suppress(Exception):
                stream.cancel()
            self._active_action_stream = None

        # Flush experiment metrics if enabled (disk output; behavior unchanged)
        if self._metrics.experiment is not None and self.config.metrics_path:
            self._metrics.experiment.flush(self.config.metrics_path)

    def reset_for_new_experiment(self, metrics_path: str | None = None) -> None:
        """Reset internal state for a new experiment without reconnecting.
        
        Call this between experiments when keeping the robot connected.
        The robot and policy server connection remain active.
        
        Args:
            metrics_path: Optional path for new experiment metrics CSV.
        """
        # Clear shutdown event so threads can run again
        self.shutdown_event.clear()
        self._active_action_stream = None

        # Reset action state
        self.action_step = -1
        self.control_step = 0
        self.obs_cooldown = 0
        self.action_schedule = ActionSchedule()

        # Reset registers (avoid leaking prior experiment values)
        self._obs_request_reg = LWWRegister(initial_control_step=-1, initial_value=None)
        self._action_reg = LWWRegister(initial_control_step=-1, initial_value=None)
        self._obs_request_reader = self._obs_request_reg.reader()
        self._action_reader = self._action_reg.reader()

        # Reset latency estimator with current config values
        self.latency_estimator = make_latency_estimator(
            kind=self.config.latency_estimator_type,
            fps=self.config.fps,
            alpha=self.config.latency_alpha,
            beta=self.config.latency_beta,
            k=self.config.latency_k,
            action_chunk_size=self.config.actions_per_chunk,
        )
        self.latency_estimator.update(0.5)

        # Reset experiment metrics
        if metrics_path:
            self.config.metrics_path = metrics_path
        self._metrics.experiment = (
            ExperimentMetricsWriter(path=self.config.metrics_path)
            if self.config.metrics_path
            else None
        )

        # Reset fault injection simulators for new experiment
        self._obs_drop_sim = DropSimulator(config=self.config.drop_obs_config)
        self._action_drop_sim = DropSimulator(config=self.config.drop_action_config)
        self._obs_dup_sim = DuplicateSimulator(config=self.config.dup_obs_config)
        self._action_dup_sim = DuplicateSimulator(config=self.config.dup_action_config)
        self._obs_reorder_sim = ReorderSimulator(config=self.config.reorder_obs_config)
        self._action_reorder_sim = ReorderSimulator(config=self.config.reorder_action_config)

        # Reset action filter
        self._filter_prev_action = None
        self._butter_zi = None
        self._action_filter = self._create_action_filter()

        # Reset debug tracking
        _max_queue_history = self.config.fps * 600  # 10 minutes
        self.action_queue_sizes = deque(maxlen=_max_queue_history)
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Create new barrier for thread synchronization
        self.start_barrier = threading.Barrier(3)

    # -------------------------------------------------------------------------
    # Observation Sender Thread
    # -------------------------------------------------------------------------

    def observation_sender(self) -> None:
        """Captures, encodes, and sends observations to the policy server."""
        self.start_barrier.wait()

        last_good_observation: RawObservation | None = None
        last_good_observation_time: float | None = None
        consecutive_capture_failures = 0
        reader = self._obs_request_reg.reader()
        idle_start = time.perf_counter()

        while self.running:
            try:
                state, _, is_new = reader.read_if_newer()
                request = state.value
                if not is_new or request is None:
                    time.sleep(0.01)
                    continue

                # Emit wait time (how long obs sender was idle waiting for work)
                self._metrics.diagnostic.timing_s("obs_wait_ms", time.perf_counter() - idle_start)

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
                        self._metrics.diagnostic.counter("obs_fallback_used", 1)
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
                encoded_observation, encode_stats = encode_images_for_transport(
                    raw_observation, jpeg_quality=60
                )
                t_encode_done = time.perf_counter()
                self._metrics.diagnostic.timing_s("obs_encode_ms", t_encode_done - t_encode_start)

                # Create timed observation
                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    control_step=request.control_step,
                    observation=encoded_observation,
                    chunk_start_step=request.chunk_start_step,
                )

                # Check if observation should be dropped (simulation/experiments)
                if self._obs_drop_sim.should_drop():
                    self._metrics.diagnostic.counter("obs_dropped_sim", 1)
                    continue

                # Reorder injection (hold-and-swap before send)
                obs_items = self._obs_reorder_sim.process(timed_obs)
                if not obs_items:
                    self._metrics.diagnostic.counter("obs_reorder_held", 1)
                    continue
                if len(obs_items) > 1:
                    self._metrics.diagnostic.counter("obs_reorder_swapped", 1)

                # Send each item (1 normally, 2 when a swap completes)
                t_send_start = time.perf_counter()
                for obs_item in obs_items:
                    self._send_observation(obs_item)

                    # Duplicate injection (after send)
                    if self._obs_dup_sim.should_duplicate():
                        self._send_observation(obs_item)
                        self._metrics.diagnostic.counter("obs_duplicated_sim", 1)
                t_send_done = time.perf_counter()
                self._metrics.diagnostic.timing_s("obs_capture_ms", t_capture_done - t_capture_start)
                self._metrics.diagnostic.timing_s("obs_send_ms", t_send_done - t_send_start)
                idle_start = time.perf_counter()

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
                    self._metrics.diagnostic.counter("trajectory_chunk_send_rpc_error", 1)

            except Exception as e:
                self._metrics.diagnostic.counter("trajectory_chunk_sender_error", 1)

    def _queue_trajectory_chunk(
        self,
        src_control_step: int,
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
            source_step=src_control_step,
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
        last_chunk_time: float | None = None
        while self.running:
            try:
                t_rpc_start = time.perf_counter()
                stream = self.stub.StreamActionsDense(services_pb2.Empty())
                self._active_action_stream = stream  # Store for cancellation on stop
                t_rpc_done = time.perf_counter()
                self._metrics.diagnostic.timing_s("rpc_ms", t_rpc_done - t_rpc_start)

                for dense in stream:
                    if not self.running:
                        break
                    t_chunk_received = time.perf_counter()
                    # Emit chunk gap timing (time since last chunk)
                    if last_chunk_time is not None:
                        self._metrics.diagnostic.timing_s("chunk_gap_ms", t_chunk_received - last_chunk_time)
                    last_chunk_time = t_chunk_received

                    # Reorder injection (hold-and-swap before handle)
                    dense_items = self._action_reorder_sim.process(dense)
                    if not dense_items:
                        self._metrics.diagnostic.counter("action_reorder_held", 1)
                        continue
                    if len(dense_items) > 1:
                        self._metrics.diagnostic.counter("action_reorder_swapped", 1)

                    for dense_item in dense_items:
                        self._handle_actions_dense(dense_item, rpc_ms=0.0)

                        # Duplicate injection (after handle)
                        if self._action_dup_sim.should_duplicate():
                            self._handle_actions_dense(dense_item, rpc_ms=0.0)
                            self._metrics.diagnostic.counter("action_chunk_duplicated_sim", 1)

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

        num_actions = int(dense.num_actions)
        action_dim = int(dense.action_dim)
        if num_actions <= 0 or action_dim <= 0:
            return

        t_deser_start = time.perf_counter()
        actions = np.frombuffer(dense.actions_f32, dtype=np.float32)
        if actions.size != num_actions * action_dim:
            raise ValueError(
                f"ActionsDense buffer size mismatch: {actions.size} != {num_actions*action_dim}"
            )
        actions = actions.reshape(num_actions, action_dim)
        t_deser_done = time.perf_counter()

        timestamp = float(dense.timestamp)
        source_control_step = int(dense.source_control_step)
        chunk_start_step = int(dense.chunk_start_step)
        dt = float(dense.dt)

        measured_latency = receive_time - timestamp
        timed_actions = [
            TimedAction(
                timestamp=timestamp + i * dt,
                control_step=source_control_step,
                action_step=chunk_start_step + i,
                action=actions[i],
            )
            for i in range(num_actions)
        ]

        self._metrics.diagnostic.timing_ms("rpc_ms", rpc_ms)
        self._metrics.diagnostic.timing_s("deser_ms", t_deser_done - t_deser_start)
        self._metrics.diagnostic.timing_s("total_latency_rtt_ms", measured_latency)

        # Check if action chunk should be dropped (simulation/experiments)
        if self._action_drop_sim.should_drop():
            self._metrics.diagnostic.counter("action_chunk_dropped_sim", 1)
            return

        self._publish_received_actions(
            timed_actions=timed_actions,
            src_control_step=source_control_step,
            chunk_start_step=chunk_start_step,
            measured_latency=measured_latency,
        )

    def _publish_received_actions(
        self,
        *,
        timed_actions: list[TimedAction],
        src_control_step: int,
        chunk_start_step: int,
        measured_latency: float,
    ) -> None:
        chunk = ReceivedActionChunk(
            actions=timed_actions,
            src_control_step=src_control_step,
            chunk_start_step=chunk_start_step,
            measured_latency=measured_latency,
        )
        self._action_reg.update_if_newer(control_step=src_control_step, value=chunk)

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

        task = task or self.config.task

        prev_loop_start: float | None = None
        next_tick: float | None = time.perf_counter() if self.config.control_use_deadline_clock else None

        while self.running:
            t_loop_start = time.perf_counter()
            if prev_loop_start is not None:
                self._metrics.diagnostic.timing_s("loop_dt_ms", t_loop_start - prev_loop_start)
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

                    # Apply action filter to reduce jitter from policy micro-updates
                    frozen = self._peek_frozen_actions() if self.config.action_filter_use_frozen_lookahead else []
                    ctx = FilterContext(action=action, frozen_actions=frozen)
                    filtered_action = self._action_filter.apply(ctx)

                    t_send_start = time.perf_counter()
                    self.robot.send_action(self._action_array_to_dict(filtered_action))
                    t_send_done = time.perf_counter()

                    # Keep action_step aligned with the schedule's action-step keys.
                    # Only the main control loop thread writes this.
                    self.action_step = step
                    self._metrics.diagnostic.timing_s("send_action_ms", t_send_done - t_send_start)

                    # Stream executed action to the visualization server (best-effort).
                    if self._trajectory_viz_client is not None:
                        self._trajectory_viz_client.on_executed_action(
                            EvExecutedAction(
                                step=step,
                                action=filtered_action.tolist(),
                                timestamp=time.time(),
                            )
                        )

                    # Record executed action for experiment trajectory visualization
                    if self._metrics.experiment is not None:
                        self._metrics.experiment.record_executed_action(
                            step=step,
                            action=filtered_action,
                        )

            t_phase1_end = time.perf_counter()
            _phase_exec_ms = self._ms(t_phase1_end - t_phase1_start)

            # Track queue size for debugging and starvation detection
            schedule_size = self.action_schedule.get_size()
            self.action_queue_sizes.append(schedule_size)
            is_starved = schedule_size == 0
            if is_starved:
                self._metrics.diagnostic.counter("starvation", 1)

            # ---------------------------------------------------------------------
            # Step 2: Check inference trigger condition
            # ---------------------------------------------------------------------
            t_phase2_start = time.perf_counter()
            latency_steps = self.latency_estimator.estimate_steps
            epsilon = self.config.epsilon
            s_min = self.config.s_min
            H = self.config.actions_per_chunk


            trigger_threshold = H - s_min
            if self.config.cooldown_enabled:
                should_trigger = schedule_size <= trigger_threshold and self.obs_cooldown == 0
            else:
                # Classic async baseline: always trigger when schedule is low
                should_trigger = schedule_size <= trigger_threshold

            if should_trigger:
                current_step = self.current_action_step

                # Clamp to 0 so the server produces chunks starting at 0 on startup (consistent with the
                # original async inference implementation that uses max(latest_action, 0)).
                rtc_meta: dict[str, Any] | None = None
                if self.config.rtc_enabled:
                    t_rtc_start = time.perf_counter()

                    # RTC paper: effective execution horizon is s = max(s_min, d)
                    # - d = latency_steps = hard mask region (weight 1.0)
                    # - overlap_end = H - s = where fresh region starts
                    # - Soft mask region: [d, overlap_end) with decaying weight
                    d = int(latency_steps)
                    s = max(s_min, d)  # Effective execution horizon
                    overlap_end = H - s  # Where fresh region starts

                    # Get masking spans from schedule (handles multi-chunk prefixes)
                    # Returns list of (src_step, start_idx, end_idx) for server cache lookup
                    prefix_chunks = self.action_schedule.get_masking_chunk_spans(
                        current_step=current_step, max_len=overlap_end
                    )

                    rtc_meta = {
                        "enabled": True,
                        "latency_steps": d,  # Hard mask region [0, d)
                        "prefix_chunks": prefix_chunks,  # List of (src_step, start, end) or None
                        "overlap_end": overlap_end,  # H - max(s_min, d): where fresh region starts
                    }
                    t_rtc_end = time.perf_counter()
                    self._metrics.diagnostic.timing_s("rtc_build_ms", t_rtc_end - t_rtc_start)

                request = ObservationRequest(
                    control_step=self.control_step,
                    chunk_start_step=max(current_step, 0),
                    task=task,
                    rtc_meta=rtc_meta,
                )

                # Always reset cooldown when trigger fires (before attempting put)
                # Cooldown = latency_steps + epsilon (buffer to prevent over-triggering)
                if self.config.cooldown_enabled:
                    self.obs_cooldown = latency_steps + epsilon

                # Publish newest request (monotone w.r.t. control_step t)
                self._obs_request_reg.update_if_newer(control_step=request.control_step, value=request)

                _tick_obs_sent = True
                self._metrics.diagnostic.counter("obs_triggered", 1)
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
            state, _, is_new = self._action_reader.read_if_newer()
            chunk = state.value
            if is_new and chunk is not None:

                current_step = self.current_action_step
                latency_steps = self.latency_estimator.estimate_steps

                # Update latency estimate
                self.latency_estimator.update(chunk.measured_latency)

                # Merge actions into schedule
                merge_stats = self.action_schedule.merge(
                    incoming_actions=chunk.actions,
                    src_control_step=chunk.src_control_step,
                    chunk_start_step=chunk.chunk_start_step,
                    current_action_step=current_step,
                )

                new_estimate = self.latency_estimator.estimate_steps
                _tick_action_received = True
                _tick_measured_latency_ms = self._ms(chunk.measured_latency)

                # Track discrepancy stats from the merge
                _tick_chunk_overlap_count = merge_stats.overlap_count
                _tick_chunk_mean_l2 = merge_stats.mean_l2
                _tick_chunk_max_l2 = merge_stats.max_l2

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
                        src_control_step=chunk.src_control_step,
                        actions=actions_arrays,
                        frozen_len=latency_steps,
                    )

                # Record chunk for experiment trajectory visualization
                if self._metrics.experiment is not None and chunk.actions:
                    actions_arrays = [ta.action for ta in chunk.actions]
                    self._metrics.experiment.record_chunk(
                        src_control_step=chunk.src_control_step,
                        actions=actions_arrays,
                        frozen_len=int(latency_steps),
                    )

            t_phase3_end = time.perf_counter()
            _phase_merge_ms = self._ms(t_phase3_end - t_phase3_start)

            # Diagnostic phase timings (avg/max only; printed periodically by DiagnosticMetrics)
            self._metrics.diagnostic.timing_ms("phase_exec_ms", _phase_exec_ms)
            self._metrics.diagnostic.timing_ms("phase_trigger_ms", _phase_trigger_ms)
            self._metrics.diagnostic.timing_ms("phase_merge_ms", _phase_merge_ms)

            # Advance the control-loop clock (always monotone, even when no action executes)
            self.control_step += 1

            # ---------------------------------------------------------------------
            # Step 4: Maintain control frequency
            # ---------------------------------------------------------------------
            elapsed = time.perf_counter() - t_loop_start
            if next_tick is None:
                sleep_s = max(0.0, self.config.environment_dt - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    self._metrics.diagnostic.counter("overrun", 1)
            else:
                # Deadline-based clock: reduces drift and jitter when occasional overruns happen.
                next_tick += self.config.environment_dt
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # If we're behind, count an overrun and re-anchor to now to avoid runaway lag.
                    self._metrics.diagnostic.counter("overrun", 1)
                    next_tick = now

            self._metrics.diagnostic.set_context(
                step=self.current_action_step,
                schedule_size=self.action_schedule.get_size(),
                latency_steps=self.latency_estimator.estimate_steps,
                cooldown=self.obs_cooldown,
            )

            # Record experiment metrics for this tick
            if self._metrics.experiment is not None:
                self._metrics.experiment.record_tick(
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


def async_client_improved(cfg: RobotClientImprovedConfig) -> None:
    """Run the improved async inference client."""

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


if __name__ == "__main__":
    import draccus

    draccus.wrap()(async_client_improved)()
