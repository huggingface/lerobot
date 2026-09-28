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

"""Real-Time Chunking inference engine.

A background thread produces action chunks asynchronously via
:meth:`policy.predict_action_chunk`.  The main control loop polls
``get_action`` for the next ready action; observations flow the other
way via ``notify_observation``.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import traceback
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import Any, Protocol, cast

import torch

from lerobot.inference.contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot
from lerobot.inference.execution import ChunkRuntime, trained_overlap_valid
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine, PolicyQuery, QueryKind

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Consecutive unusable trained-RTC chunks tolerated before declaring the delay unsupportable.
_RTC_MAX_CONSECUTIVE_DISCARDS: int = 5
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


class _FatalRTCInferenceError(RuntimeError):
    """Base class for RTC errors that cannot become valid after a retry."""


class _TrainedRTCDelayExceededError(_FatalRTCInferenceError):
    """Raised when measured latency persistently exceeds a trained RTC checkpoint's support."""


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


class _RTCPredictActionChunk(Protocol):
    """Call shape of ``predict_action_chunk`` on an RTC-capable policy."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        *,
        inference_delay: int | None,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> torch.Tensor: ...


def supports_rtc_inference(policy: PreTrainedPolicy) -> bool:
    """Whether a policy declares RTC support and accepts the RTC call shape."""
    supports_rtc = getattr(policy, "supports_rtc", None)
    if not callable(supports_rtc) or not supports_rtc():
        return False

    try:
        inspect.signature(policy.predict_action_chunk).bind(
            object(),
            inference_delay=0,
            prev_chunk_left_over=None,
        )
    except (TypeError, ValueError):
        return False
    return True


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad (holding the last action) or truncate RTC prefix actions to a fixed length.

    Zero-padding would decode to the dataset mean inside the RTC guided region.
    """
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, _ = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    if steps == 0:
        raise ValueError("Cannot pad an empty prefix: no last action to hold.")
    hold = prev_actions[-1:].expand(target_steps - steps, -1)
    return torch.cat([prev_actions, hold], dim=0)


def _trained_rtc_chunk_can_merge(
    *,
    conditioned_delay: int,
    measured_delay: int,
    training_max_delay: int,
    has_previous_actions: bool,
) -> bool:
    """Whether a trained RTC chunk still covers the overlap that actually elapsed.

    A chunk is unusable either because inference outran the prefix it was conditioned on, or
    because the elapsed delay left the range the checkpoint was trained for. Both are transient
    by nature (a latency spike), so this reports them the same way and lets the caller retry;
    only a persistent run of unusable chunks is fatal.
    """
    return trained_overlap_valid(conditioned_delay, measured_delay, training_max_delay, has_previous_actions)


def _estimate_rtc_delay(
    *,
    latency: float,
    time_per_step: float,
    mode: str,
    training_max_delay: int,
    has_previous_actions: bool,
) -> int:
    """Estimate overlap, using the trained capacity to bootstrap the first transition."""
    if latency:
        return math.ceil(latency / time_per_step)
    if mode == "trained" and has_previous_actions:
        return training_max_delay
    return 0


def _clamp_trained_rtc_delay(*, conditioned_delay: int, available_steps: int, training_max_delay: int) -> int:
    """Clamp the hard prefix to what both the checkpoint and the queue can back.

    Past ``training_max_delay`` the model has never seen a prefix that long, and past
    ``available_steps`` ``_normalize_prev_actions_length`` pads the tail by holding the last
    action, so the extra steps would be inpainted as if a frozen hold had been committed.
    Clamping keeps the chunk usable; ``_trained_rtc_chunk_can_merge`` still discards it if the
    delay that actually elapsed outran this prefix.
    """
    clamped = min(conditioned_delay, training_max_delay, available_steps)
    if clamped < conditioned_delay:
        logger.warning(
            "Trained RTC wanted a %d-step prefix but the checkpoint supports %d and the queue "
            "holds %d committed actions; conditioning on %d. Raise --inference.queue_threshold "
            "and --inference.rtc.execution_horizon, or retrain with a larger "
            "--policy.rtc_training_max_delay, to keep the full overlap.",
            conditioned_delay,
            training_max_delay,
            available_steps,
            clamped,
        )
    return clamped


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Async RTC inference: a background thread produces action chunks.

    ``get_action`` pops the next action from the shared queue (or
    returns ``None`` if the queue is empty).  The main loop should call
    ``notify_observation`` every tick and ``pause``/``resume`` around
    human-intervention phases.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        dataset_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
        max_observation_age_s: float = 5.0,
        action_timeout_s: float = 10.0,
        startup_timeout_s: float = 120.0,
        language_timeout_s: float = 120.0,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._policy_spec = policy.chunk_inference_spec()
        if not self._policy_spec.current_observation_only:
            raise ValueError("Local asynchronous inference does not implement temporal observation sampling")
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        # Same feature spec sync uses, so both engines order observation.state identically.
        self._obs_features = dataset_features
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold
        if not math.isfinite(language_timeout_s) or language_timeout_s <= 0:
            raise ValueError("Language deadline must be finite and positive")
        self._language_timeout_s = language_timeout_s
        self._action_timeout_s = action_timeout_s
        mode = ExecutionMode(f"rtc_{rtc_config.mode}") if rtc_config.enabled else ExecutionMode.CHUNK
        if mode not in self._policy_spec.modes:
            raise ValueError(f"Policy does not support asynchronous execution mode {mode.value!r}")
        self._runtime = ChunkRuntime(
            mode=mode,
            action_interval=1 / fps,
            refill_seconds=max(0, rtc_queue_threshold) / fps,
            max_observation_age_s=max_observation_age_s,
            # Cold compilation uses the startup budget; it is not an ordinary
            # action-latency measurement or a steady-state request deadline.
            action_timeout_s=startup_timeout_s if use_torch_compile else action_timeout_s,
            startup_timeout_s=startup_timeout_s,
            training_max_delay=self._policy_spec.training_max_delay,
        )
        self._runtime.queue.cfg = rtc_config
        self._reset_pending = False
        self._hold_requested = False
        self._hold_acknowledged = Event()
        self._language_deadline: float | None = None
        self._active_query_generation = 0
        self._language_preprocessor, self._language_postprocessor = deepcopy((preprocessor, postprocessor))
        if isinstance(robot_wrapper, ThreadSafeRobot):
            robot_wrapper.configure_position_hold()

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        # Bumped by reset() under _obs_lock, so a chunk whose inference started before a
        # reset is discarded instead of merged into the fresh queue.
        self._reset_epoch = 0
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._failure_traceback: str | None = None
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None
        self._started = False

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )

        # Processor introspection for relative-action re-anchoring.
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once torch.compile warmup is complete (or immediately if compile is disabled)."""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set() or self._runtime.failure is not None

    @property
    def failure_traceback(self) -> str | None:
        """Traceback captured when the RTC thread died (see ``failed``).

        Kept as data, not just logged, so consumers can re-surface it when someone looks.
        """
        return self._failure_traceback or self._runtime.failure

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        if self._started:
            raise RuntimeError("RTC inference engines cannot be restarted; create a new session")
        self._started = True
        self._action_queue = self._runtime.queue
        self._runtime.invalidate()
        self._obs_holder = {
            "obs": None,
            "robot_type": self._robot.robot_type,
        }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        self._runtime.active = False
        self._runtime.invalidate(held=True)
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            else:
                logger.info("RTC inference thread stopped")
            self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread."""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()
        self._runtime.active = False
        self._runtime.invalidate(held=True)
        with self._obs_lock:
            self._obs_holder["obs"] = None

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        if self.failed:
            return
        self._runtime.active = True
        self._runtime.held = self._hold_requested
        self._runtime.started_at = time.monotonic()
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue.

        Safe to call with the RTC thread paused or running.  Also drops the last published
        observation — a chunk computed from a stale one would jerk the robot toward an old
        pose — and bumps the reset epoch so an in-flight chunk is discarded instead of
        merged into the cleared queue.
        """
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self.drop_pending_query()
        with self._obs_lock:
            # Clear and bump in one critical section, mirroring _rtc_loop's epoch
            # check-and-merge, so a reset cannot leak a pre-reset chunk into the fresh
            # queue.  Lock order is _obs_lock -> queue.lock on both sides.
            self._runtime.invalidate()
            self._obs_holder["obs"] = None
            self._reset_epoch += 1
            self._reset_pending = True
            self._hold_requested = False
            self._hold_acknowledged.clear()
            self._language_deadline = None
        # The queue is empty, so a pending task change has nothing stale to blend against.
        self._discard_task_change()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
        if self._action_queue is self._runtime.queue:
            queued_action = self._runtime.pop()
            if queued_action is None:
                self._surface_fault()
                return None
            action, provenance = queued_action
            self._set_dispatched_task(provenance.task)
            return action
        queued = self._action_queue.get_with_task()
        if queued is None:
            return None
        # The queue pairs each action with its chunk's task under the queue lock, so a
        # concurrent merge cannot cross labels between chunks.
        action, task = queued
        if task is None:
            # Every merge here labels its chunk, so a missing label means a foreign
            # writer: fail loudly rather than corrupt dispatched_task and frame labels.
            raise RuntimeError("RTC action queue returned an action without task provenance")
        self._set_dispatched_task(task)
        return action

    def notify_observation(self, obs: dict) -> None:
        """Publish an owned snapshot and its original control-thread capture time."""
        captured_at = getattr(self._robot, "observation_time", None)
        if captured_at is None:
            captured_at = time.monotonic()
        owned = deepcopy(obs)
        with self._obs_lock:
            self._obs_holder["obs"] = owned
            self._obs_holder["capture_time"] = captured_at

    def _surface_fault(self) -> None:
        if self._runtime.failure is not None:
            self._rtc_error.set()
            self._failure_traceback = self._runtime.failure

    def dispatch_allowed(self) -> bool:
        """Revoke dispatch on every motor tick, including interpolation-only ticks."""
        if self.has_pending_query and self._policy_active.is_set():
            self._request_language_hold()
        if self._language_deadline is not None and time.monotonic() > self._language_deadline:
            self._runtime.fault("Language request deadline exceeded; restart the inference session")
        permitted = self._runtime.dispatch_allowed()
        self._surface_fault()
        return permitted and self.ready and not self._hold_requested

    def acknowledge_hold(self) -> None:
        """Let the worker submit text only after the control thread held the robot."""
        with self._obs_lock:
            if self._hold_requested and not self._hold_acknowledged.is_set():
                self._obs_holder["obs"] = None
                self._hold_acknowledged.set()
        if self.failed and self._global_shutdown_event is not None:
            self._global_shutdown_event.set()

    def _request_language_hold(self) -> None:
        with self._obs_lock:
            if not self._hold_requested:
                self._runtime.invalidate(held=True)
                self._hold_requested = True
                self._hold_acknowledged.clear()
                self._language_deadline = time.monotonic() + self._language_timeout_s

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation() and bool(getattr(self._robot, "supports_hold", False))

    def _queue_query(self, query: PolicyQuery) -> bool:
        if not self.supports_text_queries or self.failed:
            return False
        queued = super()._queue_query(query)
        if queued and self._policy_active.is_set():
            self._request_language_hold()
        return queued

    def pump_query(self, obs_processed: dict | None = None) -> bool:
        """Request a hold as soon as the control-thread autosteer sequencer queues text."""
        served = super().pump_query(obs_processed)
        if self.has_pending_query and self._policy_active.is_set() and not self.failed:
            self._request_language_hold()
        return served

    @property
    def control_thread_owns_policy(self) -> bool:
        """The RTC background thread owns the policy; it services queries in ``_rtc_loop``."""
        return False

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head.  Called on the RTC thread (see ``_rtc_loop``)."""
        obs_batch = build_dataset_frame(self._obs_features, obs_processed, prefix="observation")
        # Live task, read without consuming the task-changed edge: that belongs to the
        # chunk path.
        task = self.task
        obs_batch = prepare_observation_for_inference(
            obs_batch, torch.device(self._device), task, self._robot.robot_type
        )
        obs_batch = self._mark_query(obs_batch, query)
        generation = self._runtime.generation
        self._active_query_generation = generation
        preprocessed = self._language_preprocessor(obs_batch)
        with torch.inference_mode():
            # No str() coercion: _service_query validates the return value.
            result = self._policy.generate_text(preprocessed)
        if self.failed or generation != self._runtime.generation:
            raise RuntimeError("Language result belongs to an invalidated execution generation")
        if isinstance(result, str) and len(result) > 8192:
            raise ValueError("Language result exceeds the configured output bound")
        return result

    def _query_context_valid(self, query: PolicyQuery) -> bool:
        return (
            not self.failed
            and self._policy_active.is_set()
            and self._active_query_generation == self._runtime.generation
            and query.task_version == self.task_version
            and (
                query.kind is not QueryKind.NEXT_SUBTASK
                or query.intent_generation == self.query_intent_generation
            )
        )

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Own all policy/processor mutations and use the shared chunk runtime."""
        try:
            policy_device = torch.device(self._device)
            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0
            consecutive_discards = 0
            while not self._shutdown_event.is_set() and not self.failed:
                # Reset is ordered behind any model call already executing. The
                # control thread has already revoked its old generation locally.
                with self._obs_lock:
                    reset_pending = self._reset_pending
                    self._reset_pending = False
                if reset_pending:
                    self._policy.reset()
                    self._preprocessor.reset()
                    self._postprocessor.reset()
                    self._language_preprocessor.reset()
                    self._language_postprocessor.reset()
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue
                if self.has_pending_query:
                    self._request_language_hold()
                if self._hold_requested:
                    if not self._hold_acknowledged.is_set():
                        time.sleep(_RTC_IDLE_SLEEP_S)
                        continue
                    with self._obs_lock:
                        obs = self._obs_holder.get("obs")
                        generation = self._runtime.generation
                    if obs is None:
                        time.sleep(_RTC_IDLE_SLEEP_S)
                        continue
                    self._policy.drop_queued_actions()
                    self._service_query(obs)
                    with self._obs_lock:
                        # Pause/reset/fault during text never grants permission to
                        # resume. A healthy active run must obtain fresh actions.
                        if generation == self._runtime.generation and not self.failed:
                            self._hold_requested = False
                            self._hold_acknowledged.clear()
                            self._language_deadline = None
                            self._runtime.held = False
                            self._runtime.started_at = time.monotonic()
                            self._obs_holder["obs"] = None
                    continue
                if self._rtc_queue_threshold < 0 or not self._runtime.should_request():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                    capture_time = self._obs_holder.get("capture_time")
                    epoch_before = self._reset_epoch
                if obs is None or capture_time is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue
                with self._task_lock:
                    task, task_version = self._task, self._task_version
                    self._task_changed = False
                frame = build_dataset_frame(self._obs_features, obs, prefix="observation")
                observation = ObservationSnapshot(frame, capture_time, task, task_version)
                with self._obs_lock:
                    if epoch_before != self._reset_epoch or self._reset_pending:
                        continue
                    request = self._runtime.begin(observation)
                if request is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue
                previous = request.continuation.model_actions
                has_previous = previous is not None and previous.numel() > 0
                batch = prepare_observation_for_inference(
                    {key: value.copy() for key, value in observation.features.items()},
                    policy_device,
                    task,
                    self._robot.robot_type,
                )
                batch["task"] = [task]
                with torch.inference_mode():
                    preprocessed = self._preprocessor(batch)
                    if has_previous and self._relative_step is not None:
                        raw_state = self._relative_step.get_cached_state()
                        canonical = request.continuation.canonical_actions
                        if raw_state is None or canonical is None:
                            raise RuntimeError(
                                "Relative RTC requires a current anchor and canonical continuation"
                            )
                        previous = reanchor_relative_rtc_prefix(
                            canonical, raw_state, self._relative_step, self._normalizer_step, policy_device
                        )
                    if has_previous and previous is not None:
                        previous = _normalize_prev_actions_length(
                            previous, self._rtc_config.execution_horizon
                        )
                    else:
                        previous = None
                    if request.mode is ExecutionMode.CHUNK:
                        actions = self._policy.predict_action_chunk(preprocessed)
                    else:
                        predict = cast(_RTCPredictActionChunk, self._policy.predict_action_chunk)
                        actions = predict(
                            preprocessed, inference_delay=request.delay, prev_chunk_left_over=previous
                        )
                    if (
                        actions.ndim != 3
                        or actions.shape[0] != 1
                        or actions.shape[1] != self._policy_spec.prediction_steps
                        or not actions.is_floating_point()
                        or not torch.isfinite(actions).all()
                    ):
                        raise ValueError("Policy output violates its declared chunk inference contract")
                    original = actions.squeeze(0).clone()
                    canonical = self._postprocessor(actions)
                    if canonical.shape != actions.shape or not torch.isfinite(canonical).all():
                        raise ValueError("Canonical processor output violates the chunk inference contract")
                    processed = canonical.squeeze(0)
                if request.mode is ExecutionMode.CHUNK:
                    execution_steps = self._policy_spec.execution_steps
                    original, processed = original[:execution_steps], processed[:execution_steps]
                inference_count += 1
                if inference_count <= warmup_required:
                    # Warmup cannot leave motion or processor state for rollout.
                    with self._runtime.lock:
                        if self._runtime.pending is request:
                            self._runtime.pending = None
                    if inference_count == warmup_required:
                        self._policy.reset()
                        self._preprocessor.reset()
                        self._postprocessor.reset()
                        self._runtime.action_timeout = self._action_timeout_s
                        self._compile_warmup_done.set()
                    continue
                chunk = ActionChunk(
                    model_actions=original,
                    canonical_actions=processed,
                    execution_steps=len(processed),
                    provenance=ActionProvenance(capture_time, task, task_version),
                )
                accepted = self._runtime.accept(request, chunk, task_version=self.task_version)
                if accepted:
                    consecutive_discards = 0
                elif request.generation != self._runtime.generation:
                    logger.info("Discarding action chunk computed before an engine reset or hold")
                elif (
                    self._rtc_config.mode == "trained"
                    and request.observation.task_version == self.task_version
                ):
                    consecutive_discards += 1
                    if consecutive_discards >= _RTC_MAX_CONSECUTIVE_DISCARDS:
                        raise _TrainedRTCDelayExceededError(
                            "Trained RTC inference repeatedly exceeded the conditioned/checkpoint overlap; "
                            "increase playback coverage, reduce action FPS, or use guided RTC."
                        )
                self._surface_fault()
        except Exception as exc:
            self._failure_traceback = traceback.format_exc()
            logger.exception("Fatal error in RTC thread: %s", exc)
            self._runtime.fault(str(exc))
            self._rtc_error.set()
            self._compile_warmup_done.set()
