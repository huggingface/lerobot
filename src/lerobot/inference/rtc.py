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

"""Local asynchronous RTC prediction with serialized policy and processor ownership."""

from __future__ import annotations

import inspect
import logging
import math
import time
import traceback
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, InferenceRobot, PolicyQuery, QueryKind
from .contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot
from .execution import ChunkRuntime
from .prediction import chunk_inference_context, predict_chunk

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


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Background RTC prediction with control-thread capture and action consumption.

    Notify each capture; pause/resume around intervention and honor dispatch gates.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: InferenceRobot,
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
        action_starvation_grace_s: float = 1.0,
    ) -> None:
        """Own local asynchronous policy work and configure supported waiting."""
        super().__init__(task=task)
        self._policy = policy
        self._policy_spec = policy.chunk_inference_spec()
        if not self._policy_spec.current_observation_only:
            raise ValueError("Local asynchronous inference does not implement temporal observation sampling")
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._model_action_dim: int | None = None
        action_feature = getattr(policy.config, "action_feature", None)
        self._canonical_action_dim = (
            action_feature.shape[0]
            if action_feature is not None
            else dataset_features.get("action", {}).get("shape", (len(robot_wrapper.action_features),))[0]
        )
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        # Same feature spec sync uses, so both engines order observation.state identically.
        self._obs_features = dataset_features
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        minimum_warmup = 3 if rtc_config.enabled else 1
        self._compile_warmup_inferences = max(minimum_warmup, compile_warmup_inferences)
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
            action_starvation_grace_s=action_starvation_grace_s,
            training_max_delay=self._policy_spec.training_max_delay,
        )
        self._runtime.queue.cfg = rtc_config
        self._reset_pending = False
        self._hold_requested = False
        self._hold_acknowledged = Event()
        self._hold_started = 0.0
        self._language_deadline: float | None = None
        self._hold_reason = "Language request"
        self._active_query_generation = 0
        self._language_preprocessor, self._language_postprocessor = deepcopy((preprocessor, postprocessor))
        if getattr(robot_wrapper, "supports_position_hold", False):
            robot_wrapper.configure_position_hold()
        if not getattr(robot_wrapper, "supports_hold", False):
            self._runtime.starvation_grace = 0.0
            if action_starvation_grace_s:
                logger.warning(
                    "Local RTC action-starvation grace is unavailable without supported robot hold; "
                    "buffer exhaustion remains terminal"
                )

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
                self._compile_warmup_inferences,
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
    def action_queue(self) -> ActionQueue:
        """The shared action queue between the RTC thread and the main loop."""
        return self._runtime.queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        if self._started:
            raise RuntimeError("RTC inference engines cannot be restarted; create a new session")
        self._started = True
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
        self._runtime.deactivate()
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
        self.drop_pending_query()
        self._runtime.deactivate()
        with self._obs_lock:
            self._reset_epoch += 1
            self._obs_holder["obs"] = None
            self._hold_requested = False
            self._hold_acknowledged.clear()
            self._language_deadline = None

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        with self._obs_lock:
            if self._runtime.activate(held=self._hold_requested):
                self._policy_active.set()

    def set_task(self, task: str) -> bool:
        """Retarget an in-flight prediction through a planned hold when supported.

        Old-task results remain invalid. Without a supported local hold, ordinary
        local RTC retains its buffer/deadline limits and may exhaust on retarget.
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
            self._task_version += 1
            with self._obs_lock, self._runtime.lock:
                pending = self._runtime.pending
                # Warmup queues no motion and its control loop cannot acknowledge a hold.
                needs_hold = (
                    pending is not None
                    and pending.observation.task_version != self._task_version
                    and self._policy_active.is_set()
                    and self.ready
                    and bool(getattr(self._robot, "supports_hold", False))
                )
            # Keep the task lock through invalidation so an old result cannot be
            # accepted while the control thread is transitioning to its hold.
            if needs_hold:
                self._request_hold("Instruction change", self._action_timeout_s)
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

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
        queued = self._runtime.pop()
        if self._runtime.starvation_deadline is not None and self._runtime.held:
            self._request_hold("Action starvation", self._runtime.starvation_grace)
        if queued is None:
            return None
        action, provenance = queued
        self._set_dispatched_task(provenance.task)
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

    def dispatch_allowed(self) -> bool:
        """Revoke dispatch on every motor tick, including interpolation-only ticks."""
        if self.has_pending_query and self._policy_active.is_set():
            self._request_language_hold()
        with self._obs_lock:
            if self._language_deadline is not None and self._runtime.clock() > self._language_deadline:
                self._runtime.fault(f"{self._hold_reason} deadline exceeded; restart the inference session")
        permitted = self._runtime.dispatch_allowed()
        return permitted and self.ready and not self._hold_requested

    def acknowledge_hold(self) -> None:
        """Let the worker submit text only after the control thread held the robot."""
        with self._obs_lock:
            if self._hold_requested and not self._hold_acknowledged.is_set():
                self._obs_holder["obs"] = None
                self._runtime.acknowledge_starvation_hold()
                self._hold_acknowledged.set()
        if self.failed and self._global_shutdown_event is not None:
            self._global_shutdown_event.set()

    def _request_language_hold(self) -> None:
        self._request_hold("Language request", self._language_timeout_s)

    def _request_hold(self, reason: str, timeout_s: float) -> None:
        with self._obs_lock:
            if self._hold_requested:
                if (
                    self._hold_reason in {"Instruction change", "Action starvation"}
                    and reason == "Language request"
                ):
                    # This is still the same physical hold and invalidated generation.
                    # Widen its budget once, without renewing it on every motor tick.
                    self._language_deadline = self._hold_started + max(self._action_timeout_s, timeout_s)
                    self._hold_reason = reason
                return
            if reason != "Action starvation":
                self._runtime.invalidate(held=True, preserve_starvation=True)
            self._hold_requested = True
            self._hold_acknowledged.clear()
            self._hold_started = self._runtime.clock()
            self._language_deadline = (
                None if reason == "Action starvation" else self._hold_started + timeout_s
            )
            self._hold_reason = reason

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when both the policy text head and a local position hold are available."""
        return self._policy.supports_text_generation() and bool(getattr(self._robot, "supports_hold", False))

    def _queue_query(self, query: PolicyQuery) -> bool:
        if self._policy.supports_text_generation() and not getattr(self._robot, "supports_hold", False):
            logger.warning("Local RTC text queries require a supported robot position hold")
            return False
        if not self.supports_text_queries or self.failed:
            return False
        queued = super()._queue_query(query)
        if queued and self._policy_active.is_set():
            self._request_language_hold()
        return queued

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """Only enable language scheduling when its planned local hold is supported."""
        if not self.supports_text_queries:
            raise ValueError("Local RTC autosteering requires a policy text head and supported position hold")
        super().start_autosteer(goal, interval_s)

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
        generation = self._active_query_generation
        if generation != self._runtime.generation:
            raise RuntimeError("Language result belongs to an invalidated execution generation")
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
            # Exercise both continuation branches and a nonzero delay before motion.
            # These representative graphs cannot cover every task/shape specialization.
            warmup_required = self._compile_warmup_inferences if self._use_torch_compile else 0
            inference_count = 0
            warmup_previous: tuple[torch.Tensor, torch.Tensor] | None = None
            consecutive_discards = 0
            while not self._shutdown_event.is_set() and not self.failed:
                with self._runtime.lock:
                    wait_events = list(self._runtime.wait_events)
                    self._runtime.wait_events.clear()
                for event in wait_events:
                    if event["event"] == "waiting":
                        logger.warning(
                            "RTC actions exhausted; retaining last target for up to %.3fs", event["grace_s"]
                        )
                    else:
                        logger.info("RTC fresh actions ready after %.3fs waiting", event["wait_s"])
                # Reset is ordered behind any model call already executing. The
                # control thread has already revoked its old generation locally.
                with self._obs_lock:
                    reset_pending = self._reset_pending
                    self._reset_pending = False
                if reset_pending:
                    warmup_previous = None
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
                        self._active_query_generation = generation
                        self._runtime.complete_starvation_invalidation(generation)
                        starvation = self._hold_reason == "Action starvation"
                    if starvation:
                        self._policy.drop_queued_actions()
                        with self._obs_lock:
                            if self._hold_reason == "Action starvation" and self._runtime.release_hold(
                                generation
                            ):
                                self._hold_requested = False
                                self._hold_acknowledged.clear()
                                self._language_deadline = None
                                self._obs_holder["obs"] = None
                        continue
                    if obs is None:
                        time.sleep(_RTC_IDLE_SLEEP_S)
                        continue
                    self._policy.drop_queued_actions()
                    self._service_query(obs)
                    with self._obs_lock:
                        # Pause/reset/fault during text never grants permission to
                        # resume. A healthy active run must obtain fresh actions.
                        if self._runtime.release_hold(generation):
                            self._hold_requested = False
                            self._hold_acknowledged.clear()
                            self._language_deadline = None
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
                canonical_previous = request.continuation.canonical_actions
                delay = request.delay
                if inference_count < warmup_required and warmup_previous is not None:
                    previous, canonical_previous = warmup_previous
                    delay = 0 if inference_count == 1 else 1
                    if self._runtime.mode is ExecutionMode.RTC_TRAINED and delay:
                        delay = min(self._policy_spec.training_max_delay, self._rtc_config.execution_horizon)
                batch = prepare_observation_for_inference(
                    {key: value.copy() for key, value in observation.features.items()},
                    policy_device,
                    task,
                    self._robot.robot_type,
                )
                batch["task"] = [task]
                with chunk_inference_context(request.mode):
                    preprocessed = self._preprocessor(batch)
                    prediction = predict_chunk(
                        self._policy,
                        self._postprocessor,
                        preprocessed,
                        spec=self._policy_spec,
                        mode=request.mode,
                        canonical_action_dim=self._canonical_action_dim,
                        model_action_dim=self._model_action_dim,
                        device=policy_device,
                        rtc_horizon=self._rtc_config.execution_horizon if self._rtc_config.enabled else 0,
                        inference_delay=delay,
                        model_continuation=previous,
                        canonical_continuation=canonical_previous,
                        relative_step=self._relative_step,
                        normalizer_step=self._normalizer_step,
                    )
                    original, processed = prediction.model_actions, prediction.canonical_actions
                    self._model_action_dim = original.shape[1]
                if inference_count < warmup_required:
                    # Warmup cannot leave motion or processor state for rollout.
                    with self._runtime.lock:
                        if self._runtime.pending is not request or self._runtime.failure:
                            continue
                        self._runtime.pending = None
                    inference_count += 1
                    horizon = self._rtc_config.execution_horizon
                    warmup_previous = (original[-horizon:], processed[-horizon:])
                    if inference_count == warmup_required:
                        self._policy.reset()
                        self._preprocessor.reset()
                        self._postprocessor.reset()
                        with self._runtime.lock:
                            self._runtime.action_timeout = self._action_timeout_s
                        warmup_previous = None
                        self._compile_warmup_done.set()
                    continue
                chunk = ActionChunk(
                    model_actions=original,
                    canonical_actions=processed,
                    execution_steps=len(processed),
                    provenance=ActionProvenance(capture_time, task, task_version),
                )
                with self._task_lock:
                    accepted = self._runtime.accept(request, chunk, task_version=self._task_version)
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
        except Exception as exc:
            self._failure_traceback = traceback.format_exc()
            logger.exception("Fatal error in RTC thread: %s", exc)
            self._runtime.fault(str(exc))
            self._rtc_error.set()
            self._compile_warmup_done.set()
