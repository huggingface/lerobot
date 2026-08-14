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
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


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
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
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
        return self._rtc_error.is_set()

    @property
    def failure_traceback(self) -> str | None:
        """Traceback captured when the RTC thread died (see ``failed``).

        Kept as data, not just logged, so consumers can re-surface it when someone looks.
        """
        return self._failure_traceback

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        self._action_queue = ActionQueue(self._rtc_config)
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

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue.

        Safe to call with the RTC thread paused or running.  Also drops the last published
        observation — a chunk computed from a stale one would jerk the robot toward an old
        pose — and bumps the reset epoch so an in-flight chunk is discarded instead of
        merged into the cleared queue.
        """
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._obs_lock:
            # Clear and bump in one critical section, mirroring _rtc_loop's epoch
            # check-and-merge, so a reset cannot leak a pre-reset chunk into the fresh
            # queue.  Lock order is _obs_lock -> queue.lock on both sides.
            if self._action_queue is not None:
                self._action_queue.clear()
            self._obs_holder["obs"] = None
            self._reset_epoch += 1
        # The queue is empty, so a pending task change has nothing stale to blend against.
        self._discard_task_change()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
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
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation()

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
        preprocessed = self._preprocessor(obs_batch)
        with torch.inference_mode():
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(preprocessed)

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            # Excluded from the latency tracker: cold starts spike it.
            latency_warmup_required = max(1, warmup_required)
            inference_count = 0
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                    epoch_before = self._reset_epoch
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                # Serve a queued text query here — this is the thread that owns the policy.  Above the
                # refill branch on purpose: a query issued while the queue is full would otherwise wait
                # for it to drain.  A next-subtask answer is applied via ``set_task``, so the chunk path
                # below uses it, and that path re-runs the preprocessor on its own observation, so
                # stateful steps (relative-action anchoring) are not left holding this query's.
                if self._service_query(obs):
                    # The generation took seconds, so the snapshot above is stale: re-read
                    # the observation, and the epoch so the discard guard below also covers
                    # a reset that landed during the query.
                    with self._obs_lock:
                        obs = self._obs_holder.get("obs")
                        epoch_before = self._reset_epoch
                    if obs is None:  # a reset mid-query dropped the observation
                        continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()

                        latency = latency_tracker.max()
                        delay = math.ceil(latency / time_per_chunk) if latency else 0

                        task, task_changed = self._take_task()
                        if task_changed:
                            # No queue flush on purpose: dropping queued actions would
                            # leave the robot uncommanded for a full inference latency.
                            # With RTC blending on this chunk is merged over the previous
                            # chunk's leftover prefix, so the switch lands within one
                            # inference; with blending off the queue drains first.
                            logger.info("Task changed to '%s' — applied from the next merged chunk", task)

                        obs_batch = build_dataset_frame(self._obs_features, obs, prefix="observation")
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, task, self._robot.robot_type
                        )
                        obs_batch["task"] = [task]

                        preprocessed = self._preprocessor(obs_batch)

                        if prev_actions is not None and self._relative_step is not None:
                            # Rebase against the raw cached state so the leftover tail stays in
                            # the training-time coordinate frame.
                            raw_state = self._relative_step.get_cached_state()
                            if raw_state is not None:
                                prev_abs = queue.get_processed_left_over()
                                if prev_abs is not None and prev_abs.numel() > 0:
                                    prev_actions = reanchor_relative_rtc_prefix(
                                        prev_actions_absolute=prev_abs,
                                        current_state=raw_state,
                                        relative_step=self._relative_step,
                                        normalizer_step=self._normalizer_step,
                                        policy_device=policy_device,
                                    )

                        if prev_actions is not None:
                            prev_actions = _normalize_prev_actions_length(
                                prev_actions, target_steps=self._rtc_config.execution_horizon
                            )

                        actions = self._policy.predict_action_chunk(
                            preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )

                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        new_latency = time.perf_counter() - current_time
                        new_delay = math.ceil(new_latency / time_per_chunk)

                        inference_count += 1
                        consecutive_errors = 0
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        if inference_count <= latency_warmup_required:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        with self._obs_lock:
                            # Check and merge in one critical section, mirroring reset()'s
                            # clear-and-bump, so a reset cannot land between them and leak
                            # a pre-reset chunk.  Lock order: _obs_lock -> queue.lock.
                            epoch_unchanged = epoch_before == self._reset_epoch
                            if epoch_unchanged:
                                queue.merge(original, processed, new_delay, idx_before, task=task)
                        if not epoch_unchanged:
                            logger.info("Discarding action chunk computed before an engine reset")

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # Persistent failure: stop retrying and propagate shutdown.
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            self._failure_traceback = traceback.format_exc()
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(self._failure_traceback)
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
