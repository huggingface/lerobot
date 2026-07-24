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
from .base import InferenceEngine

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


class _FatalRTCInferenceError(RuntimeError):
    """Base class for RTC errors that cannot become valid after a retry."""


class _TrainedRTCDelayExceededError(_FatalRTCInferenceError):
    """Raised when measured latency exceeds a trained RTC checkpoint's support."""


class _TrainedRTCPrefixUnavailableError(_FatalRTCInferenceError):
    """Raised when the queue cannot provide the prefix used for conditioning."""


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def _trained_rtc_chunk_can_merge(
    *,
    conditioned_delay: int,
    measured_delay: int,
    training_max_delay: int,
    has_previous_actions: bool,
) -> bool:
    """Check that a trained RTC chunk covers the overlap observed during inference."""
    if not has_previous_actions:
        return True
    if measured_delay > training_max_delay:
        raise _TrainedRTCDelayExceededError(
            f"Measured RTC inference delay ({measured_delay}) exceeds the checkpoint's "
            f"rtc_training_max_delay ({training_max_delay})."
        )
    return measured_delay <= conditioned_delay


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


def _validate_trained_rtc_prefix_available(*, conditioned_delay: int, available_steps: int) -> None:
    """Reject hard-prefix inference when the real queue is shorter than its delay."""
    if conditioned_delay > available_steps:
        raise _TrainedRTCPrefixUnavailableError(
            f"Trained RTC needs {conditioned_delay} committed prefix actions, but the queue has "
            f"only {available_steps}. Increase --inference.queue_threshold and "
            "--inference.rtc.execution_horizon."
        )


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
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
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
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_queue is not None:
            self._action_queue.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

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
            inference_count = 0
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()
                        has_previous_actions = prev_actions is not None and prev_actions.numel() > 0

                        training_max_delay = int(getattr(self._policy.config, "rtc_training_max_delay", 0))
                        latency = latency_tracker.max()
                        delay = _estimate_rtc_delay(
                            latency=latency,
                            time_per_step=time_per_chunk,
                            mode=self._rtc_config.mode,
                            training_max_delay=training_max_delay,
                            has_previous_actions=has_previous_actions,
                        )
                        if self._rtc_config.mode == "trained" and delay > 0:
                            available_steps = 0 if prev_actions is None else prev_actions.shape[0]
                            _validate_trained_rtc_prefix_available(
                                conditioned_delay=delay,
                                available_steps=available_steps,
                            )

                        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, self._task, self._robot.robot_type
                        )
                        obs_batch["task"] = [self._task]

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
                        is_initial_trained_chunk = (
                            self._rtc_config.mode == "trained" and not has_previous_actions
                        )
                        if is_warmup or is_initial_trained_chunk:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        if (
                            not is_warmup
                            and self._rtc_config.mode == "trained"
                            and not _trained_rtc_chunk_can_merge(
                                conditioned_delay=delay,
                                measured_delay=new_delay,
                                training_max_delay=training_max_delay,
                                has_previous_actions=has_previous_actions,
                            )
                        ):
                            logger.warning(
                                "Discarding trained RTC chunk: measured delay %d exceeded "
                                "conditioned delay %d; retrying with updated latency",
                                new_delay,
                                delay,
                            )
                            continue

                        queue.merge(original, processed, new_delay, idx_before)

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except _FatalRTCInferenceError:
                        raise
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
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
