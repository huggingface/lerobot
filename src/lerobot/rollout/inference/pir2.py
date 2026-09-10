#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""piR2 real-time inference (arXiv 2607.26055, Sec. 3.3).

Where :mod:`.rtc` denoises a whole chunk per call and splices it into a queue, this engine
keeps one partially-denoised buffer alive for the whole episode. Every call spends a single
denoising step on it, hands the finished front to the robot, slides the buffer forward, and
appends fresh noise at the back -- so the schedule reproduces itself and the robot is fed
continuously without ever waiting for a full chunk.

The vision-language prefix runs on its own thread as fast as the backbone allows, and each
denoising step conditions on whichever cache is newest. On pi0.5 that prefix carries joint state
too, since proprioception is discretized into the tokenized prompt; the clamped clean front of
the buffer is what keeps the expert anchored to where the robot actually is. That covers
configuration but not the gap between commanded and actual position, so this engine will not
react to an external disturbance the way a dedicated proprioception channel would
(arXiv 2607.26055, Sec. 3.2).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.001
_JOIN_TIMEOUT_S = 5.0
_MAX_CONSECUTIVE_ERRORS = 3
# Floor for the finished-action cushion, so a d=1 schedule still absorbs one slow call.
_MIN_PENDING_ACTIONS = 2


def estimate_pir2_delay(latencies: deque[float], time_per_step: float, max_delay: int) -> int:
    """Derive the per-call delay ``d`` from a rolling window of action-head latencies.

    The paper takes the mean of the window rather than a maximum: overshooting ``d`` costs
    reactivity on every subsequent call, whereas a single slow call just means the buffer is
    slightly behind its schedule for one cycle, which the next step absorbs.
    """
    if not latencies:
        return 1
    mean_latency = sum(latencies) / len(latencies)
    return max(1, min(max_delay, round(mean_latency / time_per_step)))


class PiR2InferenceEngine(InferenceEngine):
    """Async piR2 inference: a background thread advances one shared action buffer."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        latency_window: int = 20,
        max_delay: int | None = None,
        shutdown_event: Event | None = None,
    ) -> None:
        required_methods = (
            "encode_prefix",
            "warm_start_realtime_buffer",
            "realtime_substep",
        )
        for required in required_methods:
            if not hasattr(policy, required):
                raise NotImplementedError(
                    f"{type(policy).__name__} does not support piR2 inference (missing "
                    f"{required!r}). Train a pi0.5 checkpoint with "
                    "--policy.rtc_training_schedule=staircase and use it here."
                )
        if getattr(policy.config, "rtc_training_schedule", "prefix") != "staircase":
            raise ValueError(
                "piR2 inference requires a checkpoint trained with "
                "--policy.rtc_training_schedule=staircase; this one reports "
                f"{getattr(policy.config, 'rtc_training_schedule', 'prefix')!r}. A prefix-trained "
                "checkpoint has never seen a ramped noise schedule and will not denoise it."
            )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"

        chunk_size = int(policy.config.chunk_size)
        # 2 * d <= H is what keeps a non-empty ramp between the clean front and the noise tail.
        self._max_delay = min(max_delay or chunk_size // 2, chunk_size // 2)
        self._latencies: deque[float] = deque(maxlen=latency_window)

        # A prefix older than the chunk it is steering is worth surfacing: the clean front can
        # only anchor the expert for so long before the plan behind it is answering a dead question.
        self._stale_prefix_steps = chunk_size
        self._last_stale_warning = 0.0

        self._buffer: torch.Tensor | None = None
        self._prefix: Any = None
        self._prefix_lock = Lock()
        self._emitted: deque[torch.Tensor] = deque()
        self._emitted_lock = Lock()
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._error = Event()
        self._global_shutdown_event = shutdown_event
        self._thread: Thread | None = None
        self._vlm_thread: Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the buffer has been warm-started and actions are flowing."""
        return self._buffer is not None

    @property
    def failed(self) -> bool:
        """True if the background thread exited due to an unrecoverable error."""
        return self._error.is_set()

    @property
    def control_thread_owns_policy(self) -> bool:
        """The VLM thread owns the policy; it services queries in ``_vlm_loop``."""
        return False

    def start(self) -> None:
        """Launch the background denoising thread and the vision-language thread."""
        self._obs_holder = {"obs": None, "robot_type": self._robot.robot_type}
        self._shutdown_event.clear()
        self._thread = Thread(target=self._denoise_loop, daemon=True, name="PiR2Inference")
        self._thread.start()
        self._vlm_thread = Thread(target=self._vlm_loop, daemon=True, name="PiR2VLM")
        self._vlm_thread.start()
        logger.info("piR2 inference started (max delay %d)", self._max_delay)

    def stop(self) -> None:
        """Signal the background threads to stop and wait for them."""
        self._shutdown_event.set()
        self._policy_active.clear()
        for name, thread in (("denoise", self._thread), ("VLM", self._vlm_thread)):
            if thread is not None and thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning("piR2 %s thread did not join within %.1fs", name, _JOIN_TIMEOUT_S)
        self._thread = None
        self._vlm_thread = None

    def pause(self) -> None:
        """Pause background denoising."""
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume background denoising."""
        self._policy_active.set()

    def reset(self) -> None:
        """Drop the buffer so the next episode warm-starts from pure noise."""
        logger.info("Resetting piR2 inference state (policy + processors + buffer)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._buffer = None
        self._latencies.clear()
        with self._prefix_lock:
            self._prefix = None
        with self._emitted_lock:
            self._emitted.clear()

    # ------------------------------------------------------------------
    # Action production (called from the main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next finished action (ignores ``obs_frame``)."""
        with self._emitted_lock:
            if not self._emitted:
                return None
            return self._emitted.popleft()

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the denoising thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    def pending_actions(self) -> int:
        """Number of emitted actions not yet sent to the robot."""
        with self._emitted_lock:
            return len(self._emitted)

    # ------------------------------------------------------------------
    # Background denoising thread
    # ------------------------------------------------------------------

    def _prepare_batch(self, obs: dict) -> dict:
        """Turn a raw observation into a policy batch (normalize, tokenize, move to device)."""
        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
        obs_batch = prepare_observation_for_inference(
            obs_batch, torch.device(self._device), self._task, self._robot.robot_type
        )
        return self._preprocessor(obs_batch)

    def _latest_observation(self) -> dict | None:
        with self._obs_lock:
            return self._obs_holder.get("obs")

    def _vlm_loop(self) -> None:
        """Refresh the vision-language cache as fast as the backbone allows, off the critical path."""
        try:
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                obs = self._latest_observation()
                if obs is None:
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                # Served here because this is the thread that owns the policy.  Ahead of the
                # refresh below on purpose: this loop never blocks on its own, so a query
                # queued behind it would otherwise never be picked up.
                self._service_query(obs)
                prefix = self._policy.encode_prefix(self._prepare_batch(obs))
                with self._prefix_lock:
                    self._prefix = prefix
        except Exception:
            logger.exception("piR2 VLM thread terminating")
            self._error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()

    def _current_prefix(self) -> Any | None:
        """Return the newest prefix cache, warning when it falls badly behind."""
        with self._prefix_lock:
            prefix = self._prefix
        if prefix is None:
            return None
        now = time.perf_counter()
        age_s = 0.0 if prefix.captured_at is None else now - prefix.captured_at
        age_steps = int(age_s * self._fps)
        if age_steps > self._stale_prefix_steps and now - self._last_stale_warning > 5.0:
            self._last_stale_warning = now
            logger.warning(
                "piR2 vision-language prefix is %d control steps old (%.2fs); the expert is "
                "steering on stale vision. Reduce image resolution or camera count.",
                age_steps,
                age_s,
            )
        return prefix

    def _denoise_loop(self) -> None:
        try:
            time_per_step = 1.0 / self._fps
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                try:
                    delay = estimate_pir2_delay(self._latencies, time_per_step, self._max_delay)

                    # A substep emits `delay` actions but can finish in less than the `delay`
                    # control ticks the robot needs to consume them, so the expert outruns the
                    # control loop. Running ahead buys nothing: it only makes each executed action
                    # older, which is what this engine exists to avoid. Hold one substep of
                    # cushion against latency jitter and idle otherwise.
                    if self.pending_actions() >= max(2 * delay, _MIN_PENDING_ACTIONS):
                        time.sleep(_IDLE_SLEEP_S)
                        continue

                    prefix = self._current_prefix()
                    if prefix is None:
                        # The vision-language thread has not produced its first cache yet.
                        time.sleep(_IDLE_SLEEP_S)
                        continue

                    started = time.perf_counter()
                    if self._buffer is None:
                        # Episode start: nothing is in flight, so fall back to a full denoise
                        # and re-noise the result onto the staircase.
                        self._buffer = self._policy.warm_start_realtime_buffer(prefix, delay)

                    emitted, self._buffer = self._policy.realtime_substep(prefix, self._buffer, delay)
                    self._publish(emitted)

                    self._latencies.append(time.perf_counter() - started)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    logger.exception("piR2 denoising step failed (%d)", consecutive_errors)
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_IDLE_SLEEP_S)
        except Exception:
            logger.exception("piR2 inference thread terminating")
            self._error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()

    def _publish(self, emitted: torch.Tensor) -> None:
        """Post-process the finished actions and queue them for the control loop."""
        processed = self._postprocessor(emitted).squeeze(0)
        with self._emitted_lock:
            for step in range(processed.shape[0]):
                self._emitted.append(processed[step])
