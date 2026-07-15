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
from contextlib import nullcontext
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.predictors import (
    PredictorConfig,
    make_flow_token_warp_fn,
    make_token_warp_fn,
    shift_cube_in_frame,
)
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
        predictor_config: PredictorConfig | None = None,
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

        # Optional overhead cube-position predictor (Tier 3). When enabled, the
        # predictor runs on the control-loop observation frame and the cube is
        # advanced forward by the inference latency (the "PE gap") before the
        # frame is fed to the policy. The predictor needs no extra camera thread.
        # Disabled by default -> behaviour is unchanged.
        self._predictor_config = predictor_config
        self._predictor = None
        self._flow_estimator = None
        self._predictor_camera: str | None = None
        self._predictor_mode: str = "image_shift"
        self._predictor_lead_s: float = 0.0
        self._latent_mask_threshold: float = 0.0
        self._flow_motion_threshold: float = 0.0
        if predictor_config is not None and predictor_config.enabled:
            self._predictor_camera = predictor_config.camera
            self._predictor_mode = predictor_config.mode
            self._predictor_lead_s = predictor_config.lead_s
            self._latent_mask_threshold = predictor_config.latent_mask_threshold
            self._flow_motion_threshold = predictor_config.flow_motion_threshold
            if self._predictor_mode == "latent_flow":
                self._flow_estimator = predictor_config.make_flow()
            else:
                self._predictor = predictor_config.make()
            if self._predictor_mode in ("latent_warp", "latent_flow") and not hasattr(
                policy, "set_latent_warp"
            ):
                logger.warning(
                    "RTC predictor mode %r requested but policy %s has no set_latent_warp "
                    "hook; falling back to no time-advance.",
                    self._predictor_mode,
                    type(policy).__name__,
                )
            logger.info(
                "RTC overhead predictor enabled: camera=%s mode=%s lead_s=%.3f "
                "(time-advanced observation; advance = latency + lead_s)",
                predictor_config.camera,
                self._predictor_mode,
                self._predictor_lead_s,
            )

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
        # Rebuild the predictor / flow estimator so per-episode state (previous
        # frame, cube track) is cleared.
        if self._predictor_config is not None and self._predictor_config.enabled:
            if self._predictor_mode == "latent_flow":
                self._flow_estimator = self._predictor_config.make_flow()
            else:
                self._predictor = self._predictor_config.make()

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
    # Overhead cube-position prediction (optional)
    # ------------------------------------------------------------------

    def _time_advanced_obs(self, obs: dict, delay: int, time_per_chunk: float) -> dict:
        """Advance the cube in the RGB frame by the effective lead (``image_shift``).

        Runs the cube predictor on the configured camera, extrapolates the cube to
        where it will be after ``lead_s = delay * time_per_chunk + predictor.lead_s``
        seconds and shifts it there, returning a shallow copy with only the predictor
        camera replaced (the shared observation holder is never mutated). ``delay *
        time_per_chunk`` compensates for the inference latency (PE gap); the extra
        ``predictor.lead_s`` aims ahead of the cube over the open-loop execution
        window and the arm's reach (see :class:`PredictorConfig`). The latent modes
        advance in feature space instead (:meth:`_latent_warp_context`), so this is a
        no-op for them. Falls back to the original ``obs`` when disabled, the frame is
        missing, the cube is not tracked yet, or the effective lead is zero.
        """
        lead_s = delay * time_per_chunk + self._predictor_lead_s
        if self._predictor is None or self._predictor_mode != "image_shift" or lead_s <= 0:
            return obs
        frame = obs.get(self._predictor_camera)
        if frame is None:
            return obs
        try:
            output = self._predictor(frame)
            if output.center_px is None or output.velocity_px_s is None:
                return obs
            offset = (output.velocity_px_s[0] * lead_s, output.velocity_px_s[1] * lead_s)
            shifted = shift_cube_in_frame(frame, self._predictor.red_mask(frame), offset)
        except Exception as e:  # noqa: BLE001 - predictor must never crash the RTC loop
            logger.debug("RTC predictor skipped frame: %s", e)
            return obs
        advanced = dict(obs)
        advanced[self._predictor_camera] = shifted
        return advanced

    def _latent_warp_context(self, obs: dict, delay: int, time_per_chunk: float, preprocessed: dict):
        """Context manager that advances the cube in feature space for one inference.

        For ``latent_warp`` / ``latent_flow``, builds a per-image patch-token warp from
        the predictor-camera frame and installs it via ``policy.set_latent_warp`` for
        the duration of one ``predict_action_chunk``. Returns a no-op context for other
        modes, when the policy lacks the hook, or when the warp can't be built (it must
        never crash the RTC loop).
        """
        lead_s = delay * time_per_chunk + self._predictor_lead_s
        if self._predictor_mode not in ("latent_warp", "latent_flow") or lead_s <= 0:
            return nullcontext()
        if not hasattr(self._policy, "set_latent_warp"):
            return nullcontext()
        frame = obs.get(self._predictor_camera)
        if frame is None:
            return nullcontext()
        try:
            warp_fns = self._build_latent_warp_fns(frame, lead_s, preprocessed)
        except Exception as e:  # noqa: BLE001 - latent warp must never crash the RTC loop
            logger.debug("RTC latent warp skipped frame: %s", e)
            return nullcontext()
        if warp_fns is None:
            return nullcontext()
        return self._policy.set_latent_warp(warp_fns)

    def _build_latent_warp_fns(self, frame, lead_s: float, preprocessed: dict):
        """Build the per-image patch-token warp list for ``policy.set_latent_warp``.

        Only the predictor camera gets a warp function (others ``None``), indexed in the
        order the policy embeds present image features (matching SmolVLA's
        ``embed_prefix``). ``latent_warp`` rigidly shifts the colour-masked cube by the
        analytic offset; ``latent_flow`` advances each patch by its own dense-flow
        velocity. The mask / offset / flow field are projected into the policy's
        encoder-input space (the resize/pad used by ``prepare_images``) so the patch
        grid lines up with what the encoder sees. Returns ``None`` when the cube/flow is
        unavailable or the predictor camera is absent from the present features.
        """
        image_features = getattr(self._policy.config, "image_features", {}) or {}
        present = [key for key in image_features if key in preprocessed]
        cam = self._predictor_camera
        idx = next(
            (i for i, key in enumerate(present) if key == cam or key.split(".")[-1] == cam),
            None,
        )
        if idx is None:
            return None

        resize = getattr(self._policy.config, "resize_imgs_with_padding", None)
        target = (int(resize[0]), int(resize[1])) if resize is not None else None
        if target is not None:
            # Match prepare_images: resize_with_pad(img, width, height) -> (.., height, width).
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        if self._predictor_mode == "latent_warp":
            output = self._predictor(frame)
            if output.center_px is None or output.velocity_px_s is None:
                return None
            mask = self._predictor.red_mask(frame)  # (H, W) bool
            h0, w0 = mask.shape
            offset = (output.velocity_px_s[0] * lead_s, output.velocity_px_s[1] * lead_s)
            pixel_mask, image_hw = torch.as_tensor(mask, dtype=torch.bool), (h0, w0)
            if target is not None:
                target_w, target_h = target
                m = torch.as_tensor(mask, dtype=torch.float32)[None, None]
                pixel_mask = resize_with_pad(m, target_w, target_h, pad_value=0.0)[0, 0] > 0.5
                scale = min(target_w / w0, target_h / h0)  # aspect-preserving resize
                offset = (offset[0] * scale, offset[1] * scale)
                image_hw = (target_h, target_w)
            warp_fn = make_token_warp_fn(
                pixel_mask, offset, image_hw, mask_threshold=self._latent_mask_threshold
            )
        else:  # latent_flow
            out = self._flow_estimator(frame)
            if out is None:  # first frame of the episode -> no flow yet
                return None
            flow = torch.as_tensor(out.flow, dtype=torch.float32) * (lead_s / out.dt if out.dt > 0 else 0.0)
            h0, w0 = int(flow.shape[0]), int(flow.shape[1])
            image_hw = (h0, w0)
            if target is not None:
                target_w, target_h = target
                f = resize_with_pad(flow.permute(2, 0, 1)[None], target_w, target_h, pad_value=0.0)[0]
                scale = min(target_w / w0, target_h / h0)  # aspect-preserving resize
                flow = (f * scale).permute(1, 2, 0)
                image_hw = (target_h, target_w)
            warp_fn = make_flow_token_warp_fn(
                flow, image_hw, motion_threshold=self._flow_motion_threshold
            )

        fns: list = [None] * len(present)
        fns[idx] = warp_fn
        return fns

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

                        latency = latency_tracker.max()
                        delay = math.ceil(latency / time_per_chunk) if latency else 0

                        # Advance the cube forward by the inference latency (PE gap)
                        # so the policy plans against the cube's execution-time pose.
                        inference_obs = self._time_advanced_obs(obs, delay, time_per_chunk)
                        obs_batch = build_dataset_frame(self._hw_features, inference_obs, prefix="observation")
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

                        with self._latent_warp_context(obs, delay, time_per_chunk, preprocessed):
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
                        if is_warmup:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        queue.merge(original, processed, new_delay, idx_before)

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
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
