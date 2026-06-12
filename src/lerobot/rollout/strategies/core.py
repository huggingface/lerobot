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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)

# ── Action smoothing ─────────────────────────────────────────────────────────
# Post-hoc EMA + velocity clipping applied to every action before it reaches
# the robot.  Reduces jerky movement caused by per-frame prediction noise
# without retraining.  Tune these values per-robot.
#
# Actions are **absolute goal positions** (degrees for SO-101).  The EMA
# smooths the commanded position trajectory; the clip limits the per-frame
# *change* in commanded position (velocity), NOT the absolute position.

# EMA alpha: 0.0 = raw (no smoothing), 1.0 = frozen (never move).
# Lower = smoother but more perceptible lag.  At 10 fps, alpha=0.3 means the
# commanded position reaches 90 % of target in ~6 frames (~600 ms).
EMA_SMOOTHING_ALPHA = 0.25

# Max per-frame change in commanded goal position, in the action's native
# units.  SO-101 with use_degrees=True → degrees.  At 15 fps:
#   0.75°/frame ≈ 11°/s  (slow, deliberate)
#   1.00°/frame ≈ 15°/s  (moderate)
#   1.25°/frame ≈ 19°/s  (crisp)
#   2.00°/frame ≈ 30°/s  (fast)
MAX_JOINT_DELTA = 1.25

# If the raw action differs from the stored EMA state by more than this
# threshold (degrees), the EMA state has drifted too far and is reset.
# The delta is still velocity-clipped; the reset only skips EMA blending
# (which would transmit the stale trend across the gap) and seeds _prev
# with the raw target for fast re-convergence.
# DAGGER correction boundaries are handled separately: the strategy calls
# reset_action_smoothing() on resume, which sets _prev=None so the first
# frame passes through cleanly (v ≈ physical position, fresh observation).
SMOOTHING_RESET_THRESHOLD = 10.0


def _smooth_action(
    action_dict: dict,
    alpha: float = EMA_SMOOTHING_ALPHA,
    limit: float = MAX_JOINT_DELTA,
    reset_threshold: float = SMOOTHING_RESET_THRESHOLD,
) -> dict:
    """Apply EMA smoothing + per-joint velocity clipping to *action_dict*.

    Modifies *action_dict* in place and returns it for convenience.
    Smoothing state (previous frame's smoothed action) is stored as a
    function attribute so it persists across calls within a rollout.

    Call :func:`reset_action_smoothing` at the start of every new rollout.

    Args:
        action_dict: Joint goal positions, modified in place.
        alpha: EMA blending factor (0=raw, 1=frozen).
        limit: Max per-frame delta in action units (degrees for SO-101).
        reset_threshold: Gap above which EMA blending is skipped.
    """
    prev = getattr(_smooth_action, "_prev", None)
    frame = getattr(_smooth_action, "_frame", 0) + 1
    _smooth_action._frame = frame

    if prev is None:
        _smooth_action._prev = {k: v for k, v in action_dict.items()}
        return action_dict

    clipped = False
    reset_triggered = False
    for k, v in action_dict.items():
        prev_v = prev.get(k, v)
        if abs(v - prev_v) > reset_threshold:
            # EMA state has drifted too far (policy direction change, or
            # recovery after correction — though the strategy resets _prev
            # to None on correction resume, so this path mainly catches
            # mid-autonomous drift).  Skip EMA blending but still clip
            # velocity, and seed _prev with the raw target.
            reset_triggered = True
            delta = v - prev_v
            if abs(delta) > limit:
                clipped = True
            delta = max(-limit, min(limit, delta))
            action_dict[k] = prev_v + delta
            prev[k] = v  # seed EMA state with raw target for fast re-convergence
            continue
        sv = alpha * v + (1.0 - alpha) * prev_v
        delta = sv - prev_v
        if abs(delta) > limit:
            clipped = True
        delta = max(-limit, min(limit, delta))
        sv = prev_v + delta
        action_dict[k] = sv

    _smooth_action._prev = {k: v for k, v in action_dict.items()}

    if frame % 150 == 1:
        extra = " [RESET]" if reset_triggered else ""
        logger.info(
            "clip=%s limit=%.2f°/frame alpha=%.2f frame=%d%s",
            clipped, limit, alpha, frame, extra,
        )
    return action_dict


def reset_action_smoothing() -> None:
    """Reset EMA state.  Call at the start of each new rollout."""
    _smooth_action._prev = None


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self._engine.reset()
        self._engine.start()
        self._warmup_flushed = False
        self._cached_obs_processed = None
        logger.info("Inference engine started")

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, loop_start: float, control_interval: float) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self._return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def _return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> None:
        """Smoothly interpolate the robot back to its initial position."""
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to Rerun if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_rerun_data(
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    if interpolator.needs_new_action():
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    # Apply action smoothing if enabled (EMA + velocity clip).
    # Config fields default to module-level constants if not present.
    cfg = ctx.runtime.cfg
    if getattr(cfg, "action_smoothing_enabled", True):
        _smooth_action(
            processed,
            alpha=getattr(cfg, "action_smoothing_alpha", EMA_SMOOTHING_ALPHA),
            limit=getattr(cfg, "action_smoothing_max_delta", MAX_JOINT_DELTA),
            reset_threshold=getattr(cfg, "action_smoothing_reset_threshold", SMOOTHING_RESET_THRESHOLD),
        )
    ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
