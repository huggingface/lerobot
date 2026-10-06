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
import contextlib
import logging
import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.inference import InferenceEngine
from lerobot.lerobot_types import RobotObservation
from lerobot.teleoperators import Teleoperator
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

if TYPE_CHECKING:
    from ..configs import RolloutConfig, RolloutStrategyConfig
    from ..context import (
        DatasetContext,
        HardwareContext,
        ProcessorContext,
        RolloutContext,
        RuntimeContext,
    )

logger = logging.getLogger(__name__)


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.  This is also the extension point
    for third-party strategies: subclass it next to a registered
    :class:`RolloutStrategyConfig` and ``lerobot-rollout --strategy.type=<name>``
    drives it with no edit to LeRobot (see "Bring your own strategy" in
    ``docs/source/inference.mdx``).

    Lifecycle: ``setup()`` once, then ``run()``, then ``teardown()`` once.
    A strategy whose config declares ``supports_interactive = True`` is also
    driven by ``--interactive=true``, which calls ``run()`` once per
    start/stop segment.  Such a strategy must keep ``run()`` restartable:

    - never finalize the dataset in ``run()`` — that belongs in ``teardown()``;
      at most save a partial tail episode when a segment ends;
    - keep state that must survive a segment on the instance, not in ``run()``
      locals (the ``CycleTimer`` is deliberately the other way round, see ``run()``);
    - never bind keyboard/terminal listeners — stdin belongs to the command prompt;
    - call ``engine.pump_query(obs_processed)`` once at the end of every tick, see
      ``run()``.

    One-shot strategies (``supports_interactive = False``, the default) are
    free to finalize on ``run()`` exit, e.g. via ``VideoEncodingManager``.

    Every autonomous motor tick must use :func:`send_next_action`, including
    ticks spent holding or interpolating. It enforces dispatch permission,
    invalidates interpolation when permission is revoked, acknowledges the
    supported local hold, and records dispatch provenance. Calling only
    ``engine.get_action()`` bypasses these asynchronous lifecycle guarantees.
    Custom dispatch implementations must honor the same per-tick protocol.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: RobotObservation | None = None

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
        self.reset_control_state()
        self._engine.start()
        self._warmup_flushed = False
        logger.info("Inference engine started")

    def _require_engine(self) -> InferenceEngine:
        """The inference engine attached by :meth:`_init_engine`."""
        if self._engine is None:
            raise RuntimeError(f"{type(self).__name__}: inference engine not attached; call setup() first")
        return self._engine

    def _require_interpolator(self) -> ActionInterpolator:
        """The action interpolator created by :meth:`_init_engine`."""
        if self._interpolator is None:
            raise RuntimeError(f"{type(self).__name__}: action interpolator not attached; call setup() first")
        return self._interpolator

    def _require_dataset_cfg(self, cfg: RolloutConfig) -> DatasetRecordConfig:
        """The ``--dataset.*`` config a recording strategy records with; ``RolloutConfig`` enforces ``dataset_mode = "required"``."""
        if cfg.dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}: no dataset config; a recording strategy must declare "
                'dataset_mode = "required" on its config'
            )
        return cfg.dataset

    def _require_dataset(self, data: DatasetContext) -> LeRobotDataset:
        """The dataset a recording strategy writes to (built from :meth:`_require_dataset_cfg`)."""
        if data.dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}: no dataset in the rollout context; a recording strategy "
                'must declare dataset_mode = "required" on its config'
            )
        return data.dataset

    def _require_teleop(self, hw: HardwareContext) -> Teleoperator:
        """The connected teleoperator; ``RolloutConfig`` enforces it via ``requires_teleop``."""
        if hw.teleop is None:
            raise RuntimeError(
                f"{type(self).__name__}: no teleoperator in the rollout context; a strategy that "
                "reads the teleop must declare requires_teleop = True on its config"
            )
        return hw.teleop

    def reset_control_state(self) -> None:
        """Clear episode-scoped control state so a paused session can restart cleanly.

        Resets the inference engine (policy hidden state, action queues), the action
        interpolator and the cached processed observation; pacing state is untouched.
        ``RolloutController`` calls it on its serve thread before each run segment.
        Only call while the control loop is not running: these resets are not synchronized
        against a live loop.  A caller that resets control state while a loop runs — or a
        strategy that hoists its timer onto the instance — must also call ``timer.restart()``.
        """
        if self._engine is not None:
            self._engine.reset()
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None

    def hold_control_state(self, hw: HardwareContext) -> None:
        """Apply a segment-end hold on the control thread after background dispatch is revoked."""
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None
        if hw.robot_wrapper.supports_hold and hw.robot_wrapper.hardware_failure is None:
            hw.robot_wrapper.hold()
            self._require_engine().acknowledge_hold()

    @contextlib.contextmanager
    def _pause_for_recording(
        self, ctx: RolloutContext, *, resume_allowed: Callable[[], bool] | None = None
    ) -> Iterator[None]:
        """Invalidate async motion across a blocking mid-run save; resume from a fresh capture.

        Call only inside an active run. Strategies with operator-controlled phases
        supply their current permission to resume. Save/hold failures propagate and
        leave inference paused; shutdown, takeover and faults never trigger resumption.
        """
        engine = self._require_engine()
        if engine.control_thread_owns_policy:
            yield
            return
        was_autonomous = resume_allowed is None or resume_allowed()
        engine.pause()
        self.hold_control_state(ctx.hardware)
        logger.info("Inference paused for episode save; buffered motion invalidated")
        yield
        if (
            was_autonomous
            and not ctx.runtime.shutdown_event.is_set()
            and not engine.failed
            and (resume_allowed is None or resume_allowed())
        ):
            engine.resume()
            logger.info("Episode save complete; inference resumes from the next fresh observation")

    def _process_observation_and_notify(
        self, processors: ProcessorContext, obs_raw: RobotObservation
    ) -> RobotObservation:
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
        if self._cached_obs_processed is None or self._require_interpolator().needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._require_engine().notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  Warmup ticks are paced through *timer* so the loop cadence
        stays anchored.  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        if not use_torch_compile:
            return False
        engine = self._require_engine()
        interpolator = self._require_interpolator()
        # A compiled model can hang inside its first call. Warmup bypasses the
        # normal dispatch path, so explicitly run the nonblocking deadline gate.
        engine.dispatch_allowed()
        if engine.failed:
            # Let send_next_action apply the robot hold and acknowledge it; do
            # not keep waiting, reset model state, or resume a faulted engine.
            return False
        if not engine.ready:
            timer.wait()
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            timer.restart()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """End policy execution, perform configured local shutdown movement, and disconnect.

        A terminal inference fault does not imply failed local hardware. Honor
        the return setting in that case; never home after a failed command.
        Homing uses ordinary observation acquisition and can fail with it.
        Disconnect torque behavior belongs to the robot driver, not the hold.
        """
        wrapper = hw.robot_wrapper
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None
        try:
            if self._engine is not None:
                logger.info("Stopping inference engine...")
                if self._engine.failed and wrapper.supports_hold and wrapper.hardware_failure is None:
                    try:
                        wrapper.hold()
                    except Exception:
                        logger.exception("Fault hold failed; skipping further shutdown movement")
                self._engine.stop()
        finally:
            robot = wrapper.inner
            try:
                if robot.is_connected:
                    try:
                        if wrapper.hardware_failure is not None:
                            logger.warning(
                                "Skipping return-to-initial-position: robot I/O failed: %s",
                                wrapper.hardware_failure,
                            )
                        elif not return_to_initial_position:
                            logger.info("Skipping return-to-initial-position: disabled by config")
                        elif not hw.initial_position:
                            logger.info("Skipping return-to-initial-position: no initial position captured")
                        else:
                            logger.info("Returning robot to initial position before shutdown...")
                            self.return_to_initial_position(hw)
                    finally:
                        torque_setting = getattr(
                            getattr(robot, "config", None), "disable_torque_on_disconnect", "unspecified"
                        )
                        logger.info(
                            "Disconnecting robot: disable_torque_on_disconnect=%s; "
                            "a prior hold command does not guarantee pose retention after disconnect",
                            torque_setting,
                        )
                        robot.disconnect()
            finally:
                teleop = hw.teleop
                if teleop is not None and teleop.is_connected:
                    logger.info("Disconnecting teleoperator...")
                    teleop.disconnect()

    @staticmethod
    def return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> bool:
        """Smoothly interpolate the robot back to its initial position.

        Returns ``True`` when the interpolation completed, ``False`` when it failed
        partway — the robot is then at an arbitrary pose, so callers must not report
        a completed reset on ``False``.
        """
        robot = hw.robot_wrapper
        if robot.hardware_failure is not None:
            logger.warning(
                "Cannot return to initial position after robot I/O failure: %s", robot.hardware_failure
            )
            return False
        target = hw.initial_position
        if target is None:
            logger.warning("Could not return to initial position: none was captured at connect time")
            return False
        try:
            current_obs = robot.get_observation()
            current_pos = {k: float(current_obs[k]) for k in target}
            if not all(math.isfinite(current_pos[k]) and math.isfinite(target[k]) for k in target):
                raise ValueError("Return movement requires finite current and initial positions")
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
            return False
        return True

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to the visualization backend if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_visualization_data(
            cfg.display_mode,
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.).

        The default only attaches and starts the inference engine; an override must
        call ``self._init_engine(ctx)`` (or ``super().setup(ctx)``) first.
        """
        self._init_engine(ctx)

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires.

        Implementations must call ``engine.resume()`` before entering their loop
        (async backends start paused, and the interactive controller pauses again at
        the end of every segment), and ``engine.pump_query(obs_processed)`` at the end
        of every tick — the text-query channel only advances through it, and a
        multi-second generation must not sit inside the action path.

        Publish each captured observation with ``engine.notify_observation``
        and dispatch through :func:`send_next_action` on every autonomous motor
        tick, even when no new action is expected. Do not skip the dispatch
        gate on interpolation or held ticks: hold acknowledgments unblock
        planned language work and propagate terminal faults to shutdown.

        Each ``run()`` call builds its own ``CycleTimer`` and reports it through
        ``timer.log_run_summary()`` from its ``finally``: a fresh timer's start-up
        exemption is what absorbs the interpolator that ``reset_control_state()``
        re-primes at every ``/start``, and each segment gets its own cadence report.
        """

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: finalize dataset, stop threads, disconnect hardware."""


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
    timer: CycleTimer | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Call once per autonomous motor tick, including interpolation and held ticks.
    This is the strategy's dispatch boundary: it starts the diagnostic tick,
    checks permission before pulling and sending, clears interpolation on a
    denial, performs a supported local hold, acknowledges it, and records the
    dispatched command. A direct ``get_action`` call does not replace it.

    When *timer* is given, the engine pull and the robot send are timed as the
    ``infer`` and ``send`` steps of its cadence summary, and a tick with no action
    to send is counted there.  Note that on async backends ``infer`` is only a
    queue pull — inference runs off-thread, so its latency surfaces as starved
    ticks rather than as loop-body time.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    engine.begin_control_tick()
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    # ``nullcontext`` accepts (and ignores) the section name, so it stands in for
    # ``timer.section`` verbatim when no timer was passed.
    section = timer.section if timer is not None else contextlib.nullcontext

    def dispatch_permitted() -> bool:
        if engine.dispatch_allowed():
            return True
        interpolator.reset()
        if ctx.hardware.robot_wrapper.supports_hold:
            ctx.hardware.robot_wrapper.hold()
        engine.acknowledge_hold()
        return False

    if not dispatch_permitted():
        return None

    if interpolator.needs_new_action():
        with section("infer"):
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    if not dispatch_permitted():
        return None

    interp = interpolator.get()
    if interp is None:
        if timer is not None:
            timer.note_starved_tick()
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        if not dispatch_permitted():
            return None
        dispatched = ctx.hardware.robot_wrapper.send_action(processed)
        engine.record_dispatch(
            action_dict, dispatched if isinstance(dispatched, dict) else processed, obs_raw
        )
    return action_dict
