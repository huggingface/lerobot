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
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


class CycleTimer:
    """Paces control-loop ticks and reports timing against the policy/dataset cadence.

    With ``interpolation_multiplier == N`` the control loop runs ``N`` ticks per
    policy cycle: robot commands go out every tick at ``fps × N`` Hz, while
    policy inference and dataset recording advance once per cycle at ``fps`` Hz.
    Inference runs on the tick that refills the interpolator; strategies record
    their frame on the tick that emits the policy's own end-point action (the
    last tick of the cycle), pairing it with the observation that produced it.
    That tick is identified by
    :attr:`~lerobot.utils.action_interpolator.ActionInterpolator.emitted_policy_action`,
    so recording carries no phase state of its own.

    The timer keeps two independent notions of time, because pacing and
    reporting want different anchors:

    **Pacing** uses ``_cycle_start``, re-anchored whenever the caller reports a
    new cycle, so each tick's deadline is an absolute offset from the tick that
    produced the current policy action.  A slow tick therefore borrows budget
    from the ticks that follow it instead of pushing the whole cycle back.  At
    30 FPS with multiplier 2, a 25 ms policy tick followed by a 5 ms
    interpolated tick still fits the 33.3 ms cycle.

    **Reporting** sums the *work* of ``N`` consecutive ticks — the time the loop
    body actually spends, excluding the pacing sleeps — and warns when that sum
    exceeds the ``1/fps`` budget, which is what makes the loop unable to hold
    the frame rate.  The 25 ms + 5 ms cycle above sums to 30 ms and stays
    silent.

    Two properties make this measure the right one, and both are load-bearing:

    - It is **phase-invariant**.  Groups are counted per tick and never
      re-anchored, so they drift out of step with cycles: the interpolator's
      first buffer holds a single action, which permanently offsets the two by
      one tick.  Summed work is the same whichever tick a group starts on,
      whereas a wall-clock span over an offset group would swallow a full
      pacing sleep and report a healthy loop as slow.
    - It stays meaningful when the interpolator is **starved or frozen** (an
      async backend yielding no action, or DAgger's paused and correcting
      phases), where every tick reports ``new_cycle=True``.  Tying the
      measurement to cycle completion would make the warning unreachable
      exactly when the loop is slow.

    Time lost to the OS *during* a pacing sleep is deliberately outside the
    warning, matching the pre-interpolation behaviour of these loops, which also
    measured only the work between the top of the loop body and its sleep: it is
    not something the caller can act on, and on a loaded machine it would fire
    constantly.  It is not invisible, though — the achieved start-of-group to
    start-of-group cadence, sleeps included, is logged at DEBUG whenever it
    misses the budget that the work sum met.  An individual tick overrun only
    costs interpolation smoothness, and is likewise a DEBUG note.

    Usage::

        timer = CycleTimer(cfg.fps, interpolator.multiplier)
        while ...:
            timer.tick(new_cycle=interpolator.needs_new_action())
            with timer.section("observe"):
                ...  # one block per big loop-body step
            timer.wait()
        timer.log_summary()

    ``new_cycle=True`` marks the ticks where the interpolator requests a fresh
    policy action, keeping the pacing anchor aligned with the actual inference
    cadence (the interpolator's first, single-action buffer would otherwise
    phase-shift every later cycle by one tick).

    Alongside the per-tick telemetry above, the timer accumulates run-level
    statistics — budget hit-rate, per-:meth:`section` timings, pacing slack —
    that :meth:`log_summary` reports once when the loop ends.
    """

    def __init__(self, fps: float, multiplier: int = 1, records_data: bool = True) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.fps = fps
        self.multiplier = multiplier
        self.tick_interval = 1.0 / (fps * multiplier)
        self.cycle_interval = 1.0 / fps
        self.records_data = records_data
        # Pacing anchor — re-anchored by ``new_cycle``.
        self._cycle_start: float | None = None
        self._tick_start: float | None = None
        self._ticks_done = 0
        # Reporting accumulator — advanced strictly per tick, never re-anchored.
        self._group_ticks = 0
        self._group_work = 0.0
        self._groups_closed = 0
        # Wall-clock anchor + last closed group's work, for the achieved-cadence
        # telemetry that covers what the work sum cannot see.
        self._group_start: float | None = None
        self._last_group_work = 0.0
        # Run-level statistics for ``log_summary``.  They describe the whole
        # run, so unlike the reporting accumulator they survive ``restart()``.
        self._stat_ticks = 0
        self._stat_work_total = 0.0
        self._stat_slot_overruns = 0
        self._stat_groups_judged = 0
        self._stat_groups_over = 0
        self._stat_group_work_sum = 0.0
        self._stat_group_work_max = 0.0
        self._stat_span_misses = 0
        self._stat_sleep_total = 0.0
        self._stat_sleep_max = 0.0
        self._stat_first_tick_start: float | None = None
        self._stat_last_tick_start = 0.0
        # Section name -> (calls, total seconds, worst seconds), in first-use
        # order, which follows the loop-body order.
        self._stat_sections: dict[str, tuple[int, float, float]] = {}

    def restart(self) -> None:
        """Re-arm the start-up exemption after control state was reset mid-run.

        Call wherever the interpolator is reset while the loop keeps running
        (the warm-up flush, DAgger returning to autonomous): it re-primes with a
        single-action buffer, so inference runs on two consecutive ticks again
        and the group spanning them legitimately exceeds budget.  Also call it
        after any other one-off blocking work inside the loop body (DAgger's
        smooth-handover ramps), whose cost is not the steady-state cadence.
        Only the reporting accumulator is cleared — pacing state is left alone so
        a restart between ``tick()`` and ``wait()`` cannot skip a pacing sleep.
        """
        self._group_ticks = 0
        self._group_work = 0.0
        self._groups_closed = 0
        self._group_start = None
        self._last_group_work = 0.0

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time one named step of the loop body for :meth:`log_summary`.

        Wrap each big step between :meth:`tick` and :meth:`wait` (observe,
        process, actuate, record) so the run summary can attribute the
        loop-body work to it.  Steps that only run on some ticks (recording,
        the engine pull) simply report fewer calls.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            calls, total, worst = self._stat_sections.get(name, (0, 0.0, 0.0))
            self._stat_sections[name] = (calls + 1, total + elapsed, max(worst, elapsed))

    def _report_achieved_cadence(self, group_start: float) -> None:
        """Log the cadence a closed group actually achieved, sleeps included.

        Measured start-of-group to start-of-group, so unlike the work sum this
        span covers the pacing sleeps.  When it misses the budget but the work
        did not, the time went missing *outside* the loop body — an oversleeping
        timer, or the OS descheduling the process mid-sleep — which is the one
        shortfall :meth:`wait`'s warning cannot see.  It is not the loop's own
        fault and not actionable in the same way, so it stays at DEBUG.
        """
        if self._group_start is None:
            return
        span = group_start - self._group_start
        if span <= self.cycle_interval or self._last_group_work > self.cycle_interval:
            return
        self._stat_span_misses += 1
        logger.debug(
            "Control loop held only %.1f Hz against a %g Hz target, though its loop-body work "
            "(%.1f ms) fit the %.1f ms budget: %.1f ms went missing outside the loop body "
            "(sleep overshoot or CPU starvation while pacing).",
            1.0 / span,
            self.fps,
            self._last_group_work * 1000,
            self.cycle_interval * 1000,
            (span - self.cycle_interval) * 1000,
        )

    def tick(self, new_cycle: bool = False) -> None:
        """Mark the start of a control tick.  Call at the top of the loop body."""
        self._tick_start = time.perf_counter()
        if new_cycle or self._cycle_start is None:
            self._cycle_start = self._tick_start
            self._ticks_done = 0

    def wait(self) -> None:
        """Sleep until this tick's deadline.  Call at the bottom of the loop body.

        A group of ``multiplier`` ticks whose work exceeds the ``1/fps`` budget
        means the policy/recording cadence cannot be held — the only case that
        warns.
        """
        now = time.perf_counter()
        if self._cycle_start is None or self._tick_start is None:
            return
        tick_dt = now - self._tick_start
        if self._group_ticks == 0:
            self._report_achieved_cadence(self._tick_start)
            self._group_start = self._tick_start
        if self._stat_first_tick_start is None:
            self._stat_first_tick_start = self._tick_start
        self._stat_last_tick_start = self._tick_start
        self._tick_start = None
        self._ticks_done += 1
        self._group_ticks += 1
        self._group_work += tick_dt
        self._stat_ticks += 1
        self._stat_work_total += tick_dt
        if tick_dt > self.tick_interval:
            self._stat_slot_overruns += 1
        deadline = self._cycle_start + self._ticks_done * self.tick_interval
        if self._ticks_done >= self.multiplier:
            self._cycle_start = None

        if self._group_ticks >= self.multiplier:
            group_work = self._group_work
            self._group_ticks = 0
            self._group_work = 0.0
            self._last_group_work = group_work
            self._groups_closed += 1
            # The first group is start-up, not steady state: the interpolator
            # primes its buffer with a single action, so inference runs on two
            # consecutive ticks, and one-off costs (lazy device init, camera
            # ramp-up) land here too.  Reporting it would warn on every healthy
            # launch.
            if self._groups_closed > 1:
                self._stat_groups_judged += 1
                self._stat_group_work_sum += group_work
                self._stat_group_work_max = max(self._stat_group_work_max, group_work)
                if group_work > self.cycle_interval:
                    self._stat_groups_over += 1
                    consequence = (
                        "Dataset frames might be dropped and robot control might be unstable."
                        if self.records_data
                        else "Robot control might be unstable."
                    )
                    logger.warning(
                        f"Control loop is running slower ({1 / group_work:.1f} Hz) than the target FPS "
                        f"({self.fps:g} Hz). {consequence} Common causes are: 1) Camera FPS not keeping up "
                        "2) Policy inference taking too long 3) CPU starvation"
                    )
        elif now > deadline and tick_dt > self.tick_interval:
            logger.debug(
                "Control tick overran its %.1f ms slot (took %.1f ms). Interpolated commands are sent "
                "less smoothly; the %g Hz %s cadence is judged per group of %d ticks.",
                self.tick_interval * 1000,
                tick_dt * 1000,
                self.fps,
                "policy/recording" if self.records_data else "policy",
                self.multiplier,
            )
        if (sleep_t := deadline - now) > 0:
            sleep_start = time.perf_counter()
            precise_sleep(sleep_t)
            slept = time.perf_counter() - sleep_start
            self._stat_sleep_total += slept
            self._stat_sleep_max = max(self._stat_sleep_max, slept)

    def log_summary(self) -> None:
        """Log run-level cadence statistics at INFO.  Call once when a loop ends.

        Reports the achieved tick cadence, how often the loop blew the
        ``1/fps`` work budget, how the loop-body work splits across the steps
        wrapped in :meth:`section`, and how much pacing slack was left per
        tick — enough to spot the bottleneck and to sanity-check the timer
        itself.  Groups exempted at start-up or by :meth:`restart` are not
        judged; everything else spans the whole run.
        """
        if self._stat_ticks == 0:
            return
        ms = 1000.0
        lines = [
            f"Cadence summary — {self.fps:g} Hz × {self.multiplier} "
            f"({self.tick_interval * ms:.1f} ms tick slot, {self.cycle_interval * ms:.1f} ms cycle "
            f"work budget): {self._stat_ticks} ticks, {self._stat_groups_judged} groups judged"
        ]
        span = self._stat_last_tick_start - (self._stat_first_tick_start or 0.0)
        if self._stat_ticks > 1 and span > 0:
            lines.append(
                f"  achieved tick cadence: {(self._stat_ticks - 1) / span:.2f} Hz "
                f"vs {self.fps * self.multiplier:g} Hz target"
            )
        if self._stat_groups_judged:
            lines.append(
                f"  groups over the work budget: {self._stat_groups_over}/{self._stat_groups_judged} "
                f"({100 * self._stat_groups_over / self._stat_groups_judged:.1f}%) — "
                f"group work mean {self._stat_group_work_sum / self._stat_groups_judged * ms:.1f} ms, "
                f"worst {self._stat_group_work_max * ms:.1f} ms"
            )
        if self.multiplier > 1:
            lines.append(
                f"  ticks over their {self.tick_interval * ms:.1f} ms slot: "
                f"{self._stat_slot_overruns}/{self._stat_ticks} (interpolation smoothness only)"
            )
        if self._stat_span_misses:
            lines.append(
                "  groups whose cadence slipped outside the loop body (sleep overshoot / OS): "
                f"{self._stat_span_misses}"
            )
        if self._stat_sections:
            lines.append("  loop-body steps (share of measured work):")
            width = max(len(name) for name in self._stat_sections)
            for name, (calls, total, worst) in self._stat_sections.items():
                share = 100 * total / self._stat_work_total if self._stat_work_total else 0.0
                lines.append(
                    f"    {name:<{width}}  mean {total / calls * ms:.1f} ms · worst {worst * ms:.1f} ms · "
                    f"{share:.1f}% · {calls} calls"
                )
        lines.append(
            f"  pacing sleep per tick: mean {self._stat_sleep_total / self._stat_ticks * ms:.1f} ms, "
            f"max {self._stat_sleep_max * ms:.1f} ms (headroom — near zero means the loop is saturated)"
        )
        logger.info("\n".join(lines))


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

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  Warmup ticks are paced through *timer* so the loop cadence
        stays anchored.  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
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
    timer: CycleTimer | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    When *timer* is given, the engine pull and the robot send are timed as
    ``get_action`` / ``send_action`` sections in its run summary.  Note that
    on async backends ``get_action`` is only a queue pull — inference runs
    off-thread, so its latency shows up as ``None`` returns here (a starved
    interpolator), not as loop-body time.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    if interpolator.needs_new_action():
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        with timer.section("get_action") if timer else contextlib.nullcontext():
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with timer.section("send_action") if timer else contextlib.nullcontext():
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
