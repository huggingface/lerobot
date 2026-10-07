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

"""Cadence pacing and reporting for real-time control loops.

Every loop in LeRobot that drives hardware at a fixed rate — teleoperation,
recording, replay, policy rollout — has the same two jobs beyond its own body:
sleep the right amount so the next iteration starts on time, and tell the user
when it could not keep up.  :class:`CycleTimer` owns both, so those loops share
one pacing rule, one slow-loop warning, and one end-of-run report.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from collections.abc import Callable, Iterator

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _SectionStat:
    """Call count and timings for one named loop-body step."""

    calls: int = 0
    total: float = 0.0
    worst: float = 0.0


@dataclasses.dataclass
class _CadenceStats:
    """Cadence counters accumulated over one reporting window.

    A window is the stretch of loop between two episode boundaries; the run-level
    instance is every closed window folded together.  Every field is additive so
    both share one formatter.

    Elapsed time is accumulated as a **sum of tick-to-tick gaps** rather than a
    first/last timestamp pair, which keeps the effective cadence free of stretches
    the loop was not running: the untimed reset phase between episodic's episodes,
    and the one-off blocking work that :meth:`CycleTimer.restart` drops.
    """

    ticks: int = 0
    work: float = 0.0
    span: float = 0.0
    span_ticks: int = 0
    slot_overruns: int = 0
    starved_ticks: int = 0
    groups_judged: int = 0
    groups_over: int = 0
    group_work: float = 0.0
    group_work_worst: float = 0.0
    span_misses: int = 0
    sleep: float = 0.0
    sleep_worst: float = 0.0
    sections: dict[str, _SectionStat] = dataclasses.field(default_factory=dict)

    def record_section(self, name: str, elapsed: float) -> None:
        """Accumulate one timed run of the *name* step."""
        stat = self.sections.setdefault(name, _SectionStat())
        stat.calls += 1
        stat.total += elapsed
        stat.worst = max(stat.worst, elapsed)

    def fold(self, other: _CadenceStats) -> None:
        """Merge a finished window into this cumulative one."""
        self.ticks += other.ticks
        self.work += other.work
        self.span += other.span
        self.span_ticks += other.span_ticks
        self.slot_overruns += other.slot_overruns
        self.starved_ticks += other.starved_ticks
        self.groups_judged += other.groups_judged
        self.groups_over += other.groups_over
        self.group_work += other.group_work
        self.group_work_worst = max(self.group_work_worst, other.group_work_worst)
        self.span_misses += other.span_misses
        self.sleep += other.sleep
        self.sleep_worst = max(self.sleep_worst, other.sleep_worst)
        for name, stat in other.sections.items():
            mine = self.sections.setdefault(name, _SectionStat())
            mine.calls += stat.calls
            mine.total += stat.total
            mine.worst = max(mine.worst, stat.worst)


class CycleTimer:
    """Paces control-loop ticks and reports timing against the loop's target cadence.

    At the default ``multiplier == 1`` every tick is one cycle, and that is the whole
    contract: the timer sleeps so each iteration takes ``1/fps``, warns when the loop
    body alone cannot fit that budget, and accumulates the statistics that
    :meth:`log_episode_summary` and :meth:`log_run_summary` report.  Teleoperation,
    recording and replay use it exactly that way — ``tick()``, sections, ``wait()``.

    Policy rollout is what the rest of this docstring is about, because it adds
    action interpolation.  With ``interpolation_multiplier == N`` the control loop runs
    ``N`` ticks per
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

    Alongside that per-tick telemetry the timer accumulates cadence statistics —
    effective frame rate, how often the budget was missed, where the loop-body
    work went — and reports them twice: a one-line summary per episode
    (:meth:`log_episode_summary`) and a full block for the whole run
    (:meth:`log_run_summary`, called from the loop's ``finally`` so a
    ``KeyboardInterrupt`` still produces it).  Both go to ``logger.info``, or to the
    ``report`` sink when one is given — for callers that mute the logger.  Only the
    summaries are routed; the slow-loop warning and the DEBUG telemetry stay on the
    logger.

    Usage::

        timer = CycleTimer(cfg.fps, interpolator.multiplier)
        try:
            while ...:
                timer.tick(new_cycle=interpolator.needs_new_action())
                with timer.section("observe"):
                    ...  # one section per big loop-body step
                timer.wait()
                if episode_boundary:
                    timer.log_episode_summary()
        finally:
            timer.log_run_summary()

    Loops with no interpolator just call ``timer.tick()``: at multiplier 1 the
    anchor is re-taken every tick either way, so the flag makes no difference.

    ``new_cycle=True`` marks the ticks where the interpolator requests a fresh
    policy action, keeping the pacing anchor aligned with the actual inference
    cadence (the interpolator's first, single-action buffer would otherwise
    phase-shift every later cycle by one tick).
    """

    #: Fraction of the cycle budget a group's wall-clock span may overshoot before
    #: :meth:`_report_achieved_cadence` speaks up.  ``precise_sleep`` spins to its
    #: deadline, but scheduler jitter still costs tens of microseconds per tick, so
    #: with no tolerance the note fires on nearly every group — a healthy 30 Hz run
    #: logged it for 556 groups out of 576.
    SPAN_TOLERANCE = 0.01

    def __init__(
        self,
        fps: float,
        multiplier: int = 1,
        records_data: bool = True,
        report: Callable[[str], None] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.fps = fps
        self.multiplier = multiplier
        self.tick_interval = 1.0 / (fps * multiplier)
        self.cycle_interval = 1.0 / fps
        self.records_data = records_data
        self._report = report if report is not None else logger.info
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
        # Statistics: ``_window`` is the episode in progress, ``_run`` every window
        # closed so far.  Only ``_window`` is touched per tick, so the per-tick
        # bookkeeping stays on one object; boundaries fold it into ``_run``.
        self._window = _CadenceStats()
        self._run = _CadenceStats()
        self._windows_closed = 0
        self._prev_tick_start: float | None = None
        # Boundary effects take hold at the next ``tick()`` — see :meth:`tick`.
        self._pending_close: str | None = None
        self._drop_next_gap = False

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
        Statistics describe the whole run and deliberately survive; the one thing
        dropped is the elapsed gap that *ends at the next tick*, which is the one
        containing the blocking work, so it is not billed to the effective cadence.
        """
        self._group_ticks = 0
        self._group_work = 0.0
        self._groups_closed = 0
        self._group_start = None
        self._last_group_work = 0.0
        self._drop_next_gap = True

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time one big step of the loop body for the cadence summaries.

        Wrap the coarse steps between :meth:`tick` and :meth:`wait` — observing,
        processing, inference, actuation, recording — so the summary can say where
        the loop-body work went.  Keep the sections **flat and disjoint**: shares
        are reported against the total measured work, so nesting one inside another
        double-counts.  Steps that only run on some ticks (recording, the engine
        pull) simply report fewer calls.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._window.record_section(name, time.perf_counter() - start)

    def note_starved_tick(self) -> None:
        """Record a tick that had no action to send because the engine yielded none.

        Called by :func:`send_next_action`.  Such a tick commands nothing and
        records nothing, so a run with many of them writes a dataset *shorter* than
        the wall clock it was captured over — invisible in the dataset itself, whose
        timestamps are synthesised from the frame index.  Surfacing the count is the
        only warning a user gets.
        """
        self._window.starved_ticks += 1

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
        if (
            span <= self.cycle_interval * (1.0 + self.SPAN_TOLERANCE)
            or self._last_group_work > self.cycle_interval
        ):
            return
        self._window.span_misses += 1
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
        """Mark the start of a control tick.  Call at the top of the loop body.

        This is also where episode boundaries and :meth:`restart` take effect, and
        where elapsed time is accumulated — both for the same reason.  Callers reach
        both from *inside* a loop body: a rotation happens after the blocking
        ``save_episode`` and before :meth:`wait`.  The tick in progress is therefore
        the closing episode's last tick and its cost belongs there, while the gap
        that must be dropped is the one *ending at the next tick* (the one that
        contains the blocking work), not the healthy gap already behind us.
        Deferring to here gets both right whether the caller was mid-body or between
        ticks.
        """
        self._tick_start = time.perf_counter()
        if self._pending_close is not None:
            self._close_window(self._pending_close)
            self._pending_close = None
            self._drop_next_gap = True
        # Elapsed time is measured tick-start to tick-start, summed per gap rather
        # than taken from a first/last pair so that dropping one is possible at all.
        if self._prev_tick_start is not None and not self._drop_next_gap:
            self._window.span += self._tick_start - self._prev_tick_start
            self._window.span_ticks += 1
        self._drop_next_gap = False
        self._prev_tick_start = self._tick_start
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
        tick_start = self._tick_start
        tick_dt = now - tick_start
        if self._group_ticks == 0:
            self._report_achieved_cadence(tick_start)
            self._group_start = tick_start
        self._tick_start = None
        self._ticks_done += 1
        self._group_ticks += 1
        self._group_work += tick_dt

        stats = self._window
        stats.ticks += 1
        stats.work += tick_dt
        if tick_dt > self.tick_interval:
            stats.slot_overruns += 1

        deadline = self._cycle_start + self._ticks_done * self.tick_interval
        if self._ticks_done >= self.multiplier:
            self._cycle_start = None

        warned = False
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
            # launch, and averaging it in would skew the run summary.
            if self._groups_closed > 1:
                stats.groups_judged += 1
                stats.group_work += group_work
                stats.group_work_worst = max(stats.group_work_worst, group_work)
                if group_work > self.cycle_interval:
                    stats.groups_over += 1
                    warned = True
                    consequence = (
                        "Dataset frames might be dropped and robot control might be unstable."
                        if self.records_data
                        else "Robot control might be unstable."
                    )
                    logger.warning(
                        f"Control loop is running slower ({1 / group_work:.1f} Hz) than the target FPS "
                        f"({self.fps:g} Hz). {consequence} Common causes are: 1) Camera FPS not keeping up "
                        "2) Policy inference (action or text) taking too long 3) CPU starvation"
                    )
        # A late tick that did not blow the cycle budget costs only interpolation
        # smoothness, so it is a DEBUG note — and at multiplier 1 there is no
        # smoothness to lose, the warning above is the whole story.  Group-closing
        # ticks are included unless they already warned; an ``elif`` here used to
        # swallow every multiplier-th tick's overrun.
        if self.multiplier > 1 and not warned and now > deadline and tick_dt > self.tick_interval:
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
            stats.sleep += slept
            stats.sleep_worst = max(stats.sleep_worst, slept)

    # ------------------------------------------------------------------
    # Cadence summaries
    # ------------------------------------------------------------------

    def _effective_hz(self, stats: _CadenceStats) -> float | None:
        """Achieved *command* rate in Hz, or ``None`` when too little was measured."""
        if stats.span_ticks == 0 or stats.span <= 0:
            return None
        return stats.span_ticks / stats.span

    @property
    def _judged(self) -> str:
        """What one judged group of ``multiplier`` ticks is called in the reports.

        Groups of more than one tick are *cycles*.  They are phase-shifted from the
        interpolator's actual policy cycles by one tick (see the class docstring), but
        there is exactly one per policy action either way, so the counts a reader cares
        about are the same.  At multiplier 1 a group *is* a tick, and calling it a cycle
        would invent a concept the loop does not have.
        """
        return "cycle" if self.multiplier > 1 else "tick"

    def _summary_line(self, stats: _CadenceStats) -> str:
        """One-line digest of a window: cadence held, budget missed, elapsed."""
        ms = 1e3
        parts: list[str] = []
        hz = self._effective_hz(stats)
        ticks = f"{stats.ticks} tick{'s' if stats.ticks != 1 else ''}"
        if hz is None:
            parts.append(f"{ticks}, too short to measure a rate")
        else:
            # Only an interpolating loop has two rates to tell apart.
            rate = f"{hz / self.multiplier:.2f} Hz policy" if self.multiplier > 1 else f"{hz:.2f} Hz"
            parts.append(f"{rate} vs {self.fps:g} Hz target")
            parts.append(f"{ticks}, {stats.span:.1f} s measured")
        if stats.groups_judged:
            parts.append(
                f"{stats.groups_over}/{stats.groups_judged} {self._judged}s over the "
                f"{self.cycle_interval * ms:.1f} ms budget (work mean "
                f"{stats.group_work / stats.groups_judged * ms:.1f} ms, worst "
                f"{stats.group_work_worst * ms:.1f} ms)"
            )
        if stats.starved_ticks:
            parts.append(f"starved ticks: {stats.starved_ticks}")
        return " · ".join(parts)

    def _summary_lines(self, stats: _CadenceStats, heading: str) -> list[str]:
        """Full multi-line report for a window (see :meth:`_judged` on *cycles* vs *ticks*)."""
        ms = 1e3
        # An interpolating loop has two budgets to state, a plain one has a single
        # per-tick budget and no second rate anywhere in the block.
        if self.multiplier > 1:
            target = (
                f"target {self.fps:g} Hz × {self.multiplier} ({self.tick_interval * ms:.1f} ms tick "
                f"slot, {self.cycle_interval * ms:.1f} ms cycle budget)"
            )
            judged = f"{stats.groups_judged} cycles judged"
        else:
            target = f"target {self.fps:g} Hz ({self.cycle_interval * ms:.1f} ms budget per tick)"
            judged = f"{stats.groups_judged} judged"
        # Sample size goes in the heading, unconditionally: every other number here is
        # a rate or an average over it, so a reader needs it to judge any of them —
        # and the effective-cadence line below is skipped when a window is too short
        # to have measured a rate at all.
        lines = [
            f"Cadence summary — {heading} · {target}: "
            f"{stats.ticks} tick{'s' if stats.ticks != 1 else ''}, {judged}"
        ]
        hz = self._effective_hz(stats)
        if hz is not None:
            rate = (
                f"{hz / self.multiplier:.2f} Hz policy / {hz:.2f} Hz commands"
                if self.multiplier > 1
                else f"{hz:.2f} Hz"
            )
            # The span is not ticks/rate: it sums tick-to-tick gaps, so it is short by
            # one gap per window boundary and per ``restart()``.
            lines.append(f"  effective cadence: {rate} over {stats.span:.1f} s measured")
        if stats.groups_judged:
            lines.append(
                f"  {self._judged}s over the {self.cycle_interval * ms:.1f} ms work budget: "
                f"{stats.groups_over}/{stats.groups_judged} "
                f"({100 * stats.groups_over / stats.groups_judged:.1f}%) — work mean "
                f"{stats.group_work / stats.groups_judged * ms:.1f} ms, worst "
                f"{stats.group_work_worst * ms:.1f} ms"
            )
        if self.multiplier > 1:
            lines.append(
                f"  ticks over their {self.tick_interval * ms:.1f} ms slot: "
                f"{stats.slot_overruns}/{stats.ticks} (costs interpolation smoothness only)"
            )
        if stats.starved_ticks:
            lines.append(
                f"  ticks with no action to send (inference engine starved): {stats.starved_ticks} — "
                "each commanded nothing and recorded no frame"
            )
        if stats.span_misses:
            lines.append(
                f"  {self._judged}s whose cadence slipped outside the loop body (sleep overshoot / "
                f"CPU starvation while pacing): {stats.span_misses}"
            )
        if stats.sections:
            lines.append("  loop-body steps (share of measured work):")
            width = max(len(name) for name in stats.sections)
            for name, stat in stats.sections.items():
                if stat.calls == 0:
                    continue
                share = 100 * stat.total / stats.work if stats.work > 0 else 0.0
                lines.append(
                    f"    {name:<{width}}  mean {stat.total / stat.calls * ms:6.2f} ms · worst "
                    f"{stat.worst * ms:6.2f} ms · {share:5.1f}% of work · {stat.calls} calls"
                )
        if stats.ticks:
            lines.append(
                f"  pacing headroom: {stats.sleep / stats.ticks * ms:.1f} ms slept per tick on average "
                f"(max {stats.sleep_worst * ms:.1f} ms) — near zero means the loop is saturated"
            )
        return lines

    def _close_window(self, label: str) -> None:
        """Report the finished window's digest and fold it into the run total."""
        stats = self._window
        if stats.ticks == 0:
            return
        self._windows_closed += 1
        self._report(f"Cadence ({label}): {self._summary_line(stats)}")
        self._run.fold(stats)
        self._window = _CadenceStats()

    def log_episode_summary(self, label: str | None = None) -> None:
        """Mark an episode boundary; its one-line cadence digest goes to the report sink.

        Call at each episode boundary, right after ``save_episode``.  The digest is
        emitted when the *next* tick starts — or by :meth:`log_run_summary` if the
        loop ends first — because callers are mid-tick here: the tick in progress is
        the closing episode's last one, and its ``save_episode`` cost belongs to that
        episode rather than to the one about to begin.  See :meth:`tick`.

        Once the boundary lands, the window is folded into the run total and a fresh
        one begins, so every episode is measured on its own while
        :meth:`log_run_summary` still covers the lot.  A boundary on an empty window
        reports nothing, so it is safe to call on a rotation that recorded nothing.

        Args:
            label: How to name this window in the log.  Defaults to the count of
                windows closed so far; strategies that track episodes should pass
                the dataset's own numbering instead.
        """
        self._pending_close = label or f"episode {self._windows_closed + 1}"

    def log_run_summary(self) -> None:
        """Report the whole-run cadence summary.  Call once, from the loop's ``finally``.

        Being in ``finally`` is the point: a duration limit, a ``KeyboardInterrupt``
        and a crash all still produce the summary.  A boundary still pending, and any
        window still open, are closed first — reported on their own line when the run
        had episode boundaries, folded in silently when it had none, since a
        boundary-less loop's single window *is* the run.
        """
        if self._pending_close is not None:
            self._close_window(self._pending_close)
            self._pending_close = None
        if self._window.ticks:
            if self._windows_closed:
                self._close_window("final episode")
            else:
                self._run.fold(self._window)
                self._window = _CadenceStats()
        if self._run.ticks == 0:
            return
        closed = self._windows_closed
        heading = f"whole run, {closed} episode{'s' if closed != 1 else ''}" if closed else "whole run"
        self._report("\n".join(self._summary_lines(self._run, heading)))
