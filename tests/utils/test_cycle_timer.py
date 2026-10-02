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

"""Pacing, warnings and cadence summaries of the control-loop timer.

Every test here drives the timer on the virtual `clock` fixture (see
tests/fixtures/cadence.py).  How the strategies and the CLI loops wire it up is
covered by tests/test_rollout.py and tests/test_control_robot.py.
"""

import logging

import pytest
import torch

from lerobot.utils.cycle_timer import CycleTimer

_TIMER_LOGGER = "lerobot.utils.cycle_timer"


def _timer_warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _debug_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]


def _info_messages(caplog):
    """INFO messages from the timer only (i.e. the cadence reports)."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.name == _TIMER_LOGGER]


def _drive(timer, clock, work=0.0, ticks=1, new_cycle=True):
    """Run *ticks* iterations of a loop body costing *work* seconds each.

    Stands in for a strategy's loop: ``tick``, burn *work* on the virtual clock,
    ``wait``.  *new_cycle* is what the interpolator would report — ``True`` models a
    starved or frozen one (every tick asks for a fresh action), and a callable
    ``tick_index -> bool`` covers the cadences in between.
    """
    for i in range(ticks):
        timer.tick(new_cycle=new_cycle(i) if callable(new_cycle) else new_cycle)
        clock.advance(work)
        timer.wait()


# ---------------------------------------------------------------------------
# Pacing and per-tick telemetry
# ---------------------------------------------------------------------------


def test_cycle_timer_paces_ticks_to_base_fps(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        _drive(timer, clock, ticks=4, new_cycle=lambda i: i % 2 == 0)  # two full cycles
    assert clock.now == pytest.approx(2 * (1 / 10.0))
    assert not _timer_warnings(caplog)


def test_cycle_timer_spaces_interpolated_commands_evenly(clock):
    # Interpolation exists to smooth motion, so every tick must be spaced by
    # 1/(fps × multiplier) — not batched at the start of each cycle.
    timer = CycleTimer(10.0, 2)  # 50 ms slots
    stamps = []
    for tick in range(4):
        timer.tick(new_cycle=tick % 2 == 0)
        stamps.append(clock.now)
        timer.wait()
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert gaps == pytest.approx([0.05, 0.05, 0.05])


def test_cycle_timer_slow_policy_tick_borrows_from_interpolated_ticks(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_TIMER_LOGGER):
        timer.tick(new_cycle=True)
        clock.advance(0.06)  # policy tick overruns its 50 ms slot
        timer.wait()
        _drive(timer, clock, new_cycle=False)  # instant interpolated tick absorbs it
    # The 60 ms + 0 ms cycle fits the 100 ms budget: no user-facing warning,
    # only a DEBUG note about the tick that missed its slot.
    assert not _timer_warnings(caplog)
    assert any("slot" in m for m in _debug_messages(caplog))
    # The cycle still ends on its deadline, so the policy cadence is held.
    assert clock.now == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("fps", "multiplier", "work", "ticks", "expected"),
    [
        # Every tick is its own cycle at multiplier 1, so each slow tick past the
        # exempt start-up one warns: 80 ms of work against a 50 ms budget.
        (20.0, 1, 0.08, 3, 2),
        # At multiplier 2 a single 120 ms tick already blows the 100 ms cycle
        # budget, but it is judged per group of two: 4 ticks, 2 groups, 1 exempt.
        (10.0, 2, 0.12, 4, 1),
    ],
    ids=["multiplier-1", "multiplier-2"],
)
def test_cycle_timer_warns_once_per_over_budget_cycle(caplog, clock, fps, multiplier, work, ticks, expected):
    timer = CycleTimer(fps, multiplier)
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=work, ticks=ticks)
    warnings = _timer_warnings(caplog)
    assert len(warnings) == expected
    assert f"target FPS ({fps:g}" in warnings[0].getMessage()
    assert "Dataset frames" in warnings[0].getMessage()


def test_cycle_timer_does_not_report_the_startup_group(caplog, clock):
    # The interpolator primes its buffer with a single action, so inference runs
    # on two consecutive ticks at start-up and the first group legitimately runs
    # over budget.  Reporting it would warn on every healthy launch.
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.12, ticks=2)
    assert not _timer_warnings(caplog)


def test_cycle_timer_reports_a_slot_overrun_on_the_tick_that_closes_a_group(caplog, clock):
    # Regression: the overrun note used to sit in an `elif` behind the group-close
    # branch, so every multiplier-th tick's overrun went unreported even when the
    # group as a whole was healthy.  Here the *closing* tick is the slow one.
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_TIMER_LOGGER):
        _drive(timer, clock, ticks=2, new_cycle=lambda i: i == 0)  # start-up group, exempt
        timer.tick(new_cycle=True)
        timer.wait()  # instant opening tick
        timer.tick(new_cycle=False)
        clock.advance(0.06)  # closing tick overruns its 50 ms slot...
        timer.wait()
    assert not _timer_warnings(caplog)  # ...while 60 ms still fits the 100 ms budget
    assert [m for m in _debug_messages(caplog) if "overran its" in m]

    # The counter behind that note is otherwise only visible in the summary, so it
    # needs its own assertion: without one, disabling the increment leaves the
    # suite green.
    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        timer.log_run_summary()
    assert (
        "ticks over their 50.0 ms slot: 1/4 (costs interpolation smoothness only)"
        in (_info_messages(caplog)[-1])
    )


def test_cycle_timer_warns_when_every_tick_reports_a_new_cycle(caplog, clock):
    # Regression: when the interpolator is starved or frozen (async backend
    # yielding no action, DAgger paused/correcting), every tick reports
    # new_cycle=True.  Tying the slow-loop warning to cycle completion made it
    # structurally unreachable in exactly that regime.  Groups of `multiplier`
    # ticks are measured regardless, so a genuinely slow loop still warns.
    timer = CycleTimer(10.0, 2)  # 100 ms budget per 2 ticks
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.07, ticks=6)  # 140 ms per 2-tick group, 3 groups
    warnings = _timer_warnings(caplog)
    assert len(warnings) == 2  # one per closed group, not one per tick
    assert "target FPS (10" in warnings[0].getMessage()


@pytest.mark.parametrize("multiplier", [2, 3])
def test_cycle_timer_silent_on_healthy_loop_driven_by_the_real_interpolator(caplog, clock, multiplier):
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Regression: the reporting window must not depend on WHERE it starts
    # relative to the policy cycle.  The real interpolator primes with a
    # single-action buffer, which permanently offsets groups from cycles, so
    # hand-aligned `new_cycle` flags (what a naive test supplies) hide the bug.
    # Drive the flags from the real interpolator instead, with an inference tick
    # deliberately longer than one 1/(fps × N) slot — the regime interpolation
    # exists for — while total work per cycle stays well inside the 1/fps budget.
    fps = 20.0
    policy_work = 1.0 / (fps * multiplier) * 1.5  # overruns its slot
    interp_work = 0.002
    assert policy_work + (multiplier - 1) * interp_work < 1.0 / fps  # healthy loop

    interpolator = ActionInterpolator(multiplier=multiplier)
    timer = CycleTimer(fps, multiplier)
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        for i in range(4 * multiplier):
            needs_action = interpolator.needs_new_action()
            timer.tick(new_cycle=needs_action)
            if needs_action:
                interpolator.add(torch.tensor([float(i)]))
                clock.advance(policy_work)
            else:
                clock.advance(interp_work)
            interpolator.get()
            timer.wait()
    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]


def test_cycle_timer_reports_time_lost_inside_the_pacing_sleep(caplog, clock):
    # Loop-body work fits the budget comfortably, but every pacing sleep returns
    # 10 ms late, so the achieved cadence really is below target.  That shortfall
    # is not the caller's doing and would fire constantly on a loaded machine, so
    # it is reported at DEBUG rather than as the slow-loop warning.
    clock.overshoot = 0.01
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.01, ticks=6)

    assert not _timer_warnings(caplog)
    debugs = _debug_messages(caplog)
    assert any("went missing outside the loop body" in m for m in debugs), debugs


def test_cycle_timer_tolerates_a_sliver_of_sleep_overshoot(caplog, clock):
    # Sleeps that return a few hundred microseconds late are the ordinary state of
    # a paced loop, not a cadence problem.  With no tolerance at all this note
    # fired on nearly every group of a healthy run, which made both the message
    # and the count in the summary worthless.
    clock.overshoot = 0.0004  # 0.8 ms per 100 ms cycle: inside the 1% tolerance
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.DEBUG, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.01, ticks=6)

    assert not caplog.records


def test_cycle_timer_new_cycle_reanchors_pacing(clock):
    # The interpolator's startup buffer holds a single action, so the second
    # tick requests a fresh action one tick early.  Re-anchoring must restart
    # the pacing slots from that tick rather than keep the stale anchor.
    timer = CycleTimer(10.0, 2)  # 50 ms slots
    timer.tick(new_cycle=True)
    timer.wait()  # sleeps to the first slot deadline
    reanchor = clock.now
    timer.tick(new_cycle=True)
    timer.wait()
    # Paced from the re-anchor, so a full slot — not the 0 ms a stale anchor
    # (already past its second deadline) would have produced.
    assert clock.now - reanchor == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Cadence statistics and summaries
# ---------------------------------------------------------------------------


def test_cycle_timer_run_summary_reports_effective_cadence_and_sections(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    for i in range(6):
        timer.tick(new_cycle=i % 2 == 0)
        with timer.section("observe"):
            clock.advance(0.01)
        with timer.section("infer"):
            clock.advance(0.02 if i % 2 == 0 else 0.0)  # inference on policy ticks only
        timer.wait()

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    # The sample size is in the heading, so it survives even when the rate below
    # cannot be computed.
    assert summary.startswith(
        "Cadence summary — whole run · target 10 Hz × 2 (50.0 ms tick slot, 100.0 ms cycle budget): "
        "6 ticks, 2 cycles judged"
    )
    # A correctly paced loop holds its target exactly on a virtual clock.
    # 5 gaps of 50 ms — the span sums gaps, so it is one gap short of 6 ticks' worth.
    assert "effective cadence: 10.00 Hz policy / 20.00 Hz commands over 0.2 s measured" in summary
    # Three groups of two, the first exempt as start-up; 40 ms of work each.
    assert "cycles over the 100.0 ms work budget: 0/2 (0.0%)" in summary
    assert "work mean 40.0 ms, worst 40.0 ms" in summary
    # Sections split the measured work; both ran on every tick, 30 ms each cycle.
    # `infer` alternates 20/0 ms, so its mean and worst have to differ.
    assert "observe  mean  10.00 ms · worst  10.00 ms ·  50.0% of work · 6 calls" in summary
    assert "infer    mean  10.00 ms · worst  20.00 ms ·  50.0% of work · 6 calls" in summary
    # Headroom is the pacing sleep, which is the whole budget minus the work: the
    # policy ticks sleep 20 ms of their 50 ms slot, the interpolated ticks 40 ms.
    assert "pacing headroom: 30.0 ms slept per tick on average (max 40.0 ms)" in summary


def test_cycle_timer_episode_summaries_fold_into_the_run_total(caplog, clock):
    timer = CycleTimer(10.0, 1)  # every tick is a cycle, 100 ms budget
    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.01, ticks=4)
        timer.log_episode_summary("episode 1")
        _drive(timer, clock, work=0.01, ticks=4)
        timer.log_episode_summary("episode 2")
        _drive(timer, clock, work=0.01, ticks=2)
        timer.log_run_summary()  # closes the window still open

    messages = _info_messages(caplog)
    assert len(messages) == 4
    assert [m.split(":")[0] for m in messages[:3]] == [
        "Cadence (episode 1)",
        "Cadence (episode 2)",
        "Cadence (final episode)",
    ]
    assert messages[0].startswith("Cadence (episode 1): 10.00 Hz vs 10 Hz target · 4 ticks")
    # The run total covers every tick of all three windows, and only the run block
    # carries the full breakdown.
    assert messages[3].startswith("Cadence summary — whole run, 3 episodes")
    assert "10 ticks" in messages[3]
    assert "pacing headroom: 90.0 ms slept per tick on average (max 90.0 ms)" in messages[3]


def test_cycle_timer_report_sink_takes_the_summaries_off_the_log(caplog, clock):
    """Both summaries go to the sink; per-tick warnings stay on the logger."""
    reported: list[str] = []
    timer = CycleTimer(10.0, 1, report=reported.append)  # 100 ms budget per tick

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.01, ticks=4)
        timer.log_episode_summary("episode 1")
        _drive(timer, clock, work=0.5, ticks=2)  # 500 ms of work: over budget
        timer.log_run_summary()

    assert not _info_messages(caplog)
    assert len(reported) == 3
    assert reported[0].startswith("Cadence (episode 1): 10.00 Hz vs 10 Hz target · 4 ticks")
    assert reported[1].startswith("Cadence (final episode):")
    assert reported[2].startswith("Cadence summary — whole run, 2 episodes")
    assert len(_timer_warnings(caplog)) == 2
    assert not any("running slower" in message for message in reported)


def test_cycle_timer_empty_window_reports_nothing(clock, caplog):
    # Boundaries can fall before any tick completes (a rotation on the very first
    # iteration), and a strategy should not have to guard against that.
    timer = CycleTimer(10.0, 1)
    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        timer.log_episode_summary("episode 1")
        timer.log_run_summary()
    assert not _info_messages(caplog)


def test_cycle_timer_restart_inside_a_loop_body_drops_the_stall_it_follows(caplog, clock):
    # Regression: `restart()` is only ever called from *inside* a loop body — after
    # the blocking work and before `wait()` (a ring-buffer flush, DAgger's handover
    # ramp).  Elapsed time is a backward-looking gap, so cutting the link at that
    # moment discarded the healthy gap already behind the tick and billed the stall
    # to the next one, understating the achieved cadence instead of excluding it.
    timer = CycleTimer(10.0, 1)  # 100 ms budget
    _drive(timer, clock, work=0.005, ticks=4)
    timer.tick(new_cycle=True)
    clock.advance(0.005)  # honest work...
    clock.advance(0.5)  # ...then the blocking one-off, still inside the body
    timer.restart()
    timer.wait()
    _drive(timer, clock, work=0.005, ticks=4)

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "9 ticks" in summary
    assert "effective cadence: 10.00 Hz over 0.7 s measured" in summary
    assert not _timer_warnings(caplog)  # the stalled group is exempted too


def test_cycle_timer_episode_boundary_inside_a_loop_body_belongs_to_the_closing_episode(caplog, clock):
    # Regression: strategies request a boundary after the blocking `save_episode()`
    # and before `wait()`, so the tick in progress is the closing episode's last one.
    # Folding the window there booked that tick — the entire save — against the
    # episode about to start, whose headline cadence then collapsed while the work
    # column beside it still read healthy.
    timer = CycleTimer(10.0, 1)
    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        _drive(timer, clock, work=0.005, ticks=4)
        timer.tick(new_cycle=True)  # the rotation tick
        clock.advance(0.005)
        clock.advance(0.5)  # blocking save_episode
        timer.log_episode_summary("episode 1")
        timer.restart()
        timer.wait()
        _drive(timer, clock, work=0.005, ticks=4)
        timer.log_run_summary()

    episode_1, episode_2, run = _info_messages(caplog)
    # The rotation tick counts toward the episode it finalised, and its stalled group
    # is exempt, so the work column stays honest at 5 ms rather than 505 ms.
    assert episode_1 == (
        "Cadence (episode 1): 10.00 Hz vs 10 Hz target · 5 ticks, 0.4 s measured · "
        "0/3 ticks over the 100.0 ms budget (work mean 5.0 ms, worst 5.0 ms)"
    )
    # ...and the next episode is not slandered by a stall it never paid for.
    assert episode_2.startswith("Cadence (final episode): 10.00 Hz vs 10 Hz target · 4 ticks, 0.3 s measured")
    assert "9 ticks" in run and "effective cadence: 10.00 Hz over" in run
