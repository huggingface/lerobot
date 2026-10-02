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

"""Virtual clock for driving :class:`~lerobot.utils.cycle_timer.CycleTimer` loops.

Shared by the timer's own tests and by every loop that paces through it — the
rollout strategies, ``lerobot-record``, ``lerobot-replay``, ``lerobot-teleoperate``.
"""

import logging
import time

import pytest

TIMER_LOGGER = "lerobot.utils.cycle_timer"


class FakeClock:
    """Virtual clock standing in for ``time.perf_counter`` and ``precise_sleep``.

    ``CycleTimer``'s contract is pure arithmetic over deadlines, so exercising it
    against the wall clock only adds scheduler noise: every margin has to be wide
    enough for a loaded CI machine, which makes the assertions loose and the suite
    slow.  Driving it here instead makes the pacing exact — ``advance`` stands in
    for loop-body work, and the timer's own sleeps move the same clock forward.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.overshoot = 0.0
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Simulate *seconds* of work inside the loop body."""
        self.now += seconds

    def precise_sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        # `overshoot` models a sleep that returns late — the OS descheduling the
        # process, or a coarse timer granularity.
        self.now += seconds + self.overshoot

    def __getattr__(self, name):
        # Anything else CycleTimer's module reaches for on `time` still works.
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    """Patch the cycle timer's clock, scoped to that module's namespace only."""
    from lerobot.utils import cycle_timer

    fake = FakeClock()
    monkeypatch.setattr(cycle_timer, "time", fake)
    monkeypatch.setattr(cycle_timer, "precise_sleep", fake.precise_sleep)
    return fake


class _MessageHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def cadence_log():
    """Collect the cadence reports of a run, as a list of messages.

    Attached to the timer's own logger rather than going through ``caplog``, because the
    CLI entry points call ``init_logging()``, which clears the root handlers — pytest's
    capture handler included.
    """
    logger = logging.getLogger(TIMER_LOGGER)
    handler = _MessageHandler()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
