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

"""Waiting behaviour of `precise_sleep`.

The control loops in lerobot.utils.cycle_timer and lerobot.rollout.strategies pace
themselves with this helper and the branch it takes depends on the host platform.
Both branches are driven here on a fake clock so the suite never really waits.
"""

import platform
import time

import pytest

from lerobot.utils.robot_utils import precise_sleep

SPIN_PLATFORMS = ["Darwin", "Windows"]
ALL_PLATFORMS = ["Linux", *SPIN_PLATFORMS]


class FakeClock:
    """Monotonic clock that ticks whenever it is read or slept on.

    Reading has to move time forward because the spin branch of `precise_sleep` does
    nothing but re-read the clock. On a clock that only advanced inside sleep the spin
    would never reach its deadline.
    """

    def __init__(self, tick: float = 1e-4):
        self.now = 0.0
        self.tick = tick
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        value = self.now
        self.now += self.tick
        return value

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            # Match time.sleep, which rejects a negative duration rather than
            # returning immediately. A fake that quietly accepted one would let a
            # missing guard in precise_sleep run the clock backwards forever.
            raise ValueError("sleep length must be non-negative")
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, "perf_counter", fake.perf_counter)
    monkeypatch.setattr(time, "sleep", fake.sleep)
    return fake


def set_platform(monkeypatch, name: str) -> None:
    monkeypatch.setattr(platform, "system", lambda: name)


@pytest.mark.parametrize("system", ALL_PLATFORMS)
@pytest.mark.parametrize("seconds", [0.0, -0.5])
def test_non_positive_duration_returns_without_sleeping(clock, monkeypatch, system, seconds):
    set_platform(monkeypatch, system)

    precise_sleep(seconds)

    assert clock.sleeps == []


@pytest.mark.parametrize("system", ALL_PLATFORMS)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spin_threshold": -0.001}, "spin_threshold must be >= 0"),
        ({"sleep_margin": -0.001}, "sleep_margin must be >= 0"),
    ],
)
def test_negative_tuning_values_are_rejected(clock, monkeypatch, system, kwargs, message):
    set_platform(monkeypatch, system)

    with pytest.raises(ValueError, match=message):
        precise_sleep(0.05, **kwargs)

    assert clock.sleeps == []


def test_a_zero_duration_short_circuits_before_the_tuning_is_checked(clock, monkeypatch):
    """Nothing is waited on so the invalid value never gets a chance to matter."""
    set_platform(monkeypatch, "Darwin")

    precise_sleep(0.0, spin_threshold=-1.0)

    assert clock.sleeps == []


def test_linux_hands_the_whole_duration_to_time_sleep(clock, monkeypatch):
    set_platform(monkeypatch, "Linux")

    precise_sleep(0.05)

    assert clock.sleeps == [0.05]


def test_linux_ignores_the_spin_tuning(clock, monkeypatch):
    """The spin loop is a macOS and Windows concern so Linux sleeps once either way."""
    set_platform(monkeypatch, "Linux")

    precise_sleep(0.05, spin_threshold=0.5, sleep_margin=0.4)

    assert clock.sleeps == [0.05]


@pytest.mark.parametrize("system", SPIN_PLATFORMS)
def test_spinning_platforms_leave_a_margin_then_spin_out_the_rest(clock, monkeypatch, system):
    set_platform(monkeypatch, system)

    precise_sleep(0.05, spin_threshold=0.010, sleep_margin=0.005)

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] < 0.05
    assert clock.now >= 0.05


@pytest.mark.parametrize("system", SPIN_PLATFORMS)
def test_spinning_platforms_never_sleep_past_the_deadline(clock, monkeypatch, system):
    """Oversleeping lands the next control tick late so no single sleep may overshoot."""
    set_platform(monkeypatch, system)

    precise_sleep(0.05)

    assert sum(clock.sleeps) <= 0.05


@pytest.mark.parametrize("system", SPIN_PLATFORMS)
def test_a_wait_under_the_spin_threshold_is_spun_out_entirely(clock, monkeypatch, system):
    set_platform(monkeypatch, system)

    precise_sleep(0.005, spin_threshold=0.010)

    assert clock.sleeps == []
    assert clock.now >= 0.005


@pytest.mark.parametrize("system", SPIN_PLATFORMS)
def test_a_sleep_margin_wider_than_the_wait_never_goes_negative(clock, monkeypatch, system):
    """time.sleep raises on a negative argument which is what the max() call prevents."""
    set_platform(monkeypatch, system)

    precise_sleep(0.05, spin_threshold=0.001, sleep_margin=1.0)

    assert all(slept == 0.0 for slept in clock.sleeps)
    assert clock.now >= 0.05


def test_a_real_wait_is_not_cut_short():
    """One unmocked call on whatever platform the suite runs on."""
    start = time.perf_counter()
    precise_sleep(0.02)
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.019
