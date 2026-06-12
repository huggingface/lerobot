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

"""Unit tests for the RoundRobinScheduler fairness guarantees.

The scheduler only reads ``client_uuid`` from sessions, so minimal fakes
stand in for real Session objects.
"""

from __future__ import annotations

from collections import Counter

from lerobot.policy_server.scheduler import RoundRobinScheduler


class FakeSession:
    """The scheduler touches nothing but client_uuid."""

    def __init__(self, client_uuid: str):
        self.client_uuid = client_uuid


def picks(scheduler: RoundRobinScheduler, ready: list[FakeSession], n: int) -> list[str]:
    served = []
    for _ in range(n):
        chosen = scheduler.select(list(ready))
        assert len(chosen) == 1
        served.append(chosen[0].client_uuid)
    return served


def test_empty_ready_returns_empty():
    scheduler = RoundRobinScheduler()
    assert scheduler.select([]) == []


def test_empty_ready_after_serving_returns_empty():
    scheduler = RoundRobinScheduler()
    scheduler.select([FakeSession("a")])
    assert scheduler.select([]) == []


def test_single_session_picked_repeatedly():
    scheduler = RoundRobinScheduler()
    only = FakeSession("solo")
    for _ in range(5):
        assert scheduler.select([only]) == [only]


def test_three_sessions_served_fairly_in_sorted_uuid_order():
    scheduler = RoundRobinScheduler()
    a, b, c = FakeSession("a"), FakeSession("b"), FakeSession("c")
    # Pass ready in non-sorted order: the ring is sorted by uuid internally.
    served = picks(scheduler, [c, a, b], 9)

    assert served == ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
    assert Counter(served) == {"a": 3, "b": 3, "c": 3}


def test_session_leaving_between_calls_keeps_fairness():
    scheduler = RoundRobinScheduler()
    a, b, c = FakeSession("a"), FakeSession("b"), FakeSession("c")
    assert scheduler.select([a, b, c]) == [a]

    # 'a' leaves; remaining sessions alternate with no crash or starvation.
    served = picks(scheduler, [b, c], 4)
    assert served == ["b", "c", "b", "c"]


def test_departed_last_served_uuid_resumes_after_it():
    scheduler = RoundRobinScheduler()
    a, b, c = FakeSession("a"), FakeSession("b"), FakeSession("c")
    picks(scheduler, [a, b, c], 2)  # last served is 'b'

    # 'b' leaves; the next pick is the first uuid greater than 'b'.
    assert scheduler.select([a, c]) == [c]
    assert scheduler.select([a, c]) == [a]


def test_wraparound_from_last_uuid_back_to_first():
    scheduler = RoundRobinScheduler()
    a, b, c = FakeSession("a"), FakeSession("b"), FakeSession("c")
    assert scheduler.select([c]) == [c]  # last served is the highest uuid

    # Everyone is <= last served: wrap back to the first sorted uuid.
    assert scheduler.select([a, b, c]) == [a]


def test_newly_ready_session_joins_ring_fairly():
    scheduler = RoundRobinScheduler()
    a, c = FakeSession("a"), FakeSession("c")
    served = picks(scheduler, [a, c], 2)
    assert served == ["a", "c"]

    # 'b' becomes ready; wrap-around lands on 'a', then 'b' gets its turn.
    b = FakeSession("b")
    served = picks(scheduler, [a, b, c], 3)
    assert served == ["a", "b", "c"]


def test_no_starvation_over_many_cycles():
    scheduler = RoundRobinScheduler()
    ready = [FakeSession(f"u{i:02d}") for i in range(5)]
    served = picks(scheduler, ready, 50)
    assert Counter(served) == {f"u{i:02d}": 10 for i in range(5)}
