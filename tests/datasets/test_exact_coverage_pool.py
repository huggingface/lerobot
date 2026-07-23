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

"""ExactCoveragePool: exactly-once frame coverage over a bounded episode pool."""

from collections import Counter

from lerobot.streaming.episode_video import ExactCoveragePool

EPISODES = [(0, 5), (1, 3), (2, 8), (3, 1), (4, 6), (5, 4), (6, 7), (7, 2)]
TOTAL = sum(n for _, n in EPISODES)
EXPECTED = Counter((ep, i) for ep, n in EPISODES for i in range(n))


def _drain(pool):
    out, max_resident = [], 0
    while True:
        try:
            out.append(next(pool))
        except StopIteration:
            break
        max_resident = max(max_resident, len(pool.resident))
    return out, max_resident


def test_exact_once_coverage():
    out, _ = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=42))
    assert len(out) == TOTAL
    assert Counter(out) == EXPECTED  # every (episode, frame) exactly once, no dups/misses


def test_pool_never_exceeds_size():
    _, max_resident = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=42))
    assert max_resident <= 3


def test_deterministic_per_seed_and_epoch():
    a, _ = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=7))
    b, _ = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=7))
    c, _ = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=8))
    d, _ = _drain(ExactCoveragePool(EPISODES, pool_size=3, seed=7, epoch=1))
    assert a == b
    assert a != c and a != d  # seed and epoch both change the order
    assert Counter(c) == EXPECTED and Counter(d) == EXPECTED  # ... but coverage is preserved


def test_admission_and_eviction_events():
    pool = ExactCoveragePool(EPISODES, pool_size=3, seed=0)
    admitted_ever, evicted_ever = set(), set()
    # first three episodes admitted at construction
    admitted_ever.update(pool.newly_admitted)
    assert len(admitted_ever) == 3
    while True:
        pool.newly_admitted.clear()
        pool.evicted.clear()
        try:
            next(pool)
        except StopIteration:
            break
        admitted_ever.update(pool.newly_admitted)
        evicted_ever.update(pool.evicted)
    assert admitted_ever == {ep for ep, _ in EPISODES}  # every episode admitted exactly once
    # every episode except the pool_size still resident at the end is evicted on exhaustion
    assert len(evicted_ever) >= len(EPISODES) - 3


def test_uniform_mixing_matches_coupon_collector():
    # 64 equal episodes, pool 64, first 64 draws -> ~64*(1-(1-1/64)^64) ~= 41 distinct
    big = [(e, 100) for e in range(64)]
    pool = ExactCoveragePool(big, pool_size=64, seed=0)
    head = [next(pool)[0] for _ in range(64)]
    assert len(set(head)) >= 30  # far above sequential (=1); ~41 expected


def test_large_epoch_bounded_and_complete():
    big = [(e, 90) for e in range(500)]
    out, max_resident = _drain(ExactCoveragePool(big, pool_size=64, seed=3))
    assert len(out) == 500 * 90
    assert len(set(out)) == 500 * 90  # exactly once
    assert max_resident <= 64


def test_zero_length_episodes_skipped():
    pool = ExactCoveragePool([(0, 3), (1, 0), (2, 2)], pool_size=8, seed=0)
    out, _ = _drain(pool)
    assert Counter(out) == Counter({(0, 0): 1, (0, 1): 1, (0, 2): 1, (2, 0): 1, (2, 1): 1})
