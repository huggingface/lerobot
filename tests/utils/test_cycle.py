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

from itertools import islice

import pytest

from lerobot.utils.utils import cycle


@pytest.mark.timeout(5)
@pytest.mark.parametrize("empty", [[], (), iter(())])
def test_cycle_empty_iterable_stops(empty):
    assert list(cycle(empty)) == []


@pytest.mark.timeout(5)
def test_cycle_one_shot_iterator_stops_after_exhaustion():
    assert list(cycle(iter([1, 2]))) == [1, 2]


def test_cycle_stops_at_first_empty_pass():
    class Epochs:
        def __init__(self):
            self.epochs = iter(([1, 2], [], [3]))

        def __iter__(self):
            return iter(next(self.epochs))

    assert list(islice(cycle(Epochs()), 3)) == [1, 2]


def test_cycle_fetches_new_items_on_each_pass():
    class Epochs:
        epoch = 0

        def __iter__(self):
            self.epoch += 1
            return iter([self.epoch])

    assert list(islice(cycle(Epochs()), 4)) == [1, 2, 3, 4]


def test_cycle_preserves_none_values():
    assert list(islice(cycle([None, 1]), 5)) == [None, 1, None, 1, None]


def test_cycle_propagates_source_errors():
    def source():
        yield 1
        raise ValueError("source failed")

    iterator = cycle(source())
    assert next(iterator) == 1
    with pytest.raises(ValueError, match="source failed"):
        next(iterator)
