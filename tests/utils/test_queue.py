#!/usr/bin/env python

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

from queue import Queue

from lerobot.common.utils.queue import get_last_item_from_queue


def test_get_last_item_single_item():
    """Test getting the last item when queue has only one item."""
    queue = Queue()
    queue.put("single_item")

    result = get_last_item_from_queue(queue)

    assert result == "single_item"
    assert queue.empty()


def test_get_last_item_multiple_items():
    """Test getting the last item when queue has multiple items."""
    queue = Queue()
    items = ["first", "second", "third", "fourth", "last"]

    for item in items:
        queue.put(item)

    result = get_last_item_from_queue(queue)

    assert result == "last"
    assert queue.empty()


def test_get_last_item_different_types():
    """Test with different data types in the queue."""
    queue = Queue()
    items = [1, 2.5, "string", {"key": "value"}, [1, 2, 3], ("tuple", "data")]

    for item in items:
        queue.put(item)

    result = get_last_item_from_queue(queue)

    assert result == ("tuple", "data")
    assert queue.empty()


def test_get_last_item_maxsize_queue():
    """Test with a queue that has a maximum size."""
    queue = Queue(maxsize=5)

    # Fill the queue
    for i in range(5):
        queue.put(i)

    result = get_last_item_from_queue(queue)

    assert result == 4
    assert queue.empty()


def test_get_last_item_with_none_values():
    """Test with None values in the queue."""
    queue = Queue()
    items = [1, None, 2, None, 3]

    for item in items:
        queue.put(item)

    result = get_last_item_from_queue(queue)

    assert result == 3
    assert queue.empty()
