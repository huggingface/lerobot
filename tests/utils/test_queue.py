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

import threading
import time
from queue import Queue

from lerobot.utils.queue import get_last_item_from_queue


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

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue)

    assert result == 4
    assert queue.empty()


def test_get_last_item_with_none_values():
    """Test with None values in the queue."""
    queue = Queue()
    items = [1, None, 2, None, 3]

    for item in items:
        queue.put(item)

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue)

    assert result == 3
    assert queue.empty()


def test_get_last_item_blocking_timeout():
    """Test get_last_item_from_queue returns None on timeout."""
    queue = Queue()
    result = get_last_item_from_queue(queue, block=True, timeout=0.1)
    assert result is None


def test_get_last_item_non_blocking_empty():
    """Test get_last_item_from_queue with block=False on an empty queue returns None."""
    queue = Queue()
    result = get_last_item_from_queue(queue, block=False)
    assert result is None


def test_get_last_item_non_blocking_success():
    """Test get_last_item_from_queue with block=False on a non-empty queue."""
    queue = Queue()
    items = ["first", "second", "last"]
    for item in items:
        queue.put(item)

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue, block=False)
    assert result == "last"
    assert queue.empty()


def test_get_last_item_blocking_waits_for_item():
    """Test that get_last_item_from_queue waits for an item if block=True."""
    queue = Queue()
    result = []

    def producer():
        queue.put("item1")
        queue.put("item2")

    def consumer():
        # This will block until the producer puts the first item
        item = get_last_item_from_queue(queue, block=True, timeout=0.2)
        result.append(item)

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    assert result == ["item2"]
    assert queue.empty()
