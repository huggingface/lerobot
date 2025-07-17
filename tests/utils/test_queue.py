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

import pytest
from torch.multiprocessing import Queue as TorchMPQueue

from lerobot.utils.queue import get_last_item_from_queue


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_single_item(queue_cls):
    """Test getting the last item when queue has only one item."""
    queue = queue_cls()
    queue.put("single_item")

    result = get_last_item_from_queue(queue)

    assert result == "single_item"
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_multiple_items(queue_cls):
    """Test getting the last item when queue has multiple items."""
    queue = queue_cls()
    items = ["first", "second", "third", "fourth", "last"]

    for item in items:
        queue.put(item)

    result = get_last_item_from_queue(queue)

    assert result == "last"
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_different_types(queue_cls):
    """Test with different data types in the queue."""
    queue = queue_cls()
    items = [1, 2.5, "string", {"key": "value"}, [1, 2, 3], ("tuple", "data")]

    for item in items:
        queue.put(item)

    result = get_last_item_from_queue(queue)

    assert result == ("tuple", "data")
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_maxsize_queue(queue_cls):
    """Test with a queue that has a maximum size."""
    queue = queue_cls(maxsize=5)

    # Fill the queue
    for i in range(5):
        queue.put(i)

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue)

    assert result == 4
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_with_none_values(queue_cls):
    """Test with None values in the queue."""
    queue = queue_cls()
    items = [1, None, 2, None, 3]

    for item in items:
        queue.put(item)

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue)

    assert result == 3
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_blocking_timeout(queue_cls):
    """Test get_last_item_from_queue returns None on timeout."""
    queue = queue_cls()
    result = get_last_item_from_queue(queue, block=True, timeout=0.1)
    assert result is None


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_non_blocking_empty(queue_cls):
    """Test get_last_item_from_queue with block=False on an empty queue returns None."""
    queue = queue_cls()
    result = get_last_item_from_queue(queue, block=False)
    assert result is None


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_non_blocking_success(queue_cls):
    """Test get_last_item_from_queue with block=False on a non-empty queue."""
    queue = queue_cls()
    items = ["first", "second", "last"]
    for item in items:
        queue.put(item)

    # Give the queue time to fill
    time.sleep(0.1)

    result = get_last_item_from_queue(queue, block=False)
    assert result == "last"
    assert queue.empty()


@pytest.mark.parametrize("queue_cls", [Queue, TorchMPQueue])
def test_get_last_item_blocking_waits_for_item(queue_cls):
    """Test that get_last_item_from_queue waits for an item if block=True."""
    queue = queue_cls()
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
