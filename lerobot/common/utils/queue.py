#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
from queue import Empty, Queue


def get_last_item_from_queue(queue: Queue):
    item = queue.get()
    counter = 1

    # Drain queue and keep only the most recent parameters
    try:
        while True:
            item = queue.get_nowait()
            counter += 1
    except Empty:
        pass

    logging.debug(f"Drained {counter} items from queue")

    return item
