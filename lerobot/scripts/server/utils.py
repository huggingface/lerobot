#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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
import signal
import sys
from queue import Empty
from typing import TypedDict

import torch
from torch.multiprocessing import Queue

shutdown_event_counter = 0


def setup_process_handlers(use_threads: bool) -> any:
    if use_threads:
        from threading import Event
    else:
        from multiprocessing import Event

    shutdown_event = Event()

    # Define signal handler
    def signal_handler(signum, frame):
        logging.info("Shutdown signal received. Cleaning up...")
        shutdown_event.set()
        global shutdown_event_counter
        shutdown_event_counter += 1

        if shutdown_event_counter > 1:
            logging.info("Force shutdown")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request (kill)
    signal.signal(signal.SIGHUP, signal_handler)  # Terminal closed/Hangup
    signal.signal(signal.SIGQUIT, signal_handler)  # Ctrl+\

    def signal_handler(signum, frame):
        logging.info("Shutdown signal received. Cleaning up...")
        shutdown_event.set()

    return shutdown_event


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


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    truncated: bool
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def move_transition_to_device(transition: Transition, device: str = "cpu") -> Transition:
    device = torch.device(device)
    non_blocking = device.type == "cuda"

    # Move state tensors to device
    transition["state"] = {
        key: val.to(device, non_blocking=non_blocking) for key, val in transition["state"].items()
    }

    # Move action to device
    transition["action"] = transition["action"].to(device, non_blocking=non_blocking)

    # Move reward and done if they are tensors
    if isinstance(transition["reward"], torch.Tensor):
        transition["reward"] = transition["reward"].to(device, non_blocking=non_blocking)

    if isinstance(transition["done"], torch.Tensor):
        transition["done"] = transition["done"].to(device, non_blocking=non_blocking)

    if isinstance(transition["truncated"], torch.Tensor):
        transition["truncated"] = transition["truncated"].to(device, non_blocking=non_blocking)

    # Move next_state tensors to device
    transition["next_state"] = {
        key: val.to(device, non_blocking=non_blocking) for key, val in transition["next_state"].items()
    }

    # Move complementary_info tensors if present
    if transition.get("complementary_info") is not None:
        for key, val in transition["complementary_info"].items():
            if isinstance(val, torch.Tensor):
                transition["complementary_info"][key] = val.to(device, non_blocking=non_blocking)
            elif isinstance(val, (int, float, bool)):
                transition["complementary_info"][key] = torch.tensor(val, device=device)
            else:
                raise ValueError(f"Unsupported type {type(val)} for complementary_info[{key}]")
    return transition


def move_state_dict_to_device(state_dict, device="cpu"):
    """
    Recursively move all tensors in a (potentially) nested
    dict/list/tuple structure to the CPU.
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    elif isinstance(state_dict, dict):
        return {k: move_state_dict_to_device(v, device=device) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [move_state_dict_to_device(v, device=device) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in state_dict)
    else:
        return state_dict
