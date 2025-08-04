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

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from lerobot.processor.pipeline import EnvTransition, TransitionKey


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return (
        isinstance(x, numbers.Real)
        or isinstance(x, (np.integer, np.floating))
        or (isinstance(x, np.ndarray) and x.ndim == 0)
    )


def log_rerun_data(
    data: list[dict[str | Any] | EnvTransition] | dict[str | Any] | EnvTransition | None = None,
    *,
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    # Normalize "data" to a list for uniform parsing
    items = data if isinstance(data, list) else ([data] if data is not None else [])

    # Seed with explicit kwargs (if provided)
    obs = {} if observation is None else dict(observation)
    act = {} if action is None else dict(action)

    # Parse list/dict/EnvTransition inputs
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # EnvTransition-like (TransitionKey keys)
        if any(isinstance(k, TransitionKey) for k in item.keys()):
            o = item.get(TransitionKey.OBSERVATION) or {}
            a = item.get(TransitionKey.ACTION) or {}
            if isinstance(o, dict):
                obs.update(o)
            if isinstance(a, dict):
                act.update(a)
            continue

        # Plain dict: check for prefixes
        keys = list(item.keys())
        has_obs = any(str(k).startswith("observation.") for k in keys)
        has_act = any(str(k).startswith("action.") for k in keys)

        if has_obs or has_act:
            if has_obs:
                obs.update(item)
            if has_act:
                act.update(item)
        else:
            # No prefixes: assume first is observation, second is action, others -> observation
            if idx == 0:
                obs.update(item)
            elif idx == 1:
                act.update(item)
            else:
                obs.update(item)

    for k, v in obs.items():
        if v is None:
            continue
        key = k if str(k).startswith("observation.") else f"observation.{k}"

        if _is_scalar(v):
            rr.log(key, rr.Scalar(float(v)))
        elif isinstance(v, np.ndarray):
            arr = v
            # Convert CHW -> HWC when needed
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 1:
                for i, vi in enumerate(arr):
                    rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
            else:
                rr.log(key, rr.Image(arr), static=True)

    for k, v in act.items():
        if v is None:
            continue
        key = k if str(k).startswith("action.") else f"action.{k}"

        if _is_scalar(v):
            rr.log(key, rr.Scalar(float(v)))
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                for i, vi in enumerate(v):
                    rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
            else:
                # Fall back to flattening higher-d arrays
                flat = v.flatten()
                for i, vi in enumerate(flat):
                    rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
