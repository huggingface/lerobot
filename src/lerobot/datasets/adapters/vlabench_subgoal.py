"""Wraps a `LeRobotDataset`(-like) instance with a per-frame `observation.subgoal_state` label,
precomputed offline by `examples/safediff_vla/compute_subgoal_labels.py`.

Why this exists: SafeDiff-VLA's state-predictor head (see `lerobot.policies.safediff_vla`) needs a
real regression target. True object positions aren't recoverable for episodes already in
`lerobot/vlabench_unified` -- VLABench's per-episode object layout isn't reproducible from
`env.reset(seed=...)` (verified empirically: the same seed yields a different target object and
position on repeated resets, because the upstream task samplers use Python's global `random`
module, reseeded from OS entropy on retries -- see `lerobot.envs.vlabench._ensure_env`'s
docstring). Every episode's own `action` gripper channel, however, already encodes exactly when
the demonstrated pick/place happens (open->close = grasp, close->open = release): the
`observation.state` at that frame is a zero-cost, already-in-hand proxy target, computed once
and reused across training runs.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch


class SubgoalLabelDataset:
    """Attaches a precomputed `observation.subgoal_state` feature to every returned frame.

    Composes with (rather than subclasses) the base dataset so it works for both `LeRobotDataset`
    and `StreamingLeRobotDataset` alike; every attribute other than `__len__`/`__getitem__` is
    passed straight through to the wrapped instance.

    Labels are kept as one contiguous `numpy` array plus a sorted index for
    `numpy.searchsorted` lookups, *not* as a dict of millions of individual `torch.Tensor`
    objects: a `DataLoader` with `num_workers > 0` has to send the dataset to each worker
    process, and PyTorch shares every individual tensor's storage through its own
    `/dev/shm`-backed mmap -- millions of one-off tensors exhausts shared-memory file
    descriptors well before it exhausts actual memory (`unable to mmap ... Cannot allocate
    memory`). A `numpy` array is pickled by value like any other array, with no per-element
    sharing machinery, so this costs one copy of a few hundred MB per worker instead.
    """

    def __init__(self, base_dataset: Any, labels_path: str) -> None:
        self.base_dataset = base_dataset
        labels = pd.read_parquet(labels_path, columns=["index", "subgoal_state"]).sort_values("index")
        self._label_indices = labels["index"].to_numpy()
        self._label_values = np.stack(labels["subgoal_state"].to_numpy()).astype(np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _attach(self, frame: dict[str, Any]) -> dict[str, Any]:
        global_index = int(frame["index"])
        row = np.searchsorted(self._label_indices, global_index)
        if row >= len(self._label_indices) or self._label_indices[row] != global_index:
            raise KeyError(
                f"No precomputed subgoal_state label for dataset index {global_index}. Re-run "
                "examples/safediff_vla/compute_subgoal_labels.py so it covers this episode range."
            )
        frame["observation.subgoal_state"] = torch.from_numpy(self._label_values[row]).clone()
        return frame

    def __getitem__(self, idx: int | slice):
        item = self.base_dataset[idx]
        if isinstance(item, list):
            return [self._attach(frame) for frame in item]
        return self._attach(item)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined above (meta, stats, episodes, repo_id, ...).
        # Guard against `base_dataset` (or any dunder) being looked up before `__init__` has run
        # -- e.g. a `DataLoader` worker spawned via multiprocessing reconstructs this object
        # through `__reduce_ex__`/`__getstate__`, which probes for those *before* `__dict__` is
        # populated. Without this guard, `self.base_dataset` itself falls through to
        # `__getattr__` again (it isn't set yet either), recursing until the stack overflows.
        if name == "base_dataset" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.base_dataset, name)
