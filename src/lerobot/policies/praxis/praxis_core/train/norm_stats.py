# Vendored from the praxis repository (praxis/train/norm_stats.py, the base-checkpoint tooling): the statistics only. The
# window sampler of the LeRobot port is `examples/praxis/compute_norm_stats.py`.

"""The normalization statistics of a dataset for a new single-arm rig.

The statistics are those of the ``normalization`` section of a single-arm config. They
come from the same windows that the fine-tune trains on:

* ``action_q01`` and ``action_q99``: the 1 % and 99 % quantiles of each action value over
  all rows of the windows. The actions are the commanded poses relative to the chunk start
  (``xyz``, ``rot6d``, ``grip``). Only the dims of ``Q99_DIMS`` (``xyz`` and the gripper)
  use them.
* ``proprio_mean`` and ``proprio_std``: the mean and the standard deviation of each
  proprio value (the absolute pose at the chunk starts).

The gripper (``GRIP_DIM``) is not a quantile: it is mapped to [0, 1] and the decode clamps it
to [q01, q99], so it gets ``q01 = 0`` and ``q99 = 1``, the stats of the DROID checkpoint. A
quantile would keep a policy from a full close when the gripper is closed in less than 1 %
of the rows. An xyz dim with ``q99 <= q01`` (a constant motion ``c``) gets the unit range
``[c - 0.5, c + 0.5]``: the normalization stays finite, and ``c`` normalizes to 0. A proprio
dim with a standard deviation below ``MIN_STD`` gets ``MIN_STD``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..rig import ACTION_DIM, GRIP_DIM, Q99_DIMS

log = logging.getLogger(__name__)

#: The smallest proprio standard deviation of the statistics.
MIN_STD = 1e-3


def norm_stats(actions: np.ndarray, proprio: np.ndarray) -> dict[str, list[float]]:
    """Return the four normalization vectors of the action and proprio rows."""
    if actions.shape[1:] != (ACTION_DIM,) or proprio.shape[1:] != (ACTION_DIM,):
        raise ValueError(f"the rows must have {ACTION_DIM} values")
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    q01[GRIP_DIM], q99[GRIP_DIM] = 0.0, 1.0
    for d in Q99_DIMS:
        if not q99[d] > q01[d]:
            # centre the unit range on the value: zero motion stays at 0
            log.warning("action dim %d is constant (%g): unit range", d, q01[d])
            q01[d] -= 0.5
            q99[d] = q01[d] + 1.0
    std = proprio.std(axis=0)
    if (small := np.flatnonzero(std < MIN_STD)).size:
        log.warning("proprio dims %s have a std below %g", small.tolist(), MIN_STD)
    return {
        "action_q01": q01.tolist(),
        "action_q99": q99.tolist(),
        "proprio_mean": proprio.mean(axis=0).tolist(),
        "proprio_std": np.maximum(std, MIN_STD).tolist(),
    }
