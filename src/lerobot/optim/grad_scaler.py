#!/usr/bin/env python

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
"""Loss-scaler state persistence for fp16 mixed precision.

The scaler itself is created and owned by the `Accelerator`: with
``mixed_precision="fp16"`` accelerate builds a plain `torch.amp.GradScaler` for every
backend LeRobot uses, FSDP2 included. It persists that scaler only inside
``Accelerator.save_state``/``load_state``, which write a whole checkpoint directory in
accelerate's own layout (and restore RNG state, which LeRobot owns through its own channel).
LeRobot therefore keeps its own checkpoint layout and calls the scaler's state-dict API
directly here — the same API accelerate calls one layer up.

The state is five scalars, so it is stored as JSON beside ``scheduler_state.json``.
"""

import logging
from pathlib import Path

from torch.amp import GradScaler

from lerobot.utils.constants import SCALER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object, write_json


def save_scaler_state(scaler: GradScaler, save_dir: Path) -> None:
    """Write the loss-scaler state (scale, growth/backoff factors, growth tracker).

    Carries no rank gate of its own — the caller's single ``is_main_process()`` region owns
    it. Saving from one rank is correct because the state is identical everywhere: the
    ``found_inf`` flag is reduced across the whole mesh before ``update()`` runs, so every
    rank derives the same scale from the same skip decisions.

    Args:
        scaler (GradScaler): The accelerator-owned scaler to snapshot.
        save_dir (Path): The `training_state/` directory to write `scaler_state.json` into.
    """
    write_json(scaler.state_dict(), save_dir / SCALER_STATE)


def load_scaler_state(scaler: GradScaler, save_dir: Path) -> GradScaler:
    """Restore the loss-scaler state, tolerating checkpoints that carry none.

    A checkpoint has no scaler state when it predates fp16 support or was written by a run in
    another precision. Recalibrating from the configured ``init_scale`` only costs a handful
    of skipped updates, so that case warns and continues instead of failing the resume.

    Deliberately does NOT materialize ``_scale``/``_growth_tracker``:
    ``GradScaler.load_state_dict`` records the values as the scaler's init fields while it is
    still lazy, and the first ``scaler.scale(loss)`` inside ``accelerator.backward()`` creates
    both tensors on the loss's own device. Nothing between here and that first call touches
    the scaler — the sharded checkpoint channel is handed the unwrapped optimizer precisely so
    that stays true (see :mod:`lerobot.distributed.checkpoint`) — which is also why this
    call's position within the resume sequence is not load-bearing.

    Args:
        scaler (GradScaler): The accelerator-owned scaler to restore into.
        save_dir (Path): The checkpoint's `training_state/` directory.

    Returns:
        GradScaler: The same scaler, restored when the checkpoint carried a state.
    """
    state_path = save_dir / SCALER_STATE
    if not state_path.is_file():
        logging.warning(
            "No %s in the checkpoint: resuming fp16 from scale=%s. The first few optimizer "
            "updates may be skipped while the loss scale recalibrates.",
            SCALER_STATE,
            scaler.get_scale(),
        )
        return scaler

    scaler.load_state_dict(deserialize_json_into_object(state_path, scaler.state_dict()))
    return scaler
