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

"""Public reading and legacy compatibility for frame-signal outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _pyarrow_available

from .schema import (
    attach_metadata,
    decode_provenance,
    decode_signal_descriptors,
    is_frame_signal_table,
    require_pyarrow,
    validate_frame_signal_table,
)
from .types import SignalDescriptor

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq
else:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

_LEGACY_PROGRESS_COLUMNS = ("progress_sparse", "progress_dense")
_IDENTITY_COLUMNS = ("index", "episode_index", "frame_index")


def get_signal_descriptors(table: pa.Table) -> dict[str, SignalDescriptor]:
    """Return the semantic descriptors stored with a frame-signal table."""
    return decode_signal_descriptors(table)


def get_scoring_provenance(table: pa.Table) -> dict[str, Any]:
    """Return the dataset, model, and adapter lineage stored with a table."""
    return decode_provenance(table)


def _read_legacy_progress(table: pa.Table, path: Path) -> pa.Table:
    progress_columns = [name for name in _LEGACY_PROGRESS_COLUMNS if name in table.column_names]
    if not progress_columns or any(name not in table.column_names for name in _IDENTITY_COLUMNS):
        raise ValueError(f"Not a recognized LeRobot frame-signal output: {path}")

    descriptors: dict[str, SignalDescriptor] = {}
    for name in progress_columns:
        values = table[name].to_numpy(zero_copy_only=False)
        descriptors[name] = SignalDescriptor(
            description="Progress imported from a legacy LeRobot reward-model sidecar.",
            direction="higher",
            missing_values="nan"
            if np.issubdtype(values.dtype, np.floating) and np.isnan(values).any()
            else "forbidden",
        )
    old_metadata = table.schema.metadata or {}
    provenance: dict[str, Any] = {"legacy_output": True, "source_path": str(path)}
    reward_model_path = old_metadata.get(b"reward_model_path")
    if reward_model_path is not None:
        provenance["reward_model_path"] = reward_model_path.decode(errors="replace")

    normalized = attach_metadata(
        table,
        descriptors=descriptors,
        provenance=provenance,
        episode_indices=sorted(set(table["episode_index"].to_pylist())),
        legacy_source="progress_sparse/progress_dense parquet",
    )
    validate_frame_signal_table(normalized)
    return normalized


def read_frame_signals(path: str | Path) -> pa.Table:
    """Read and validate a current or recognized legacy frame-signal output."""
    require_pyarrow()
    if pq is None:
        raise ImportError("Reading frame signals requires pyarrow.parquet")

    output_path = Path(path)
    table = pq.read_table(output_path)
    if is_frame_signal_table(table):
        validate_frame_signal_table(table)
        return table
    return _read_legacy_progress(table, output_path)
