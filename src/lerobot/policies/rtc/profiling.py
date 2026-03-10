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

"""Profiling utilities for remote RTC runs."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RTCProfilingRecord:
    """Per-request timing and queue metrics for remote RTC."""

    request_idx: int
    timestamp: float
    label: str
    payload_bytes: int | None = None

    queue_size_before: int | None = None
    queue_size_after: int | None = None
    action_index_before: int | None = None
    inference_delay_requested: int | None = None
    realized_delay: int | None = None

    client_observation_ms: float | None = None
    client_pickle_ms: float | None = None
    client_send_ms: float | None = None
    client_get_actions_ms: float | None = None
    client_unpickle_ms: float | None = None
    client_total_ms: float | None = None

    server_queue_wait_ms: float | None = None
    server_preprocess_ms: float | None = None
    server_inference_ms: float | None = None
    server_postprocess_ms: float | None = None
    server_pickle_ms: float | None = None
    server_total_ms: float | None = None


class RTCProfiler:
    """Stores profiling records and writes parquet + plot artifacts."""

    def __init__(self, enabled: bool, output_dir: str, run_name: str):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self._records: list[RTCProfilingRecord] = []

    def add(self, record: RTCProfilingRecord) -> None:
        if not self.enabled:
            return
        self._records.append(record)

    def finalize(self) -> dict[str, str]:
        if not self.enabled or not self._records:
            return {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = self.output_dir / f"{self.run_name}_profiling.parquet"
        plot_path = self.output_dir / f"{self.run_name}_profiling.png"

        self._save_parquet(parquet_path, [asdict(r) for r in self._records])
        self._save_plot(plot_path)

        return {"parquet": str(parquet_path), "plot": str(plot_path)}

    def _save_parquet(self, path: Path, rows: list[dict[str, Any]]) -> None:
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError:
            logger.warning("pyarrow not installed, skipping parquet export for %s", path.name)
            return

        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)

    def _save_plot(self, path: Path) -> None:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            logger.warning("matplotlib not installed, skipping profiling plot.")
            return

        x = np.arange(len(self._records))
        has_queue = any(r.queue_size_before is not None or r.queue_size_after is not None for r in self._records)

        nrows = 2 if has_queue else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(16, 5 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        ax_timing = axes[0]
        self._plot_field(ax_timing, x, "client_total_ms", "Client Total")
        self._plot_field(ax_timing, x, "client_observation_ms", "Client Observation")
        self._plot_field(ax_timing, x, "client_send_ms", "Client Send")
        self._plot_field(ax_timing, x, "client_get_actions_ms", "Client GetActions Wait")
        self._plot_field(ax_timing, x, "server_total_ms", "Server Total")
        self._plot_field(ax_timing, x, "server_inference_ms", "Server Inference")
        self._plot_field(ax_timing, x, "server_preprocess_ms", "Server Preprocess")
        self._plot_field(ax_timing, x, "server_postprocess_ms", "Server Postprocess")
        ax_timing.set_ylabel("Milliseconds")
        ax_timing.set_title("Remote RTC Timing Breakdown")
        ax_timing.grid(True, alpha=0.3)
        ax_timing.legend(loc="upper right")

        if has_queue:
            ax_queue = axes[1]
            self._plot_field(ax_queue, x, "queue_size_before", "Queue Before")
            self._plot_field(ax_queue, x, "queue_size_after", "Queue After")
            self._plot_field(ax_queue, x, "inference_delay_requested", "Requested Delay")
            self._plot_field(ax_queue, x, "realized_delay", "Realized Delay")
            ax_queue.set_ylabel("Steps")
            ax_queue.set_title("Queue and Delay Dynamics")
            ax_queue.grid(True, alpha=0.3)
            ax_queue.legend(loc="upper right")

        axes[-1].set_xlabel("Request Index")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_field(self, ax, x: np.ndarray, field_name: str, label: str) -> None:
        values = [getattr(r, field_name) for r in self._records]
        if not any(v is not None for v in values):
            return
        arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        ax.plot(x, arr, label=label, linewidth=2, alpha=0.85)
