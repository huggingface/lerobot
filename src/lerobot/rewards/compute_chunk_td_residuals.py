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

"""Compute model-agnostic N-step temporal-difference residuals for chunks.

The input may come from any model or annotation pipeline. A score can represent
either a value (larger is better) or a cost-to-go (smaller is better). The
utility converts it to value semantics and computes:

    delta_t = sum(k=0..N-1) gamma^k r_t+k + gamma^N V_t+N - V_t

This temporal-difference residual can be used as an action-chunk advantage
proxy when its assumptions are appropriate. It is not tied to a particular
reward model or policy-improvement algorithm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_TD_METADATA_KEY = b"chunk_td_residual"
ScoreSemantics = Literal["value", "cost_to_go"]


def _discounted_chunk_return(
    num_steps: int,
    *,
    fps: float,
    gamma: float,
    reward_per_second: float,
) -> float:
    """Return a discounted constant-rate reward over a chunk."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not 0 < gamma <= 1:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    reward_per_step = reward_per_second / fps
    if gamma == 1.0:
        return num_steps * reward_per_step
    return reward_per_step * (1.0 - gamma**num_steps) / (1.0 - gamma)


def _to_value(score: np.ndarray, semantics: ScoreSemantics) -> np.ndarray:
    if semantics == "value":
        return score
    if semantics == "cost_to_go":
        return -score
    raise ValueError(f"Unknown score semantics: {semantics}")


def _compute_episode_chunks(
    episode_df: pd.DataFrame,
    *,
    score_column: str,
    score_semantics: ScoreSemantics,
    chunk_size: int,
    stride: int,
    fps: float,
    gamma: float,
    reward_per_second: float,
    include_incomplete: bool,
) -> pd.DataFrame:
    """Compute chunk-boundary TD residuals for one dense episode."""
    episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)
    frame_indices = episode_df["frame_index"].to_numpy(dtype=np.int64)
    scores = episode_df[score_column].to_numpy(dtype=np.float64)
    values = _to_value(scores, score_semantics)

    if len(frame_indices) == 0:
        return pd.DataFrame()
    episode_index = int(episode_df["episode_index"].iloc[0])
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError(f"Episode {episode_index} has duplicate frame indices")
    if len(frame_indices) > 1 and np.any(np.diff(frame_indices) != 1):
        raise ValueError(f"Episode {episode_index} is not dense and contiguous")

    rows = []
    for chunk_index, start_pos in enumerate(range(0, len(episode_df), stride)):
        end_pos = start_pos + chunk_size
        valid_chunk = end_pos < len(episode_df)
        if not valid_chunk and not include_incomplete:
            continue

        bounded_end_pos = min(end_pos, len(episode_df) - 1)
        elapsed_steps = bounded_end_pos - start_pos
        if valid_chunk:
            chunk_return = _discounted_chunk_return(
                chunk_size,
                fps=fps,
                gamma=gamma,
                reward_per_second=reward_per_second,
            )
            td_residual = chunk_return + gamma**chunk_size * values[end_pos] - values[start_pos]
        else:
            chunk_return = np.nan
            td_residual = np.nan

        rows.append(
            {
                "episode_index": episode_index,
                "chunk_index": chunk_index,
                "chunk_start_frame": int(frame_indices[start_pos]),
                "chunk_end_frame": int(frame_indices[bounded_end_pos]),
                "horizon_frames": chunk_size,
                "elapsed_time_s": elapsed_steps / fps,
                "score_start": float(scores[start_pos]),
                "score_end": float(scores[bounded_end_pos]),
                "value_start": float(values[start_pos]),
                "value_end": float(values[bounded_end_pos]),
                "chunk_return": chunk_return,
                "chunk_td_residual": td_residual,
                "valid_chunk": valid_chunk,
            }
        )

    return pd.DataFrame(rows)


def compute_chunk_td_residuals(
    input_path: str | Path,
    output_path: str | Path,
    *,
    score_column: str,
    score_semantics: ScoreSemantics,
    reward_per_second: float,
    fps: float,
    chunk_size: int,
    stride: int | None = None,
    gamma: float = 1.0,
    include_incomplete: bool = True,
) -> Path:
    """Read dense scores and write one N-step TD residual per chunk."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    stride = chunk_size if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")

    source_path = Path(input_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Score parquet not found: {source_path}")
    source_table = pq.read_table(source_path)
    required = {"episode_index", "frame_index", score_column}
    missing = required.difference(source_table.column_names)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")

    source_df = source_table.select(sorted(required)).to_pandas()
    chunks = [
        _compute_episode_chunks(
            episode_df,
            score_column=score_column,
            score_semantics=score_semantics,
            chunk_size=chunk_size,
            stride=stride,
            fps=fps,
            gamma=gamma,
            reward_per_second=reward_per_second,
            include_incomplete=include_incomplete,
        )
        for _, episode_df in source_df.groupby("episode_index", sort=True)
    ]
    chunks = [chunk_df for chunk_df in chunks if not chunk_df.empty]
    if not chunks:
        raise ValueError("Input parquet contains no episodes with frames")

    output_df = pd.concat(chunks, ignore_index=True)
    output_table = pa.Table.from_pandas(output_df, preserve_index=False)
    metadata = dict(source_table.schema.metadata or {})
    metadata[CHUNK_TD_METADATA_KEY] = json.dumps(
        {
            "source_path": str(source_path),
            "score_column": score_column,
            "score_semantics": score_semantics,
            "reward_per_second": reward_per_second,
            "fps": fps,
            "chunk_size": chunk_size,
            "stride": stride,
            "gamma": gamma,
            "include_incomplete": include_incomplete,
        },
        sort_keys=True,
    ).encode()
    output_table = output_table.replace_schema_metadata(metadata)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--score-semantics", choices=["value", "cost_to_go"], required=True)
    parser.add_argument("--reward-per-second", type=float, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument(
        "--stride",
        type=int,
        help="Distance between chunk starts. Defaults to --chunk-size.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--drop-incomplete", action="store_true")
    args = parser.parse_args()

    destination = compute_chunk_td_residuals(
        args.input_path,
        args.output_path,
        score_column=args.score_column,
        score_semantics=args.score_semantics,
        reward_per_second=args.reward_per_second,
        fps=args.fps,
        chunk_size=args.chunk_size,
        stride=args.stride,
        gamma=args.gamma,
        include_incomplete=not args.drop_incomplete,
    )
    output = pq.read_table(destination)
    valid = output["valid_chunk"].to_numpy(zero_copy_only=False)
    print(f"Wrote {len(output)} chunks ({int(valid.sum())} valid) to {destination}")


if __name__ == "__main__":
    main()
