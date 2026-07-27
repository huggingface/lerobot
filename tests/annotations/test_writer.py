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
"""Writer correctness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ``pyarrow`` and the ``lerobot.annotations`` -> ``lerobot.datasets`` chain
# (-> the HF ``datasets`` library) only ship under the ``dataset`` extra.
# Skip this module in tiers without it instead of erroring at import.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.steerable_pipeline.reader import iter_episodes  # noqa: E402
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging  # noqa: E402
from lerobot.annotations.steerable_pipeline.writer import (  # noqa: E402
    LanguageColumnsWriter,
    speech_atom,
)


def _stage_episode(
    staging_dir: Path,
    episode_index: int,
    *,
    plan: list[dict] | None = None,
    interjections: list[dict] | None = None,
    vqa: list[dict] | None = None,
) -> None:
    staging = EpisodeStaging(staging_dir, episode_index)
    if plan is not None:
        staging.write("plan", plan)
    if interjections is not None:
        staging.write("interjections", interjections)
    if vqa is not None:
        staging.write("vqa", vqa)


def test_writer_persistence_identity(fixture_dataset_root: Path, tmp_path: Path) -> None:
    """Every frame in an episode has a byte-identical persistent list."""
    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        plan=[
            {
                "role": "assistant",
                "content": "grasp the sponge",
                "style": "subtask",
                "timestamp": 0.0,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "1. wipe\n2. dry",
                "style": "plan",
                "timestamp": 0.0,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "wiped the counter",
                "style": "memory",
                "timestamp": 0.5,
                "tool_calls": None,
            },
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    LanguageColumnsWriter().write_all(records, staging_dir, fixture_dataset_root)

    table = pq.read_table(fixture_dataset_root / "data" / "chunk-000" / "file-000.parquet")
    persistent = table.column("language_persistent").to_pylist()
    first = persistent[0]
    assert first  # non-empty
    for row in persistent:
        assert row == first, "persistent slice must be byte-identical across all frames"


def test_writer_events_exact_timestamp(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        interjections=[
            speech_atom(0.0, "Got it."),
            {
                "role": "user",
                "content": "skip the dishes",
                "style": "interjection",
                "timestamp": 0.5,
                "tool_calls": None,
            },
            speech_atom(0.5, "Skipping the dishes."),
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    LanguageColumnsWriter().write_all(records, staging_dir, fixture_dataset_root)

    table = pq.read_table(fixture_dataset_root / "data" / "chunk-000" / "file-000.parquet")
    timestamps = table.column("timestamp").to_pylist()
    events = table.column("language_events").to_pylist()
    for ts, ev in zip(timestamps, events, strict=True):
        if abs(ts - 0.0) < 1e-9:
            assert any(r["role"] == "assistant" and r.get("style") is None for r in ev), ev
        elif abs(ts - 0.5) < 1e-9:
            assert any(r.get("style") == "interjection" for r in ev), ev
            assert any(r.get("style") is None for r in ev), ev
        else:
            assert ev == []


def test_writer_column_routing(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        plan=[
            {
                "role": "assistant",
                "content": "do X",
                "style": "subtask",
                "timestamp": 0.0,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "1. do X",
                "style": "plan",
                "timestamp": 0.0,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "did X",
                "style": "memory",
                "timestamp": 0.3,
                "tool_calls": None,
            },
        ],
        interjections=[
            speech_atom(0.0, "OK"),
            {
                "role": "user",
                "content": "wait",
                "style": "interjection",
                "timestamp": 0.2,
                "tool_calls": None,
            },
            speech_atom(0.2, "Waiting"),
        ],
        vqa=[
            {
                "role": "user",
                "content": "where is the cup?",
                "style": "vqa",
                "timestamp": 0.4,
                "camera": "observation.images.front",
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [1, 2, 3, 4]}]},
                    sort_keys=True,
                ),
                "style": "vqa",
                "timestamp": 0.4,
                "camera": "observation.images.front",
                "tool_calls": None,
            },
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    LanguageColumnsWriter().write_all(records, staging_dir, fixture_dataset_root)
    table = pq.read_table(fixture_dataset_root / "data" / "chunk-000" / "file-000.parquet")

    persistent = table.column("language_persistent").to_pylist()[0]
    persistent_styles = {r["style"] for r in persistent}
    assert persistent_styles == {"subtask", "plan", "memory"}

    all_events = [r for ev in table.column("language_events").to_pylist() for r in ev]
    event_styles = {r.get("style") for r in all_events}
    assert event_styles == {None, "interjection", "vqa"}


def test_writer_drops_subtask_index_idempotent(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        plan=[
            {
                "role": "assistant",
                "content": "do X",
                "style": "subtask",
                "timestamp": 0.0,
                "tool_calls": None,
            },
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    writer = LanguageColumnsWriter()
    writer.write_all(records, staging_dir, fixture_dataset_root)

    path = fixture_dataset_root / "data" / "chunk-000" / "file-000.parquet"
    table_a = pq.read_table(path)
    assert "subtask_index" not in table_a.column_names
    assert "language_persistent" in table_a.column_names
    assert "language_events" in table_a.column_names
    # The writer no longer emits a dataset-level ``tools`` column; the
    # ``say`` tool schema lives as a code constant (``SAY_TOOL_SCHEMA``)
    # so the parquet stays small and the pipeline doesn't extend the schema.
    assert "tools" not in table_a.column_names

    # second pass — must produce identical bytes for the language columns
    records_again = list(iter_episodes(fixture_dataset_root))
    writer.write_all(records_again, staging_dir, fixture_dataset_root)
    table_b = pq.read_table(path)
    assert (
        table_a.column("language_persistent").to_pylist() == table_b.column("language_persistent").to_pylist()
    )
    assert table_a.column("language_events").to_pylist() == table_b.column("language_events").to_pylist()


def test_writer_normalize_rejects_misrouted_persistent_style() -> None:
    """``_normalize_persistent_row`` must reject any non-persistent style."""
    from lerobot.annotations.steerable_pipeline.writer import _normalize_persistent_row

    with pytest.raises(ValueError, match="non-persistent style"):
        _normalize_persistent_row(
            {"role": "assistant", "content": "oops", "style": "vqa", "timestamp": 0.0, "tool_calls": None}
        )


def test_writer_normalize_rejects_misrouted_event_style() -> None:
    """``_normalize_event_row`` must reject any persistent style."""
    from lerobot.annotations.steerable_pipeline.writer import _normalize_event_row

    with pytest.raises(ValueError):
        _normalize_event_row({"role": "assistant", "content": "oops", "style": "subtask", "tool_calls": None})


def test_say_tool_schema_constant_is_well_formed() -> None:
    """``SAY_TOOL_SCHEMA`` (and ``DEFAULT_TOOLS``) replace the parquet
    ``tools`` column — chat-template consumers import them directly.
    """
    from lerobot.annotations.steerable_pipeline.writer import (
        DEFAULT_TOOLS,
        SAY_TOOL_SCHEMA,
    )

    assert DEFAULT_TOOLS == [SAY_TOOL_SCHEMA]
    assert SAY_TOOL_SCHEMA["function"]["name"] == "say"
    params = SAY_TOOL_SCHEMA["function"]["parameters"]
    assert params["properties"]["text"]["type"] == "string"
    assert params["required"] == ["text"]


def test_writer_does_not_add_tools_column(fixture_dataset_root: Path, tmp_path: Path) -> None:
    """Re-running on a parquet that already has a legacy ``tools`` column
    must drop it cleanly so reruns converge to the v3.1 schema.
    """
    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        plan=[
            {"role": "assistant", "content": "x", "style": "subtask", "timestamp": 0.0, "tool_calls": None}
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    LanguageColumnsWriter().write_all(records, staging_dir, fixture_dataset_root)
    table = pq.read_table(fixture_dataset_root / "data" / "chunk-000" / "file-000.parquet")
    assert "tools" not in table.column_names


def test_annotation_metadata_sync_allows_non_streaming_load(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Annotated parquet columns must be declared in ``meta/info.json``.

    ``LeRobotDataset`` loads non-streaming datasets by casting parquet
    against metadata-derived HF features. If the annotation writer adds
    language columns but metadata stays stale, that cast fails with a column
    mismatch.
    """
    from lerobot.annotations.steerable_pipeline.executor import Executor
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_info, load_nested_dataset
    from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT, language_feature_info

    info_path = fixture_dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    info_path.write_text(json.dumps(info, indent=2))

    staging_dir = tmp_path / "stage"
    _stage_episode(
        staging_dir,
        0,
        plan=[
            {"role": "assistant", "content": "do X", "style": "subtask", "timestamp": 0.0, "tool_calls": None}
        ],
    )
    records = list(iter_episodes(fixture_dataset_root))
    LanguageColumnsWriter().write_all(records, staging_dir, fixture_dataset_root)

    Executor._ensure_annotation_metadata_in_info(fixture_dataset_root)

    synced = load_info(fixture_dataset_root)
    for key, feature in language_feature_info().items():
        assert synced["features"][key] == feature

    hf_features = get_hf_features_from_features(synced["features"])
    dataset = load_nested_dataset(fixture_dataset_root / "data", features=hf_features)

    assert LANGUAGE_PERSISTENT in dataset.column_names
    assert LANGUAGE_EVENTS in dataset.column_names
    assert len(dataset) == 24


def _build_packed_dataset(root: Path, episode_lengths: list[int], *, fps: int = 10) -> Path:
    """Pack several episodes into a single shard (vs build_annotation_dataset's one-per-file),
    so the writer's rewrite must re-emit one row group per episode instead of collapsing them."""
    from lerobot.datasets.io_utils import write_tasks
    from lerobot.utils.io_utils import write_json

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_index, frame_index, timestamp, task_index, subtask_index = [], [], [], [], []
    for ep, length in enumerate(episode_lengths):
        episode_index += [ep] * length
        frame_index += list(range(length))
        timestamp += [round(i / fps, 6) for i in range(length)]
        task_index += [0] * length
        subtask_index += [0] * length  # legacy column the writer must drop
    pd.DataFrame(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "task_index": task_index,
            "subtask_index": subtask_index,
        }
    ).to_parquet(data_dir / "file-000.parquet", index=False)

    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the thing"], name="task"))
    write_tasks(tasks_df, root)
    write_json(
        {"codebase_version": "v3.1", "fps": fps, "features": {}, "total_episodes": len(episode_lengths)},
        root / "meta" / "info.json",
    )
    return root


def test_writer_one_row_group_per_episode(tmp_path: Path) -> None:
    """Rewriting a packed shard must keep one row group per episode, not collapse
    every episode into a single giant row group."""
    episode_lengths = [4, 6, 5]  # unequal lengths, all in one shard
    root = _build_packed_dataset(tmp_path / "ds", episode_lengths)
    shard = root / "data" / "chunk-000" / "file-000.parquet"
    assert pq.ParquetFile(shard).metadata.num_row_groups == 1, "fixture should start collapsed"

    staging_dir = tmp_path / "stage"
    for ep in range(len(episode_lengths)):
        _stage_episode(
            staging_dir,
            ep,
            plan=[
                {
                    "role": "assistant",
                    "content": f"subtask for ep {ep}",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "tool_calls": None,
                }
            ],
        )

    records = list(iter_episodes(root))
    LanguageColumnsWriter().write_all(records, staging_dir, root)

    # One row group per episode, with row counts matching the episode lengths.
    md = pq.ParquetFile(shard).metadata
    assert md.num_row_groups == len(episode_lengths)
    assert [md.row_group(i).num_rows for i in range(md.num_row_groups)] == episode_lengths
    # Language columns are still present after the per-episode rewrite.
    table = pq.read_table(shard)
    assert "language_persistent" in table.column_names
    assert "language_events" in table.column_names


def test_speech_atom_shape_matches_plan_spec() -> None:
    atom = speech_atom(2.5, "I'm cleaning up!")
    assert atom["role"] == "assistant"
    assert atom["style"] is None
    assert atom["content"] is None
    assert atom["timestamp"] == 2.5
    assert isinstance(atom["tool_calls"], list)
    call = atom["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "say"
    assert call["function"]["arguments"]["text"] == "I'm cleaning up!"
