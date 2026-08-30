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

import pytest
import torch

# The script imports ``lerobot.datasets`` (via the annotation frame provider),
# which only ships under the ``dataset`` extra.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from unittest.mock import MagicMock, patch  # noqa: E402

from lerobot.annotations.camera_curation.config import CameraCurationConfig  # noqa: E402
from lerobot.scripts.lerobot_curate_cameras import (  # noqa: E402
    _central_indices,
    _discover_subpaths,
    _empty_aggregate,
    _seed_aggregate_from_report,
    _summary_report,
    _to_uint8_frame,
    _update_aggregate,
)


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (0, 4, []),
        (5, 0, []),
        (10, 1, [4]),  # single frame -> middle of the episode
        (100, 1, [49]),
        (100, 4, [0, 33, 66, 99]),  # whole episode by default (endpoints included)
    ],
)
def test_central_indices_whole_episode(n, k, expected):
    assert _central_indices(n, k) == expected


def test_central_indices_custom_window():
    # A narrowed window skips the very start/end.
    idxs = _central_indices(100, 4, (0.25, 0.75))
    assert idxs == [25, 41, 58, 74]
    assert idxs[0] > 0 and idxs[-1] < 99


def test_to_uint8_frame_scales_floats():
    frame = torch.ones(3, 4, 4, dtype=torch.float32)  # [0,1] float
    out = _to_uint8_frame(frame)
    assert out.dtype == torch.uint8
    assert int(out.max()) == 255


def test_to_uint8_frame_passthrough_uint8():
    frame = torch.zeros(3, 4, 4, dtype=torch.uint8)
    out = _to_uint8_frame(frame)
    assert out is frame  # uint8 passes through untouched


def test_discover_subpaths_nested_and_single():
    api = MagicMock()
    # Nested collection: two sub-datasets, sorted, prefixes stripped.
    api.list_repo_files.return_value = [
        "b/y/meta/info.json",
        "a/x/meta/info.json",
        "a/x/videos/observation.images.cam/chunk-000/file-000.mp4",
        "README.md",
    ]
    with patch("huggingface_hub.HfApi", return_value=api):
        assert _discover_subpaths("repo") == ["a/x", "b/y"]

    # Standard single dataset (root meta/info.json) -> None.
    api.list_repo_files.return_value = ["meta/info.json", "data/chunk-000/file-000.parquet"]
    with patch("huggingface_hub.HfApi", return_value=api):
        assert _discover_subpaths("repo") is None


import json  # noqa: E402


def test_summary_report_tallies_no_per_dataset_detail():
    cfg = CameraCurationConfig(repo_id="u/collection", mode="rename")
    agg = _empty_aggregate()
    _update_aggregate(agg, "u/ok", {"cameras": {"a": {"proposed_new_key": "observation.images.top"}}})
    _update_aggregate(agg, "u/noop", {"cameras": {"a": {"proposed_new_key": None}}})
    _update_aggregate(agg, "u/bad", {"error": "boom"})
    _update_aggregate(
        agg,
        "u/flag",
        {
            "cameras": {"a": {"proposed_new_key": "observation.images.top_1"}},
            "has_unusable": True,
            "has_name_collision": True,
            "unusable_views": {"observation.images.phone": "placeholder graphic"},
            "conflicting_views": {"top": ["observation.images.top_1", "observation.images.top_2"]},
        },
    )
    report = _summary_report(cfg, ["u/ok", "u/noop", "u/bad", "u/flag"], agg)
    assert report["n_total"] == 4
    assert report["n_done"] == 3  # ok, noop, flag (bad failed -> not done)
    assert report["failed"] == {"u/bad": "boom"} and report["n_failed"] == 1
    assert set(report["renamed"]) == {"u/ok", "u/flag"} and report["n_renamed"] == 2
    assert report["with_unusable"] == ["u/flag"] and report["n_with_unusable"] == 1
    assert report["with_name_collision"] == ["u/flag"] and report["n_with_name_collision"] == 1
    assert set(report["completed"]) == {"u/ok", "u/noop", "u/flag"}  # resume source
    assert "subdatasets" not in report  # summary only — no per-camera dump for clean cams
    # but the flagged datasets DO carry the reason + which cameras conflict
    assert report["unusable_views"] == {"u/flag": {"phone": "placeholder graphic"}}
    assert report["conflicting_views"] == {"u/flag": {"top": ["top_1", "top_2"]}}


def test_update_aggregate_retry_clears_failure():
    # A dataset that failed then succeeds on retry moves out of failed into done.
    agg = _empty_aggregate()
    _update_aggregate(agg, "u/x", {"error": "boom"})
    assert agg["failed"] == {"u/x": "boom"} and "u/x" not in agg["completed"]
    _update_aggregate(agg, "u/x", {"cameras": {"a": {"proposed_new_key": "observation.images.top"}}})
    assert agg["failed"] == {} and "u/x" in agg["completed"] and "u/x" in agg["renamed"]


def test_seed_aggregate_from_report_roundtrip(tmp_path):
    report = tmp_path / "progress.json"
    report.write_text(
        json.dumps(
            {
                "completed": ["u/a", "u/b"],
                "renamed": ["u/a"],
                "with_unusable": ["u/b"],
                "failed": {"u/c": "boom"},
            }
        ),
        encoding="utf-8",
    )
    agg = _empty_aggregate()
    _seed_aggregate_from_report(agg, report)
    assert agg["completed"] == {"u/a", "u/b"}  # what --resume skips
    assert agg["renamed"] == {"u/a"}
    assert agg["with_unusable"] == {"u/b"}
    assert agg["failed"] == {"u/c": "boom"}  # u/c retried (not in completed)


def test_seed_aggregate_missing_or_corrupt(tmp_path):
    agg = _empty_aggregate()
    _seed_aggregate_from_report(agg, tmp_path / "nope.json")
    assert agg["completed"] == set()
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    _seed_aggregate_from_report(agg, corrupt)
    assert agg["completed"] == set()


def test_persist_progress_to_hub_uploads(tmp_path):
    import threading

    from lerobot.scripts.lerobot_curate_cameras import _persist_progress_to_hub

    report = tmp_path / "camera_curation_collection.json"
    report.write_text('{"subdatasets": {}}', encoding="utf-8")
    cfg = CameraCurationConfig(repo_id="u/collection", branch="curated")
    api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=api):
        _persist_progress_to_hub(cfg, report, threading.Lock())
    kwargs = api.upload_file.call_args.kwargs
    assert kwargs["repo_id"] == "u/collection"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "curated"
    assert kwargs["path_in_repo"] == "camera_curation_collection.json"


def test_persist_progress_to_hub_swallows_errors(tmp_path):
    import threading

    from lerobot.scripts.lerobot_curate_cameras import _persist_progress_to_hub

    report = tmp_path / "p.json"
    report.write_text("{}", encoding="utf-8")
    cfg = CameraCurationConfig(repo_id="u/c")
    api = MagicMock()
    api.upload_file.side_effect = RuntimeError("network down")
    with patch("huggingface_hub.HfApi", return_value=api):
        _persist_progress_to_hub(cfg, report, threading.Lock())  # must not raise


def test_fetch_progress_from_hub_copies(tmp_path):
    from lerobot.scripts.lerobot_curate_cameras import _fetch_progress_from_hub

    remote = tmp_path / "remote.json"
    remote.write_text('{"subdatasets": {"u/ok": {"cameras": {}}}}', encoding="utf-8")
    dest = tmp_path / "out" / "camera_curation_collection.json"
    cfg = CameraCurationConfig(repo_id="u/collection")
    with patch("huggingface_hub.hf_hub_download", return_value=str(remote)):
        assert _fetch_progress_from_hub(cfg, dest) is True
    assert dest.exists() and "u/ok" in dest.read_text()


def test_fetch_progress_from_hub_missing_returns_false(tmp_path):
    from lerobot.scripts.lerobot_curate_cameras import _fetch_progress_from_hub

    cfg = CameraCurationConfig(repo_id="u/collection")
    with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("404")):
        assert _fetch_progress_from_hub(cfg, tmp_path / "x.json") is False
