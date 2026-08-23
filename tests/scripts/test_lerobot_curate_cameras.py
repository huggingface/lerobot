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
    _build_collection_report,
    _central_indices,
    _discover_subpaths,
    _load_completed_subdatasets,
    _to_uint8_frame,
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


def test_load_completed_subdatasets_skips_errors(tmp_path):
    report = tmp_path / "progress.json"
    report.write_text(
        json.dumps(
            {
                "subdatasets": {
                    "u/ok": {
                        "cameras": {"observation.images.a": {"proposed_new_key": "observation.images.top"}}
                    },
                    "u/clean": {"cameras": {}},
                    "u/bad": {"error": "boom"},
                }
            }
        ),
        encoding="utf-8",
    )
    done = _load_completed_subdatasets(report)
    # completed = every entry without an "error" (bad one is retried on resume)
    assert set(done) == {"u/ok", "u/clean"}


def test_load_completed_subdatasets_missing_or_corrupt(tmp_path):
    assert _load_completed_subdatasets(tmp_path / "nope.json") == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert _load_completed_subdatasets(corrupt) == {}


def test_build_collection_report_tallies():
    cfg = CameraCurationConfig(repo_id="u/collection", mode="rename")
    collection = {
        "u/ok": {"cameras": {"observation.images.a": {"proposed_new_key": "observation.images.top"}}},
        "u/noop": {"cameras": {"observation.images.top": {"proposed_new_key": None}}},
        "u/bad": {"error": "boom"},
        "u/conf": {"cameras": {}, "collisions": {"observation.images.x": "reason"}},
    }
    report = _build_collection_report(cfg, ["u/ok", "u/noop", "u/bad", "u/conf"], collection)
    assert report["n_total"] == 4
    assert report["n_done"] == 4
    assert report["n_failed"] == 1 and report["failed"] == {"u/bad": "boom"}
    assert report["n_conflicts"] == 1
    assert report["renamed"] == ["u/ok"]  # only the one with a real proposed_new_key
    assert report["n_renamed"] == 1


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
