# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the local v3 dataset video-integrity command."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av", reason="PyAV is required (install lerobot[dataset])")
pa = pytest.importorskip("pyarrow", reason="PyArrow is required (install lerobot[dataset])")
pq = pytest.importorskip("pyarrow.parquet", reason="PyArrow is required (install lerobot[dataset])")
np = pytest.importorskip("numpy", reason="NumPy is required (install lerobot[dataset])")

from lerobot import available_extras  # noqa: E402
from lerobot.scripts import lerobot_check_dataset_videos as checker  # noqa: E402

VIDEO_KEY = "observation.images.image"
VIDEO_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _write_video(path: Path, *, frames: int, rate: Fraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            pixels = np.full((32, 32, 3), index * 13 % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_dataset(
    root: Path, intervals: list[tuple[float, float]], *, frames: int = 10, video: str = "valid"
) -> Path:
    episode_dir = root / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "video_path": VIDEO_TEMPLATE,
        "features": {VIDEO_KEY: {"dtype": "video"}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = [
        {
            "episode_index": index,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": start,
            f"videos/{VIDEO_KEY}/to_timestamp": end,
        }
        for index, (start, end) in enumerate(intervals)
    ]
    pq.write_table(pa.Table.from_pylist(rows), episode_dir / "file-000.parquet")

    path = root / VIDEO_TEMPLATE.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    if video == "valid":
        _write_video(path, frames=frames, rate=Fraction(10))
    elif video == "corrupt":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an mp4")
    return path


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_command_is_registered_and_help_does_not_require_dataset_io() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["lerobot-check-dataset-videos"] == (
        "lerobot.scripts.lerobot_check_dataset_videos:main"
    )
    assert "lerobot-check-dataset-videos" in available_extras["dataset"]
    assert importlib.util.find_spec("lerobot.scripts.lerobot_check_dataset_videos") is not None

    result = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_check_dataset_videos", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Local LeRobot v3.0 dataset root" in result.stdout


def test_valid_shared_video_is_reported_and_dataset_is_read_only(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [(0.0, 0.5), (0.5, 1.0)])
    with av.open(str(path)) as container:
        assert len(list(container.decode(video=0))) == 10
    before = _snapshot(tmp_path)

    report = checker.inspect_dataset(tmp_path)

    assert report["ok"]
    assert report["episodes_checked"] == 2
    assert report["intervals_checked"] == 2
    assert report["unique_videos"] == 1
    assert _snapshot(tmp_path) == before


def test_missing_final_episode_is_attributed_to_its_episode(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0), (1.0, 2.0)])

    report = checker.inspect_dataset(tmp_path)

    assert [(issue["code"], issue["episode_index"]) for issue in report["issues"]] == [
        ("interval_past_video_end", 1)
    ]


def test_one_missing_frame_is_not_hidden_by_tolerance(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0)], frames=9)
    assert checker.inspect_dataset(tmp_path)["issues"][0]["code"] == "interval_past_video_end"


@pytest.mark.parametrize("rate", [Fraction(10), Fraction(30000, 1001)])
def test_fractional_rate_roundoff_is_within_tolerance(tmp_path: Path, rate: Fraction) -> None:
    duration = float(10 / rate)
    path = tmp_path / "fractional.mp4"
    _write_video(path, frames=10, rate=rate)
    with av.open(str(path)) as container:
        actual_duration = float(container.streams.video[0].duration * container.streams.video[0].time_base)
    assert actual_duration <= duration + 1e-4

    _write_dataset(tmp_path / "dataset", [(0.0, duration + 1e-6)])
    report = checker.inspect_dataset(tmp_path / "dataset")
    assert report["ok"]


@pytest.mark.parametrize(
    ("video", "expected"), [("missing", "missing_video"), ("corrupt", "unreadable_video")]
)
def test_missing_and_corrupt_video_are_reported(tmp_path: Path, video: str, expected: str) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0)], video=video)
    assert checker.inspect_dataset(tmp_path)["issues"][0]["code"] == expected


@pytest.mark.parametrize("interval", [(-0.1, 0.5), (0.5, 0.5), (0.5, 0.4), (0.0, float("nan"))])
def test_invalid_interval_is_reported(tmp_path: Path, interval: tuple[float, float]) -> None:
    _write_dataset(tmp_path, [interval])
    assert checker.inspect_dataset(tmp_path)["issues"][0]["code"] == "invalid_interval"


def test_non_video_dataset_has_nothing_to_check(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "meta"
    metadata_dir.mkdir()
    (metadata_dir / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {"observation.state": {"dtype": "float32"}}}),
        encoding="utf-8",
    )

    report = checker.inspect_dataset(tmp_path)

    assert report["ok"]
    assert report["unique_videos"] == 0
    assert report["message"] == "No video features found"


def test_invalid_tolerance_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tolerance_s must be finite and nonnegative"):
        checker.inspect_dataset(tmp_path, tolerance_s=-1)


def test_parquet_shards_and_shared_video_probes_are_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path, [(0.0, 0.5)])
    source = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    row = pq.read_table(source).to_pylist()[0]
    row["episode_index"] = 1
    second_shard = source.with_name("file-001.parquet")
    pq.write_table(pa.Table.from_pylist([row]), second_shard)
    real_probe = checker.probe_video
    calls = 0

    def counting_probe(path: Path) -> dict:
        nonlocal calls
        calls += 1
        return real_probe(path)

    monkeypatch.setattr(checker, "probe_video", counting_probe)
    report = checker.inspect_dataset(tmp_path)

    assert report["episodes_checked"] == 2
    assert report["unique_videos"] == 1
    assert calls == 1


def test_all_video_features_are_checked(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0)])
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    other_key = "observation.images.hand_image"
    info["features"][other_key] = {"dtype": "video"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    parquet_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    row = pq.read_table(parquet_path).to_pylist()[0]
    row.update(
        {
            column.replace(VIDEO_KEY, other_key): value
            for column, value in list(row.items())
            if column.startswith("videos/")
        }
    )
    pq.write_table(pa.Table.from_pylist([row]), parquet_path)

    report = checker.inspect_dataset(tmp_path)

    assert report["intervals_checked"] == 2
    assert [(issue["code"], issue["video_key"]) for issue in report["issues"]] == [
        ("missing_video", other_key)
    ]


def test_invalid_file_index_is_not_silently_truncated(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0)])
    path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    row = pq.read_table(path).to_pylist()[0]
    row[f"videos/{VIDEO_KEY}/file_index"] = 0.5
    pq.write_table(pa.Table.from_pylist([row]), path)
    assert checker.inspect_dataset(tmp_path)["issues"][0]["code"] == "invalid_metadata"


@pytest.mark.parametrize("fault", ["unsupported_version", "missing_column", "outside_root"])
def test_invalid_dataset_layout_raises_a_clear_error(tmp_path: Path, fault: str) -> None:
    _write_dataset(tmp_path, [(0.0, 1.0)])
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if fault == "unsupported_version":
        info["codebase_version"] = "v2.1"
    elif fault == "outside_root":
        info["video_path"] = "../outside.mp4"
    else:
        parquet_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        pq.write_table(pq.read_table(parquet_path).drop([f"videos/{VIDEO_KEY}/to_timestamp"]), parquet_path)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError):
        checker.inspect_dataset(tmp_path)


@pytest.mark.parametrize(("interval", "expected"), [((0.0, 1.0), 0), ((0.0, 2.0), 1)])
def test_cli_reports_json_and_returns_issue_status(
    tmp_path: Path, interval: tuple[float, float], expected: int
) -> None:
    _write_dataset(tmp_path, [interval])
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_check_dataset_videos", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stderr
    assert json.loads(result.stdout)["ok"] is (expected == 0)


def test_cli_returns_fatal_status_for_non_dataset_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_check_dataset_videos", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "fatal_error" in json.loads(result.stdout)
