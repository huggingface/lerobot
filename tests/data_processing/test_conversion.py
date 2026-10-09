# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import io
import json
import tarfile

import numpy as np
import pytest

pytest.importorskip("datasets")
import av  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lerobot.configs import RGBEncoderConfig  # noqa: E402
from lerobot.data_processing.artifacts import file_checksum  # noqa: E402
from lerobot.data_processing.conversion import ConvertConfig, convert_dataset, letterbox  # noqa: E402
from lerobot.data_processing.sources.acquisition import acquire_file, unpack_tar  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def raw_source(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    episodes = []
    for episode in range(2):
        values = np.arange(12, dtype=np.float32).reshape(6, 2) + 100 * episode
        np.save(root / f"{episode}.npy", values)
        with av.open(root / f"{episode}.mp4", "w") as container:
            stream = container.add_stream("libx264", rate=6)
            stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
            for index in range(6):
                image = np.full((64, 96, 3), 40 + 20 * episode + 2 * index, np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        episodes.append(
            {
                "id": f"episode-{episode}",
                "task": "move object",
                "arrays": {"action": f"{episode}.npy"},
                "videos": {"observation.images.front": f"{episode}.mp4"},
            }
        )
    manifest = root / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "fps": 6,
                "robot_type": "fixture",
                "features": {"action": {"dtype": "float32", "shape": [2], "names": ["joint1", "joint2"]}},
                "episodes": episodes,
            }
        )
    )
    return manifest


@pytest.mark.parametrize("workers", [1, 2])
def test_real_conversion_native_actions_video_and_resume(tmp_path, workers):
    manifest = raw_source(tmp_path)
    cfg = ConvertConfig(
        source={"manifest": str(manifest)},
        output=tmp_path / "converted",
        size=64,
        encoder=RGBEncoderConfig(vcodec="h264"),
    )
    cfg.runtime.workers, cfg.runtime.shard_size = workers, 1
    output = convert_dataset(cfg)
    dataset = LeRobotDataset(cfg.repo_id, root=output, video_backend="pyav")
    assert len(dataset) == 12 and dataset.num_episodes == 2
    assert dataset[0]["observation.images.front"].shape == (3, 64, 64)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), [0, 1])
    np.testing.assert_array_equal(dataset[6]["action"].numpy(), [100, 101])
    mapping = pq.read_table(output / "meta/source_episode_map.parquet").to_pylist()
    assert [row["source_episode_id"] for row in mapping] == ["episode-0", "episode-1"]
    geometry = pq.read_table(output / "meta/camera_geometry.parquet").to_pylist()[0]
    assert geometry["scaled_width"] == 64 and geometry["scaled_height"] == 43 and geometry["pad_top"] == 10
    movie = next((output / "videos").rglob("*.mp4")).read_bytes()
    assert movie.index(b"moov") < movie.index(b"mdat")
    accepted_before = list((tmp_path / "converted.processing").rglob("checkpoints/*.json"))
    cfg.output = tmp_path / "second-release"
    convert_dataset(cfg)
    assert list((tmp_path / "converted.processing").rglob("checkpoints/*.json")) == accepted_before


def test_letterbox_no_distortion_and_pinned_acquisition(tmp_path):
    image = np.full((10, 20, 3), 255, np.uint8)
    converted, geometry = letterbox(image, 64)
    assert geometry["scaled_height"] == 32
    assert np.all(converted[:16] == 0) and np.all(converted[16:48] == 255)
    source = tmp_path / "input"
    source.write_bytes(b"pinned raw input")
    target = tmp_path / "copy"
    acquire_file(str(source), file_checksum(source)[0], target)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        acquire_file(str(source), "a" * 64, tmp_path / "wrong")


def test_tar_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("../outside")
        member.size = 4
        stream.addfile(member, io.BytesIO(b"oops"))
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_tar(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()


def test_already_lerobot_native_subtasks(tmp_path):
    root = tmp_path / "original"
    dataset = LeRobotDataset.create(
        "publisher/native",
        fps=6,
        root=root,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
        },
        use_videos=False,
    )
    for index in range(4):
        dataset.add_frame(
            {
                "action": np.array([index, -index], np.float32),
                "subtask_index": np.array([7], np.int64),
                "task": "native task",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    from lerobot.datasets.language import language_array, language_persistent_arrow_type

    data_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    atom = {
        "role": "assistant",
        "content": "Native subtask",
        "style": "subtask",
        "timestamp": 0.0,
        "camera": None,
        "tool_calls": None,
    }
    pq.write_table(
        table.append_column(
            "language_persistent", language_array([[atom]] * 4, language_persistent_arrow_type())
        ),
        data_path,
    )
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["language_persistent"] = {"dtype": "language", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))
    (root / "README.md").write_text("Publisher card; do not infer a licence from this text.\n")
    (root / "NOTICE").write_text("Original publisher attribution.\n")
    before = next((root / "data").rglob("*.parquet")).read_bytes()
    cfg = ConvertConfig(
        source_factory="lerobot.data_processing.sources.lerobot:LeRobotSource",
        source={"root": str(root)},
        output=tmp_path / "normalized",
    )
    converted = convert_dataset(cfg)
    loaded = LeRobotDataset(cfg.repo_id, root=converted)
    assert int(loaded[0]["subtask_index"]) == 7
    assert loaded[0]["language_persistent"][0]["content"] == "Native subtask"
    assert (converted / "meta/source/README.md").read_bytes() == (root / "README.md").read_bytes()
    assert (converted / "meta/source/NOTICE").read_bytes() == (root / "NOTICE").read_bytes()
    assert not (converted / "meta/source/LICENSE").exists()
    assert next((root / "data").rglob("*.parquet")).read_bytes() == before


def test_clock_failure_retries_do_not_publish_partial_dataset(tmp_path):
    manifest = raw_source(tmp_path)
    # Truncate native data while leaving all six camera frames present.
    np.save(manifest.parent / "0.npy", np.zeros((4, 2), np.float32))
    cfg = ConvertConfig(
        source={"manifest": str(manifest)},
        output=tmp_path / "output",
        size=64,
        encoder=RGBEncoderConfig(vcodec="h264"),
    )
    with pytest.raises(ValueError, match="more frames"):
        convert_dataset(cfg)
    assert not cfg.output.exists()


def test_lerobot_source_episode_selection_preserves_native_identity(tmp_path):
    from lerobot.data_processing.sources.lerobot import LeRobotSource

    root = tmp_path / "native-selection"
    dataset = LeRobotDataset.create(
        "publisher/native",
        fps=6,
        root=root,
        features={"action": {"dtype": "float32", "shape": (2,), "names": None}},
        use_videos=False,
    )
    for episode in range(3):
        for frame in range(2):
            dataset.add_frame({"action": np.array([episode, frame], np.float32), "task": "move"})
        dataset.save_episode()
    dataset.finalize()
    source = LeRobotSource(root, episode_indices=[2, 0])
    source.acquire()
    items = list(source.discover(None, None, None))
    assert [item.key for item in items] == ["0", "2"]
    assert [item.payload["dataset_from_index"] for item in items] == [0, 4]
    np.testing.assert_array_equal(list(source.frames(items[1].payload))[0]["action"], [2, 0])
    with pytest.raises(ValueError, match="absent"):
        LeRobotSource(root, episode_indices=[3]).acquire()
    for selection in ([], [0, 0], [-1], [True]):
        with pytest.raises(ValueError, match="episode_indices"):
            LeRobotSource(root, episode_indices=selection)


def test_fresh_worker_counts_preserve_part_hashes(tmp_path):
    manifest = raw_source(tmp_path)
    hashes = []
    for workers in (1, 2):
        cfg = ConvertConfig(
            source={"manifest": str(manifest)},
            output=tmp_path / f"output-{workers}",
            size=64,
            encoder=RGBEncoderConfig(vcodec="h264"),
        )
        cfg.runtime.workers, cfg.runtime.shard_size = workers, 1
        convert_dataset(cfg)
        records = {}
        for path in (tmp_path / f"output-{workers}.processing").rglob("checkpoints/*.json"):
            for row in json.loads(path.read_text())["results"]:
                records[row["item_id"]] = [
                    (artifact["name"], artifact["sha256"]) for artifact in row["artifacts"]
                ]
        hashes.append(records)
    assert hashes[0] == hashes[1]
