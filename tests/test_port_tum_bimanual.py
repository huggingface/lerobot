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

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from examples.port_datasets.port_tum_bimanual import (
    HandData,
    PortingError,
    PortOptions,
    SessionData,
    TimedArray,
    build_parser,
    convert_dataset,
    discover_sessions,
    main,
    make_features,
    nearest_indices,
    prepare_output,
    read_clamp,
    read_imu,
    read_intrinsics,
    read_session,
    read_tum_pose,
    read_video_indices,
    read_video_timestamps,
    slerp_series,
    sync_session,
)
from lerobot.datasets import LeRobotDataset


def create_hand_tree(session: Path, hand_name: str) -> Path:
    hand = session / hand_name
    (hand / "Calibration").mkdir(parents=True)
    (hand / "Clamp_Data").mkdir()
    (hand / "IMU").mkdir()
    (hand / "Merged_Trajectory").mkdir()
    (hand / "RGB_Images").mkdir()
    return hand


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_discover_sessions_returns_sorted_complete_sessions(tmp_path: Path) -> None:
    create_hand_tree(tmp_path / "session_000002", "left_hand_device_b")
    create_hand_tree(tmp_path / "session_000002", "right_hand_device_b")
    create_hand_tree(tmp_path / "session_000001", "left_hand_device_a")
    create_hand_tree(tmp_path / "session_000001", "right_hand_device_a")

    sessions = discover_sessions(tmp_path)

    assert [item.name for item in sessions] == ["session_000001", "session_000002"]
    assert sessions[0].left.name == "left_hand_device_a"
    assert sessions[0].right.name == "right_hand_device_a"


def test_discover_sessions_rejects_missing_right_hand(tmp_path: Path) -> None:
    create_hand_tree(tmp_path / "session_000001", "left_hand_device_a")

    with pytest.raises(PortingError, match="exactly one right_hand"):
        discover_sessions(tmp_path)


def test_discover_sessions_rejects_no_sessions(tmp_path: Path) -> None:
    with pytest.raises(PortingError, match="no session_"):
        discover_sessions(tmp_path)


def test_read_tum_pose_normalizes_quaternion(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "pose.txt",
        "0.0 1 2 3 0 0 0 2\n1.0 2 3 4 0 0 0 2\n",
    )

    series = read_tum_pose(path)

    np.testing.assert_allclose(series.values[:, 3:], [[0, 0, 0, 1]] * 2)


def test_numeric_reader_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "pose.txt",
        "1.0 1 2 3 0 0 0 1\n0.0 2 3 4 0 0 0 1\n",
    )

    with pytest.raises(PortingError, match="strictly increasing"):
        read_tum_pose(path)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("0 1 2\n", "expected 2 numeric columns"),
        ("0 nan\n", "finite"),
    ],
)
def test_read_clamp_rejects_invalid_rows(tmp_path: Path, content: str, message: str) -> None:
    path = write_text(tmp_path / "clamp.txt", content)

    with pytest.raises(PortingError, match=message):
        read_clamp(path)


def test_read_tum_pose_rejects_zero_norm_quaternion(tmp_path: Path) -> None:
    path = write_text(tmp_path / "pose.txt", "0 1 2 3 0 0 0 0\n")

    with pytest.raises(PortingError, match="zero norm"):
        read_tum_pose(path)


def test_read_imu_accepts_compact_and_covariance_rows(tmp_path: Path) -> None:
    compact = write_text(
        tmp_path / "compact.txt",
        "0 1 2 3 4 5 6\n1 2 3 4 5 6 7\n",
    )
    covariance = write_text(
        tmp_path / "covariance.txt",
        "0 1 2 3 (0, 0, 0, 0, 0, 0, 0, 0, 0) "
        "4 5 6 (0, 0, 0, 0, 0, 0, 0, 0, 0)\n"
        "1 2 3 4 (0, 0, 0, 0, 0, 0, 0, 0, 0) "
        "5 6 7 (0, 0, 0, 0, 0, 0, 0, 0, 0)\n",
    )

    np.testing.assert_allclose(read_imu(compact).values[0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(read_imu(covariance).values[0], [1, 2, 3, 4, 5, 6])


def test_read_video_timestamps_accepts_supported_headers_and_rejects_nan(
    tmp_path: Path,
) -> None:
    simple = write_text(tmp_path / "simple.csv", "timestamp\n0.0\n0.5\n")
    indexed = write_text(
        tmp_path / "indexed.csv",
        "frame_index,seq,header_stamp\n0,10,0.0\n1,11,0.5\n",
    )
    np.testing.assert_allclose(read_video_timestamps(simple), [0.0, 0.5])
    np.testing.assert_allclose(read_video_timestamps(indexed), [0.0, 0.5])

    invalid = write_text(tmp_path / "bad.csv", "timestamp\n0.0\nnan\n")
    with pytest.raises(PortingError, match="finite"):
        read_video_timestamps(invalid)


def test_read_video_timestamps_requires_contiguous_frame_indices(
    tmp_path: Path,
) -> None:
    path = write_text(
        tmp_path / "timestamps.csv",
        "frame_index,seq,header_stamp\n0,10,0.0\n2,11,0.5\n",
    )

    with pytest.raises(PortingError, match="contiguous"):
        read_video_timestamps(path)


def test_read_intrinsics_requires_positive_integer_dimensions(tmp_path: Path) -> None:
    valid = write_text(tmp_path / "valid.json", '{"width": 10, "height": 8}')
    assert read_intrinsics(valid) == (8, 10)

    invalid = write_text(tmp_path / "invalid.json", '{"width": 0, "height": 8}')
    with pytest.raises(PortingError, match="positive integer"):
        read_intrinsics(invalid)


def _write_hand_streams(hand: Path, start: float) -> None:
    write_text(
        hand / "Merged_Trajectory" / "merged_trajectory.txt",
        f"{start} 0 0 0 0 0 0 1\n{start + 1} 1 0 0 0 0 0 1\n",
    )
    write_text(
        hand / "Clamp_Data" / "clamp_data_tum.txt",
        f"{start} 0\n{start + 1} 1\n",
    )
    write_text(
        hand / "IMU" / "imu.txt",
        f"{start} 1 2 3 4 5 6\n{start + 1} 2 3 4 5 6 7\n",
    )
    write_text(
        hand / "RGB_Images" / "timestamps.csv",
        f"timestamp\n{start}\n{start + 1}\n",
    )
    write_text(
        hand / "Calibration" / "rgb_intrinsic.json",
        '{"width": 10, "height": 8}',
    )


def test_read_session_aligns_relative_time_to_left_trajectory(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session_000001"
    left = create_hand_tree(session, "left_hand_device_a")
    right = create_hand_tree(session, "right_hand_device_a")
    _write_hand_streams(left, start=10.0)
    _write_hand_streams(right, start=10.0)
    write_text(
        session / "relative_transforms_left_to_right.txt",
        "1000.0 0 0 0 0 0 0 1\n1001.0 1 0 0 0 0 0 1\n",
    )

    data = read_session(discover_sessions(tmp_path)[0])

    np.testing.assert_allclose(data.relative_pose.timestamps, [10.0, 11.0])


def _pose_series(start: float, end: float) -> TimedArray:
    return TimedArray(
        timestamps=np.array([start, end], dtype=np.float64),
        values=np.array(
            [
                [0, 0, 0, 0, 0, 0, 1],
                [1, 2, 3, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        ),
    )


def _vector_series(start: float, end: float, width: int) -> TimedArray:
    return TimedArray(
        timestamps=np.array([start, end], dtype=np.float64),
        values=np.vstack(
            [
                np.arange(width, dtype=np.float64),
                np.arange(width, dtype=np.float64) + 1,
            ]
        ),
    )


def synthetic_session_data(
    left_video_range: tuple[float, float] = (0.0, 2.0),
    right_video_range: tuple[float, float] = (0.0, 2.0),
    relative_pose_range: tuple[float, float] = (0.0, 2.0),
) -> SessionData:
    left = HandData(
        video_path=Path("left.mp4"),
        video_timestamps=np.array(left_video_range),
        trajectory=_pose_series(0.0, 2.0),
        clamp=_vector_series(0.0, 2.0, 1),
        imu=_vector_series(0.0, 2.0, 6),
        image_size=(8, 10),
    )
    right = HandData(
        video_path=Path("right.mp4"),
        video_timestamps=np.array(right_video_range),
        trajectory=_pose_series(0.25, 1.75),
        clamp=_vector_series(0.25, 1.75, 1),
        imu=_vector_series(0.25, 1.75, 6),
        image_size=(8, 10),
    )
    return SessionData(
        name="session_000001",
        left=left,
        right=right,
        relative_pose=_pose_series(*relative_pose_range),
    )


def test_nearest_indices_breaks_ties_toward_earlier_frame() -> None:
    result = nearest_indices(np.array([0.0, 1.0]), np.array([0.5]))
    np.testing.assert_array_equal(result, [0])


def test_slerp_uses_shortest_path_and_unit_norm() -> None:
    series = TimedArray(
        np.array([0.0, 1.0]),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]),
    )

    result = slerp_series(series, np.array([0.5]))

    np.testing.assert_allclose(result, [[0.0, 0.0, 0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0], atol=1e-6)


def test_sync_session_uses_common_overlap_without_extrapolation() -> None:
    data = synthetic_session_data(
        left_video_range=(0.0, 2.0),
        right_video_range=(0.25, 1.75),
        relative_pose_range=(0.5, 1.5),
    )

    episode = sync_session(data, fps=2)

    np.testing.assert_allclose(episode.timestamps, [0.5, 1.0, 1.5])


def test_actions_are_next_state_with_last_state_repeated() -> None:
    episode = sync_session(synthetic_session_data(), fps=2)

    np.testing.assert_array_equal(episode.actions[:-1], episode.states[1:])
    np.testing.assert_array_equal(episode.actions[-1], episode.states[-1])


def test_features_match_documented_shapes() -> None:
    features = make_features(8, 10)

    assert features["observation.state"]["shape"] == (16,)
    assert features["observation.imu"]["shape"] == (12,)
    assert features["observation.relative_pose"]["shape"] == (7,)
    assert features["action"]["shape"] == (16,)
    assert features["source_timestamp"]["shape"] == (1,)
    assert features["observation.images.left_hand"]["shape"] == (8, 10, 3)
    assert features["observation.images.right_hand"]["shape"] == (8, 10, 3)


def test_sync_session_rejects_non_positive_fps_and_empty_overlap() -> None:
    with pytest.raises(PortingError, match="positive"):
        sync_session(synthetic_session_data(), fps=0)

    data = synthetic_session_data(relative_pose_range=(3.0, 4.0))
    with pytest.raises(PortingError, match="common time"):
        sync_session(data, fps=2)


def write_test_video(path: Path, colors_bgr: list[tuple[int, int, int]], fps: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (10, 8),
    )
    assert writer.isOpened()
    try:
        for color in colors_bgr:
            frame = np.empty((8, 10, 3), dtype=np.uint8)
            frame[:] = color
            writer.write(frame)
    finally:
        writer.release()
    return path


def create_complete_synthetic_source(raw_dir: Path) -> Path:
    session = raw_dir / "session_000001"
    left = create_hand_tree(session, "left_hand_device_a")
    right = create_hand_tree(session, "right_hand_device_a")
    for hand, offset in ((left, 0.0), (right, 0.25)):
        write_text(
            hand / "Merged_Trajectory" / "merged_trajectory.txt",
            f"0.0 {offset} 0 0 0 0 0 1\n0.5 {offset + 0.5} 0 0 0 0 0 1\n1.0 {offset + 1.0} 0 0 0 0 0 1\n",
        )
        write_text(
            hand / "Clamp_Data" / "clamp_data_tum.txt",
            "0.0 0\n0.5 0.5\n1.0 1\n",
        )
        write_text(
            hand / "IMU" / "imu.txt",
            "0.0 1 2 3 4 5 6\n0.5 2 3 4 5 6 7\n1.0 3 4 5 6 7 8\n",
        )
        write_text(
            hand / "RGB_Images" / "timestamps.csv",
            "frame_index,seq,header_stamp\n0,10,0.0\n1,11,0.5\n2,12,1.0\n",
        )
        write_text(
            hand / "Calibration" / "rgb_intrinsic.json",
            '{"width": 10, "height": 8}',
        )
        write_test_video(
            hand / "RGB_Images" / "video.mp4",
            [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
        )
    write_text(
        session / "relative_transforms_left_to_right.txt",
        "1000.0 0 0 0 0 0 0 1\n1000.5 0.5 0 0 0 0 0 1\n1001.0 1 0 0 0 0 0 1\n",
    )
    return raw_dir


def test_read_video_indices_returns_rgb_in_requested_order(tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "video.mp4", colors_bgr=[(0, 0, 255), (0, 255, 0)])

    frames = list(read_video_indices(video, np.array([0, 1])))

    assert frames[0].shape == (8, 10, 3)
    assert frames[0].dtype == np.uint8
    assert int(frames[0][0, 0, 0]) > int(frames[0][0, 0, 1])
    assert int(frames[1][0, 0, 1]) > int(frames[1][0, 0, 0])


def test_read_video_indices_rejects_decreasing_indices(tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "video.mp4", colors_bgr=[(0, 0, 255), (0, 255, 0)])

    with pytest.raises(PortingError, match="nondecreasing"):
        list(read_video_indices(video, np.array([1, 0])))


def test_convert_dataset_round_trips_one_synthetic_episode(
    tmp_path: Path,
) -> None:
    raw_dir = create_complete_synthetic_source(tmp_path / "raw")
    output = tmp_path / "dataset"

    report = convert_dataset(
        PortOptions(
            raw_dir=raw_dir,
            repo_id="namespace/synthetic_bimanual",
            root=output,
            fps=2,
            task="move both hands",
        )
    )
    dataset = LeRobotDataset(
        repo_id="namespace/synthetic_bimanual",
        root=output,
    )

    assert dataset.num_episodes == 1
    assert len(dataset) == 3
    assert report["episodes"] == 1
    assert report["frames"] == 3
    assert report["validation"] == {
        "loadable": True,
        "episodes": 1,
        "frames": 3,
    }
    assert report["sessions"][0]["source_ranges"]["left_video"] == [0.0, 1.0]
    assert (output / "meta" / "info.json").is_file()
    assert (output / "conversion_report.json").is_file()
    assert "observation.images.left_hand" in dataset.features
    assert "observation.images.right_hand" in dataset.features


def test_convert_dataset_rejects_video_timestamp_count_mismatch(
    tmp_path: Path,
) -> None:
    raw_dir = create_complete_synthetic_source(tmp_path / "raw")
    timestamps = raw_dir / "session_000001" / "left_hand_device_a" / "RGB_Images" / "timestamps.csv"
    write_text(
        timestamps,
        "frame_index,seq,header_stamp\n0,10,0.0\n1,11,0.5\n2,12,1.0\n3,13,1.5\n",
    )

    with pytest.raises(PortingError, match="video frames.*timestamps"):
        convert_dataset(
            PortOptions(
                raw_dir=raw_dir,
                repo_id="namespace/mismatch",
                root=tmp_path / "dataset",
                fps=2,
            )
        )


def test_parser_exposes_upstream_style_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--raw-dir",
            "/path/to/raw_root",
            "--repo-id",
            "namespace/dataset_name",
            "--root",
            "/path/to/output",
            "--fps",
            "30",
            "--task",
            "move both hands",
            "--push-to-hub",
        ]
    )

    assert args.repo_id == "namespace/dataset_name"
    assert args.fps == 30
    assert args.push_to_hub is True


def test_conversion_rejects_overlapping_input_and_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pytest.raises(PortingError, match="overlap"):
        prepare_output(raw_dir, raw_dir / "output", overwrite=False)


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output = tmp_path / "output"
    raw_dir.mkdir()
    output.mkdir()

    with pytest.raises(PortingError, match="already exists"):
        prepare_output(raw_dir, output, overwrite=False)


def test_explicit_overwrite_recreates_only_the_output_path(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output = tmp_path / "output"
    sibling = tmp_path / "keep.txt"
    raw_dir.mkdir()
    output.mkdir()
    write_text(output / "old.txt", "old")
    write_text(sibling, "keep")

    _, resolved_output = prepare_output(raw_dir, output, overwrite=True)

    assert resolved_output == output.resolve()
    assert not output.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_skip_invalid_session_converts_valid_session_and_reports_failure(
    tmp_path: Path,
) -> None:
    raw_dir = create_complete_synthetic_source(tmp_path / "raw")
    invalid = raw_dir / "session_000002"
    create_hand_tree(invalid, "left_hand_device_b")
    create_hand_tree(invalid, "right_hand_device_b")
    write_text(
        invalid / "relative_transforms_left_to_right.txt",
        "0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n",
    )
    output = tmp_path / "dataset"

    report = convert_dataset(
        PortOptions(
            raw_dir=raw_dir,
            repo_id="namespace/synthetic_skip",
            root=output,
            fps=2,
            skip_invalid_session=True,
        )
    )

    assert report["episodes"] == 1
    assert [item["status"] for item in report["sessions"]] == [
        "success",
        "failed",
    ]
    assert report["sessions"][1]["session"] == "session_000002"
    assert "<raw_dir>" in report["sessions"][1]["error"]


def test_main_returns_data_error_exit_code(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--raw-dir",
            str(tmp_path / "missing"),
            "--repo-id",
            "namespace/missing",
            "--root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert "porting error:" in capsys.readouterr().err


def test_contribution_contains_no_private_or_production_identifiers() -> None:
    root = Path(__file__).resolve().parents[1]
    contribution_files = (
        root / "examples" / "port_datasets" / "port_tum_bimanual.py",
        root / "examples" / "port_datasets" / "README_tum_bimanual.md",
        Path(__file__).resolve(),
    )
    assert all(path.is_file() for path in contribution_files)

    forbidden_literals = (
        "/Us" + "ers/",
        "xwe" + "chat_files",
        "session_" + "115",
    )
    forbidden_brand = "lu" + "ming"
    for path in contribution_files:
        text = path.read_text(encoding="utf-8").lower()
        assert forbidden_brand not in text
        for literal in forbidden_literals:
            assert literal.lower() not in text
        assert re.search(r"(?<!\d)\d{12,}(?!\d)", text) is None
