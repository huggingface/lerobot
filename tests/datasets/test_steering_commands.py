# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.datasets.steering_commands import SteeringCommandDataset, SteeringCommands


def manifest():
    return {
        "version": 1,
        "source": {"repo_id": "test/data", "revision": "abc"},
        "segments": [
            {
                "episode_index": 3,
                "start_frame": 0,
                "end_frame": 5,
                "review": {"verdict": "accepted", "reviewer": "test-reviewer"},
                "commands": [
                    {"style": "subtask", "text": "reach for the tape", "evidence": "test video"},
                    {"style": "motion", "text": "move the left gripper left", "evidence": "test FK"},
                ],
            }
        ],
    }


def test_fresh_commands_preserve_action_targets_and_requested_mixture():
    index = SteeringCommands(manifest())
    sample = {
        "index": 400,
        "episode_index": 3,
        "frame_index": 2,
        "task": "put tape in bin",
        "action": object(),
    }
    np.random.seed(123)
    results = [index.sample(sample, 0.2) for _ in range(2000)]
    assert all(r["action"] is sample["action"] for r in results)
    assert len({r["task"] for r in results}) == 3
    assert 330 < sum(r["task"] == sample["task"] for r in results) < 470
    assert index.sample(sample, 0.2, deterministic=True) == index.sample(sample, 0.2, deterministic=True)


def test_missing_and_unreviewed_commands_fail_instead_of_changing_the_mixture():
    data = manifest()
    data["segments"][0]["review"]["verdict"] = "uncertain"
    with pytest.raises(ValueError, match="accepted"):
        SteeringCommands(data)
    index = SteeringCommands(manifest())
    with pytest.raises(ValueError, match="Missing"):
        index.at(3, 5)
    with pytest.raises(ValueError, match="Missing"):
        index.at(4, 0)
    data = manifest()
    data["segments"].append(copy.deepcopy(data["segments"][0]))
    with pytest.raises(ValueError, match="Overlapping"):
        SteeringCommands(data)


def test_command_mask_uses_real_offsets_preserves_targets_and_episode_padding():
    index = SteeringCommands(manifest())
    action = torch.arange(8).reshape(4, 2).float()
    original_pad = torch.tensor([False, False, True, False])
    sample = {
        "episode_index": 3,
        "frame_index": 2,
        "task": "overall task",
        "action": action,
        "action_is_pad": original_pad,
    }
    command = index.sample(sample, 0, action_offsets=[-2, 0, 2, 4])
    assert command["action"] is action
    assert command["action_is_pad"].tolist() == [False, False, True, True]
    assert original_pad.tolist() == [False, False, True, False]
    task = index.sample(sample, 1, action_offsets=[-2, 0, 2, 4])
    assert task["task"] == "overall task"
    assert torch.equal(task["action_is_pad"], original_pad)
    with pytest.raises(ValueError, match="No demonstrated"):
        index.sample(sample, 0, action_offsets=[3, 4, 5, 6])
    with pytest.raises(ValueError, match="horizon"):
        index.sample(sample, 0, action_offsets=[0])


def test_command_mask_does_not_cross_interval_start_or_end():
    data = manifest()
    data["segments"][0].update(start_frame=3, end_frame=5)
    result = SteeringCommands(data).sample(
        {"episode_index": 3, "frame_index": 3, "task": "task", "action": torch.zeros(4, 2)},
        0,
        action_offsets=[-1, 0, 1, 2],
    )
    assert result["action_is_pad"].tolist() == [True, False, False, True]


def test_camera_scoped_points_keep_original_coordinate_frame():
    data = manifest()
    command = {
        "style": "point",
        "text": "reach with the left gripper to",
        "evidence": "test tracker",
        "camera": "observation.images.base",
        "image_size": [640, 480],
        "points": [[100, 200]],
    }
    data["segments"][0]["commands"] = [command]
    result = SteeringCommands(data).sample({"episode_index": 3, "frame_index": 2, "task": "task"}, 0)
    assert result["task"] == "In base view (640x480 pixels), reach with the left gripper to: [100, 200]."
    command["points"] = [[640, 200]]
    with pytest.raises(ValueError, match="outside"):
        SteeringCommands(data)


def moving_point_manifest():
    data = manifest()
    data["segments"][0]["commands"] = [
        {
            "style": "point",
            "text": "pick at the first point and place at the second point",
            "evidence": {"source": "reviewed object tracks"},
            "camera": "observation.images.base",
            "image_size": [640, 480],
            "points_by_frame": {str(frame): [[100 + frame, 200], [300, 150]] for frame in range(5)},
        }
    ]
    return data


def test_moving_points_follow_sample_frame_without_shortening_action_chunk():
    data = moving_point_manifest()
    before = copy.deepcopy(data)
    index = SteeringCommands(data)
    action = torch.arange(56).reshape(4, 14).float()
    for frame in (1, 2):
        sample = {"index": frame, "episode_index": 3, "frame_index": frame, "task": "task", "action": action}
        result = index.sample(sample, 0, deterministic=True, action_offsets=[0, 1, 2, 3])
        assert f"[{100 + frame}, 200], [300, 150]" in result["task"]
        assert result["action"] is action
        assert result["action_is_pad"].tolist() == [False, False, False, frame == 2]
        assert index.at(3, frame)[0]["points"] == [[100 + frame, 200], [300, 150]]
    assert data == before
    profile = index.annotation_profile()
    assert profile["coordinate_frames_by_camera_style"]["observation.images.base"]["point"] == 5
    assert profile["expected_style_fraction_given_steering"]["point"] == 1


@pytest.mark.parametrize("bad_frame", [None, [], [[640, 200]], [[float("nan"), 200]]])
def test_missing_or_invalid_moving_geometry_is_never_filled(bad_frame):
    data = moving_point_manifest()
    data["segments"][0]["commands"][0]["points_by_frame"]["2"] = bad_frame
    with pytest.raises(ValueError):
        SteeringCommands(data)


def test_moving_geometry_rejects_wrong_frames_and_unresolved_rendering():
    from lerobot.utils.steering import render_steering_command

    for removed, added in [("2", None), ("2", "5"), ("2", "02"), ("2", 2)]:
        data = moving_point_manifest()
        series = data["segments"][0]["commands"][0]["points_by_frame"]
        points = series.pop(removed)
        if added is not None:
            series[added] = points
        with pytest.raises(ValueError, match="exactly every frame"):
            SteeringCommands(data)
    data = moving_point_manifest()
    command = data["segments"][0]["commands"][0]
    with pytest.raises(ValueError, match="Resolve"):
        render_steering_command({**command, "style": "combination"})
    command["points"] = [[1, 2]]
    with pytest.raises(ValueError, match="either static"):
        SteeringCommands(data)
    data = moving_point_manifest()
    data["segments"][0]["commands"][0]["points_by_frame"]["2"] = [[100, 200]]
    with pytest.raises(ValueError, match="same target count"):
        SteeringCommands(data)


@pytest.mark.parametrize("shape", [[480, 640, 3], [240, 320, 3]])
def test_moving_coordinates_check_actual_dataset_camera_shape(monkeypatch, tmp_path, shape):
    path = tmp_path / "moving.json"
    path.write_text(json.dumps(moving_point_manifest()))

    def init(dataset, repo_id, **kwargs):
        dataset.repo_id = repo_id
        dataset.episodes = [3]
        dataset.delta_timestamps = None
        dataset.meta = SimpleNamespace(
            fps=30,
            episodes={3: {"length": 5}},
            features={"observation.images.base": {"shape": shape}},
        )

    monkeypatch.setattr("lerobot.datasets.language_task.RecipeTaskDataset.__init__", init)
    if shape == [240, 320, 3]:
        with pytest.raises(ValueError, match="camera dimensions"):
            SteeringCommandDataset("test/data", revision="abc", steering_manifest=str(path))
    else:
        dataset = SteeringCommandDataset("test/data", revision="abc", steering_manifest=str(path))
        assert dataset.steering_coverage["expected_style_fraction"]["point"] == 0.8


def test_composition_keeps_per_frame_geometry_for_targets_traces_and_combinations():
    compose = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/prepare_steering.py"))[
        "compose_segment"
    ]
    series = {"0": [[100, 200], [300, 150]], "1": [[110, 210], [300, 150]]}
    features = {
        "episode_index": 3,
        "start_frame": 0,
        "end_frame": 2,
        "subtask": "place tape in bin",
        "subtask_evidence": "video",
        "views": [
            {
                "camera": "observation.images.base",
                "image_size": [640, 480],
                "targets": [
                    {"points_by_frame": series, "instruction": "pick and place", "evidence": "tracks"}
                ],
                "traces": [{"points_by_frame": series, "arm": "left", "evidence": "gripper tracks"}],
            }
        ],
    }
    before = copy.deepcopy(features)
    segment = compose(features)
    assert features == before
    assert [c["style"] for c in segment["commands"]] == [
        "subtask",
        "point",
        "combination",
        "trace",
        "combination",
    ]
    assert all(c["points_by_frame"] == series and "points" not in c for c in segment["commands"][1:])
    segment["review"] = {"verdict": "accepted", "reviewer": "test-only"}
    data = manifest()
    data["segments"] = [segment]
    assert all(c["points"] == series["1"] for c in SteeringCommands(data).at(3, 1)[1:])


@pytest.mark.parametrize("geometry", [{"point": [100, 200]}, {"points": [[100, 200], [300, 150]]}])
def test_composition_preserves_target_order_and_combined_grounding(geometry):
    compose = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/prepare_steering.py"))[
        "compose_segment"
    ]
    features = {
        "episode_index": 3,
        "start_frame": 0,
        "end_frame": 5,
        "subtask": "place tape in the bin",
        "subtask_evidence": {"source": "timestamped video"},
        "views": [
            {
                "camera": "observation.images.base",
                "image_size": [640, 480],
                "targets": [
                    {
                        **geometry,
                        "instruction": "pick the object at the first point and place it at the second point"
                        if "points" in geometry
                        else "reach to the target",
                        "evidence": {"source": "reviewed object masks", "frame": 0},
                    }
                ],
            }
        ],
    }
    before = copy.deepcopy(features)
    segment = compose(features)
    assert features == before
    point, combination = segment["commands"][1:]
    assert point["style"] == "point" and combination["style"] == "combination"
    assert point["points"] == combination["points"] == geometry.get("points", [geometry.get("point")])
    assert combination["evidence"] == [features["subtask_evidence"], point["evidence"]]
    assert "review" not in segment
    # Structure alone never accepts a command. This review is only a test fixture.
    segment["review"] = {"verdict": "accepted", "reviewer": "test-only"}
    data = manifest()
    data["segments"] = [segment]
    SteeringCommands(data)
    target = features["views"][0]["targets"][0]
    target.update(point=[1, 2], points=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="either one point"):
        compose(features)


def test_coverage_detects_absent_episodes_holes_and_out_of_range_intervals():
    index = SteeringCommands(manifest())
    assert index.coverage({3: 5})["complete"]
    report = index.coverage({3: 7, 4: 9})
    assert not report["complete"]
    assert report["covered_frames"] == 5
    assert report["total_frames"] == 16
    assert report["gaps"] == [
        {"episode_index": 3, "start_frame": 5, "end_frame": 7},
        {"episode_index": 4, "start_frame": 0, "end_frame": 9},
    ]
    with pytest.raises(ValueError, match="exceeds"):
        index.coverage({3: 4})


def test_fk_uses_calibrated_measured_joints_and_rejects_reversals(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/fk_motion.py"))
    urdf = tmp_path / "test.urdf"
    urdf.write_text("test fixture, not a robot model")
    config = {
        "arm": "left",
        "units": "radians",
        "calibration": "test calibration",
        "frame": "left_base",
        "urdf": str(urdf),
        "calibration_status": "verified",
        "joint_names": ["joint"],
        "state_keys": ["left.joint"],
        "signs": [-1],
        "offset_degrees": [10],
        "tool_frame": "tcp",
        "deadband_m": 0.01,
        "axis_directions": [{"positive": "right", "negative": "left"}] * 3,
    }

    class Kinematics:
        def forward_kinematics(self, joints):
            result = np.eye(4)
            result[0, 3] = joints[0] / 100
            return result

    extract = module["extract_motion"]
    motion = extract([[0], [np.pi / 2]], ["left.joint"], config, kinematics=Kinematics())
    assert motion[0]["text"] == "move the left gripper left"
    assert motion[0]["evidence"]["displacement_m"][0] == pytest.approx(-0.9)
    assert motion[0]["evidence"]["calibration_status"] == "verified"
    assert motion[0]["evidence"]["joint_mapping"]["units"] == "radians"
    assert motion[0]["evidence"]["joint_mapping"]["offset_degrees"] == [10]
    with pytest.raises(ValueError, match="reversing"):
        extract([[0], [np.pi / 2], [0.1]], ["left.joint"], config, kinematics=Kinematics())
    for status in ("unverified", None):
        config["calibration_status"] = status
        with pytest.raises(ValueError, match="verified calibration"):
            extract([[0], [np.pi / 2]], ["left.joint"], config, kinematics=Kinematics())
        positions = module["measured_positions"](
            [[0], [np.pi / 2]], ["left.joint"], config, kinematics=Kinematics()
        )
        assert positions[-1, 0] - positions[0, 0] == pytest.approx(-0.9)


@pytest.mark.parametrize("style", ["motion", "combination"])
@pytest.mark.parametrize("status", [None, "unverified", "verified"])
def test_accepted_visual_review_does_not_override_fk_calibration(style, status):
    data = manifest()
    evidence = {
        "method": "measured-joint FK",
        "calibration": "fixture calibration",
        "calibration_status": status,
    }
    data["segments"][0]["commands"] = [
        {
            "style": style,
            "text": "move the left gripper upward",
            "evidence": evidence if style == "motion" else ["fixture video", [evidence]],
        }
    ]
    if status == "verified":
        SteeringCommands(data)
    else:
        with pytest.raises(ValueError, match="verified calibration"):
            SteeringCommands(data)


def test_annotation_profile_weights_frames_and_variants_and_excludes_other_episodes():
    data = manifest()
    point = {
        "style": "point",
        "text": "reach here",
        "evidence": "fixture",
        "camera": "observation.images.base",
        "image_size": [640, 480],
        "points": [[30, 40]],
    }
    later = copy.deepcopy(data["segments"][0])
    later.update(
        start_frame=5,
        end_frame=20,
        commands=[
            {"style": "subtask", "text": "reach", "evidence": "fixture"},
            {"style": "subtask", "text": "approach", "evidence": "fixture"},
            point,
        ],
    )
    heldout = copy.deepcopy(later)
    heldout.update(
        episode_index=4,
        start_frame=0,
        end_frame=50,
        commands=[{**point, "style": "trace", "points": [[1, 2], [3, 4]]}],
    )
    data["segments"].extend([later, heldout])
    profile = SteeringCommands(data).coverage({3: 20})["annotation_profile"]
    assert profile["annotated_frames"] == 20
    assert profile["frames_with_style"] == {
        "subtask": 20,
        "motion": 5,
        "point": 15,
        "trace": 0,
        "combination": 0,
    }
    assert profile["alternatives_by_style"]["subtask"] == 3
    fractions = profile["expected_style_fraction_given_steering"]
    assert fractions["subtask"] == pytest.approx(0.625)
    assert fractions["motion"] == pytest.approx(0.125)
    assert fractions["point"] == pytest.approx(0.25)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert set(profile["missing_styles"]) == {"trace", "combination"}
    assert profile["coordinate_frames_by_camera_style"]["observation.images.base"]["point"] == 15
    assert profile["physical_capabilities_verified"] is False


def test_required_styles_use_selected_split_and_profile_includes_task_mixture(monkeypatch, tmp_path):
    data = manifest()
    other = copy.deepcopy(data["segments"][0])
    other.update(
        episode_index=4,
        commands=[
            {
                "style": "point",
                "text": "reach",
                "evidence": "fixture",
                "camera": "observation.images.base",
                "image_size": [640, 480],
                "points": [[1, 2]],
            }
        ],
    )
    data["segments"].append(other)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))

    def init(dataset, repo_id, *, episodes, **kwargs):
        dataset.repo_id = repo_id
        dataset.episodes = episodes
        dataset.delta_timestamps = None
        dataset.meta = SimpleNamespace(
            fps=30,
            episodes={3: {"length": 5}, 4: {"length": 5}},
            features={"observation.images.base": {"shape": [480, 640, 3]}},
        )

    monkeypatch.setattr("lerobot.datasets.language_task.RecipeTaskDataset.__init__", init)
    with pytest.raises(ValueError, match="absent from selected episodes"):
        SteeringCommandDataset(
            "test/data", episodes=[3], revision="abc", steering_manifest=str(path), required_styles=["point"]
        )
    dataset = SteeringCommandDataset(
        "test/data",
        episodes=[3],
        revision="abc",
        steering_manifest=str(path),
        required_styles=["motion", "subtask"],
    )
    profile = dataset.steering_coverage
    assert profile["expected_style_fraction"]["task"] == 0.2
    assert profile["expected_style_fraction"]["motion"] == 0.4
    assert profile["expected_style_fraction"]["subtask"] == 0.4
    assert profile["expected_style_fraction"]["point"] == 0
    assert profile["required_styles"] == ["motion", "subtask"]
    assert len(profile["manifest_sha256"]) == 64


@pytest.mark.parametrize("task_probability", [0, 1])
def test_dataloader_batch_applies_steering_and_command_boundary_mask(monkeypatch, task_probability):
    rows = [
        {
            "index": i,
            "episode_index": 3,
            "frame_index": frame,
            "timestamp": frame / 30,
            "task": "original task",
            "action": torch.ones(4, 2),
            "action_is_pad": torch.zeros(4, dtype=torch.bool),
        }
        for i, frame in enumerate([1, 4])
    ]
    reader = SimpleNamespace(
        get_items=lambda indices: [rows[i] for i in indices], get_item=lambda index: rows[index]
    )
    monkeypatch.setattr(LeRobotDataset, "_ensure_reader", lambda self: reader)
    dataset = object.__new__(SteeringCommandDataset)
    dataset.task_recipe = TrainingRecipe.from_dict(
        {"messages": [{"role": "user", "content": "corrected overall task", "stream": "low_level"}]}
    )
    data = manifest()
    data["segments"][0]["commands"] = data["segments"][0]["commands"][:1]
    dataset.steering = SteeringCommands(data)
    dataset.task_probability = task_probability
    dataset.deterministic = True
    dataset.steering_action_offsets = [0, 1, 2, 3]
    loader = torch.utils.data.DataLoader(dataset, sampler=[0, 1], batch_size=2, collate_fn=list)
    batch = next(iter(loader))
    expected = "corrected overall task" if task_probability else "reach for the tape"
    assert [row["task"] for row in batch] == [expected, expected]
    assert not batch[0]["action_is_pad"].any()
    assert batch[1]["action_is_pad"].tolist() == [False] + [not task_probability] * 3
    assert all(row["action"] is original["action"] for row, original in zip(batch, rows, strict=True))
    assert not rows[1]["action_is_pad"].any()
