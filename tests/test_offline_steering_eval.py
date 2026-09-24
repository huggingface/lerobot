# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Offline comparisons must not leak training episodes or compare unmatched targets."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def api():
    return runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/offline_steering_eval.py"))


@pytest.fixture
def manifest():
    return {
        "version": 1,
        "source": {"repo_id": "test/source", "revision": "pinned"},
        "segments": [
            {
                "episode_index": ep,
                "start_frame": start,
                "end_frame": end,
                "review": {"verdict": "accepted", "reviewer": "test fixture"},
                "commands": [{"style": style, "text": style, "evidence": "fixture"} for style in styles],
            }
            for ep, start, end, styles in [(90, 2, 10, ["subtask"]), (98, 7, 9, ["subtask", "motion"])]
        ],
    }


def test_panel_is_fixed_and_each_style_has_identical_anchor_and_task_reference(api, manifest):
    original = copy.deepcopy(manifest)
    panel = api["make_panel"](manifest)
    assert manifest == original
    assert panel == api["make_panel"](manifest)
    assert [(s["episode_index"], s["frame_index"]) for s in panel["samples"]] == [
        (90, 2),
        (90, 5),
        (90, 9),
        (98, 7),
        (98, 8),
    ]
    assert panel["prompt_counts"] == {"task": 5, "subtask": 5, "motion": 2}
    assert panel["annotation_profile"]["missing_styles"] == ["combination", "point", "trace"]
    assert all(s["commands"][0]["style"] == "task" for s in panel["samples"])
    assert len({s["seed"] for s in panel["samples"]}) == len(panel["samples"])


def test_panel_rejects_training_episodes_and_unreviewed_labels(api, manifest):
    manifest["segments"][0]["episode_index"] = 89
    with pytest.raises(ValueError, match="held-out"):
        api["make_panel"](manifest)
    manifest["segments"][0]["episode_index"] = 90
    manifest["segments"][0]["review"]["verdict"] = "uncertain"
    with pytest.raises(ValueError, match="accepted"):
        api["make_panel"](manifest)


def test_error_ignores_out_of_interval_targets_and_episode_padding(api):
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0], [float("nan"), float("nan")]])
    prediction = torch.zeros(4, 2)
    kwargs = {"frame": 4, "start": 4, "end": 6, "scale": torch.tensor([1.0, 2.0])}
    error = api["action_errors"](prediction, target, torch.tensor([False, True, False, True]), **kwargs)
    assert error == {
        "valid_action_steps": 1,
        "valid_action_indices": [0],
        "mae_per_dimension": [1.0, 2.0],
        "normalized_mse": 1.0,
    }
    target[1:] = -1e10
    assert error == api["action_errors"](
        prediction, target, torch.tensor([False, True, False, True]), **kwargs
    )
    with pytest.raises(ValueError, match="No valid"):
        api["action_errors"](prediction, target, torch.ones(4, dtype=torch.bool), **kwargs)


def test_exported_indices_preserve_holes_in_valid_demonstration_targets(api):
    target = torch.tensor([[0.0, 0.0], [float("nan"), float("nan")], [2.0, 4.0], [999.0, 999.0]])
    error = api["action_errors"](
        torch.zeros_like(target),
        target,
        torch.tensor([False, True, False, False]),
        frame=4,
        start=4,
        end=7,
        scale=torch.tensor([1.0, 2.0]),
    )
    assert error["valid_action_indices"] == [0, 2]
    assert target[error["valid_action_indices"]].tolist() == [[0.0, 0.0], [2.0, 4.0]]
    assert error["normalized_mse"] == 2.0


def test_paired_metric_does_not_compare_different_motion_coverage_or_overweight_paraphrases(api):
    rows = [
        {"episode_index": 90, "frame_index": frame, "style": style, "normalized_mse": mse}
        for frame, style, mse in [(1, "task", 100), (2, "task", 10), (2, "motion", 3), (2, "motion", 5)]
    ]
    summary = api["summarize"](rows)
    assert summary["task"]["mean_normalized_mse"] == 55
    assert summary["motion"] == {
        "anchors": 1,
        "command_variants": 2,
        "mean_normalized_mse": 4,
        "mean_paired_delta_vs_task": -6,
    }


def test_saved_split_cannot_label_training_data_as_heldout(api):
    meta = SimpleNamespace(total_episodes=100, episodes=[{"tasks": ["pick"]} for _ in range(100)])
    assert api["training_episodes"]({"eval_split": 0.1}, meta) == list(range(90))
    assert 98 in api["training_episodes"]({}, meta)
    assert 98 not in api["training_episodes"]({"exclude_episodes": list(range(90, 100))}, meta)


def test_development_panel_uses_explicit_episode_identities(api, manifest):
    manifest["segments"][0]["episode_index"] = 6
    manifest["segments"][1]["episode_index"] = 50
    panel = api["make_panel"](manifest, development_episodes=[50, 6])
    assert panel["evaluation_split"] == "development"
    assert panel["heldout_episodes"] == [50, 6]
    assert api["evaluation_episodes"](panel, trained_episodes=[0, 1, 2]) == [6, 50]
    assert panel["prompt_counts"] == {"task": 5, "subtask": 5, "motion": 2}
    with pytest.raises(ValueError, match="training split"):
        api["evaluation_episodes"](panel, trained_episodes=[0, 6])


@pytest.mark.parametrize("episodes", [[], [6, 6], [90], [-1], [100], [True], [6.0]])
def test_invalid_development_holdouts_are_rejected(api, manifest, episodes):
    with pytest.raises(ValueError, match="held-out"):
        api["make_panel"](manifest, development_episodes=episodes)


def test_loaded_panel_cannot_hide_training_samples_behind_final_declaration(api, manifest):
    panel = api["make_panel"](manifest)
    panel["samples"][0]["episode_index"] = 6
    with pytest.raises(ValueError, match="outside the declared"):
        api["evaluation_episodes"](panel, trained_episodes=list(range(90)))


def test_leaking_checkpoint_is_rejected_before_model_or_optional_dependencies(
    api, manifest, tmp_path, monkeypatch
):
    manifest["segments"][0]["episode_index"] = 6
    manifest["segments"][1]["episode_index"] = 50
    panel = api["make_panel"](manifest, development_episodes=[6, 50])
    (tmp_path / "train_config.json").write_text(json.dumps({"dataset": manifest["source"]}))
    meta = SimpleNamespace(total_episodes=100, episodes=[{"tasks": ["pick"]} for _ in range(100)])
    evaluate = api["evaluate"]
    monkeypatch.setitem(evaluate.__globals__, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)

    def unexpected_model_dependency(*args, **kwargs):
        pytest.fail("A leaking checkpoint must fail before loading model dependencies")

    monkeypatch.setitem(evaluate.__globals__, "require_package", unexpected_model_dependency)
    with pytest.raises(ValueError, match="training split"):
        evaluate(panel, tmp_path, tmp_path, tmp_path / "output.json", "cpu")


def test_point_interventions_change_coordinates_only_and_keep_repeat_control(api, manifest):
    manifest["segments"] = [manifest["segments"][0]]
    manifest["segments"][0]["commands"] = [
        {
            "style": "point",
            "text": "pick at the first point and place at the second",
            "camera": "observation.images.base",
            "image_size": [640, 480],
            "points": [[100, 120], [320, 240]],
            "evidence": "fixture",
        }
    ]
    original = copy.deepcopy(manifest)
    panel = api["make_point_sensitivity_panel"](manifest, anchors_per_style=1)
    assert manifest == original
    assert panel["diagnostic"] == "point_sensitivity"
    commands = {c["variant"]: c for c in panel["samples"][0]["commands"]}
    assert commands["reference"]["task"] == commands["repeat"]["task"]
    assert commands["mirror_pick_x"]["points"] == [[539, 120], [320, 240]]
    assert commands["mirror_place_x"]["points"] == [[100, 120], [319, 240]]
    assert commands["swap_points"]["points"] == [[320, 240], [100, 120]]
    assert all(
        c["task"].startswith("In base view (640x480 pixels), pick at the first") for c in commands.values()
    )
    assert set(panel["prompt_counts"].values()) == {1}


def test_sensitivity_rejects_split_leakage_and_absent_points(api, manifest):
    with pytest.raises(ValueError, match="No reviewed point"):
        api["make_point_sensitivity_panel"](manifest)
    manifest["segments"][0]["episode_index"] = 6
    with pytest.raises(ValueError, match="held-out"):
        api["make_point_sensitivity_panel"](manifest)


@pytest.fixture
def sensitivity_rows():
    return [
        {
            "episode_index": 90,
            "frame_index": 2,
            "variant": name,
            "task": "same" if name in {"reference", "repeat"} else name,
            "seed": 7,
            "observation_state": [0.0] * 14,
            "predicted_action_chunk": [[value] * 14] * 2,
        }
        for name, value in [
            ("reference", 1),
            ("repeat", 1),
            ("mirror_pick_x", 3),
            ("mirror_place_x", 5),
            ("swap_points", 7),
        ]
    ]


def test_sensitivity_measures_prediction_changes_without_demonstration_labels(api, sensitivity_rows):
    result = api["point_sensitivity"](sensitivity_rows, torch.full((14,), 2.0))
    assert {k: v["mean_normalized_rms_change"] for k, v in result.items()} == {
        "reference": 0,
        "repeat": 0,
        "mirror_pick_x": 1,
        "mirror_place_x": 2,
        "swap_points": 3,
    }
    assert all(
        "normalized_mse" not in row and "valid_demonstrated_actions" not in row for row in sensitivity_rows
    )


@pytest.mark.parametrize("change", ["seed", "observation", "repeat", "missing", "duplicate", "nonfinite"])
def test_invalid_sensitivity_controls_are_rejected(api, sensitivity_rows, change):
    if change == "seed":
        sensitivity_rows[2]["seed"] += 1
    elif change == "observation":
        sensitivity_rows[2]["observation_state"][0] = 1
    elif change == "repeat":
        sensitivity_rows[1]["task"] = "different instruction"
    elif change == "missing":
        sensitivity_rows.pop()
    elif change == "duplicate":
        sensitivity_rows.append(copy.deepcopy(sensitivity_rows[0]))
    else:
        sensitivity_rows[2]["predicted_action_chunk"][0][0] = float("nan")
    with pytest.raises(ValueError):
        api["point_sensitivity"](sensitivity_rows, torch.ones(14))
