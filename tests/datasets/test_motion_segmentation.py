# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def segment():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/fk_motion.py"))[
        "segment_motion"
    ]


def test_reversal_gripper_and_semantic_splits_preserve_every_frame(segment):
    times = np.arange(7, dtype=float)
    left = np.zeros((7, 3))
    left[:, 0] = [0, 1, 2, 1, 0, 0, 0]
    right = np.zeros((7, 3))
    results = segment(
        times,
        {"left": left, "right": right},
        {"left": np.array([0, 0, 0, 0, -30, -30, -30]), "right": np.zeros(7)},
        boundaries=[1, 5],
        translation_speed_m_s=0.1,
        gripper_speed_deg_s=10,
        median_window=1,
    )
    assert [i for s in results for i in range(s["start_frame"], s["end_frame"])] == list(range(7))
    assert {1, 2, 3, 4, 5, 6} <= {s["start_frame"] for s in results}
    assert results[0]["channel_signs"]["left.x"] == 1
    assert next(s for s in results if s["start_frame"] == 2)["channel_signs"]["left.x"] == -1
    assert next(s for s in results if s["start_frame"] == 3)["channel_signs"]["left.gripper"] == -1
    assert all(s["review"] == "pending" for s in results)
    assert results[-1]["review_flags"] == ["no_outgoing_observation"]
    assert results[-1]["channel_signs"] is None


def test_short_gripper_pulse_is_preserved_when_median_suppresses_it(segment):
    results = segment(
        np.arange(9) * 0.1,
        {"left": np.zeros((9, 3))},
        {"left": np.array([0, 0, 0, 20, 0, 0, 0, 0, 0])},
        boundaries=[],
        translation_speed_m_s=0.01,
        gripper_speed_deg_s=10,
        median_window=5,
    )
    events = [s for s in results[:-1] if s["raw_velocity_signs"]["left.gripper"] != [0]]
    assert len(events) == 2
    assert {s["raw_velocity_signs"]["left.gripper"][0] for s in events} == {-1, 1}
    assert all("raw_and_filtered_velocity_disagree" in s["review_flags"] for s in events)
    assert all("short_interval" in s["review_flags"] for s in events)


def test_filter_never_smooths_across_semantic_boundary(segment):
    xyz = np.zeros((5, 3))
    xyz[:, 0] = [0, 1, 2, 1, 0]
    results = segment(
        np.arange(5),
        {"right": xyz},
        {"right": np.zeros(5)},
        boundaries=[2],
        translation_speed_m_s=0.1,
        gripper_speed_deg_s=10,
        median_window=5,
    )
    assert [(s["start_frame"], s["end_frame"]) for s in results] == [(0, 2), (2, 4), (4, 5)]
    assert [s["channel_signs"]["right.x"] for s in results[:-1]] == [1, -1]


def test_jitter_is_not_a_new_interval_but_large_brief_reversal_is(segment):
    times = np.arange(21) * 0.033
    xyz = np.zeros((21, 3))
    xyz[:, 0] = np.arange(21) % 2 * 0.0001
    arguments = {
        "boundaries": [],
        "translation_speed_m_s": 0.001,
        "gripper_speed_deg_s": 10,
        "translation_reversal_m": 0.01,
        "median_window": 5,
        "minimum_duration_s": 0.15,
    }
    quiet = segment(times, {"left": xyz}, {"left": np.zeros(21)}, **arguments)
    assert len(quiet) == 2  # one unlabelled noisy interval plus the terminal observation
    xyz[10, 0] = 0.03
    pulse = segment(times, {"left": xyz}, {"left": np.zeros(21)}, **arguments)
    assert 10 in {s["start_frame"] for s in pulse}
    assert sum(s["end_frame"] - s["start_frame"] for s in pulse) == 21


@pytest.mark.parametrize("times", [[0, 1, 1], [0, 1, 10], [0, float("nan"), 2]])
def test_invalid_measurement_timing_is_not_interpolated(segment, times):
    with pytest.raises(ValueError):
        segment(
            times,
            {"left": np.zeros((3, 3))},
            {"left": np.zeros(3)},
            boundaries=[],
            translation_speed_m_s=0.01,
            gripper_speed_deg_s=10,
        )


def test_gripper_only_segmentation_has_no_invented_cartesian_channels(segment):
    results = segment(
        np.arange(6),
        {},
        {"left": np.array([0, -30, -60, -30, 0, 0]), "right": np.zeros(6)},
        boundaries=[2],
        translation_speed_m_s=0.01,
        gripper_speed_deg_s=10,
        median_window=1,
    )
    assert [i for s in results for i in range(s["start_frame"], s["end_frame"])] == list(range(6))
    assert {0, 2, 4, 5} <= {s["start_frame"] for s in results}
    for interval in results[:-1]:
        assert set(interval["channel_signs"]) == {"left.gripper", "right.gripper"}
        assert set(interval["measured_delta"]) == {"left.gripper", "right.gripper"}
    assert results[-1]["review_flags"] == ["no_outgoing_observation"]


@pytest.fixture
def gripper_extractor():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/fk_motion.py"))[
        "extract_gripper_motion"
    ]


@pytest.fixture
def gripper_config():
    return {
        "arm": "left",
        "state_key": "left_gripper.pos",
        "units": "degrees",
        "opening_sign": -1,
        "deadband_degrees": 10,
        "semantics_review": {
            "verdict": "accepted",
            "reviewer": {"kind": "model", "id": "test-reviewer"},
            "notes": "Fixture: decreasing angle opens the recorded left gripper.",
            "evidence": [{"source": "test fixture"}],
        },
    }


@pytest.mark.parametrize("units", ["degrees", "radians"])
@pytest.mark.parametrize("angles,verb", [([0, -30, -60], "open"), ([-60, -30, 0], "close")])
def test_gripper_command_preserves_review_and_measured_evidence(
    gripper_extractor, gripper_config, units, angles, verb
):
    gripper_config["units"] = units
    values = np.asarray(angles, dtype=float)
    if units == "radians":
        values = np.deg2rad(values)
    result = gripper_extractor(values[:, None], ["left_gripper.pos"], gripper_config)
    assert result[0]["text"] == f"{verb} the left gripper"
    evidence = result[0]["evidence"]
    assert evidence["measured_delta_degrees"] == pytest.approx(angles[-1] - angles[0])
    assert evidence["semantics_review"] == gripper_config["semantics_review"]
    assert evidence["review"] == "pending" and not evidence["accepted_training_labels"]
    assert "human_verified" not in evidence
    gripper_config["semantics_review"]["notes"] = "changed"
    assert evidence["semantics_review"]["notes"] != "changed"


@pytest.mark.parametrize(
    "angles", [[0, -60, 0], [0, -100, -20], [0, -100, -80, -180], [0, float("nan"), -60]]
)
def test_gripper_reversals_and_missing_measurements_cannot_be_commands(
    gripper_extractor, gripper_config, angles
):
    with pytest.raises(ValueError):
        gripper_extractor(np.asarray(angles)[:, None], ["left_gripper.pos"], gripper_config)


def test_gripper_hold_is_not_an_open_or_close_command(gripper_extractor, gripper_config):
    assert gripper_extractor([[0], [-0.2], [0]], ["left_gripper.pos"], gripper_config) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"semantics_review": {}},
        {"opening_sign": 0},
        {"opening_sign": True},
        {"arm": "right"},
        {"units": "normalized"},
        {"deadband_degrees": 0},
    ],
)
def test_gripper_labels_require_supported_mapping_and_review(gripper_extractor, gripper_config, changes):
    config = copy.deepcopy(gripper_config)
    config.update(changes)
    with pytest.raises(ValueError):
        gripper_extractor([[0], [-60]], ["left_gripper.pos"], config)
