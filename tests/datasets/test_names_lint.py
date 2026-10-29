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
"""Pure-Python regression coverage for non-blocking dataset channel-name lints."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.datasets.names_lint import lint_dataset_info, lint_feature_names, write_names_lint_report


def _feature(shape, names, dtype="float32"):
    return {"dtype": dtype, "shape": shape, "names": names}


def _codes(report):
    return [issue["code"] for issue in report["issues"]]


def test_semantic_flat_names_have_no_warnings():
    report = lint_feature_names({"action": _feature((3,), ["left_elbow", "left_wrist", "gripper"])})
    assert report["issue_count"] == 0
    assert report["feature_classes"] == {"action": "count_match"}


def test_duplicate_names_in_grouped_feature():
    report = lint_feature_names(
        {"action": _feature((3,), {"arm": ["shoulder", "elbow"], "gripper": ["elbow"]})}
    )
    assert _codes(report) == ["duplicate_name"]
    assert report["issues"][0]["details"] == {"name": "elbow", "positions": [1, 2]}


def test_mismatched_shape_after_grouped_flattening():
    report = lint_feature_names({"state": _feature([2, 2], {"left": ["l1", "l2"], "right": ["r1"]})})
    assert _codes(report) == []
    assert report["feature_classes"]["state"] == "multidimensional_excluded"


def test_placeholder_names_are_unattested_not_invalid():
    report = lint_feature_names({"action": _feature((3,), ["motor_0", "joint-1", "left_gripper"])})
    assert report["feature_classes"]["action"] == "count_match"
    assert _codes(report) == ["placeholder_names"]
    assert report["issues"][0]["details"]["names"] == ["motor_0", "joint-1"]


def test_absent_names_are_distinguished_from_placeholders():
    report = lint_feature_names({"state": _feature((6,), None), "action": _feature((1,), ["motor_0"])})
    assert report["feature_classes"] == {"action": "count_match", "state": "absent"}
    assert _codes(report) == ["placeholder_names"]


def test_visual_names_are_axes_not_pixel_channels():
    report = lint_feature_names({"camera": _feature((480, 640, 3), ["height", "width", "channels"], "video")})
    assert report["issue_count"] == 0
    assert report["feature_classes"] == {"camera": "visual_excluded"}


def test_same_name_in_action_and_state_is_not_a_duplicate():
    report = lint_feature_names(
        {"action": _feature((1,), ["gripper"]), "observation.state": _feature((1,), ["gripper"])}
    )
    assert report["issue_count"] == 0


def test_invalid_names_are_warning_not_exception():
    report = lint_feature_names({"action": _feature((2,), ["gripper", 9])})
    assert _codes(report) == ["unsupported_names"]
    assert report["feature_classes"]["action"] == "unsupported"


def test_indexed_mapping_support_and_gap_warning():
    report = lint_feature_names(
        {
            "action": _feature((2,), {"gripper": 1, "wrist": 0}),
            "state": _feature((2,), {"wrist": 0, "elbow": 2}),
        }
    )
    assert _codes(report) == ["unsupported_names"]
    assert report["issues"][0]["feature"] == "state"


@pytest.mark.parametrize("names", [{"delta_x": 0}, {"axes": {"delta_x": 0}}, [{"delta_x": 0}]])
@pytest.mark.parametrize("size", [1, 4])
def test_singleton_index_mapping_preserves_channel_semantics(names, size):
    report = lint_feature_names({"action": _feature([size], names)})
    if size == 1:
        assert report["feature_classes"] == {"action": "count_match"}
        assert report["issue_count"] == 0
    else:
        assert report["feature_classes"] == {"action": "mismatch"}
        assert report["warnings_by_code"] == {"name_shape_mismatch": 1}
        assert report["issues"][0]["details"] == {"expected": 4, "actual": 1, "shape": [4]}


@pytest.mark.parametrize("names", [{"delta_x": 1}, {"delta_x": True}])
def test_invalid_singleton_index_mapping_remains_unsupported(names):
    report = lint_feature_names({"action": _feature([4], names)})
    assert report["feature_classes"] == {"action": "unsupported"}
    assert _codes(report) == ["unsupported_names"]


def test_empty_name_list_does_not_pass_shape_check():
    assert _codes(lint_feature_names({"action": _feature((2,), [])})) == ["empty_declaration"]


def test_invalid_features_are_structured_warning():
    report = lint_dataset_info({"features": "not a mapping"})
    assert report["warnings_by_code"] == {"invalid_features": 1}


def test_report_written_alongside_info_and_does_not_mutate_input(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    metadata = {"codebase_version": "v3.0", "features": {"action": _feature([2], ["motor_0", "motor_1"])}}
    info_file = meta / "info.json"
    original = json.dumps(metadata)
    info_file.write_text(original, encoding="utf-8")

    destination = write_names_lint_report(tmp_path)
    assert destination == meta / "names_lint.json"
    assert info_file.read_text(encoding="utf-8") == original
    assert json.loads(destination.read_text(encoding="utf-8"))["warnings_by_code"] == {"placeholder_names": 1}


def test_cli_emits_json_and_exits_zero_with_warnings(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps({"features": {"action": _feature([2], ["motor_0", "motor_0"])}}), encoding="utf-8"
    )
    cli = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "datasets" / "names_lint.py"
    completed = subprocess.run(
        [sys.executable, str(cli), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((meta / "names_lint.json").read_text(encoding="utf-8"))
    assert report["issue_count"] == 2
    assert set(_codes(report)) == {"duplicate_name", "placeholder_names"}


@pytest.mark.parametrize("names", [{"delta_x": 0}, {"axes": {"delta_x": 0}}])
def test_cli_singleton_index_mapping_warns_without_rejecting_dataset(tmp_path, names):
    info_file = tmp_path / "info.json"
    original = json.dumps({"features": {"action": _feature([4], names)}})
    info_file.write_text(original, encoding="utf-8")
    cli = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "datasets" / "names_lint.py"
    completed = subprocess.run(
        [sys.executable, str(cli), str(info_file)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "names_lint.json").read_text(encoding="utf-8"))
    assert report["feature_classes"] == {"action": "mismatch"}
    assert report["warnings_by_code"] == {"name_shape_mismatch": 1}
    assert info_file.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("names", [["left", "right"], {"arms": {"left": ["left"], "right": ["right"]}}])
def test_nested_and_flat_names_both_supported(names):
    assert lint_feature_names({"action": _feature((2,), names)})["issue_count"] == 0


@pytest.mark.parametrize("names", ["state", ["state"], {"axes": ["state"]}])
def test_libero_whole_vector_label_is_not_a_count_error(names):
    report = lint_feature_names({"observation.state": _feature([8], names)})
    assert report["feature_classes"]["observation.state"] == "possible_whole_vector"
    assert report["issue_count"] == 0


def test_metaworld_null_group_is_an_empty_declaration():
    report = lint_feature_names({"observation.state": _feature([4], {"axes": None})})
    assert report["feature_classes"]["observation.state"] == "empty_declaration"
    assert _codes(report) == ["empty_declaration"]


def test_multidimensional_per_axis_names_are_not_scalar_names():
    report = lint_feature_names({"waypoints": _feature([10, 2], ["way", "points"])})
    assert report["feature_classes"]["waypoints"] == "multidimensional_excluded"
    assert report["issue_count"] == 0


def test_last_axis_action_chunk_names_are_out_of_scope():
    report = lint_feature_names({"action": _feature([20, 14], [[f"arm_{i}" for i in range(14)]])})
    assert report["feature_classes"]["action"] == "multidimensional_excluded"
    assert report["issue_count"] == 0


def test_bookkeeping_feature_is_explicitly_excluded():
    report = lint_feature_names({"timestamp": _feature([1], None)})
    assert report["feature_classes"]["timestamp"] == "bookkeeping_excluded"
    assert report["issue_count"] == 0


def test_droid_joint_channel_name_count_mismatch():
    report = lint_feature_names(
        {"observation.state.joint_position": _feature([7], {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]})}
    )
    assert _codes(report) == ["name_shape_mismatch"]
    assert report["feature_classes"]["observation.state.joint_position"] == "mismatch"


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_report_cannot_overwrite_input_metadata(tmp_path, alias):
    info_file = tmp_path / "info.json"
    original = json.dumps({"features": {"action": _feature([2], ["motor_0", "motor_1"])}})
    info_file.write_text(original, encoding="utf-8")
    output = info_file
    if alias != "direct":
        output = tmp_path / "report.json"
        if alias == "symlink":
            output.symlink_to(info_file)
        else:
            output.hardlink_to(info_file)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_names_lint_report(info_file, output)
    assert info_file.read_text(encoding="utf-8") == original
