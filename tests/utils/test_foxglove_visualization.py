#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the Foxglove backend's pure helpers.

These cover topic naming, series labelling and feature-name parsing. They import
``foxglove_visualization`` directly and need NO ``foxglove`` extra: the SDK is imported lazily inside
the functions that talk to the server, so the helpers below run in the base test tier.
"""

import numpy as np

from lerobot.utils import foxglove_visualization as fv
from lerobot.utils.constants import ACTION, OBS_STATE


def test_foxglove_safe_name_collapses_dots():
    assert fv._foxglove_safe_name("observation.images.front") == "observation_images_front"
    assert fv._foxglove_safe_name("plain") == "plain"


def test_foxglove_topic_image_strips_prefix_without_doubling_images():
    # Fully-qualified camera key -> single clean segment (no doubled "images").
    assert fv._foxglove_topic("observation.images.front", is_image=True) == "/observation/images/front"
    # A nested camera name keeps its structure via safe-name collapsing.
    assert (
        fv._foxglove_topic("observation.images.wrist.left", is_image=True) == "/observation/images/wrist_left"
    )
    # Bare camera name (as real robots emit).
    assert fv._foxglove_topic("front", is_image=True) == "/observation/images/front"


def test_foxglove_topic_scalar_sources():
    assert fv._foxglove_topic(OBS_STATE) == "/observation/state"
    assert fv._foxglove_topic("observation.environment_state") == "/observation/state"
    assert fv._foxglove_topic(ACTION) == "/action/state"
    assert fv._foxglove_topic("action.delta") == "/action/state"


def test_labeled_scalars_uses_labels_then_index_fallback():
    assert fv._labeled_scalars("state", np.array([1.0, 2.0, 3.0])) == {
        "state_0": 1.0,
        "state_1": 2.0,
        "state_2": 3.0,
    }
    assert fv._labeled_scalars("state", [1.0, 2.0], ["pan", "lift"]) == {"pan": 1.0, "lift": 2.0}
    # Wrong-length labels fall back to index naming (never silently mislabels).
    assert fv._labeled_scalars("q", [1.0, 2.0], ["only_one"]) == {"q_0": 1.0, "q_1": 2.0}


def test_frame_to_scalars_matches_live_labeling_and_handles_scalar():
    frame = {OBS_STATE: np.array([1.0, 2.0])}
    # No metadata -> {short_name}_{i}, identical to the live-stream fallback.
    assert fv._frame_to_scalars(frame, OBS_STATE) == fv._labeled_scalars("state", np.array([1.0, 2.0]))
    assert fv._frame_to_scalars(frame, OBS_STATE) == {"state_0": 1.0, "state_1": 2.0}
    # Metadata labels are honored.
    assert fv._frame_to_scalars(frame, OBS_STATE, ["pan", "lift"]) == {"pan": 1.0, "lift": 2.0}
    # A 0-d scalar becomes a single entry named by the short feature name.
    assert fv._frame_to_scalars({ACTION: np.array(5.0)}, ACTION) == {"action": 5.0}
    # A missing feature yields an empty mapping.
    assert fv._frame_to_scalars({}, OBS_STATE) == {}


def test_feature_dim_names_formats():
    # Flat list of names.
    assert fv._feature_dim_names({"shape": [2], "names": ["x", "y"]}) == ["x", "y"]
    # Category mapping (dict of lists).
    assert fv._feature_dim_names({"shape": [2], "names": {"motors": ["m0", "m1"]}}) == ["m0", "m1"]
    # name -> index mapping (returned sorted by index).
    assert fv._feature_dim_names({"shape": [2], "names": {"delta_x": 0, "delta_y": 1}}) == [
        "delta_x",
        "delta_y",
    ]
    # Bool values must NOT be treated as an index map (bool is a subclass of int).
    assert fv._feature_dim_names({"shape": [2], "names": {"a": True, "b": False}}) is None
    # Mismatched length -> None (won't silently mislabel).
    assert fv._feature_dim_names({"shape": [3], "names": ["x", "y"]}) is None
    # Missing / absent names -> None.
    assert fv._feature_dim_names(None) is None
    assert fv._feature_dim_names({"shape": [2]}) is None


def test_is_scalar():
    assert fv._is_scalar(1.0)
    assert fv._is_scalar(np.float32(2.0))
    assert fv._is_scalar(np.array(3.0))  # 0-d array
    assert not fv._is_scalar(np.array([1.0, 2.0]))
    assert not fv._is_scalar("x")
