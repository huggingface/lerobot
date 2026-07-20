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

"""End-to-end dry-run tests for the dog-nav REPL + synthetic scene.

These exercise the whole navigation stack — sim scene → voxel map →
SigLIP stand-in → skills → agent → controller — with no robot, camera,
or models.
"""

from __future__ import annotations

import pytest

from lerobot.navigation.dog_cli import DogController, _build_dry_run, main
from lerobot.navigation.sim import kitchen_scene


def test_kitchen_scene_builds_with_all_objects():
    scene = kitchen_scene()
    assert {o.name for o in scene.objects} == {"couch", "chair", "lamp", "plant"}
    assert len(scene.voxel_map) > 0
    assert scene.voxel_map.feature_dim == scene.feature_dim


def test_feature_extractor_matches_object_vectors():
    scene = kitchen_scene()
    fx = scene.feature_extractor()
    couch = scene.object("couch")
    emb = fx.encode_text("couch")
    # The couch query should align with the couch's stored basis vector.
    import numpy as np

    assert float(np.dot(emb, couch.feature_vec / np.linalg.norm(couch.feature_vec))) > 0.9


def test_controller_reaches_mapped_object():
    ctl = _build_dry_run()
    result = ctl.handle_prompt("couch")
    assert result.fully_successful
    tr = result.target_results[0]
    assert tr.reached
    # Landed near the couch ground-truth (3.0, _, 2.0).
    assert tr.final_xyz is not None
    assert abs(tr.final_xyz[0] - 3.0) < 1.5
    assert abs(tr.final_xyz[2] - 2.0) < 1.5


def test_controller_navigates_to_each_object():
    for name, (gx, gz) in {
        "couch": (3.0, 2.0),
        "chair": (-2.0, -1.5),
        "plant": (-2.5, 2.5),
    }.items():
        ctl = _build_dry_run()
        result = ctl.handle_prompt(name)
        assert result.fully_successful, f"failed to reach {name}"
        fx = result.target_results[0].final_xyz
        assert abs(fx[0] - gx) < 1.5 and abs(fx[2] - gz) < 1.5


def test_controller_abstains_on_absent_object():
    ctl = _build_dry_run()
    result = ctl.handle_prompt("banana")  # not in the scene
    assert not result.fully_successful
    assert result.target_results[0].reason in {"budget_exhausted", "no_frontier"}


def test_idle_tick_explores_or_reports_no_frontier():
    ctl = _build_dry_run()
    ex = ctl.idle_tick()
    # A fully-observed synthetic floor may have no frontier; either way the
    # call must be well-formed and not raise.
    assert ex.reason in {"ok", "no frontier"}


def test_main_single_command_dry_run_returns_zero():
    assert main(["--dry-run", "--command", "couch", "--log-level", "WARNING"]) == 0


def test_main_absent_object_returns_nonzero():
    assert main(["--dry-run", "--command", "banana", "--log-level", "WARNING"]) == 1


def test_main_live_mode_refuses_until_pipeline_lands():
    with pytest.raises(SystemExit):
        main(["--command", "couch"])


def test_dogcontroller_stop_is_safe():
    ctl = _build_dry_run()
    ctl.stop()
    assert isinstance(ctl, DogController)
