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

import json
import os

import numpy as np
import pytest

pytest.importorskip("mujoco")
os.environ.setdefault("MUJOCO_GL", "egl")

from examples.so101_datagen.config import CalibrationConfig  # noqa: E402
from examples.so101_datagen.generate import GenerateConfig, generate  # noqa: E402
from examples.so101_datagen.processors import VirtualCalibration  # noqa: E402
from examples.so101_datagen.so101 import GRIPPER, JOINT_LIMITS_DEG, JOINTS  # noqa: E402
from examples.so101_datagen.worker import EPISODES_FILE  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.datasets.language_render import render_sample  # noqa: E402
from lerobot.datasets.recipe import load_recipe  # noqa: E402
from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.motors_bus import SerialMotorsBus  # noqa: E402
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT  # noqa: E402

RECIPES = ["subtask", "subtask_mem", "subtask_mem_vqa_speech"]


def _gl_available() -> bool:
    try:
        from mujoco.rendering.classic import gl_context

        gl_context.GLContext(8, 8).free()
        return True
    except Exception:
        return False


def test_virtual_calibration_reports_like_the_motors_bus():
    """What the synthetic follower reports equals what a real SO-101 bus computes from the same ticks."""
    motors = {
        joint: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if i == GRIPPER else MotorNormMode.DEGREES)
        for i, joint in enumerate(JOINTS)
    }
    rng = np.random.default_rng(0)
    limits = np.array(JOINT_LIMITS_DEG)
    calibration = VirtualCalibration(CalibrationConfig())
    for _ in range(50):
        calibration.start_episode(rng)
        bus = SerialMotorsBus(port="/dev/null", motors=motors, calibration=calibration.motor_calibration())
        values = rng.uniform(limits[:, 0], limits[:, 1])
        values[GRIPPER] = rng.uniform(0, 100)
        angles = values.copy()
        angles[GRIPPER] = limits[GRIPPER, 0] + values[GRIPPER] / 100 * (
            limits[GRIPPER, 1] - limits[GRIPPER, 0]
        )
        raw = calibration.raw(angles)
        expected = bus._normalize({i + 1: int(r) for i, r in enumerate(raw)})
        np.testing.assert_allclose(calibration.report(values), [expected[i + 1] for i in range(len(JOINTS))])


@pytest.mark.skipif(not _gl_available(), reason="needs an OpenGL context (EGL)")
def test_generate_merges_annotated_shards(tmp_path):
    """Two tiny shards are generated, merged, and every frame renders with the shipped recipes."""
    cfg = GenerateConfig(
        output_dir=tmp_path,
        num_shards=2,
        episodes_per_shard=1,
        num_workers=2,
    )
    cfg.generation.height = cfg.generation.width = 64
    cfg.generation.objects.mesh_prob = 0.0
    cfg.generation.annotation.min_visible_pixels = 4
    generate(cfg)

    merged = tmp_path / "merged"
    dataset = LeRobotDataset(cfg.repo_id, root=merged)
    assert dataset.num_episodes == 2
    assert dataset.meta.robot_type == "so_follower"
    assert dataset.meta.info.tools
    assert dataset.features["observation.state"]["names"] == [f"{j}.pos" for j in JOINTS]
    sidecar = [json.loads(line) for line in (merged / EPISODES_FILE).read_text().splitlines()]
    assert [entry["episode_index"] for entry in sidecar] == [0, 1]

    recipes = {name: load_recipe(f"src/lerobot/configs/recipes/{name}.yaml") for name in RECIPES}
    rendered_frames = dict.fromkeys(RECIPES, 0)
    vqa_frames = 0
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["observation.images.front"].shape == (3, 64, 64)
        persistent, events = item[LANGUAGE_PERSISTENT], item[LANGUAGE_EVENTS]
        assert persistent
        vqa_frames += bool(events)
        for name, recipe in recipes.items():
            # Rendering raises on ambiguous or malformed rows. None is a sub-recipe without a target at
            # this frame (e.g. memory before the first boundary), which training falls back from.
            rendered = render_sample(
                recipe=recipe,
                persistent=persistent,
                events=events,
                t=float(item["timestamp"]),
                sample_idx=i,
                task=item["task"],
            )
            rendered_frames[name] += rendered is not None
            if events and name == "subtask_mem_vqa_speech":
                assert rendered is not None, f"VQA frame {i} does not render"
    assert vqa_frames > 0
    for name, count in rendered_frames.items():
        assert count > 0.7 * len(dataset), f"{name} renders {count}/{len(dataset)} frames"
