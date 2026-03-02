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

"""
Integration tests for loading robot/teleop pipelines from the Hugging Face Hub.

These tests require network access and are marked with ``@pytest.mark.integration``.
Run with::

    pytest tests/test_pipeline_hub.py -m integration -v

The tests verify the full end-to-end flow of:
1. Loading a pipeline from the Hub via ``RobotProcessorPipeline.from_pretrained(...)``
2. Attaching it to a robot or teleoperator via ``set_output_pipeline`` / ``set_input_pipeline``
3. Verifying that ``observation_features`` / ``action_features`` differ from the raw versions

Note: The Hub repos referenced below are placeholders. Update them once actual pipelines
are published to the Hub.
"""

import pytest


# ─── Shared mock infrastructure (mirrors test_robot_pipeline.py) ──────────────

try:
    from tests.test_robot_pipeline import MockRobot, MockTeleop  # type: ignore[import]
except ImportError:
    # Fallback if tests are run from a different working directory
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_robot_pipeline import MockRobot, MockTeleop


# ─── Integration tests ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_load_robot_pipeline_from_hub(tmp_path):
    """
    Full end-to-end: load a FK observation pipeline for SO-101 from the Hub,
    attach it to a robot, and verify that observation_features are transformed.

    Prerequisites:
    - A pipeline must be published at ``lerobot/so101-fk-observation-pipeline`` on the Hub.
    - A URDF file must be available locally (update ``local_urdf_path`` to point to it).
    """
    pytest.importorskip("huggingface_hub")
    from lerobot.processor.pipeline import RobotProcessorPipeline

    local_urdf_path = tmp_path / "so101.urdf"
    # NOTE: In a real test environment, provide an actual URDF or mock the kinematics.
    # For now, this test validates the Hub loading mechanism only if a URDF is provided.
    if not local_urdf_path.exists():
        pytest.skip("URDF not available; skipping Hub loading test")

    pipeline = RobotProcessorPipeline.from_pretrained(
        "lerobot/so101-fk-observation-pipeline",
        overrides={"step_0": {"urdf_path": str(local_urdf_path)}},
    )
    robot = MockRobot()
    robot.set_output_pipeline(pipeline)

    # Pipeline-transformed features should differ from raw features (EE vs joints)
    assert robot.observation_features != robot.raw_observation_features


@pytest.mark.integration
def test_load_teleop_pipeline_from_hub(tmp_path):
    """
    Full end-to-end: load a FK action pipeline for SO-101 leader from the Hub,
    attach it to a teleoperator, and verify that action_features are transformed.

    Prerequisites:
    - A pipeline must be published at ``lerobot/so101-leader-fk-action-pipeline`` on the Hub.
    - A URDF file must be available locally (update ``local_urdf_path`` to point to it).
    """
    pytest.importorskip("huggingface_hub")
    from lerobot.processor.pipeline import RobotProcessorPipeline

    local_urdf_path = tmp_path / "so101.urdf"
    if not local_urdf_path.exists():
        pytest.skip("URDF not available; skipping Hub loading test")

    pipeline = RobotProcessorPipeline.from_pretrained(
        "lerobot/so101-leader-fk-action-pipeline",
        overrides={"step_0": {"urdf_path": str(local_urdf_path)}},
    )
    teleop = MockTeleop()
    teleop.set_output_pipeline(pipeline)

    # Pipeline-transformed features should differ from raw features (EE vs joints)
    assert teleop.action_features != teleop.raw_action_features
