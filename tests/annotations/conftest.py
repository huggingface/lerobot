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
"""Shared fixtures for annotation-pipeline tests.

The on-disk dataset builder lives with the other dataset factories in
``tests/fixtures/dataset_factories.py`` (:func:`build_annotation_dataset`);
these fixtures only wire it into pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ``build_annotation_dataset`` pulls in ``lerobot.datasets`` (HF ``datasets``
# + ``pandas``, only in the ``dataset`` extra), so it's imported lazily inside
# each fixture — this conftest stays importable without that extra. The test
# modules ``pytest.importorskip("datasets")`` so they skip rather than error.


@pytest.fixture
def fixture_dataset_root(tmp_path: Path) -> Path:
    """A tiny dataset with two episodes, 12 frames each at 10 fps."""
    from tests.fixtures.dataset_factories import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[
            (0, 12, "Could you tidy the kitchen please?"),
            (1, 12, "Please clean up the kitchen"),
        ],
        fps=10,
    )


@pytest.fixture
def single_episode_root(tmp_path: Path) -> Path:
    from tests.fixtures.dataset_factories import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds_one",
        episode_specs=[(0, 30, "Pour water from the bottle into the cup.")],
        fps=10,
    )
