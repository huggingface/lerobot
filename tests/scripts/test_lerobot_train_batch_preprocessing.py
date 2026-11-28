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

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import _preprocess_dataset_batch  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402

pytestmark = pytest.mark.dataset


def test_preprocess_dataset_batch_normalizes_and_renames_before_processor():
    seen_batch = None

    def preprocessor(batch):
        nonlocal seen_batch
        seen_batch = batch
        return batch

    batch = {
        "image": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "actions": torch.ones(1, 2),
    }
    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=["image"],
        rename_map={"image": f"{OBS_IMAGES}.camera1", "actions": ACTION},
        preprocessor=preprocessor,
    )

    assert seen_batch is processed
    assert set(processed) == {f"{OBS_IMAGES}.camera1", ACTION}
    assert processed[f"{OBS_IMAGES}.camera1"].dtype == torch.float32
    torch.testing.assert_close(processed[f"{OBS_IMAGES}.camera1"], torch.ones(1, 3, 2, 2))


def test_preprocess_dataset_batch_does_not_apply_rename_map_twice():
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep

    rename_map = {
        "observation.state": "observation.state.original",
        "observation.state.cartesian": "observation.state",
        "action": "action.original",
        "action.cartesian": ACTION,
        "observation.images.exterior_1_left": f"{OBS_IMAGES}.exterior",
        "observation.images.wrist_left": f"{OBS_IMAGES}.wrist",
    }
    step = RenameObservationsProcessorStep(rename_map=rename_map)
    pipeline = DataProcessorPipeline(steps=[step])
    batch = {
        "observation.state": torch.tensor([1.0]),
        "observation.state.cartesian": torch.tensor([2.0]),
        "observation.state_is_pad": torch.tensor([False]),
        "observation.state.cartesian_is_pad": torch.tensor([True]),
        "action": torch.tensor([3.0]),
        "action.cartesian": torch.tensor([4.0]),
        "observation.images.exterior_1_left": torch.zeros(1, 3, 2, 2),
        "observation.images.wrist_left": torch.ones(1, 3, 2, 2),
    }

    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=[],
        rename_map=rename_map,
        preprocessor=pipeline,
    )

    torch.testing.assert_close(processed["observation.state"], torch.tensor([2.0]))
    torch.testing.assert_close(processed["observation.state.original"], torch.tensor([1.0]))
    torch.testing.assert_close(processed["observation.state_is_pad"], torch.tensor([True]))
    torch.testing.assert_close(processed["observation.state.original_is_pad"], torch.tensor([False]))
    torch.testing.assert_close(processed[ACTION], torch.tensor([4.0]))
    assert f"{OBS_IMAGES}.exterior" in processed
    assert f"{OBS_IMAGES}.wrist" in processed
    assert step.rename_map is rename_map
