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

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.processor import DataProcessorPipeline, Torch2NumpyActionProcessorStep


@pytest.mark.parametrize("squeeze_batch_dim", [False, True])
def test_torch2numpy_preserves_batch_dimension_setting_after_save_load(
    tmp_path: Path, squeeze_batch_dim: bool
):
    pipeline = DataProcessorPipeline(
        steps=[Torch2NumpyActionProcessorStep(squeeze_batch_dim=squeeze_batch_dim)],
        name="action_processor",
    )
    batch = {"action": torch.tensor([[1.0, 2.0, 3.0]])}
    expected = np.array([1.0, 2.0, 3.0] if squeeze_batch_dim else [[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(pipeline(batch)["action"], expected)

    pipeline.save_pretrained(tmp_path)
    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path, config_filename="action_processor.json", local_files_only=True
    )

    np.testing.assert_array_equal(loaded(batch)["action"], expected)
    assert loaded.steps[0].squeeze_batch_dim is squeeze_batch_dim


def test_torch2numpy_legacy_empty_config_uses_default_squeeze():
    pipeline = DataProcessorPipeline.from_config(
        {
            "steps": [{"registry_name": "torch2numpy_action_processor", "config": {}}],
        }
    )

    output = pipeline({"action": torch.tensor([[1.0, 2.0, 3.0]])})["action"]

    np.testing.assert_array_equal(output, np.array([1.0, 2.0, 3.0], dtype=np.float32))
