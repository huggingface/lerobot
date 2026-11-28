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

from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

datasets = pytest.importorskip("datasets")

from lerobot.scripts.lerobot_train import make_dataloaders  # noqa: E402

pytestmark = pytest.mark.dataset


class _Frames(Dataset):
    episodes = None
    absolute_to_relative_idx = None

    def __init__(self, task_indices, *, has_task_column=True):
        self.task_indices = task_indices
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [len(task_indices)]},
            has_language_columns=False,
        )
        if has_task_column:
            self.hf_dataset = datasets.Dataset.from_dict({"task_index": task_indices})

    def __len__(self):
        return len(self.task_indices)

    def __getitem__(self, index):
        return {"index": index, "task_index": self.task_indices[index]}


@pytest.mark.parametrize(
    ("tasks", "cap", "drop", "has_task_column", "expected"),
    [
        ([0, 1, 2], 2, 0, True, [0, 1]),
        ([0, 1, 1, 1, 1, 2], 4, 0, True, [0, 1, 2, 5]),
        ([9, 1, 0, 1, 0], 2, 1, True, [1, 2]),
        ([0, 1, 2], 0, 0, True, [0, 1, 2]),
        ([2, 2, 0, 1], 2, 0, False, [0, 1]),
        ([], 2, 0, True, []),
    ],
)
def test_held_out_loader_respects_sample_cap(tasks, cap, drop, has_task_column, expected):
    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(drop_n_first_frames=drop),
        dataset=SimpleNamespace(streaming=False),
        resume=False,
        seed=42,
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
        max_eval_samples=cap,
    )
    training = _Frames([0, 0, 0])
    held_out = _Frames(tasks, has_task_column=has_task_column)
    _, loader = make_dataloaders(
        cfg, training, held_out, step=0, parallel_dims=SimpleNamespace(device_type="cpu")
    )

    assert loader is not None
    assert [index for batch in loader for index in batch["index"].tolist()] == expected
