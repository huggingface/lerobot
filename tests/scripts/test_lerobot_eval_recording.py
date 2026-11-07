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

from lerobot.scripts.lerobot_eval import _get_batch_recording_target


def test_batch_recording_target_preserves_single_batch():
    recording_dir = Path("recordings")
    repo_id = "user/eval"

    batch_dir, batch_repo_id = _get_batch_recording_target(recording_dir, repo_id, batch_ix=0, n_batches=1)

    assert batch_dir == recording_dir
    assert batch_repo_id == repo_id


def test_batch_recording_target_is_unique_for_multiple_batches():
    recording_dir = Path("recordings")
    repo_id = "user/eval"

    first_dir, first_repo_id = _get_batch_recording_target(recording_dir, repo_id, batch_ix=0, n_batches=2)
    second_dir, second_repo_id = _get_batch_recording_target(recording_dir, repo_id, batch_ix=1, n_batches=2)

    assert first_dir == recording_dir / "batch_0000"
    assert second_dir == recording_dir / "batch_0001"
    assert first_dir != second_dir
    assert first_repo_id == "user/eval_batch_0000"
    assert second_repo_id == "user/eval_batch_0001"


def test_batch_recording_target_without_recording_dir():
    batch_dir, batch_repo_id = _get_batch_recording_target(None, "user/eval", batch_ix=1, n_batches=2)

    assert batch_dir is None
    assert batch_repo_id == "user/eval"
