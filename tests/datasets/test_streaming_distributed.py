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

"""End-to-end distributed streaming smoke test under a real `accelerate launch`.

Mirrors tests/training/test_multi_gpu.py but runs on CPU and only checks the dataloading contract: with
two processes, `split_dataset_by_node` (auto-resolved from the Accelerate state) must give each rank a
disjoint set of frames that together cover the dataset. Skips if the environment can't actually spawn
>= 2 processes (e.g. local macOS multi-CPU), so it never silently passes as a single process.
"""

import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from tests.fixtures.constants import DUMMY_REPO_ID

WORKER = """
import json, sys
from accelerate import PartialState
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

root, repo_id, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
state = PartialState()
ds = StreamingLeRobotDataset(
    repo_id=repo_id, root=root, shuffle=False, episode_pool_size=8, max_num_shards=8
)
indices = [int(frame["index"]) for frame in ds]
payload = {"rank": state.process_index, "world": state.num_processes, "indices": indices}
with open(f"{out_dir}/rank_{state.process_index}.json", "w") as f:
    json.dump(payload, f)
"""


@pytest.mark.skipif(shutil.which("accelerate") is None, reason="accelerate CLI not available")
def test_accelerate_launch_ranks_are_disjoint(tmp_path, lerobot_dataset_factory):
    total_frames = 160
    repo_id = f"{DUMMY_REPO_ID}-acc"
    root = tmp_path / "ds"
    lerobot_dataset_factory(
        root=root,
        repo_id=repo_id,
        total_episodes=8,
        total_frames=total_frames,
        use_videos=False,
        data_files_size_in_mb=0.001,
        chunks_size=1,
    )

    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        "accelerate",
        "launch",
        "--num_processes=2",
        "--num_machines=1",
        "--mixed_precision=no",
        "--dynamo_backend=no",
        "--cpu",
        str(worker),
        str(root),
        repo_id,
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"accelerate launch failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    payloads = [json.loads(p.read_text()) for p in sorted(out_dir.glob("rank_*.json"))]
    if len(payloads) < 2 or any(p["world"] < 2 for p in payloads):
        pytest.skip("environment did not spawn >= 2 distributed processes (e.g. local macOS multi-CPU)")

    rank_sets = [set(p["indices"]) for p in payloads]
    assert rank_sets[0].isdisjoint(rank_sets[1]), "ranks streamed overlapping frames under accelerate launch"
    assert set().union(*rank_sets) == set(range(total_frames)), "ranks did not jointly cover all frames"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
