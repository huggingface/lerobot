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

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.language import DEFAULT_TOOLS, language_feature_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "left,right,compatible",
    [
        ("implicit", "implicit", True),
        ("custom", "custom", True),
        ("implicit", "default", True),
        ("default", "implicit", True),
        ("custom", "implicit", False),
        ("implicit", "custom", False),
        ("custom", "conflict", False),
        ("ordered", "ordered", True),
        ("ordered", "reordered", False),
    ],
)
def test_aggregate_tool_catalog(tmp_path: Path, left: str, right: str, compatible: bool) -> None:
    custom = [{"type": "function", "function": {"name": "grasp", "parameters": {"type": "object"}}}]
    conflict = copy.deepcopy(custom)
    conflict[0]["function"]["parameters"] = {"type": "object", "required": ["object"]}
    catalogs = {"implicit": None, "default": DEFAULT_TOOLS, "custom": custom, "conflict": conflict}
    catalogs.update(ordered=custom + DEFAULT_TOOLS, reordered=DEFAULT_TOOLS + custom)
    roots = []
    originals = {}
    for source, selection in enumerate((left, right)):
        root = tmp_path / f"source{source}"
        dataset = LeRobotDataset.create(
            f"local/source{source}",
            root=root,
            fps=10,
            use_videos=False,
            features={"observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}},
        )
        dataset.meta.tools = copy.deepcopy(catalogs[selection])
        for frame in range(2):
            dataset.add_frame(
                {"observation.state": np.array([source, frame], dtype=np.float32), "task": "move"}
            )
        dataset.save_episode()
        dataset.finalize()
        # Annotations are added after recording; keep real tool calls alongside their catalog.
        tool_name = dataset.meta.tools[0]["function"]["name"]
        arguments = {"text": "ready"} if tool_name == "say" else {"object": "cube"}
        row = {
            "role": "assistant",
            "content": None,
            "style": "motion",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": [
                json.dumps({"type": "function", "function": {"name": tool_name, "arguments": arguments}})
            ],
        }
        language_type = pa.list_(
            pa.struct(
                [
                    ("role", pa.string()),
                    ("content", pa.string()),
                    ("style", pa.string()),
                    ("timestamp", pa.float32()),
                    ("camera", pa.string()),
                    ("tool_calls", pa.list_(pa.string())),
                ]
            )
        )
        for path in (root / "data").glob("*/*.parquet"):
            table = pq.read_table(path)
            table = table.append_column(
                "language_persistent", pa.array([[row]] * len(table), type=language_type)
            )
            pq.write_table(table, path)
            originals[path] = path.read_bytes()
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"]["language_persistent"] = language_feature_info()["language_persistent"]
        info_path.write_text(json.dumps(info))
        originals[info_path] = info_path.read_bytes()
        roots.append(root)

    output = tmp_path / "merged"
    kwargs = {
        "repo_ids": ["local/source0", "local/source1"],
        "aggr_repo_id": "local/merged",
        "roots": roots,
        "aggr_root": output,
    }
    if not compatible:
        with pytest.raises(ValueError, match="tool catalog"):
            aggregate_datasets(**kwargs)
        assert not output.exists()
    else:
        aggregate_datasets(**kwargs)
        merged = LeRobotDataset("local/merged", root=output)
        assert len(merged) == 4
        assert merged.meta.info.tools == catalogs[left]
        assert json.loads((output / "meta/info.json").read_text()).get("tools") == catalogs[left]
        expected_tools = catalogs[left] or DEFAULT_TOOLS
        assert merged.meta.tools == expected_tools
        for index in range(4):
            item = merged[index]
            np.testing.assert_array_equal(item["observation.state"].numpy(), [index // 2, index % 2])
            call = item["language_persistent"][0]["tool_calls"][0]
            name = expected_tools[0]["function"]["name"]
            arguments = {"text": "ready"} if name == "say" else {"object": "cube"}
            assert call == {"type": "function", "function": {"name": name, "arguments": arguments}}
    for path, original in originals.items():
        assert path.read_bytes() == original
