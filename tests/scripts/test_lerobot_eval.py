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
from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts.lerobot_eval import get_eval_provenance


def test_get_eval_provenance_schema():
    """Verify that get_eval_provenance extracts all expected metadata fields."""
    mock_cfg = SimpleNamespace(
        policy=SimpleNamespace(
            path="lerobot/diffusion_pusht",
            pretrained_path=Path("lerobot/diffusion_pusht"),
            type="diffusion",
            device="cpu",
        ),
        env=SimpleNamespace(
            type="pusht",
            task="PushT-v0",
            fps=10,
        ),
        eval=SimpleNamespace(
            n_episodes=10,
            batch_size=2,
            recording=False,
            recording_repo_id=None,
            recording_private=False,
        ),
        seed=1000,
        output_dir=Path("/tmp/eval_test"),
        trust_remote_code=False,
    )

    provenance = get_eval_provenance(mock_cfg)

    # Tool identification
    assert "tool" in provenance
    assert provenance["tool"]["name"] == "lerobot-eval"
    assert "version" in provenance["tool"]

    # Policy metadata
    assert "policy" in provenance
    assert provenance["policy"]["path"] == "lerobot/diffusion_pusht"
    assert provenance["policy"]["type"] == "diffusion"

    # Environment metadata
    assert "environment" in provenance
    assert provenance["environment"]["type"] == "pusht"
    assert provenance["environment"]["task"] == "PushT-v0"
    assert provenance["environment"]["fps"] == 10

    # System metadata
    assert "system" in provenance
    assert "os" in provenance["system"]
    assert "python_version" in provenance["system"]
    assert "torch_version" in provenance["system"]

    # Eval settings
    assert "eval_settings" in provenance
    assert provenance["eval_settings"]["n_episodes"] == 10
    assert provenance["eval_settings"]["batch_size"] == 2
    assert provenance["eval_settings"]["seed"] == 1000

    # Timestamp
    assert "timestamp_utc" in provenance

    # JSON serializability check
    serialized = json.dumps(provenance)
    deserialized = json.loads(serialized)
    assert deserialized["environment"]["task"] == "PushT-v0"


def test_get_eval_provenance_with_none_policy():
    """Verify that get_eval_provenance gracefully handles minimal/None policy configuration."""
    mock_cfg = SimpleNamespace(
        policy=None,
        env=SimpleNamespace(type="aloha_sim", task="AlohaTransferCube-v0", fps=50),
        eval=SimpleNamespace(n_episodes=5, batch_size=1),
        seed=42,
    )

    provenance = get_eval_provenance(mock_cfg)

    assert provenance["policy"]["path"] is None
    assert provenance["policy"]["type"] is None
    assert provenance["environment"]["type"] == "aloha_sim"
    assert provenance["eval_settings"]["seed"] == 42
