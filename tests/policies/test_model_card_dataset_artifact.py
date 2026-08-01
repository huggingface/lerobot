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
"""`PreTrainedPolicy.generate_model_card` must never claim a Hugging Face dataset repo that doesn't
exist for artifact-backed runs (`cfg.dataset.artifact_ref` set): `dataset_repo_id` there is only the
W&B artifact's collection name, derived to satisfy the local dataset constructor (see
`_materialize_dataset_artifact`), not a real Hub repo.

`lerobot.policies.pretrained` is imported at the base install tier (no `wandb`/`datasets` extra), so
this module must not require them either.
"""

import re
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.integrations.wandb_artifacts.sidecar import ArtifactSidecar, write_sidecar
from lerobot.policies.pretrained import PreTrainedPolicy


class _FakeConfig:
    input_features: dict = {}
    output_features: dict = {}
    repo_id = "user/policy"
    license = None
    tags = None


class _FakeSelf:
    config = _FakeConfig()


class _FakeTasks:
    index = ["pick"]


class _FakeDatasetMeta:
    total_episodes = 10
    total_frames = 1000
    fps = 30
    tasks = _FakeTasks()
    robot_type = "so101"
    camera_keys = ["observation.images.front"]

    def __init__(self, repo_id: str):
        self.repo_id = repo_id


def _generate(cfg: TrainPipelineConfig | None, dataset_repo_id: str, dataset_meta) -> str:
    return str(
        PreTrainedPolicy.generate_model_card(
            _FakeSelf(), dataset_repo_id, "act", None, None, cfg=cfg, dataset_meta=dataset_meta
        )
    )


def _cfg(*, artifact_ref: str | None, repo_id: str | None, root: Path | None = None) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id=repo_id, artifact_ref=artifact_ref, root=root))
    cfg.dataset.root = root
    return cfg


def test_non_artifact_model_card_is_unchanged():
    """No `dataset.artifact_ref`: `datasets=` frontmatter and the Hub "Repository:" section/badge
    render exactly like before this change. Isolate that from the (pre-existing, unrelated)
    "Training Configuration" section, which only ever renders when `cfg` is passed at all.
    """
    dataset_meta = _FakeDatasetMeta(repo_id="user/dataset")

    with_cfg = _generate(_cfg(artifact_ref=None, repo_id="user/dataset"), "user/dataset", dataset_meta)
    without_cfg = _generate(None, "user/dataset", dataset_meta)
    with_cfg_minus_training = re.sub(
        r"\n## Training Configuration\n.*?\n---\n\n## How to Get Started",
        "\n---\n\n## How to Get Started",
        with_cfg,
        flags=re.S,
    )

    assert with_cfg_minus_training == without_cfg
    assert "datasets: user/dataset" in with_cfg
    assert "**Repository:** [user/dataset](https://huggingface.co/datasets/user/dataset)" in with_cfg
    assert "visualize-this-dataset" in with_cfg
    assert "W&B Artifact" not in with_cfg


def test_artifact_backed_model_card_omits_datasets_and_names_the_artifact_ref(tmp_path):
    """`dataset.artifact_ref` set: no `datasets=` claim, and the resolved (immutable) ref from the
    materialized directory's sidecar is named in the card body instead of the derived collection name.
    """
    write_sidecar(
        tmp_path,
        ArtifactSidecar(
            requested_ref="team/proj/pick-cube:latest",
            resolved_ref="team/proj/pick-cube:v3",
            version="v3",
            digest="deadbeef",
        ),
    )
    dataset_meta = _FakeDatasetMeta(repo_id="pick-cube")  # the derived collection name
    cfg = _cfg(artifact_ref="team/proj/pick-cube:latest", repo_id=None, root=tmp_path)

    card = _generate(cfg, "pick-cube", dataset_meta)

    assert "datasets:" not in card
    assert "**W&B Artifact:** `team/proj/pick-cube:v3`" in card
    assert "huggingface.co/datasets/pick-cube" not in card
    assert "**Repository:**" not in card
    assert "visualize-this-dataset" not in card


def test_artifact_backed_model_card_falls_back_to_requested_ref_without_a_sidecar(tmp_path):
    """No sidecar (or a mismatched one) at `cfg.dataset.root`: fall back to the requested ref rather
    than fail model-card generation, which must always succeed best-effort.
    """
    dataset_meta = _FakeDatasetMeta(repo_id="pick-cube")
    cfg = _cfg(artifact_ref="team/proj/pick-cube:latest", repo_id=None, root=tmp_path)

    card = _generate(cfg, "pick-cube", dataset_meta)

    assert "datasets:" not in card
    assert "**W&B Artifact:** `team/proj/pick-cube:latest`" in card
