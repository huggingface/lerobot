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
"""`lerobot_train.train()` materializes `dataset.artifact_ref` before building any dataset object,
on the main process only, and the resulting local directory is loaded without ever calling a
Hugging Face Hub download function.

`WandBLogger` is mocked one layer up (per the handoff), the same way `test_wandb_utils_dataset_artifact.py`
mocks the store boundary: no real wandb SDK call happens here.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import draccus
import numpy as np
import pytest

# Importing lerobot_train eagerly pulls in lerobot.datasets, which needs the `dataset` extra.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
# ...and these tests call `train()`, which needs `accelerate` from the `training` extra. Without
# this second guard they fail (rather than skip) in any tier that installs `dataset` alone.
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

import lerobot.scripts.lerobot_train as train_module  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.integrations.wandb_artifacts import MaterializedArtifact  # noqa: E402
from lerobot.policies.act.configuration_act import (
    ACTConfig,  # noqa: E402, F401  (registers --policy.type act)
)

_ACTION_FEATURE = {"dtype": "float32", "shape": (6,), "names": None}


def _build_local_dataset(root: Path) -> None:
    """A tiny, genuinely valid local LeRobot dataset `make_train_eval_datasets` can load offline."""
    dataset = LeRobotDataset.create(
        repo_id="placeholder/unused",
        fps=30,
        features={"action": _ACTION_FEATURE},
        root=root,
        robot_type="so101",
        use_videos=False,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    for frame_index in range(2):
        dataset.add_frame({"action": np.full(6, frame_index, dtype=np.float32), "task": "task-0"})
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


class _StopError(Exception):
    """Raised by the mocked `make_policy` to prove control reached it (and no further)."""


def _parse_cfg(tmp_path: Path) -> TrainPipelineConfig:
    return draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.artifact_ref",
            "team/proj/pick-cube:latest",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--wandb.enable",
            "true",
            "--wandb.project",
            "proj",
            "--output_dir",
            str(tmp_path / "run"),
        ],
    )


def test_train_materializes_artifact_before_dataset_creation_with_no_hf_calls(monkeypatch, tmp_path):
    cfg = _parse_cfg(tmp_path)
    materialized_root = tmp_path / "run" / "wandb_dataset"
    dataset_source = tmp_path / "artifact-source"
    _build_local_dataset(dataset_source)

    fake_logger = MagicMock()

    def _fake_download(ref: str, download_root: Path) -> MaterializedArtifact:
        assert download_root == materialized_root
        shutil.copytree(dataset_source, download_root)
        return MaterializedArtifact(
            requested_ref=ref,
            resolved_ref="team/proj/pick-cube:v3",
            local_path=download_root,
            version="v3",
            digest="deadbeef",
            metadata={},
        )

    fake_logger.download_dataset_artifact.side_effect = _fake_download
    monkeypatch.setattr(train_module, "WandBLogger", lambda cfg: fake_logger)

    # The real HF Hub download entry points `LeRobotDatasetMetadata`/`LeRobotDataset` fall back to.
    # If the artifact path reaches them, the test fails immediately instead of hitting the network.
    def _hf_download_forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Hugging Face Hub snapshot_download must never be called on the artifact_ref path"
        )

    monkeypatch.setattr("lerobot.datasets.dataset_metadata.snapshot_download", _hf_download_forbidden)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.snapshot_download", _hf_download_forbidden)

    # Stop right after a real dataset is built from the materialized directory, before any policy or
    # optimizer allocation (criterion 7).
    monkeypatch.setattr(train_module, "make_policy", lambda **_kwargs: (_ for _ in ()).throw(_StopError))

    with pytest.raises(_StopError):
        train_module.train(cfg)

    fake_logger.download_dataset_artifact.assert_called_once_with(
        "team/proj/pick-cube:latest", materialized_root
    )
    fake_logger.record_dataset_artifact_lineage.assert_called_once_with(
        "team/proj/pick-cube:latest", "team/proj/pick-cube:v3"
    )
    # cfg was repointed at the materialized directory. `repo_id` stays unset — it means "Hub dataset"
    # everywhere downstream; the local dataset's constructor name comes from `local_id` instead.
    assert cfg.dataset.root == materialized_root
    assert cfg.dataset.repo_id is None
    assert cfg.dataset.local_id == "pick-cube"


def _resume_cfg(output_dir: Path) -> TrainPipelineConfig:
    return draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.artifact_ref",
            "team/proj/pick-cube:latest",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--wandb.enable",
            "true",
            "--wandb.project",
            "proj",
            "--output_dir",
            str(output_dir),
            "--resume",
            "true",
        ],
    )


def test_materialize_dataset_artifact_resume_reuses_matching_sidecar_with_no_wandb_call(tmp_path):
    """A resumed run whose sidecar's `requested_ref` matches `cfg.dataset.artifact_ref` reuses the
    materialized copy and takes `resolved_ref`/`digest` straight from the sidecar: no W&B call (no
    re-resolving a possibly-moved alias) is needed to verify or restore lineage.
    """
    from lerobot.integrations.wandb_artifacts.sidecar import ArtifactSidecar, write_sidecar

    output_dir = tmp_path / "run"
    download_root = output_dir / "wandb_dataset"
    _build_local_dataset(download_root)
    write_sidecar(
        download_root,
        ArtifactSidecar(
            requested_ref="team/proj/pick-cube:latest",
            resolved_ref="team/proj/pick-cube:v3",
            version="v3",
            digest="deadbeef",
        ),
    )
    cfg = _resume_cfg(output_dir)

    fake_logger = MagicMock()

    train_module._materialize_dataset_artifact(cfg, fake_logger, is_main_process=True)

    fake_logger.download_dataset_artifact.assert_not_called()
    # No W&B call at all beyond recording lineage from the sidecar (e.g. no re-resolve).
    assert [call[0] for call in fake_logger.method_calls] == ["record_dataset_artifact_lineage"]
    fake_logger.record_dataset_artifact_lineage.assert_called_once_with(
        "team/proj/pick-cube:latest", "team/proj/pick-cube:v3"
    )
    assert cfg.dataset.root == download_root
    assert cfg.dataset.repo_id is None


def test_materialize_dataset_artifact_resume_fails_fast_on_sidecar_mismatch(tmp_path):
    """The sidecar records a different artifact than `cfg.dataset.artifact_ref` now asks for (e.g.
    `--dataset.artifact_ref` or `--output_dir` changed between runs): refuse to train on the stale
    copy instead of silently reusing unrelated data.
    """
    from lerobot.integrations.wandb_artifacts.sidecar import ArtifactSidecar, write_sidecar

    output_dir = tmp_path / "run"
    download_root = output_dir / "wandb_dataset"
    _build_local_dataset(download_root)
    write_sidecar(
        download_root,
        ArtifactSidecar(
            requested_ref="team/proj/other-dataset:latest",
            resolved_ref="team/proj/other-dataset:v1",
            version="v1",
            digest="c0ffee",
        ),
    )
    cfg = _resume_cfg(output_dir)

    fake_logger = MagicMock()

    with pytest.raises(ValueError, match="team/proj/other-dataset:latest.*team/proj/pick-cube:latest"):
        train_module._materialize_dataset_artifact(cfg, fake_logger, is_main_process=True)

    fake_logger.download_dataset_artifact.assert_not_called()
    fake_logger.record_dataset_artifact_lineage.assert_not_called()


def test_materialize_dataset_artifact_resume_fails_fast_when_sidecar_absent(tmp_path):
    """A materialized directory with no sidecar at all (e.g. left over from before this identity
    check existed) can't be proven to hold the right artifact either: fail fast rather than guess.
    """
    output_dir = tmp_path / "run"
    download_root = output_dir / "wandb_dataset"
    _build_local_dataset(download_root)
    cfg = _resume_cfg(output_dir)

    fake_logger = MagicMock()

    with pytest.raises(ValueError, match=str(download_root)):
        train_module._materialize_dataset_artifact(cfg, fake_logger, is_main_process=True)

    fake_logger.download_dataset_artifact.assert_not_called()
    fake_logger.record_dataset_artifact_lineage.assert_not_called()


def test_train_skips_materialization_entirely_when_artifact_ref_unset(monkeypatch, tmp_path):
    """Unset artifact_ref: byte-for-byte unchanged behavior, no `wandb_dataset` dir, no logger calls."""
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "team/proj/pick-cube",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--output_dir",
            str(tmp_path / "run"),
        ],
    )

    monkeypatch.setattr(
        train_module,
        "WandBLogger",
        lambda cfg: (_ for _ in ()).throw(AssertionError("WandBLogger must not be constructed")),
    )
    monkeypatch.setattr(
        train_module,
        "make_train_eval_datasets",
        lambda cfg: (_ for _ in ()).throw(_StopError),
    )

    with pytest.raises(_StopError):
        train_module.train(cfg)

    assert cfg.dataset.root is None
    assert cfg.dataset.repo_id == "team/proj/pick-cube"
    assert not (tmp_path / "run" / "wandb_dataset").exists()
