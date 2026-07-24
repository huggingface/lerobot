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
"""Convert a DCP-format checkpoint into a distributable safetensors model, offline.

Runs single-process (no GPUs, no process group). Example:

```bash
lerobot-convert-dcp --checkpoint_dir=outputs/train/run/checkpoints/005000
lerobot-convert-dcp --checkpoint_dir=... --delete_dcp=true --push_to_hub=user/my-policy
```

`--push_to_hub` publishes the converted directory as a model repo, degrading gracefully: the
core artifacts (model.safetensors, config.json, processor files) always upload; the README
model card is enriched with training/dataset metadata only when `train_config.json` (and the
dataset it names) are reachable, with a WARNING naming exactly what was skipped otherwise.
DCP shard artifacts are never uploaded — published repos carry safetensors only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.configs import parser
from lerobot.distributed.checkpoint import dcp_to_safetensors
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.hub import DCP_ARTIFACT_PATTERNS
from lerobot.utils.utils import init_logging


@dataclass
class ConvertDcpConfig:
    """CLI config for the offline DCP-to-safetensors checkpoint conversion."""

    # A checkpoint step directory (containing pretrained_model/) or a pretrained_model
    # directory itself.
    checkpoint_dir: Path
    # Remove the DCP shard directory after a successful conversion.
    delete_dcp: bool = False
    # Publish the converted directory to this Hub repo id (e.g. "user/my-policy").
    push_to_hub: str | None = None
    private: bool | None = None


def _locate_pretrained_dir(checkpoint_dir: Path) -> Path:
    """Resolve the pretrained_model/ directory from a user-supplied checkpoint path.

    Args:
        checkpoint_dir (Path): A checkpoint step directory (containing `pretrained_model/`) or a
            `pretrained_model` directory itself.

    Returns:
        Path: The nested `pretrained_model/` directory when present, otherwise `checkpoint_dir`
        unchanged.
    """
    nested = checkpoint_dir / PRETRAINED_MODEL_DIR
    return nested if nested.is_dir() else checkpoint_dir


def _publish_converted(pretrained_dir: Path, repo_id: str, private: bool | None) -> None:
    """Best-effort publish of a converted checkpoint dir, degrading gracefully.

    The core artifacts (model.safetensors, config.json, processor files) always upload; the README
    model card gains training/dataset metadata only when `train_config.json` (and the dataset it
    names) are reachable, with a WARNING naming what was skipped otherwise. DCP shard artifacts are
    excluded from the upload.

    Args:
        pretrained_dir (Path): The converted `pretrained_model/` directory to upload.
        repo_id (str): Target Hub model repo id (e.g. "user/my-policy"); created if missing.
        private (bool | None): Repo visibility passed to `create_repo`; None keeps the Hub (or
            existing repo's) default.
    """
    from lerobot.common.train_utils import generate_model_card
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig

    train_cfg = None
    dataset_meta = None
    if (pretrained_dir / TRAIN_CONFIG_NAME).is_file():
        try:
            train_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
        except Exception as e:  # noqa: BLE001 — degrade, never block the upload
            logging.warning(f"Could not parse {TRAIN_CONFIG_NAME} ({e}); README will lack training metadata.")
    else:
        logging.warning(f"{TRAIN_CONFIG_NAME} missing; README will lack training metadata.")
    if train_cfg is not None:
        try:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            dataset_meta = LeRobotDatasetMetadata(
                repo_id=train_cfg.dataset.repo_id,
                root=train_cfg.dataset.root,
                revision=train_cfg.dataset.revision,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f"Dataset '{train_cfg.dataset.repo_id}' unreachable ({e}); README will lack dataset metadata."
            )
    try:
        model_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
        card = generate_model_card(model_cfg, cfg=train_cfg, dataset_meta=dataset_meta)
        card.save(str(pretrained_dir / "README.md"))
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Could not build the model card ({e}); publishing without README.")

    api = HfApi()
    repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(pretrained_dir),
        commit_message="Upload converted policy (DCP -> safetensors)",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        ignore_patterns=["*.tmp", "*.log", *DCP_ARTIFACT_PATTERNS],
    )
    logging.info(f"Model pushed to {commit_info.repo_url.url}")


def convert_checkpoint(cfg: ConvertDcpConfig) -> Path:
    """Merge a checkpoint's DCP shards into `model.safetensors`, then optionally publish it.

    Args:
        cfg (ConvertDcpConfig): Conversion options — the checkpoint directory to convert, whether
            to delete the DCP shards after a successful merge, and the optional Hub repo id (and
            visibility) to publish the converted directory to.

    Returns:
        Path: The path to the merged `model.safetensors` file.

    Raises:
        FileNotFoundError: If the checkpoint has no DCP shard directory, i.e. it was not saved
            with `checkpoint_format=dcp` (or `safetensors_dcp`).
    """
    from accelerate.utils.constants import FSDP_MODEL_NAME

    pretrained_dir = _locate_pretrained_dir(cfg.checkpoint_dir)
    dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(
            f"No DCP shard directory at {dcp_dir}. Point --checkpoint_dir at a checkpoint "
            "saved with checkpoint_format=dcp (or safetensors_dcp)."
        )
    logging.info(f"Merging {dcp_dir} -> {pretrained_dir / 'model.safetensors'}")
    safetensors_path = dcp_to_safetensors(dcp_dir, pretrained_dir, keep_dcp=not cfg.delete_dcp)
    if cfg.push_to_hub:
        _publish_converted(pretrained_dir, cfg.push_to_hub, cfg.private)
    return safetensors_path


@parser.wrap()
def convert_dcp(cfg: ConvertDcpConfig) -> Path:
    """Parser-wrapped CLI entry: run `convert_checkpoint` on the draccus-parsed config.

    Args:
        cfg (ConvertDcpConfig): The conversion config parsed from the command line.

    Returns:
        Path: The path to the merged `model.safetensors` file.
    """
    return convert_checkpoint(cfg)


def main() -> None:
    """`lerobot-convert-dcp` console entry point: set up logging and run the conversion."""
    init_logging()
    convert_dcp()


if __name__ == "__main__":
    main()
