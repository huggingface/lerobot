#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Borrowed from https://github.com/fyhMer/fowm/blob/main/src/logger.py

# TODO(rcadene, alexander-soare): clean this file
"""

import logging
import os
import re
from dataclasses import asdict
from glob import glob
from pathlib import Path

import draccus
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_global_random_state
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode

PRETRAINED_MODEL = "pretrained_model"
TRAINING_STATE = "training_state.pth"


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"dataset:{cfg.dataset.repo_id}",
        f"seed:{cfg.seed}",
    ]
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(checkpoint_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(checkpoint_dir / "../wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


class Logger:
    """Primary logger object. Logs either locally or using wandb.

    The logger creates the following directory structure:

    provided_log_dir
    ├── checkpoints
    │   ├── specific_checkpoint_name
    │   │   ├── pretrained_model  # Hugging Face pretrained model directory
    │   │   │   ├── ...
    │   │   └── training_state.pth  # optimizer, scheduler, and random states + training step
    |   ├── another_specific_checkpoint_name
    │   │   ├── ...
    |   ├── ...
    │   └── last  # a softlink to the last logged checkpoint
    """

    pretrained_model_dir_name = PRETRAINED_MODEL
    training_state_file_name = TRAINING_STATE

    def __init__(self, cfg: TrainPipelineConfig):
        self._cfg = cfg
        self.log_dir = cfg.output_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_name = cfg.job_name
        self.checkpoints_dir = self.get_checkpoints_dir(self.log_dir)
        self.last_checkpoint_dir = self.get_last_checkpoint_dir(self.log_dir)
        self.last_pretrained_model_dir = self.get_last_pretrained_model_dir(self.log_dir)

        # Set up WandB.
        self._group = cfg_to_group(cfg)
        run_offline = not cfg.wandb.enable or not cfg.wandb.project
        if run_offline:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            os.environ["WANDB_SILENT"] = "true"
            import wandb

            wandb_run_id = None
            if cfg.resume:
                wandb_run_id = get_wandb_run_id_from_filesystem(self.checkpoints_dir)

            wandb.init(
                id=wandb_run_id,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=self.job_name,
                notes=cfg.wandb.notes,
                tags=cfg_to_group(cfg, return_list=True),
                dir=self.log_dir,
                config=asdict(self._cfg),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                resume="must" if cfg.resume else None,
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
            self._wandb = wandb

    @classmethod
    def get_checkpoints_dir(cls, log_dir: str | Path) -> Path:
        """Given the log directory, get the sub-directory in which checkpoints will be saved."""
        return Path(log_dir) / "checkpoints"

    @classmethod
    def get_last_checkpoint_dir(cls, log_dir: str | Path) -> Path:
        """Given the log directory, get the sub-directory in which the last checkpoint will be saved."""
        return cls.get_checkpoints_dir(log_dir) / "last"

    @classmethod
    def get_last_pretrained_model_dir(cls, log_dir: str | Path) -> Path:
        """
        Given the log directory, get the sub-directory in which the last checkpoint's pretrained weights will
        be saved.
        """
        return cls.get_last_checkpoint_dir(log_dir) / cls.pretrained_model_dir_name

    def save_model(self, save_dir: Path, policy: PreTrainedPolicy, wandb_artifact_name: str | None = None):
        """Save the weights of the Policy model using PyTorchModelHubMixin.

        The weights are saved in a folder called "pretrained_model" under the checkpoint directory.

        Optionally also upload the model to WandB.
        """

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        register_features_types()
        policy.save_pretrained(save_dir)
        # Also save the full config for the env configuration.
        self._cfg.save_pretrained(save_dir)
        if self._wandb and not self._cfg.wandb.disable_artifact:
            # note wandb artifact does not accept ":" or "/" in its name
            artifact = self._wandb.Artifact(wandb_artifact_name, type="model")
            artifact.add_file(save_dir / SAFETENSORS_SINGLE_FILE)
            self._wandb.log_artifact(artifact)
        if self.last_checkpoint_dir.exists():
            os.remove(self.last_checkpoint_dir)

    def save_training_state(
        self,
        save_dir: Path,
        train_step: int,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ):
        """Checkpoint the global training_step, optimizer state, scheduler state, and random state.

        All of these are saved as "training_state.pth" under the checkpoint directory.
        """
        training_state = {}
        training_state["step"] = train_step
        training_state.update(get_global_random_state())
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()
        torch.save(training_state, save_dir / self.training_state_file_name)

    def save_checkpoint(
        self,
        train_step: int,
        identifier: str,
        policy: PreTrainedPolicy,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ):
        """Checkpoint the model weights and the training state."""
        checkpoint_dir = self.checkpoints_dir / str(identifier)
        wandb_artifact_name = (
            None
            if self._wandb is None
            else f"{self._group.replace(':', '_').replace('/', '_')}-{self._cfg.seed}-{identifier}"
        )
        self.save_model(
            checkpoint_dir / self.pretrained_model_dir_name, policy, wandb_artifact_name=wandb_artifact_name
        )
        self.save_training_state(checkpoint_dir, train_step, optimizer, scheduler)

        relative_target = checkpoint_dir.relative_to(self.last_checkpoint_dir.parent)
        self.last_checkpoint_dir.symlink_to(relative_target)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        assert mode in {"train", "eval"}
        # TODO(alexander-soare): Add local text log.
        if self._wandb is not None:
            for k, v in d.items():
                if not isinstance(v, (int, float, str)):
                    logging.warning(
                        f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                    )
                    continue
                self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        assert mode in {"train", "eval"}
        assert self._wandb is not None
        wandb_video = self._wandb.Video(video_path, fps=self._cfg.env.fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)


def register_features_types():
    draccus.decode.register(FeatureType, lambda x: FeatureType[x])
    draccus.encode.register(FeatureType, lambda x: x.name)

    draccus.decode.register(NormalizationMode, lambda x: NormalizationMode[x])
    draccus.encode.register(NormalizationMode, lambda x: x.name)
