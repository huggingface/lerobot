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
# TODO(rcadene, alexander-soare): clean this file
"""Borrowed from https://github.com/fyhMer/fowm/blob/main/src/logger.py"""

import logging
import os
from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import OmegaConf
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_global_random_state, set_global_random_state


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def cfg_to_group(cfg, return_list=False):
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.name}",
        f"dataset:{cfg.dataset_repo_id}",
        f"env:{cfg.env.name}",
        f"seed:{cfg.seed}",
    ]
    return lst if return_list else "-".join(lst)


class Logger:
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, job_name, cfg):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._job_name = job_name
        self._checkpoint_dir = self._log_dir / "checkpoints"
        self._last_checkpoint_path = self._checkpoint_dir / "last"
        self._buffer_dir = self._log_dir / "buffers"
        self._save_model = cfg.training.save_model
        self._disable_wandb_artifact = cfg.wandb.disable_artifact
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._cfg = cfg
        self._eval = []
        project = cfg.get("wandb", {}).get("project")
        entity = cfg.get("wandb", {}).get("entity")
        enable_wandb = cfg.get("wandb", {}).get("enable", False)
        run_offline = not enable_wandb or not project
        if run_offline:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            os.environ["WANDB_SILENT"] = "true"
            import wandb

            wandb.init(
                project=project,
                entity=entity,
                name=job_name,
                notes=cfg.get("wandb", {}).get("notes"),
                # group=self._group,
                tags=cfg_to_group(cfg, return_list=True),
                dir=self._log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                # TODO(rcadene): add resume option
                resume="must",
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
            self._wandb = wandb

    @property
    def last_checkpoint_path(self):
        return self._last_checkpoint_path

    def save_model(self, policy: Policy, identifier: str):
        if self._save_model:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_dir = self._checkpoint_dir / str(identifier)
            policy.save_pretrained(save_dir)
            # Also save the full Hydra config for the env configuration.
            OmegaConf.save(self._cfg, save_dir / "config.yaml")
            if self._wandb and not self._disable_wandb_artifact:
                # note wandb artifact does not accept ":" or "/" in its name
                artifact = self._wandb.Artifact(
                    f"{self._group.replace(':', '_').replace('/', '_')}-{self._seed}-{identifier}",
                    type="model",
                )
                artifact.add_file(save_dir / SAFETENSORS_SINGLE_FILE)
                self._wandb.log_artifact(artifact)
        os.symlink(save_dir.absolute(), self._last_checkpoint_path)  # TODO(now): Check this works

    def save_training_state(
        self, train_step: int, optimizer: Optimizer, scheduler: LRScheduler | None, identifier: str
    ):
        training_state = {
            "step": train_step,
            "optimizer": optimizer.state_dict(),
            **get_global_random_state(),
        }
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()
        torch.save(training_state, self._checkpoint_dir / str(identifier) / "training_state.pth")

    def load_last_training_state(self, optimizer: Optimizer, scheduler: LRScheduler | None) -> int:
        """
        Load the optimizer and scheduler state_dict from the last checkpoint, set the random state, and return
        the global training step.
        """
        training_state = torch.load(self._checkpoint_dir / "last" / "training_state.pth")
        optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])
        elif "scheduler" in training_state:
            raise ValueError(
                "The checkpoint contains a scheduler state_dict, but no LRScheduler was provided."
            )
        # Small hack to get the expected keys: use `get_global_random_state`.
        set_global_random_state({k: training_state[k] for k in get_global_random_state()})
        return training_state["step"]

    def save_checkpont(
        self,
        train_step: int,
        policy: Policy,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        identifier: str,
    ):
        self.save_model(policy, identifier)
        self.save_training_state(train_step, optimizer, scheduler, identifier)

    def log_dict(self, d, step, mode="train"):
        assert mode in {"train", "eval"}
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
        wandb_video = self._wandb.Video(video_path, fps=self._cfg.fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
