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
import logging
import re
from glob import glob
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_swanlab_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the SwanLab run ID.
    paths = glob(str(log_dir / "swanlab/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous SwanLab run ID for run resumption.")
    match = re.search(r"run-([^\.]+).swanlab", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous SwanLab run ID for run resumption.")
    swanlab_run_id = match.groups(0)[0]
    return swanlab_run_id


def get_safe_swanlab_artifact_name(name: str):
    """SwanLab artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class SwanLabLogger:
    """A helper class to log object using swanlab."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.swanlab
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        import swanlab

        swanlab_run_id = (
            cfg.swanlab.run_id
            if cfg.swanlab.run_id
            else get_swanlab_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        self._run = swanlab.init(
            project=self.cfg.project,
            experiment_name=swanlab_run_id,
            description=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            logdir=str(self.log_dir),
            config=cfg.to_dict(),
            save_code=False,
            resume=cfg.resume,
            mode=self.cfg.mode if self.cfg.mode in ["cloud", "offline", "local", "disabled"] else "cloud",
        )
        run_id = self._run.public.run_id
        # NOTE: We will override the cfg.swanlab.run_id with the swanlab run id.
        # This is because we want to be able to resume the run from the swanlab run id.
        cfg.swanlab.run_id = run_id
        # Handle custom step key for rl asynchronous training.
        self._swanlab_custom_step_key: set[str] | None = None
        print(colored("Logs will be synced with swanlab.", "blue", attrs=["bold"]))
        logging.info(
            f"Track this run --> {colored(self._run.public.cloud.experiment_url, 'yellow', attrs=['bold'])}"
        )
        self._swanlab = swanlab

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to swanlab."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_swanlab_artifact_name(artifact_name)
        # SwanLab doesn't have direct artifact logging like wandb
        # We'll log the model file path as a text log for now
        model_path = str(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)
        self._swanlab.log({f"model_checkpoint/{step_id}": model_path})

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # NOTE: This is not simple. SwanLab step must always monotonically increase and it
        # increases with each swanlab.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._swanlab_custom_step_key is None:
                self._swanlab_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._swanlab_custom_step_key:
                self._swanlab_custom_step_key.add(new_custom_key)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'SwanLab logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            # Do not log the custom step key itself.
            if self._swanlab_custom_step_key is not None and k in self._swanlab_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d.get(custom_step_key)
                if value_custom_step is None:
                    logging.warning(
                        f'Custom step key "{custom_step_key}" not found in the dictionary. Skipping logging for this key.'
                    )
                    continue
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._swanlab.log(data)
                continue

            self._swanlab.log(data={f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        # SwanLab media logging - using Media.Video for video logging
        swanlab_video = self._swanlab.Video(video_path, fps=self.env_fps)
        self._swanlab.log({f"{mode}/video": swanlab_video}, step=step)
