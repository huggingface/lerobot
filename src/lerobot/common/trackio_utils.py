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
import logging
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.common.tracker_utils import cfg_to_group
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


class TrackioLogger:
    """A helper class to log objects using Hugging Face trackio.

    Mirrors :class:`lerobot.common.wandb_utils.WandBLogger`. Trackio is local-first:
    without ``space_id`` runs are stored in a local SQLite DB and can be viewed with
    ``trackio show --project <project>``; with ``space_id`` the dashboard is hosted
    on a Hugging Face Space. No API key is required for local use.
    """

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.tracker
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        import trackio

        # Trackio identifies runs by name within a project, and `resume` resolves a name to the
        # most recently written run that carries it. A fresh run therefore needs a name unique to
        # this run, not `job_name` (which repeats across runs of the same policy and env, so a
        # resume could reattach to a different one). The output dir's leaf is unique, readable
        # and already carries the job name, e.g. "15-30-14_pusht_act".
        run_name = self.cfg.run_id or (Path(self.log_dir).name if self.log_dir else self.job_name)
        if cfg.resume and not self.cfg.run_id:
            logging.warning(
                "Resuming a run whose config has no trackio run name (it predates tracker support), "
                f"so its metrics continue under a new run: {run_name}."
            )
        run = trackio.init(
            project=self.cfg.project,
            name=run_name,
            group=self._group,
            space_id=self.cfg.space_id,
            dataset_id=self.cfg.dataset_id,
            private=self.cfg.private,
            config=cfg.to_dict(),
            resume="allow" if cfg.resume else "never",
        )
        # NOTE: Store the run name in cfg so that checkpoint resume reattaches to this run.
        self.cfg.run_id = getattr(run, "name", None) or run_name
        logging.info(colored("Logs will be tracked with trackio.", "blue", attrs=["bold"]))
        if self.cfg.space_id:
            logging.info(f"Dashboard hosted on Space: {colored(self.cfg.space_id, 'yellow', attrs=['bold'])}")
        else:
            logging.info(
                f"View this run locally with: "
                f"{colored(f'trackio show --project {self.cfg.project}', 'yellow', attrs=['bold'])}"
            )
        self._trackio = trackio

    def log_policy(self, checkpoint_dir: Path):
        """Uploads the policy weights as a trackio artifact (disabled by default).

        Checkpoints are multi-GB; prefer ``--save_checkpoint_to_hub`` for weights and keep
        ``disable_artifact=true`` unless the run is hosted somewhere that can absorb them.
        """
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}".replace(":", "_").replace("/", "_")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE
        model_file = adapter_model_file if adapter_model_file.exists() else standard_model_file
        if not model_file.exists():
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in "
                f"{pretrained_model_dir}. Skipping model artifact upload to trackio."
            )
            return

        self._trackio.log_artifact(model_file, name=artifact_name, type="model")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        batch_data = {}
        for k, v in d.items():
            # The custom step key (async RL) is added below as a plain metric; trackio has no
            # equivalent of wandb.define_metric so it is simply not hidden in the dashboard.
            if custom_step_key is not None and k == custom_step_key:
                continue

            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'Trackio logging of key "{k}" was ignored as its type "{type(v)}" '
                    "is not handled by this wrapper."
                )
                continue

            batch_data[f"{mode}/{k}"] = v

        if batch_data:
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
                self._trackio.log(batch_data)
            else:
                self._trackio.log(batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        video = self._trackio.Video(video_path, fps=self.env_fps, format="mp4")
        self._trackio.log({f"{mode}/video": video}, step=step)
