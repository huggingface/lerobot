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
"""Experiment tracking with Trackio (https://huggingface.co/docs/trackio).

Runs live in a local SQLite database under ``~/.cache/huggingface/trackio`` and are viewed with
``trackio show --project <project>``. Setting ``--trackio.space_id`` syncs the run to a Hugging Face
Space instead, which is the only option that survives a remote job.
"""

import logging
import uuid
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import require_package

TRACKIO_MODES = frozenset({"train", "eval"})


def cfg_to_group(cfg: TrainPipelineConfig) -> str:
    """Return the Trackio group for a run, built from what the run trains on."""
    trainable_kind = "reward_model" if cfg.is_reward_model_training else "policy"
    parts = [f"{trainable_kind}:{cfg.trainable_config.type}", f"seed:{cfg.seed}"]
    if cfg.dataset is not None:
        parts.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        parts.append(f"env:{cfg.env.type}")
    return "-".join(parts)


def get_safe_artifact_name(name: str) -> str:
    """Trackio addresses artifacts as ``project/name:alias``, so neither separator may appear in the name."""
    return name.replace(":", "_").replace("/", "_")


class TrackioLogger:
    """Logs training metrics, evaluation videos and checkpoints to Trackio."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        require_package("trackio", "training")
        import trackio

        self.cfg = cfg.trackio
        self._group = cfg_to_group(cfg)

        # Trackio resumes by run name. The name is generated once, persisted with the checkpoint's
        # train_config.json, and reused when the checkpoint is resumed. "allow" reattaches to that run when
        # the target store has it and otherwise starts a new run under the same name (another machine, a
        # fresh Space, a wiped cache), where "must" would crash.
        generated = self.cfg.run_name is None
        if generated:
            if cfg.resume:
                logging.warning(
                    "The resumed checkpoint has no Trackio run name, starting a new run. Pass "
                    "`--trackio.run_name=<previous run>` to reattach to an existing one."
                )
            self.cfg.run_name = f"{cfg.job_name}-{uuid.uuid4().hex[:8]}"
        resume = self.cfg.resume or ("allow" if cfg.resume and not generated else "never")
        self._run = trackio.init(
            project=self.cfg.project,
            name=self.cfg.run_name,
            group=self._group,
            space_id=self.cfg.space_id,
            private=self.cfg.private,
            config=cfg.to_dict(),
            resume=resume,
            capture_console=self.cfg.capture_console,
        )
        self._trackio = trackio

        # `run.url` is the resolved Space id (`user/space`, also for a bare name or TRACKIO_SPACE_ID),
        # a self-hosted server URL, or None for the local database.
        remote = self._run.url
        logging.info(colored("Logs will be synced with Trackio.", "blue", attrs=["bold"]))
        if remote and remote.startswith(("http://", "https://")):
            where = remote
        elif remote:
            self.cfg.space_id = remote
            where = f"https://huggingface.co/spaces/{remote}"
        else:
            where = f'trackio show --project "{self.cfg.project}"'
        logging.info(f"Track this run --> {colored(where, 'yellow', attrs=['bold'])}")

    def log_policy(self, checkpoint_dir: Path) -> None:
        """Uploads the policy weights of a checkpoint as a Trackio ``model`` artifact.

        The upload runs on the calling thread (a Space receives the file before this returns) and is
        best-effort: a failure is logged, never raised, so a flaky Space cannot kill a training run.
        """
        if self.cfg.disable_artifact:
            return
        try:
            self._upload_policy_artifact(checkpoint_dir)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Skipping Trackio model artifact upload for {checkpoint_dir}: {exc}")

    def _upload_policy_artifact(self, checkpoint_dir: Path) -> None:
        step_id = checkpoint_dir.name
        artifact_name = get_safe_artifact_name(f"{self._group}-{step_id}")
        artifact = self._trackio.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # A PEFT checkpoint stores adapter files instead of model.safetensors.
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # The policy config is needed to load the adapter back.
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to Trackio."
            )
            return

        self._trackio.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None:
        """Logs scalar metrics under ``<mode>/<key>``.

        With ``step`` the metrics are placed on the run's step axis. With ``custom_step_key`` (asynchronous
        RL, where several clocks coexist) the metrics are logged on Trackio's auto-incremented step and the
        custom step is kept as a plain metric, to be selected as the x-axis in the dashboard.
        """
        if mode not in TRACKIO_MODES:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        if custom_step_key is not None and custom_step_key not in d:
            raise ValueError(f"custom_step_key {custom_step_key!r} is missing from the logged metrics.")

        batch_data = {}
        for k, v in d.items():
            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'Trackio logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue
            batch_data[f"{mode}/{k}"] = v

        if not batch_data:
            return
        if custom_step_key is not None:
            self._trackio.log(batch_data)
        else:
            self._trackio.log(batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Logs a video file under ``<mode>/video``."""
        if mode not in TRACKIO_MODES:
            raise ValueError(mode)
        video = self._trackio.Video(video_path, caption=f"{mode} step {step}")
        self._trackio.log({f"{mode}/video": video}, step=step)

    def finish(self) -> None:
        """Flushes pending logs and closes the run."""
        self._trackio.finish()
