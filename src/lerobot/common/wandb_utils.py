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
import os
import re
from glob import glob
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required.

        wandb rejects tags longer than 64 characters.
        See: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
    lst = [
        trainable_tag,
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
        # This is because we want to be able to resume the run from the wandb run id.
        cfg.wandb.run_id = run_id
        # Handle custom step key for rl asynchronous training.
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # Check if this is a PEFT model (has adapter files instead of model.safetensors)
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: add adapter files and configs
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # Also add the policy config which is needed for loading
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # Standard model: add the single safetensors file
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # NOTE: This is not simple. Wandb step must always monotonically increase and it
        # increases with each wandb.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        batch_data = {}
        for k, v in d.items():
            # Skip the custom step key here, it's added to the batch below.
            if custom_step_key is not None and k == custom_step_key:
                continue

            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            batch_data[f"{mode}/{k}"] = v

        if batch_data:
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
                self._wandb.log(batch_data)
            else:
                self._wandb.log(data=batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)

    def log_training_examples(
        self,
        batch: dict,
        step: int,
        *,
        camera_keys: list[str],
        n_samples: int = 4,
        policy=None,
        predict_actions: bool = False,
        mode: str = "train",
    ) -> None:
        """Push a ``wandb.Table`` of training-example rows for the current batch.

        Each row is one batch element with:
          * one ``wandb.Image`` column per camera in ``camera_keys`` (CHW or
            HWC, uint8 or float in [0,1] — auto-detected),
          * any text fields present in the batch (``task`` / ``subtask`` /
            ``memory`` / ``instruction``),
          * ground-truth action first/last frame (the action chunk's
            endpoints — gives a quick sense of trajectory direction),
          * if ``predict_actions=True`` and ``policy`` is supplied, the model's
            ``predict_action_chunk`` first/last frame alongside.

        This is opt-in via ``--wandb.log_examples_freq=N`` on the CLI; the
        training loop calls it once every N steps. Cheap to keep on: with
        N=4 samples and 3 cameras you upload 12 small image files per dump and (if
        enabled) run one extra inference forward pass.
        """
        import logging  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        # Batch size — first tensor-like value wins.
        bsz = next(
            (int(v.shape[0]) for v in batch.values() if hasattr(v, "shape") and v.ndim > 0),
            None,
        )
        if not bsz:
            return
        n = min(int(n_samples), bsz)

        # Optional predicted-action forward pass on the first n samples.
        pred_actions: np.ndarray | None = None
        if predict_actions and policy is not None:
            was_training = policy.training
            try:
                policy.eval()
                sub_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        sub_batch[k] = v[:n]
                    elif isinstance(v, (list, tuple)):
                        sub_batch[k] = list(v[:n])
                    else:
                        sub_batch[k] = v
                with torch.no_grad():
                    pred = policy.predict_action_chunk(sub_batch)
                pred_actions = pred.detach().cpu().float().numpy()
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "log_training_examples: predict_action_chunk failed (%s) — "
                    "skipping predicted-action columns",
                    exc,
                )
                pred_actions = None
            finally:
                if was_training:
                    policy.train()

        present_cameras = [c for c in camera_keys if c in batch]
        text_keys = [k for k in ("task", "subtask", "memory", "instruction") if k in batch]

        columns = ["sample"]
        columns.extend(c.removeprefix("observation.images.") or c for c in present_cameras)
        columns.extend(text_keys)
        columns.append("gt_action_first")
        columns.append("gt_action_last")
        if pred_actions is not None:
            columns.append("pred_action_first")
            columns.append("pred_action_last")

        table = self._wandb.Table(columns=columns)

        def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
            # Strip an outer time dim if present: (T, C, H, W) -> first frame.
            if t.ndim == 4:
                t = t[0]
            # CHW -> HWC.
            if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
                t = t.permute(1, 2, 0)
            arr = t.detach().cpu().float().numpy()
            if arr.size and float(arr.max()) <= 1.5:
                arr = arr * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)

        def _action_endpoints(a: torch.Tensor) -> tuple[str, str]:
            arr = a.detach().cpu().float().numpy()
            if arr.ndim == 2:  # (T, D)
                return (
                    str(np.round(arr[0], 3).tolist()),
                    str(np.round(arr[-1], 3).tolist()),
                )
            if arr.ndim == 1:
                rounded = np.round(arr, 3).tolist()
                return (str(rounded), str(rounded))
            return (str(arr.tolist()), str(arr.tolist()))

        for i in range(n):
            row: list = [i]
            for cam in present_cameras:
                try:
                    row.append(self._wandb.Image(_to_uint8_hwc(batch[cam][i])))
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "log_training_examples: camera %s sample %d failed (%s)",
                        cam,
                        i,
                        exc,
                    )
                    row.append(None)
            for tk in text_keys:
                v = batch[tk]
                if isinstance(v, (list, tuple)):
                    row.append(str(v[i]) if i < len(v) else "")
                else:
                    row.append(str(v))
            action = batch.get("action")
            if isinstance(action, torch.Tensor) and action.ndim >= 1:
                first, last = _action_endpoints(action[i])
                row.append(first)
                row.append(last)
            else:
                row.append("")
                row.append("")
            if pred_actions is not None:
                p = torch.from_numpy(pred_actions[i])
                pfirst, plast = _action_endpoints(p)
                row.append(pfirst)
                row.append(plast)
            table.add_data(*row)

        self._wandb.log({f"{mode}/examples": table}, step=step)
