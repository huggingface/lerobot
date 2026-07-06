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
import builtins
import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.hub import HubMixin, find_latest_hub_checkpoint
from lerobot.utils.sample_weighting import SampleWeightingConfig

from . import parser
from .default import DatasetConfig, EvalConfig, JobConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .rewards import RewardModelConfig

TRAIN_CONFIG_NAME = "train_config.json"


def _migrate_legacy_rabc_fields(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return migrated payload for legacy RA-BC fields, or None when no migration is needed."""
    legacy_fields = (
        "use_rabc",
        "rabc_progress_path",
        "rabc_kappa",
        "rabc_epsilon",
        "rabc_head_mode",
    )
    if not any(key in config for key in legacy_fields):
        return None

    migrated_config = dict(config)
    use_rabc = bool(migrated_config.pop("use_rabc", False))
    rabc_progress_path = migrated_config.pop("rabc_progress_path", None)
    rabc_kappa = migrated_config.pop("rabc_kappa", None)
    rabc_epsilon = migrated_config.pop("rabc_epsilon", None)
    rabc_head_mode = migrated_config.pop("rabc_head_mode", None)

    # New configs may already define sample_weighting explicitly. In that case,
    # legacy fields are ignored after being stripped from the payload.
    if migrated_config.get("sample_weighting") is None and use_rabc:
        sample_weighting: dict[str, Any] = {"type": "rabc"}
        if rabc_progress_path is not None:
            sample_weighting["progress_path"] = rabc_progress_path
        if rabc_kappa is not None:
            sample_weighting["kappa"] = rabc_kappa
        if rabc_epsilon is not None:
            sample_weighting["epsilon"] = rabc_epsilon
        if rabc_head_mode is not None:
            sample_weighting["head_mode"] = rabc_head_mode
        migrated_config["sample_weighting"] = sample_weighting

    return migrated_config


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    reward_model: RewardModelConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. Pass `--config_path` pointing at either a local
    # checkpoint's train_config.json or a Hub repo id holding `checkpoints/<step>/` subtrees (the
    # latest checkpoint is downloaded and resumed from). Note that when resuming, the default behavior
    # is to use the configuration from the checkpoint, regardless of what's provided with the training
    # command at the time of resumption (CLI `--*` flags still override).
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    steps: int = 100_000
    # Run policy in the simulation environment every N steps to measure reward/success (0 = disabled).
    env_eval_freq: int = 20_000
    log_freq: int = 200
    # Compute eval loss on held-out episodes every N steps (0 = disabled). Requires eval_split > 0.
    eval_steps: int = 0
    # Cap on total eval samples, split uniformly across tasks (0 = use all held-out data).
    max_eval_samples: int = 0
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # Where to run training (local default, or an HF Jobs flavor). See JobConfig.
    job: JobConfig = field(default_factory=JobConfig)
    # Push each saved checkpoint to the Hub (policy.repo_id) as it is written, not
    # just the final model (useful to monitor progress mid-run). Optional; the
    # final model is pushed regardless. Works the same locally and remotely.
    save_checkpoint_to_hub: bool = False

    # Sample weighting configuration (e.g., for RA-BC training)
    sample_weighting: SampleWeightingConfig | None = None

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def is_reward_model_training(self) -> bool:
        """True when the config targets a reward model rather than a policy."""
        return self.reward_model is not None

    @property
    def trainable_config(self) -> PreTrainedConfig | RewardModelConfig:
        """Return whichever config (policy or reward_model) is active."""
        if self.is_reward_model_training:
            return self.reward_model  # type: ignore[return-value]
        return self.policy  # type: ignore[return-value]

    def _resolve_pretrained_from_cli(self) -> None:
        """Resolve the pretrained source passed on the CLI into a loaded config.

        The pretrained paths (`--policy.path`, `--reward_model.path`) and
        `--config_path` are only recoverable by re-reading the CLI args: draccus
        has already consumed them by the time `validate()` runs, so they are not
        reflected on `self`. Exactly one source applies, in priority order:
        reward-model path, policy path, then resume.
        """
        reward_model_path = parser.get_path_arg("reward_model")
        policy_path = parser.get_path_arg("policy")

        if reward_model_path:
            cli_overrides = parser.get_cli_overrides("reward_model")
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path, cli_overrides=cli_overrides
            )
            self.reward_model.pretrained_path = str(Path(reward_model_path))
        elif policy_path:
            overrides = parser.get_yaml_overrides("policy") + (parser.get_cli_overrides("policy") or [])
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            self._resolve_resume_checkpoint()

    def _resolve_resume_checkpoint(self) -> None:
        """Point the trainable config at the checkpoint named by `--config_path`.

        `config_path` is either a local path (to a checkpoint's train_config.json or its
        pretrained_model/ dir) or a Hub repo id. For a Hub repo, the latest checkpoint is downloaded
        into a fresh local run dir and resumed from there. The download is skipped when dispatching to
        an HF Job (`job.is_remote`): the pod performs it when it runs the resume locally, and
        `submit_to_hf` resolves the source repo for the remote command.
        """
        config_path = parser.parse_arg("config_path")
        if not config_path:
            raise ValueError(
                f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
            )

        if Path(config_path).resolve().exists():
            policy_dir = Path(config_path).parent
            self.checkpoint_path = policy_dir.parent
        elif self.job.is_remote:
            return
        else:
            from lerobot.common.train_utils import resolve_resume_checkpoint

            # `self.output_dir` was loaded from the checkpoint's config and points at the original
            # run's (now-absent) local dir. Resume into a fresh local dir instead, unless the user
            # passed --output_dir explicitly.
            cli_output_dir = parser.parse_arg("output_dir")
            if cli_output_dir:
                self.output_dir = Path(cli_output_dir)
            else:
                now = dt.datetime.now()
                self.output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_resume"
            self.checkpoint_path = resolve_resume_checkpoint(config_path, self.output_dir)
            policy_dir = self.checkpoint_path / PRETRAINED_MODEL_DIR

        if self.policy is not None:
            self.policy.pretrained_path = policy_dir
        if self.reward_model is not None:
            self.reward_model.pretrained_path = str(policy_dir)

    def validate(self) -> None:
        self._resolve_pretrained_from_cli()

        if self.policy is None and self.reward_model is None:
            raise ValueError(
                "Neither policy nor reward_model is configured. "
                "Please specify one with `--policy.path` or `--reward_model.path`."
            )

        active_cfg = self.trainable_config
        if self.rename_map and active_cfg.pretrained_path is None:
            raise ValueError(
                "`rename_map` requires a pretrained policy checkpoint. "
                "Fresh initialization derives feature names from the current dataset, so no rename is applied."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{active_cfg.type}"
            else:
                self.job_name = f"{self.env.type}_{active_cfg.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = active_cfg.get_optimizer_preset()
            self.scheduler = active_cfg.get_scheduler_preset()

        if self.eval_steps > 0 and self.dataset.eval_split == 0.0:
            raise ValueError("eval_steps > 0 requires dataset.eval_split > 0.0 to hold out eval data.")

        # Remote runs auto-generate the repo_id in submit_to_hf (the policy may only be
        # resolved here, from --policy.path), so don't demand it up front for them.
        if (
            hasattr(active_cfg, "push_to_hub")
            and active_cfg.push_to_hub
            and not active_cfg.repo_id
            and not self.job.is_remote
        ):
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

        if self.save_checkpoint_to_hub and not (self.policy is not None and self.policy.repo_id):
            raise ValueError("save_checkpoint_to_hub requires --policy.repo_id.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["policy", "reward_model"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            dl_kwargs = {
                "repo_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "token": token,
                "local_files_only": local_files_only,
            }
            try:
                config_file = hf_hub_download(filename=TRAIN_CONFIG_NAME, **dl_kwargs)
            except HfHubHTTPError as e:
                # No root train_config.json: this is a repo of periodic checkpoints from an
                # interrupted run. Fall back to the latest checkpoint's config so the run can be
                # resumed straight from the repo with `--config_path=<repo>`.
                latest = find_latest_hub_checkpoint(model_id, token=token, revision=revision)
                if latest is None:
                    raise FileNotFoundError(
                        f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                    ) from e
                config_file = hf_hub_download(
                    filename=f"{latest}/{PRETRAINED_MODEL_DIR}/{TRAIN_CONFIG_NAME}", **dl_kwargs
                )

        cli_args = kwargs.pop("cli_args", [])
        # Legacy RA-BC migration only applies to framework-saved checkpoints (always JSON).
        # Hand-written YAML/TOML configs are expected to use the current sample_weighting schema.
        if config_file is not None and config_file.endswith(".json"):
            with open(config_file) as f:
                config = json.load(f)
            migrated_config = _migrate_legacy_rabc_fields(config)
            if migrated_config is not None:
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                    json.dump(migrated_config, f)
                    config_file = f.name

        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
