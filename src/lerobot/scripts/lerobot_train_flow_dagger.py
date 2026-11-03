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

"""Train a FlowDAgger noise policy (arXiv 2607.08877) from DAgger rollouts.

FlowDAgger leaves a pi05 / smolvla policy frozen and learns a small noise policy instead. The
noise policy predicts the noise the policy's flow-matching action head starts denoising from.
Every human correction (and every clean autonomous chunk) recorded by
``lerobot-rollout --strategy.type=dagger`` is inverted through the frozen action head to the
noise that reproduces it. The noise policy is then regressed onto those targets with 50/50
intervention/autonomous batches.

Example (aggregate two DAgger rounds):

```
lerobot-train-flow-dagger \\
    --policy.path=<user>/smolvla_so101_pick \\
    --dataset.repo_ids='[<user>/dagger_round1,<user>/dagger_round2]' \\
    --output_dir=outputs/flow_dagger/round2
```

Deploy for the next DAgger round:

```
lerobot-rollout --strategy.type=dagger --policy.path=<user>/smolvla_so101_pick \\
    --policy.noise_policy_path=outputs/flow_dagger/round2/noise_policy ...
```
"""

import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch
from safetensors.torch import load_file, save_file

from lerobot import policies  # noqa: F401  (registers policy configs for draccus)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.flow_dagger import SUPPORTED_BASE_POLICY_TYPES, NoisePolicy, NoisePolicyConfig
from lerobot.policies.flow_dagger.inversion import (
    InversionResult,
    invert_dataset,
    read_selection_columns,
    select_samples,
    summarize_inversion,
)
from lerobot.policies.flow_dagger.training import NoisePolicyTrainConfig, train_noise_policy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

NOISE_POLICY_DIR = "noise_policy"
INVERSION_CACHE_DIR = "inversion_cache"


@dataclass
class FlowDAggerDatasetConfig:
    # DAgger datasets to aggregate (one per round), recorded with `lerobot-rollout --strategy.type=dagger`.
    repo_ids: list[str] = field(default_factory=list)
    # Optional local roots, one per repo_id (same order). Defaults to the Hugging Face cache.
    roots: list[str] | None = None
    revision: str | None = None
    video_backend: str | None = None


@dataclass
class FlowDAggerInversionConfig:
    # Fixed-point iterations per reverse Euler step (M in the paper).
    num_fixed_point_iters: int = 5
    batch_size: int = 8
    num_workers: int = 4
    # Minimum leading chunk steps inside the frame's segment. Defaults to the policy's n_action_steps.
    min_valid_steps: int | None = None
    # Autonomous frames within this many seconds before an intervention are not used.
    exclude_pre_intervention_s: float = 2.0
    # Keep every n-th eligible autonomous frame.
    autonomous_stride: int = 1
    # Set False to train on interventions only.
    use_autonomous: bool = True
    # Re-decode every inverted noise and log the action reconstruction error.
    check_reconstruction: bool = True
    # Reuse inverted samples cached under <output_dir>/inversion_cache (or `cache_dir`).
    use_cache: bool = True
    cache_dir: str | None = None


@dataclass
class FlowDAggerNoisePolicyArchConfig:
    hidden_dim: int = 1024
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class FlowDAggerTrainPipelineConfig:
    dataset: FlowDAggerDatasetConfig = field(default_factory=FlowDAggerDatasetConfig)
    # Frozen base policy, given as --policy.path=<hub repo or local dir>. Only pi05 and smolvla.
    policy: PreTrainedConfig | None = None
    inversion: FlowDAggerInversionConfig = field(default_factory=FlowDAggerInversionConfig)
    train: NoisePolicyTrainConfig = field(default_factory=NoisePolicyTrainConfig)
    noise_policy: FlowDAggerNoisePolicyArchConfig = field(default_factory=FlowDAggerNoisePolicyArchConfig)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    rename_map: dict[str, str] = field(default_factory=dict)
    # Push the trained noise policy to the Hub as a model repo.
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if not self.job_name:
            policy_type = self.policy.type if self.policy is not None else "policy"
            self.job_name = f"flow_dagger_{policy_type}"
        if self.output_dir is None:
            now = dt.datetime.now()
            self.output_dir = Path("outputs/flow_dagger") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"

    def validate(self) -> None:
        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("FlowDAgger adapts a pretrained policy: pass --policy.path=<repo or dir>.")
        if self.policy.type not in SUPPORTED_BASE_POLICY_TYPES:
            raise ValueError(
                f"FlowDAgger supports {SUPPORTED_BASE_POLICY_TYPES} base policies, got {self.policy.type!r}."
            )
        if not self.dataset.repo_ids:
            raise ValueError("Pass at least one DAgger dataset with --dataset.repo_ids='[...]'.")
        if self.dataset.roots is not None and len(self.dataset.roots) != len(self.dataset.repo_ids):
            raise ValueError("--dataset.roots must have one entry per --dataset.repo_ids entry.")
        if self.push_to_hub and not self.repo_id:
            raise ValueError("--push_to_hub=true requires --repo_id.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Lets the parser load the policy config with `--policy.path=...`."""
        return ["policy"]


def _num_inference_steps(policy_cfg: PreTrainedConfig) -> int:
    return int(policy_cfg.num_inference_steps if policy_cfg.type == "pi05" else policy_cfg.num_steps)


def _cache_key(cfg: FlowDAggerTrainPipelineConfig, dataset: LeRobotDataset, min_valid_steps: int) -> str:
    assert cfg.policy is not None
    payload = {
        "repo_id": dataset.repo_id,
        "revision": cfg.dataset.revision,
        "root": str(dataset.root),
        "total_frames": dataset.meta.total_frames,
        "total_episodes": dataset.meta.total_episodes,
        "policy": str(cfg.policy.pretrained_path),
        "policy_revision": cfg.policy.pretrained_revision,
        "num_inference_steps": _num_inference_steps(cfg.policy),
        "chunk_size": cfg.policy.chunk_size,
        "num_fixed_point_iters": cfg.inversion.num_fixed_point_iters,
        "min_valid_steps": min_valid_steps,
        "exclude_pre_intervention_s": cfg.inversion.exclude_pre_intervention_s,
        "autonomous_stride": cfg.inversion.autonomous_stride,
        "use_autonomous": cfg.inversion.use_autonomous,
        "rename_map": cfg.rename_map,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()
    return f"{dataset.repo_id.replace('/', '__')}-{digest[:12]}"


@parser.wrap()
def train_flow_dagger(cfg: FlowDAggerTrainPipelineConfig) -> Path:
    cfg.validate()
    assert cfg.policy is not None and cfg.output_dir is not None
    logger.info(pformat(asdict(cfg)))
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.inversion.cache_dir) if cfg.inversion.cache_dir else output_dir / INVERSION_CACHE_DIR
    device = get_safe_torch_device(str(cfg.policy.device), log=True)

    roots: list[str | None] = (
        list(cfg.dataset.roots) if cfg.dataset.roots else [None] * len(cfg.dataset.repo_ids)
    )
    first_meta = LeRobotDatasetMetadata(cfg.dataset.repo_ids[0], root=roots[0], revision=cfg.dataset.revision)

    # The base policy is frozen: it is only ever run under no_grad for inversion.
    logger.info("Loading frozen base policy from %s", cfg.policy.pretrained_path)
    cfg.policy.noise_policy_path = None  # invert the base policy itself, never an adapted one
    policy = make_policy(cfg=cfg.policy, ds_meta=first_meta, rename_map=cfg.rename_map)
    policy.eval().requires_grad_(False)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    min_valid_steps = cfg.inversion.min_valid_steps or cfg.policy.n_action_steps
    results: list[InversionResult] = []
    for repo_id, root in zip(cfg.dataset.repo_ids, roots, strict=True):
        meta = LeRobotDatasetMetadata(repo_id, root=root, revision=cfg.dataset.revision)
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            revision=cfg.dataset.revision,
            delta_timestamps=resolve_delta_timestamps(cfg.policy, meta, cfg.rename_map),
            video_backend=cfg.dataset.video_backend,
        )
        cache_file = cache_dir / f"{_cache_key(cfg, dataset, min_valid_steps)}.safetensors"
        if cfg.inversion.use_cache and cache_file.exists():
            logger.info("Loading cached inversion for %s from %s", repo_id, cache_file)
            results.append(InversionResult.from_dict(load_file(str(cache_file))))
            continue

        episode_index, intervention = read_selection_columns(dataset)
        selection = select_samples(
            episode_index,
            intervention,
            chunk_size=cfg.policy.chunk_size,
            min_valid_steps=min_valid_steps,
            exclude_pre_intervention_frames=int(round(cfg.inversion.exclude_pre_intervention_s * meta.fps)),
            autonomous_stride=cfg.inversion.autonomous_stride,
            use_autonomous=cfg.inversion.use_autonomous,
        )
        logger.info(
            "%s: %d frames, %d intervention / %d autonomous samples selected",
            repo_id,
            len(episode_index),
            len(selection.intervention_positions),
            len(selection.autonomous_positions),
        )
        result = invert_dataset(
            policy,
            preprocessor,
            dataset,
            selection,
            device=device,
            batch_size=cfg.inversion.batch_size,
            num_workers=cfg.inversion.num_workers,
            num_fixed_point_iters=cfg.inversion.num_fixed_point_iters,
            check_reconstruction=cfg.inversion.check_reconstruction,
            rename_map=cfg.rename_map,
            camera_keys=meta.camera_keys,
        )
        logger.info("%s inversion: %s", repo_id, pformat(summarize_inversion(result)))
        if cfg.inversion.use_cache and len(result) > 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            save_file(result.to_dict(), str(cache_file))
        results.append(result)

    data = InversionResult.concatenate(results)
    if len(data) == 0:
        raise ValueError("No usable samples in the given datasets (check the intervention flags).")
    inversion_stats = summarize_inversion(data)
    logger.info("Aggregated inversion: %s", pformat(inversion_stats))

    # The VLM is no longer needed: the noise policy trains on the cached features alone.
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    noise_policy = NoisePolicy(
        NoisePolicyConfig(
            feature_dim=data.features.shape[1],
            chunk_size=cfg.policy.chunk_size,
            max_action_dim=cfg.policy.max_action_dim,
            base_policy_type=cfg.policy.type,
            num_inference_steps=_num_inference_steps(cfg.policy),
            hidden_dim=cfg.noise_policy.hidden_dim,
            num_layers=cfg.noise_policy.num_layers,
            dropout=cfg.noise_policy.dropout,
        )
    )
    metrics = train_noise_policy(noise_policy, data, cfg.train, device, seed=cfg.seed)

    save_dir = output_dir / NOISE_POLICY_DIR
    noise_policy.save_pretrained(save_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps({"inversion": inversion_stats, "train": metrics}, indent=2)
    )
    logger.info("Saved noise policy to %s", save_dir)

    if cfg.push_to_hub:
        from huggingface_hub import HfApi

        assert cfg.repo_id is not None
        api = HfApi()
        api.create_repo(cfg.repo_id, repo_type="model", private=cfg.private, exist_ok=True)
        api.upload_folder(repo_id=cfg.repo_id, folder_path=str(save_dir), repo_type="model")
        logger.info("Pushed noise policy to https://huggingface.co/%s", cfg.repo_id)

    logger.info(
        "Deploy with: --policy.path=%s --policy.noise_policy_path=%s",
        cfg.policy.pretrained_path,
        cfg.repo_id if cfg.push_to_hub else save_dir,
    )
    return save_dir


def main() -> None:
    init_logging()
    register_third_party_plugins()
    train_flow_dagger()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
