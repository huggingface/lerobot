# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Prepare a reloadable FLUX3 policy and processors for training a new embodiment.

This exports the base without performing a training step.

Run from the repository root. Statistics must describe consecutive command deltas
(gripper absolute) and absolute states, in the dataset's native joint units.
"""

import argparse
import json
import math
from pathlib import Path

import draccus
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features


def compute_statistics(
    dataset_repo, metadata, settings, *, root=None, action_representation="delta", absolute_dims=(-1,)
):
    """Training episodes only; split rule matches make_train_eval_datasets."""
    by_task = {}
    allowed = settings.get("episodes") or list(range(metadata.total_episodes))
    excluded = set(settings.get("exclude_episodes") or [])
    for episode in allowed:
        if episode not in excluded:
            tasks = metadata.episodes["tasks"][episode]
            by_task.setdefault(tasks[0] if tasks else "", []).append(episode)
    train = []
    for episodes in by_task.values():
        n_eval = math.ceil(len(episodes) * settings.get("eval_split", 0))
        train.extend(episodes[: len(episodes) - n_eval])
    if not train:
        raise ValueError("No training episodes remain for normalization")
    dataset = LeRobotDataset(dataset_repo, root=root, episodes=train, download_videos=False)
    values = {"action": [], "state": []}
    previous, last_episode = None, None
    for row in dataset.hf_dataset.select_columns(["episode_index", "action", "observation.state"]):
        episode = int(row["episode_index"])
        command = np.asarray(row["action"], dtype=np.float64)
        state = np.asarray(row["observation.state"], dtype=np.float64)
        delta = command.copy()
        if action_representation == "delta":
            delta = np.zeros_like(command) if episode != last_episode else command - previous
            delta[list(absolute_dims)] = command[list(absolute_dims)]
        values["action"].append(delta)
        values["state"].append(state)
        previous, last_episode = command, episode
    result = {}
    for key, rows in values.items():
        array = np.stack(rows)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {key} values in training episodes")
        result[key] = {
            "q01": np.percentile(array, 1, axis=0).tolist(),
            "q99": np.percentile(array, 99, axis=0).tolist(),
        }
    return result, train


def export_base(
    config_path, dataset_repo, statistics_path, trunk_weights, output, *, root=None, device="cpu"
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw = json.loads(Path(config_path).read_text())
    settings = dict(raw["policy"])
    settings.pop("type", None)
    # A resolved training config can point at a previous base. This export must
    # initialize from the explicitly supplied trunk, not defer a checkpoint restore.
    settings.update(
        device=device,
        trunk_weights=trunk_weights,
        use_peft=False,
        pretrained_path=None,
        pretrained_revision=None,
    )
    metadata = LeRobotDatasetMetadata(dataset_repo, root=root)
    features = dataset_to_policy_features(metadata.features)
    settings["input_features"] = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    settings["output_features"] = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    config = Flux3Config(**settings)
    if metadata.fps != config.fps:
        raise ValueError(f"Dataset fps {metadata.fps} differs from policy fps {config.fps}")
    if statistics_path:
        supplied = json.loads(Path(statistics_path).read_text())
        config.normalization_stats = {
            stream: {q: supplied[stream][q] for q in ("q01", "q99")} for stream in ("action", "state")
        }
        normalization_source = {"source": str(Path(statistics_path).resolve())}
    else:
        config.normalization_stats, episodes = compute_statistics(
            dataset_repo,
            metadata,
            raw["dataset"],
            root=root,
            action_representation=config.action_representation,
            absolute_dims=config.delta_absolute_dims,
        )
        normalization_source = {"training_episodes": episodes}
    config.validate_features()
    torch.manual_seed(raw["seed"])
    policy = Flux3Policy(config).to(device)
    pre, post = make_flux3_pre_post_processors(config, metadata.stats)
    # The processor state is the only authoritative statistics source in new exports.
    config.normalization_stats = None
    policy.save_pretrained(output)
    (output / "normalization_source.json").write_text(json.dumps(normalization_source, indent=2) + "\n")
    pre.save_pretrained(output)
    post.save_pretrained(output)
    # A resolved training config identifies the processor contract and base path. CLI --policy.path
    # loads this base; use_peft must stay false until the trainer attaches the new adapter.
    raw["policy"] = {"type": "flux3", **draccus.encode(config)}
    raw["policy"]["pretrained_path"] = str(output.resolve())
    raw["policy"]["device"] = "cuda"
    raw["dataset"]["repo_id"] = dataset_repo
    if root is not None:
        raw["dataset"]["root"] = str(root)
    (output / "so101_train.json").write_text(json.dumps(raw, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config with explicit robot/model settings")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument(
        "--statistics", help="Existing checkpoint statistics; otherwise compute from training episodes only"
    )
    parser.add_argument("--trunk-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    export_base(
        args.config,
        args.dataset,
        args.statistics,
        args.trunk_weights,
        args.output,
        root=args.dataset_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
