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

"""Evaluate a trained reward model against per-frame MC returns.

The output CSV is also accepted by ``lerobot-annotate
--advantage.predictions_path=...``. This keeps model inference faithful to the
normal LeRobot dataset and processor path, including multi-camera observations,
state normalization, and temporal delta windows.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.rewards import RewardModelConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from lerobot.rewards import make_reward_model
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.hub import find_latest_hub_checkpoint

logger = logging.getLogger(__name__)


def _resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        if (path / "pretrained_model").is_dir():
            return path / "pretrained_model"
        return path

    latest = find_latest_hub_checkpoint(path_or_repo)
    if latest is None:
        snapshot = Path(snapshot_download(repo_id=path_or_repo, repo_type="model"))
        return snapshot
    snapshot = Path(
        snapshot_download(
            repo_id=path_or_repo,
            repo_type="model",
            allow_patterns=f"{latest}/pretrained_model/*",
        )
    )
    return snapshot / latest / "pretrained_model"


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _correlation(
        pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy()
    )


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    num_positive = int(labels.sum())
    num_negative = len(labels) - num_positive
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_positive = float(ranks[labels].sum())
    return (rank_sum_positive - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)


def _held_out_episodes(metadata: LeRobotDatasetMetadata, eval_split: float) -> list[int] | None:
    if eval_split == 0:
        return None
    if not 0 < eval_split < 1:
        raise ValueError(f"eval_split must be in [0,1), got {eval_split}")
    task_to_episodes: dict[str, list[int]] = {}
    episode_tasks = metadata.episodes["tasks"]
    for episode_index in range(metadata.total_episodes):
        task = episode_tasks[episode_index][0] if episode_tasks[episode_index] else ""
        task_to_episodes.setdefault(task, []).append(episode_index)
    held_out: list[int] = []
    for episodes in task_to_episodes.values():
        count = math.ceil(len(episodes) * eval_split)
        held_out.extend(episodes[-count:])
    return held_out


def _compute_advantages(
    target: np.ndarray,
    prediction: np.ndarray,
    episode_index: np.ndarray,
    n_step: int | None,
) -> np.ndarray:
    if n_step is None:
        return target - prediction
    advantage = np.empty_like(target)
    for episode in np.unique(episode_index):
        indices = np.flatnonzero(episode_index == episode)
        for local_index, index in enumerate(indices):
            bootstrap_index = local_index + n_step
            if bootstrap_index >= len(indices):
                advantage[index] = target[index] - prediction[index]
            else:
                future = indices[bootstrap_index]
                advantage[index] = target[index] - target[future] + prediction[future] - prediction[index]
    return advantage


def _target_distribution(model, target: torch.Tensor, is_terminal: torch.Tensor) -> torch.Tensor:
    try:
        return model.compute_target_distribution(
            target,
            is_terminal,
            method=model.config.target_method,
            use_one_hot_terminal=model.config.use_one_hot_terminal,
        )
    except TypeError:
        return model.compute_target_distribution(target, is_terminal)


def _predict_logits_and_value(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "_vlm_forward"):
        logits, predicted = model._vlm_forward(batch)
        return logits, predicted.reshape(-1)
    if hasattr(model, "_get_value_readout"):
        logits = model.value_head(model._get_value_readout(batch))
        probabilities = logits.softmax(-1)
        centers = model.value_head.bin_centers.to(probabilities.dtype)
        return logits, (probabilities * centers).sum(-1)
    raise TypeError(f"{type(model).__name__} does not expose a distributional value readout")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model_path = _resolve_model_path(args.reward_model_path)
    logger.info("Loading reward model from %s", model_path)

    config = RewardModelConfig.from_pretrained(model_path)
    config.pretrained_path = str(model_path)
    config.device = device.type

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    episodes = _held_out_episodes(metadata, args.eval_split)
    if episodes is not None:
        logger.info("Evaluating %d held-out episode(s) (eval_split=%s)", len(episodes), args.eval_split)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = make_reward_model(config, dataset_meta=metadata).eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device.type}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    nll_sum = 0.0
    target_entropy_sum = 0.0
    count = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating reward model"):
            if args.max_samples > 0 and count >= args.max_samples:
                break
            for camera_key in metadata.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float().div_(255)

            raw_target = batch["mc_return"].reshape(-1)
            raw_terminal = batch["is_terminal"].reshape(-1).bool()
            raw_episode = batch["episode_index"].reshape(-1)
            raw_frame = batch["frame_index"].reshape(-1)
            if args.max_samples > 0:
                keep = min(len(raw_target), args.max_samples - count)
                if keep < len(raw_target):
                    batch = {
                        key: value[:keep]
                        if isinstance(value, torch.Tensor)
                        else value[:keep]
                        if isinstance(value, list)
                        else value
                        for key, value in batch.items()
                    }
                    raw_target = raw_target[:keep]
                    raw_terminal = raw_terminal[:keep]
                    raw_episode = raw_episode[:keep]
                    raw_frame = raw_frame[:keep]

            if ACTION in batch and isinstance(batch[ACTION], torch.Tensor):
                actions.append(batch[ACTION].reshape(len(raw_target), -1).float().cpu().numpy())
            if OBS_STATE in batch and isinstance(batch[OBS_STATE], torch.Tensor):
                state = batch[OBS_STATE]
                if state.ndim >= 3:
                    state = state[:, -1]
                states.append(state.reshape(len(raw_target), -1).float().cpu().numpy())
            if "intervention" in batch and isinstance(batch["intervention"], torch.Tensor):
                interventions.append(batch["intervention"].reshape(-1).bool().cpu().numpy())

            processed = preprocessor(batch)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, predicted = _predict_logits_and_value(model, processed)

            target_on_device = processed["mc_return"].reshape(-1)
            terminal_on_device = processed["is_terminal"].reshape(-1).bool()
            target_dist = _target_distribution(model, target_on_device, terminal_on_device)
            per_sample_nll = -(target_dist * logits.log_softmax(-1)).sum(-1)
            entropy = -(target_dist * target_dist.clamp_min(1e-12).log()).sum(-1)

            batch_count = len(predicted)
            nll_sum += float(per_sample_nll.sum())
            target_entropy_sum += float(entropy.sum())
            count += batch_count
            predictions.append(predicted.float().cpu().numpy())
            targets.append(raw_target.float().cpu().numpy())
            terminals.append(raw_terminal.cpu().numpy())
            episode_indices.append(raw_episode.cpu().numpy())
            frame_indices.append(raw_frame.cpu().numpy())

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    terminal = np.concatenate(terminals)
    episode_index = np.concatenate(episode_indices).astype(np.int64)
    frame_index = np.concatenate(frame_indices).astype(np.int64)
    residual = prediction - target
    advantage = _compute_advantages(target, prediction, episode_index, args.n_step)
    threshold = float(np.percentile(advantage, args.threshold_percentile * 100))
    advantage_label = np.where(advantage > threshold, "positive", "negative")

    terminal_success = np.isclose(target[terminal], 0.0, atol=1e-6)
    metrics = {
        "samples": len(prediction),
        "nll": nll_sum / count,
        "target_entropy": target_entropy_sum / count,
        "excess_nll": (nll_sum - target_entropy_sum) / count,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "pearson": _correlation(prediction, target),
        "spearman": _spearman(prediction, target),
        "terminal_success_auc": _binary_auc(prediction[terminal], terminal_success),
        "advantage_threshold": threshold,
        "positive_fraction": float(np.mean(advantage_label == "positive")),
    }
    for name, value in metrics.items():
        logger.info("%s: %s", name, f"{value:.6f}" if isinstance(value, float) else value)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "mc_return": target,
        "predicted_value": prediction,
        "residual": residual,
        "advantage": advantage,
        "advantage_label": advantage_label,
        "is_terminal": terminal,
    }
    if actions and states:
        action = np.concatenate(actions)
        state = np.concatenate(states)
        common_dim = min(action.shape[1], state.shape[1])
        output["command_delta_norm"] = np.linalg.norm(
            action[:, :common_dim] - state[:, :common_dim],
            axis=1,
        )
        state_motion = np.zeros(len(state), dtype=np.float32)
        same_episode_indices = np.flatnonzero(episode_index[1:] == episode_index[:-1]) + 1
        state_motion[same_episode_indices] = np.linalg.norm(
            state[same_episode_indices] - state[same_episode_indices - 1],
            axis=1,
        )
        output["state_motion_norm"] = state_motion
    if interventions and sum(map(len, interventions)) == len(prediction):
        output["intervention"] = np.concatenate(interventions)
    pd.DataFrame(output).to_csv(output_path, index=False)
    logger.info("Wrote per-frame predictions to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-model-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument("--n-step", type=int)
    parser.add_argument("--threshold-percentile", type=float, default=0.6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args)


if __name__ == "__main__":
    main()
