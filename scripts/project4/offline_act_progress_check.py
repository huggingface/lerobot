#!/usr/bin/env python
"""Offline sanity checks for a progress-conditioned ACT checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 10, 19])
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/project4_offline_check"))
    return parser.parse_args()


def resolve_device(requested: str | None, policy_device: str | None) -> str:
    if requested and is_torch_device_available(requested):
        return requested
    if policy_device and is_torch_device_available(policy_device):
        return policy_device
    return auto_select_torch_device().type


def load_policy(policy_path: str, device: str):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def select_action(policy, preprocessor, postprocessor, observation: dict, device: str, task: str) -> np.ndarray:
    obs = {key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()}
    obs = prepare_observation_for_inference(obs, torch.device(device), task=task)
    with torch.inference_mode():
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def episode_rows(dataset: LeRobotDataset, episode: int) -> list[int]:
    meta = dataset.meta.episodes[episode]
    return list(range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"])))


def plot_episode(output_dir: Path, episode: int, recorded: np.ndarray, predicted: np.ndarray, names: list[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output_dir / f"episode_{episode:03d}_pred_vs_recorded_actions.png"
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.9 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(recorded))
    for dim, name in enumerate(names):
        axes[dim].plot(x, recorded[:, dim], label="recorded", linewidth=1.2)
        axes[dim].plot(x, predicted[:, dim], label="predicted", linewidth=1.1, alpha=0.85)
        axes[dim].set_ylabel(name)
    axes[-1].set_xlabel("frame")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def reset_inference(policy, preprocessor, postprocessor) -> None:
    for obj in (policy, preprocessor, postprocessor):
        reset = getattr(obj, "reset", None)
        if callable(reset):
            reset()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    if OBS_ENV_STATE not in dataset.meta.features:
        raise ValueError(f"{args.dataset_repo_id} does not contain {OBS_ENV_STATE}. Run the augmentation first.")

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    device = resolve_device(args.device, policy_cfg.device)
    policy, preprocessor, postprocessor = load_policy(args.policy_path, device)

    hf_dataset = dataset.hf_dataset.with_format(None)
    action_names = dataset.meta.features[ACTION]["names"]
    report = {
        "policy_path": args.policy_path,
        "dataset_repo_id": args.dataset_repo_id,
        "device": device,
        "episodes": {},
    }

    for episode in args.episodes:
        reset_inference(policy, preprocessor, postprocessor)
        indices = episode_rows(dataset, episode)
        recorded = []
        predicted = []
        for idx in indices:
            row = hf_dataset[int(idx)]
            observation = {
                OBS_STATE: row[OBS_STATE],
                OBS_ENV_STATE: row[OBS_ENV_STATE],
            }
            predicted.append(select_action(policy, preprocessor, postprocessor, observation, device, args.task))
            recorded.append(np.asarray(row[ACTION], dtype=np.float32))
        recorded_arr = np.stack(recorded)
        predicted_arr = np.stack(predicted)
        diff = predicted_arr - recorded_arr
        plot_path = plot_episode(args.output_dir, episode, recorded_arr, predicted_arr, action_names)
        report["episodes"][str(episode)] = {
            "frames": len(indices),
            "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "predicted_action_std_by_dim": predicted_arr.std(axis=0).tolist(),
            "predicted_mean_l2_step_delta": float(np.linalg.norm(np.diff(predicted_arr, axis=0), axis=1).mean()),
            "plot": str(plot_path),
        }
        logger.info(
            "Episode %d: MAE=%.4f RMSE=%.4f mean_pred_step_delta=%.4f",
            episode,
            report["episodes"][str(episode)]["mae"],
            report["episodes"][str(episode)]["rmse"],
            report["episodes"][str(episode)]["predicted_mean_l2_step_delta"],
        )

    report_path = args.output_dir / "offline_act_progress_check.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote offline check report: %s", report_path)


if __name__ == "__main__":
    main()
