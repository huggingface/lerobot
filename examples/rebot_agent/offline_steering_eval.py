# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compare WALL-X prompts on fixed held-out observations without operating a robot.

Action error against a recorded demonstration is an offline diagnostic. It does not
measure physical success or prove that an alternative valid action is incorrect.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.steering_commands import STYLES, SteeringCommands
from lerobot.datasets.utils import resolve_episode_indices
from lerobot.policies import make_pre_post_processors
from lerobot.utils.import_utils import _wallx_deps_available, require_package
from lerobot.utils.steering import render_steering_command

if TYPE_CHECKING or _wallx_deps_available:
    from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy

HELDOUT_EPISODES = tuple(range(90, 100))
OVERALL_TASK = "Pick up objects from the table and place them into the bin."


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_panel(manifest: dict, anchors_per_style: int = 3, seed: int = 7) -> dict:
    """Choose deterministic anchors per episode/style, then pair every available prompt."""
    if not 1 <= anchors_per_style <= 10:
        raise ValueError("Use one to ten anchors per episode/style")
    index = SteeringCommands(manifest)
    if not index.episodes or not set(index.episodes) <= set(HELDOUT_EPISODES):
        raise ValueError("Evaluation manifest must contain only held-out episodes 90–99")
    anchors = set()
    for ep, spans in index.episodes.items():
        for style in sorted(STYLES):
            frames = [
                frame
                for span in spans
                if any(command["style"] == style for command in span["commands"])
                for frame in range(span["start_frame"], span["end_frame"])
            ]
            if frames:
                count = min(anchors_per_style, len(frames))
                positions = (
                    [(len(frames) - 1) // 2]
                    if count == 1
                    else [i * (len(frames) - 1) // (count - 1) for i in range(count)]
                )
                anchors.update((ep, frames[position]) for position in positions)
    samples = []
    for ep, frame in sorted(anchors):
        span = index.span_at(ep, frame)
        commands = [{"style": "task", "task": OVERALL_TASK}]
        commands.extend(
            {"style": command["style"], "task": render_steering_command(command)}
            for command in index.at(ep, frame)
        )
        samples.append(
            {
                "episode_index": ep,
                "frame_index": frame,
                "start_frame": span["start_frame"],
                "end_frame": span["end_frame"],
                "seed": seed + ep * 1_000_003 + frame,
                "commands": commands,
            }
        )
    return {
        "version": 1,
        "source": index.source,
        "heldout_episodes": list(HELDOUT_EPISODES),
        "anchors_per_style_per_episode": anchors_per_style,
        "seed": seed,
        "annotation_profile": index.annotation_profile(),
        "samples": samples,
        "prompt_counts": dict(Counter(c["style"] for s in samples for c in s["commands"])),
        "comparison": "Identical observations, noise seed, and in-interval action targets for each prompt at an anchor",
        "physical_success_measured": False,
    }


def action_errors(prediction, target, padding, *, frame, start, end, scale):
    """Exclude episode padding and all targets outside the reviewed command interval."""
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Predictions and targets must have matching [horizon, dimension] shapes")
    if padding.shape != target.shape[:1] or padding.dtype != torch.bool:
        raise ValueError("Require a boolean padding mask per action step")
    if scale.shape != target.shape[1:] or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Action scales must be finite and positive for each dimension")
    frames = frame + torch.arange(len(target), device=target.device)
    valid = ~padding & (frames >= start) & (frames < end)
    if not valid.any():
        raise ValueError("No valid demonstrated action targets")
    error = prediction[valid] - target[valid]
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite prediction or valid action target")
    return {
        "valid_action_steps": int(valid.sum()),
        "mae_per_dimension": error.abs().mean(0).tolist(),
        "normalized_mse": float((error / scale).square().mean()),
    }


def summarize(rows: list[dict]) -> dict:
    """Equal-anchor metrics, with paired deltas on exactly the same task anchors."""
    task = {(r["episode_index"], r["frame_index"]): r for r in rows if r["style"] == "task"}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["style"]][row["episode_index"], row["frame_index"]].append(row)
    result = {}
    for style, anchors in groups.items():
        errors, deltas = [], []
        for key, variants in anchors.items():
            error = sum(v["normalized_mse"] for v in variants) / len(variants)
            errors.append(error)
            deltas.append(error - task[key]["normalized_mse"])
        result[style] = {
            "anchors": len(anchors),
            "command_variants": sum(len(v) for v in anchors.values()),
            "mean_normalized_mse": sum(errors) / len(errors),
            "mean_paired_delta_vs_task": sum(deltas) / len(deltas),
        }
    return result


def training_episodes(config: dict, metadata) -> list[int]:
    """Reconstruct main's per-task trailing holdout split from saved training configuration."""
    episodes = resolve_episode_indices(
        config.get("episodes"), metadata.total_episodes, config.get("exclude_episodes")
    )
    if episodes is None:
        episodes = list(range(metadata.total_episodes))
    groups = defaultdict(list)
    for episode in episodes:
        tasks = metadata.episodes[episode]["tasks"]
        groups[tasks[0] if tasks else ""].append(episode)
    split = config.get("eval_split", 0)
    if not 0 <= split < 1:
        raise ValueError("Invalid checkpoint evaluation split")
    return sorted(
        ep for group in groups.values() for ep in group[: len(group) - math.ceil(len(group) * split)]
    )


def evaluate(panel: dict, checkpoint: Path, dataset_root: Path, output: Path, device: str) -> dict:
    require_package("transformers", extra="wallx")
    require_package("peft", extra="wallx")
    require_package("torchdiffeq", extra="wallx")
    require_package("qwen-vl-utils", extra="wallx", import_name="qwen_vl_utils")
    if output.exists():
        raise FileExistsError(output)
    training = json.loads((checkpoint / "train_config.json").read_text())
    if any(training["dataset"].get(k) != panel["source"][k] for k in ("repo_id", "revision")):
        raise ValueError("Checkpoint training source differs from evaluation source")
    policy = WallXPolicy.from_pretrained(checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    episodes = sorted({s["episode_index"] for s in panel["samples"]})
    ds = LeRobotDataset(
        panel["source"]["repo_id"],
        root=dataset_root,
        revision=panel["source"]["revision"],
        episodes=episodes,
        video_backend="pyav",
        delta_timestamps={"action": [i / 30 for i in range(policy.config.chunk_size)]},
    )
    if ds.meta.fps != 30 or policy.config.output_features["action"].shape != (14,):
        raise ValueError("This evaluation expects the recorded ReBot 30 Hz, 14-dimensional action space")
    trained_episodes = training_episodes(training["dataset"], ds.meta)
    if set(trained_episodes) & set(panel["heldout_episodes"]):
        raise ValueError("Checkpoint training split includes evaluation episodes")
    scale = torch.as_tensor(ds.meta.stats["action"]["std"]).float().reshape(-1)
    rows = []
    for anchor in panel["samples"]:
        ep, frame = anchor["episode_index"], anchor["frame_index"]
        absolute = int(ds.meta.episodes[ep]["dataset_from_index"]) + frame
        relative = (
            ds.absolute_to_relative_idx[absolute] if ds.absolute_to_relative_idx is not None else absolute
        )
        sample = ds[relative]
        assert int(sample["episode_index"]) == ep and int(sample["frame_index"]) == frame
        for command in anchor["commands"]:
            # Future actions and language annotation history are never model inputs.
            observation = {
                key: value.clone() for key, value in sample.items() if key.startswith("observation.")
            }
            inputs = pre({**observation, "task": command["task"]})
            if inputs.get("action_chunk") is not None:
                raise ValueError("Demonstrated future actions entered inference inputs")
            policy.reset()
            torch.manual_seed(anchor["seed"])
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
            )
            with torch.inference_mode(), autocast:
                actions = post(policy.predict_action_chunk(inputs)).detach().float().cpu()
            if actions.shape != (1, policy.config.chunk_size, 14) or not torch.isfinite(actions).all():
                raise ValueError("Invalid inferred action chunk")
            metrics = action_errors(
                actions[0],
                sample["action"],
                sample["action_is_pad"],
                frame=frame,
                start=anchor["start_frame"],
                end=anchor["end_frame"],
                scale=scale,
            )
            rows.append({"episode_index": ep, "frame_index": frame, **command, **metrics})
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
        "checkpoint_train_config_sha256": digest(checkpoint / "train_config.json"),
        "evaluation_code_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "device": device,
        "training_episodes": trained_episodes,
        "panel": panel,
        "action_features": ds.meta.features["action"],
        "normalization": {
            "source": "pinned source dataset action standard deviations",
            "std": scale.tolist(),
        },
        "samples": rows,
        "styles": summarize(rows),
        "optimizer_updates": 0,
        "robot_commands": 0,
        "physical_success_measured": False,
        "limitations": "Recorded-action agreement only. Anchors are correlated; styles have different coverage. Compare prompt deltas only on paired anchors. No grasp, success, pixel accuracy, or command-compliance claim.",
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchors-per-style", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    panel = make_panel(json.loads(args.manifest.read_text()), args.anchors_per_style, args.seed)
    panel["manifest_sha256"] = digest(args.manifest)
    if args.checkpoint:
        if not args.dataset_root:
            parser.error("--checkpoint requires --dataset-root")
        evaluate(panel, args.checkpoint, args.dataset_root, args.output, args.device)
    else:
        args.output.write_text(json.dumps(panel, indent=2) + "\n")


if __name__ == "__main__":
    main()
