# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Prepare a semantic/steering comparison on identical reviewed frames, without launching jobs.

Run from the repository root with ``python -m examples.rebot_agent.prepare_matched_training``.
This diagnostic excludes intervals without reviewed semantics, including trace-only intervals;
it is not a replacement for completing the full steering annotation dataset.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from examples.rebot_agent.train_wall_oss_flow import prepare_run
from lerobot.datasets.steering_commands import SteeringCommands


def paired_manifests(manifest: dict) -> tuple[dict, dict, list[dict]]:
    """Keep aligned intervals and equal alternative counts for identical sampling RNG draws."""
    index = SteeringCommands(manifest)
    control, steering, excluded = [], [], []
    for ep, spans in sorted(index.episodes.items()):
        if ep >= 90:
            continue
        for span in spans:
            semantics = [c for c in span["commands"] if c["style"] == "subtask"]
            if not semantics:
                excluded.append(
                    {k: span[k] for k in ("episode_index", "start_frame", "end_frame")}
                    | {"reason": "No reviewed semantic command for a matched control"}
                )
                continue
            steering.append(copy.deepcopy(span))
            baseline = copy.deepcopy(span)
            # Repeating a reviewed instruction is intentional. Both arms must consume the
            # same randint draw (including its range) so subsequent task choices stay paired.
            baseline["commands"] = [
                copy.deepcopy(semantics[i % len(semantics)]) for i in range(len(span["commands"]))
            ]
            control.append(baseline)
    if not steering:
        raise ValueError("No reviewed semantic frames below episode 90")
    common = {
        "version": 1,
        "source": copy.deepcopy(manifest["source"]),
        "training_release": "matched_coverage_diagnostic_not_full_mixture",
        "human_verified": False,
    }
    return {**common, "segments": control}, {**common, "segments": steering}, excluded


def prepare_pair(
    manifest: dict,
    output: Path,
    development_episodes: list[int],
    *,
    gpus: int = 1,
    batch_size: int = 4,
    seed: int = 7,
    dataset_root: Path | None = None,
) -> tuple[dict, dict, dict]:
    """Return manifests and configs; standard trainer uses the ordered episode tail for dev."""
    control, steering, excluded = paired_manifests(manifest)
    index = SteeringCommands(steering)
    development = sorted(development_episodes)
    if (
        not development
        or len(set(development)) != len(development)
        or any(ep not in index.episodes for ep in development)
    ):
        raise ValueError("Development episodes must be distinct reviewed episodes below 90")
    training = sorted(set(index.episodes) - set(development))
    if not training:
        raise ValueError("Development split leaves no training episodes")
    coverage = index.annotation_profile(training)
    # Round to a quarter-pass unit so the native periodic saver includes 0.25/0.5/1/2 passes.
    # Additional periodic checkpoints are retained as well; no trainer changes are needed.
    if gpus not in range(1, 5) or batch_size < 1:
        raise ValueError("Use one to four GPUs and a positive per-GPU batch size")
    quarter_steps = math.ceil(coverage["annotated_frames"] / (4 * gpus * batch_size))
    steps = 8 * quarter_steps
    configs = {}
    for name, commands in [("semantic_control", control), ("steerable", steering)]:
        config, _ = prepare_run(output / name, gpus, batch_size, smoke=False, dataset_root=dataset_root)
        if any(manifest["source"].get(k) != config["dataset"][k] for k in ("repo_id", "revision")):
            raise ValueError("Manifest source differs from the pinned ReBot training dataset")
        config["dataset"].update(
            episodes=training + development,
            # Stay away from an integer floating-point boundary in ceil(n * eval_split).
            eval_split=(len(development) - 0.5) / len(index.episodes),
            task_recipe={
                "messages": [
                    {
                        "role": "user",
                        "stream": "low_level",
                        "content": "Pick up objects from the table and place them into the bin.",
                    }
                ]
            },
            steering_manifest=str(output / name / "steering_manifest.json"),
            steering_task_probability=0.2,
            steering_skip_uncovered=True,
            steering_style_weights=None,
            steering_required_styles=sorted(
                style
                for style, count in SteeringCommands(commands)
                .annotation_profile(training)["frames_with_style"]
                .items()
                if count
            ),
            image_transforms={"enable": False},
        )
        config["policy"].update(
            steering_coordinate_format="native_points_v1",
            scheduler_warmup_steps=min(100, math.ceil(0.05 * steps)),
            scheduler_decay_steps=steps,
        )
        config.update(
            steps=steps,
            seed=seed,
            num_workers=0,
            save_freq=quarter_steps,
            eval_steps=quarter_steps,
            max_eval_samples=256,
            log_freq=min(100, quarter_steps),
        )
        configs[name] = config
    plan = {
        "status": "prepared_not_launched",
        "training_episodes": training,
        "development_episodes": development,
        "final_heldout_episodes": list(range(90, 100)),
        "training_profile": coverage,
        "development_profile": index.annotation_profile(development),
        "excluded_intervals": excluded,
        "steps": steps,
        "quarter_pass_steps": quarter_steps,
        "comparison_checkpoint_steps": [quarter_steps * n for n in (1, 2, 4, 8)],
        "nominal_passes_at_checkpoints": [
            quarter_steps * n * gpus * batch_size / coverage["annotated_frames"] for n in (1, 2, 4, 8)
        ],
        "gpus_per_run": gpus,
        "configs": configs,
        "requirements_before_training": [
            "Verify dataset metadata has one task group and native factory reproduces the exact split",
            "Verify both actual dataloaders produce identical frame order, actions, masks and task routing",
            "Run GPU forward/backward/save/reload smoke tests with these configs before full training",
        ],
        "limitations": [
            "Intervals without reviewed semantic commands are excluded; no fabricated control labels",
            "Duplicated semantic alternatives match RNG consumption; do not enable style weights",
            "num_workers=0 keeps command sampling in one reproducible stream for the paired audit",
            "Development episodes are disjoint in this new pair but may have trained older experiments",
            "Final held-out episodes already informed earlier development; disclose this in reporting",
            "Training-loop losses condition on different prompt distributions; also evaluate paired prompts",
        ],
    }
    return control, steering, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="Local native annotation derivative for both runs")
    parser.add_argument("--development-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    control, steering, plan = prepare_pair(
        json.loads(args.manifest.read_text()),
        output,
        args.development_episodes,
        gpus=args.gpus,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
    )
    plan["source_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for name, commands in [("semantic_control", control), ("steerable", steering)]:
        directory = output / name
        directory.mkdir()
        for filename, value in [("steering_manifest.json", commands), ("config.json", plan["configs"][name])]:
            (directory / filename).write_text(json.dumps(value, indent=2) + "\n")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(output / "plan.json"), "steps": plan["steps"]}))


if __name__ == "__main__":
    main()
