# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Launch the ReBot WALL-OSS-Flow candidate inside a single-node GPU allocation.

This does not request scheduler resources. Use --dry-run to inspect the configuration
on a CPU host, then run --smoke inside an allocation before starting the full run.
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from lerobot.datasets.recipe import TrainingRecipe


def prepare_run(
    output: Path,
    gpus: int,
    batch_size: int,
    smoke: bool,
    steering_manifest: Path | None = None,
    skip_uncovered: bool = False,
    style_weights: dict[str, float] | None = None,
    coordinate_format: str = "original_pixels",
) -> tuple[dict, list[str]]:
    """Resolve the checked-in recipe and build a bounded, single-node torchrun command."""
    if gpus not in range(1, 5) or batch_size < 1:
        raise ValueError("Use one to four GPUs and a positive per-GPU batch size")
    if coordinate_format not in {"original_pixels", "native_points_v1"}:
        raise ValueError("Unknown steering coordinate format")
    if skip_uncovered and steering_manifest is None:
        raise ValueError("--skip-uncovered requires --steering-manifest")
    if style_weights is not None and steering_manifest is None:
        raise ValueError("--style-weights requires --steering-manifest")
    workspace = Path(__file__).resolve().parents[2]
    session = json.loads((workspace / "examples/rebot_agent/training.json").read_text())
    candidate = next(c for c in session["candidates"] if c["name"] == "wall_oss_flow_80_20_v1")
    config = candidate["training"]
    recipe = TrainingRecipe.from_yaml(workspace / candidate["recipe_path"])
    config["dataset"]["task_recipe"] = asdict(recipe)
    if steering_manifest is not None:
        from lerobot.datasets.steering_commands import SteeringCommands, validate_style_weights

        validate_style_weights(style_weights)
        manifest = json.loads(steering_manifest.read_text())
        SteeringCommands(manifest)
        if any(manifest["source"].get(k) != config["dataset"][k] for k in ("repo_id", "revision")):
            raise ValueError("Steering manifest source differs from the training dataset")
        config["dataset"]["steering_manifest"] = str(steering_manifest.resolve())
        config["dataset"]["steering_task_probability"] = 0.2
        config["dataset"]["steering_style_weights"] = style_weights
        config["dataset"]["steering_skip_uncovered"] = skip_uncovered
        config["dataset"]["steering_required_styles"] = ["subtask", "motion", "point", "trace", "combination"]
        config["dataset"]["image_transforms"] = {"enable": False}
        # Keep the same corrected task conditioning in both experiment arms.
        config["dataset"]["task_recipe"] = asdict(recipe.blend["high_level_task"])
    config["policy"].update(
        type="wall_x",
        pretrained_name_or_path=candidate["base_model"],
        steering_coordinate_format=coordinate_format,
    )
    config.update(batch_size=batch_size, output_dir=str(output / "training"))
    config["accelerator"] = {"mixed_precision": "bf16"}
    if smoke:
        config.update(steps=10, save_freq=10, log_freq=1, eval_steps=10, max_eval_samples=20, num_workers=0)
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}"]
    argv.extend(["-m", "lerobot.scripts.lerobot_train", f"--config_path={output / 'config.json'}"])
    return config, argv


def reload_command(argv: list[str], output: Path, steps: int) -> list[str]:
    """Exercise native checkpoint/optimizer reload and one further update in a fresh process."""
    checkpoint = output / "training/checkpoints/last/pretrained_model/train_config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not save its checkpoint configuration: {checkpoint}")
    # Keep the same process topology, model, recipe and split saved by the training run.
    return [arg for arg in argv if not arg.startswith("--config_path=")] + [
        f"--config_path={checkpoint}",
        "--resume=true",
        f"--steps={steps + 1}",
        f"--eval_steps={steps + 1}",
        "--save_checkpoint=false",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this launch")
    parser.add_argument("--gpus", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU micro-batch size")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Ten updates, bounded validation, save, then native reload/update",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write config and command without training")
    parser.add_argument(
        "--steering-manifest", type=Path, help="Reviewed multi-style commands; omit for the semantic baseline"
    )
    parser.add_argument(
        "--skip-uncovered",
        action="store_true",
        help="Sample only reviewed frame intervals and report exclusions",
    )
    parser.add_argument(
        "--style-weights", type=Path, help="JSON of relative weights for all five steering styles"
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["original_pixels", "native_points_v1"],
        default="original_pixels",
        help="Checkpoint-owned coordinate encoding; native_points_v1 scales named-camera points before tokenization",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    config, argv = prepare_run(
        output,
        args.gpus,
        args.batch_size,
        args.smoke,
        args.steering_manifest,
        args.skip_uncovered,
        json.loads(args.style_weights.read_text()) if args.style_weights else None,
        args.coordinate_format,
    )
    workspace = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None:
        parser.error("Git is required to record the source revision")
    if not args.dry_run:
        import torch

        if torch.cuda.device_count() < args.gpus:
            parser.error("Fewer CUDA GPUs are visible than requested; run inside the scheduler allocation")
    output.mkdir(parents=True, exist_ok=False)
    if args.steering_manifest is not None:
        manifest_copy = output / "steering_manifest.json"
        shutil.copyfile(args.steering_manifest, manifest_copy)
        config["dataset"]["steering_manifest"] = str(manifest_copy)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    revision = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    patch = subprocess.check_output([git, "diff", "HEAD", "--binary"], cwd=workspace)
    (output / "code.patch").write_bytes(patch)
    (output / "launch.json").write_text(
        json.dumps({"argv": argv, "revision": revision, "gpus": args.gpus, "smoke": args.smoke}, indent=2)
        + "\n"
    )
    print(json.dumps({"config": str(output / "config.json"), "argv": argv}, indent=2), flush=True)
    if not args.dry_run:
        # Preserve torchrun's output for the scheduler log and propagate a failed rank's exit status.
        subprocess.run(argv, cwd=workspace, check=True)
        if args.smoke:
            reload_argv = reload_command(argv, output, config["steps"])
            (output / "reload_launch.json").write_text(json.dumps({"argv": reload_argv}, indent=2) + "\n")
            subprocess.run(reload_argv, cwd=workspace, check=True)
            (output / "smoke_completed.json").write_text(
                json.dumps(
                    {"saved_step": config["steps"], "reloaded_update_step": config["steps"] + 1}, indent=2
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
