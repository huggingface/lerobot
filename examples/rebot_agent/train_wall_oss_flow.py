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


def prepare_run(output: Path, gpus: int, batch_size: int, smoke: bool) -> tuple[dict, list[str]]:
    """Resolve the checked-in recipe and build a bounded, single-node torchrun command."""
    if gpus not in range(1, 5) or batch_size < 1:
        raise ValueError("Use one to four GPUs and a positive per-GPU batch size")
    workspace = Path(__file__).resolve().parents[2]
    session = json.loads((workspace / "examples/rebot_agent/session.json").read_text())
    candidate = next(c for c in session["candidates"] if c["name"] == "wall_oss_flow_80_20_v1")
    config = candidate["training"]
    recipe = TrainingRecipe.from_yaml(workspace / candidate["recipe_path"])
    config["dataset"]["task_recipe"] = asdict(recipe)
    config["policy"].update(type="wall_x", pretrained_name_or_path=candidate["base_model"])
    config.update(batch_size=batch_size, output_dir=str(output / "training"))
    config["accelerator"] = {"mixed_precision": "bf16"}
    if smoke:
        config.update(steps=10, save_freq=10, log_freq=1, eval_steps=10, num_workers=0)
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}"]
    argv.extend(["-m", "lerobot.scripts.lerobot_train", f"--config_path={output / 'config.json'}"])
    return config, argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this launch")
    parser.add_argument("--gpus", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU micro-batch size")
    parser.add_argument("--smoke", action="store_true", help="Ten steps, validation, and checkpoint save")
    parser.add_argument("--dry-run", action="store_true", help="Write config and command without training")
    args = parser.parse_args()
    output = args.output.resolve()
    config, argv = prepare_run(output, args.gpus, args.batch_size, args.smoke)
    workspace = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None:
        parser.error("Git is required to record the source revision")
    if not args.dry_run:
        import torch

        if torch.cuda.device_count() < args.gpus:
            parser.error("Fewer CUDA GPUs are visible than requested; run inside the scheduler allocation")
    output.mkdir(parents=True, exist_ok=False)
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


if __name__ == "__main__":
    main()
