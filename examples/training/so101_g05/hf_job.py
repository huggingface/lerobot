"""Pinned HF Job: validate, profile an effective batch of 32, train the 1k-update pilot."""

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from packaging.version import Version

from examples.training.so101_g05.prepare import prepare
from examples.training.so101_g05.split import held_out
from examples.training.so101_g05.validate import validate

WORK = Path("/data/g05")
ARTIFACTS = WORK / "artifacts"


def stage(name, **extra):
    value = {"stage": name, "time": time.time(), **extra}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "status.json").write_text(json.dumps(value, indent=2))
    print("G05_JOB", json.dumps(value), flush=True)


def execute(args, log, environment=None):
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment
        )
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return process.wait(), time.monotonic() - started


def parse_profile(log, accumulation, batch):
    text = log.read_text()
    values = [float(x) for x in re.findall(r"step_s:\s*([0-9.]+)", text)]
    memory = [float(x) for x in re.findall(r"mem_gb:\s*([0-9.]+)", text)]
    if len(values) < 2 or re.search(r"loss:\s*(?:nan|inf)", text, re.I):
        raise ValueError("The profile did not show stable finite training updates")
    return {
        "microbatch": batch,
        "accumulation": accumulation,
        "effective_batch": batch * accumulation,
        "seconds_per_microbatch": statistics.mean(values[-2:]),
        "seconds_per_update": statistics.mean(values[-2:]) * accumulation,
        "max_logged_allocated_GiB": max(memory, default=0),
    }


def check_triton_version(version, compute_major):
    if compute_major == 9 and Version("3.4.0") <= Version(version) < Version("3.7.1"):
        raise RuntimeError("Hopper G05 training requires Triton >= 3.7.1 for safe FLA backward kernels")


def reusable_validation(spec):
    cache = spec.get("cached_validation")
    if cache is None:
        return None
    for key in (
        "dataset_repo",
        "dataset_revision",
        "checkpoint_repo",
        "checkpoint_revision",
        "recipe_sha256",
    ):
        if cache[key] != spec[key]:
            raise ValueError(f"Cached validation input changed: {key}")
    for path, expected in cache["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached validation code changed: {path}")
    if not cache["report"]["full_dataset"]:
        raise ValueError("Cached validation must cover the full dataset")
    return cache["report"]


def main():
    spec = json.loads(Path(os.environ["RUN_SPEC"]).read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    import torch
    import triton

    check_triton_version(triton.__version__, torch.cuda.get_device_capability()[0])
    stage("kernel_runtime_checked", torch=torch.__version__, triton=triton.__version__)
    stage("downloading_inputs")
    dataset = Path(
        snapshot_download(
            spec["dataset_repo"],
            repo_type="dataset",
            revision=spec["dataset_revision"],
            local_dir=WORK / "dataset",
            max_workers=6,
        )
    )
    source = (
        Path(
            snapshot_download(
                spec["checkpoint_repo"],
                revision=spec["checkpoint_revision"],
                allow_patterns=[spec["checkpoint_subfolder"] + "/*"],
                local_dir=WORK / "source_checkpoint",
                max_workers=4,
            )
        )
        / spec["checkpoint_subfolder"]
    )
    checkpoint = WORK / "initial_checkpoint"
    recipe = Path(__file__).with_name("recipe.yaml")
    if hashlib.sha256(recipe.read_bytes()).hexdigest() != spec["recipe_sha256"]:
        raise ValueError("Recipe hash differs from the reviewed snapshot")
    prepare(source, checkpoint)
    report = reusable_validation(spec)
    if report is None:
        stage("validating_recipe")
        report = validate(dataset, checkpoint)
    else:
        stage("reusing_full_validation", from_job=spec["cached_validation"]["source_job"])
    (ARTIFACTS / "validation.json").write_text(json.dumps(report, indent=2))
    split = held_out(dataset)
    (ARTIFACTS / "split.json").write_text(json.dumps({"held_out_episodes": split, "seed": 1000}))
    stage("validated", **report)
    common = [
        f"--policy.path={checkpoint}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--policy.private=false",
        f"--dataset.repo_id={spec['dataset_repo']}",
        f"--dataset.root={dataset}",
        f"--dataset.revision={spec['dataset_revision']}",
        f"--dataset.exclude_episodes={json.dumps(split)}",
        "--dataset.eval_split=0",
        "--dataset.video_backend=pyav",
        "--eval_steps=0",
        "--env_eval_freq=0",
        "--num_workers=6",
        "--prefetch_factor=2",
        "--accelerator.mixed_precision=bf16",
        "--wandb.enable=false",
        "--seed=1000",
    ]
    launch = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    profile = None
    for batch, accumulation in [(32, 1), (16, 2), (8, 4)]:
        stage("profiling", microbatch=batch, accumulation=accumulation)
        log = ARTIFACTS / f"profile_b{batch}.log"
        code, elapsed = execute(
            launch
            + common
            + [
                f"--batch_size={batch}",
                f"--accelerator.gradient_accumulation.steps={accumulation}",
                f"--steps={8 * accumulation}",
                f"--log_freq={2 * accumulation}",
                "--save_checkpoint=false",
                f"--output_dir={WORK / f'profile_b{batch}'}",
            ],
            log,
        )
        if code:
            text = log.read_text()
            if re.search(r"out of memory|OutOfMemoryError", text, re.I):
                stage("profile_oom", microbatch=batch)
                continue
            raise RuntimeError(f"Profile failed with status {code}; refusing to start training")
        profile = parse_profile(log, accumulation, batch)
        profile["profile_wall_seconds"] = elapsed
        break
    if profile is None:
        raise RuntimeError("No tested microbatch fits; refusing to launch a changed training recipe")
    (ARTIFACTS / "profile.json").write_text(json.dumps(profile, indent=2))
    updates = spec["optimizer_steps"]
    accumulation = profile["accumulation"]
    batch = profile["microbatch"]
    estimated_hours = profile["seconds_per_update"] * updates / 3600
    stage("profile_complete", **profile, estimated_pilot_hours=estimated_hours)
    name = f"so101-g05-full-pilot-b32-{os.environ.get('JOB_ID', int(time.time()))}"
    environment = {**os.environ, "G05_RUN_NAME": name, "G05_ACCUMULATION": str(accumulation)}
    stage(
        "training",
        optimizer_steps=updates,
        micro_steps=updates * accumulation,
        run_name=name,
        effective_batch=profile["effective_batch"],
        estimated_pilot_hours=estimated_hours,
    )
    args = (
        [sys.executable, "-m", "examples.training.so101_g05.job_train"]
        + common
        + [
            f"--batch_size={batch}",
            f"--accelerator.gradient_accumulation.steps={accumulation}",
            f"--steps={updates * accumulation}",
            f"--save_freq={updates * accumulation}",
            f"--policy.scheduler_warmup_steps={500 * accumulation}",
            f"--log_freq={25 * accumulation}",
            "--policy.push_to_hub=true",
            f"--policy.repo_id={spec['model_repo']}",
            "--save_checkpoint_to_hub=true",
            "--wandb.enable=true",
            "--wandb.project=so101-g05",
            "--wandb.disable_artifact=true",
            f"--output_dir={WORK / 'training'}",
            "--job_name=so101-g05-full-pilot",
        ]
    )
    code, elapsed = execute(args, ARTIFACTS / "training.log", environment)
    if code:
        raise RuntimeError(f"Training failed with status {code}; published checkpoints remain available")
    stage("complete", optimizer_steps=updates, training_hours=elapsed / 3600, **profile)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        stage("failed", error=repr(error))
        raise
