#!/usr/bin/env python

"""
Submit LeRobot training to a remote Ray cluster.
- Supports submitting multiple concurrent jobs safely.
- Lets Ray assign GPUs per job (no hardcoded CUDA device).
- Parameterize Ray URL, resources, and run specifics via CLI args.
- Use an official PyPI release of lerobot by default (no local code upload), overridable via --lerobot-req.
- Supports passing arbitrary lerobot train.py args after a "--" separator.
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Tuple, List

from ray import job_submission


def _short_hash(*parts: str, n: int = 6) -> str:
    h = hashlib.md5("_".join(parts).encode()).hexdigest()
    return h[:n]


def build_training_cmd(
    model_name: str,
    dataset_repo: str,
    env_type: str,
    env_task: str,
    batch_size: int,
    steps: int,
    wandb_entity: str,
    wandb_project: str,
    run_name: str,
    seed: int | None,
    policy_path: str | None,
    wandb_notes: str | None,
    extra_train_args: list[str] | None,
) -> str:
    args = [
        "python",
        "-u",  # unbuffered for real-time logs
        "-m",
        "lerobot.scripts.train",
    ]

    # Policy selection: either from scratch via --policy.type, or pretrained via --policy.path
    if policy_path:
        args.append(f"--policy.path={policy_path}")
    else:
        args.append(f"--policy.type={model_name}")

    # Common args
    args.extend(
        [
            f"--dataset.repo_id={dataset_repo}",
            f"--env.type={env_type}",
            f"--env.task={env_task}",
            f"--batch_size={batch_size}",
            f"--steps={steps}",
            "--wandb.enable=true",
            f"--wandb.entity={wandb_entity}",
            f"--wandb.project={wandb_project}",
            "--wandb.disable_artifact=true",
            f"--job_name={run_name}",
            "--policy.push_to_hub=false",
            # "--eval_freq=1",
        ]
    )

    if seed is not None:
        args.append(f"--seed={seed}")
    if wandb_notes:
        args.append(f"--wandb.notes=\"{wandb_notes}\"")

    if extra_train_args:
        # Pass through arbitrary train.py arguments verbatim
        # Handle JSON arguments that need proper shell escaping
        for arg in extra_train_args:
            if '={' in arg and '}' in arg:
                # This looks like a JSON argument, quote it properly
                key, value = arg.split('=', 1)
                args.append(f'{key}=\'{value}\'')
            else:
                args.append(arg)

    return " ".join(args)


def build_version_prelude_cmd() -> str:
    code = (
        "import sys\n"
        "from importlib.metadata import version, PackageNotFoundError\n"
        "print(f\"Python: {sys.version.split()[0]}\")\n"
        "print(\"LeRobot version:\", end=\" \")\n"
        "try:\n    print(version(\"lerobot\"))\n"
        "except PackageNotFoundError:\n    print(\"unknown\")\n"
    )
    return f"python -c '{code}'"


def submit_training_job(
    ray_dashboard_url: str,
    model_name: str,
    dataset_repo: str,
    env_type: str,
    env_task: str,
    batch_size: int,
    steps: int,
    wandb_entity: str,
    wandb_project: str,
    job_suffix: str,
    cpus: int,
    gpus: float,
    seed: int | None,
    lerobot_req: str,
    policy_path: str | None,
    wandb_notes: str | None,
    run_base_override: str | None,
    extra_train_args: list[str] | None,
) -> Tuple[str, job_submission.JobSubmissionClient]:
    """Submit a single training job to the Ray cluster and return (job_id, client)."""

    client = job_submission.JobSubmissionClient(ray_dashboard_url)
    print(f"Connected to Ray cluster at: {ray_dashboard_url}")

    derived_run_base = run_base_override or f"{model_name}_{Path(dataset_repo).name}"
    run_name = f"{derived_run_base}_{job_suffix}_{_short_hash(derived_run_base, job_suffix)}"
    print(f"WandB run name: {run_name}")

    training_cmd = build_training_cmd(
        model_name=model_name,
        dataset_repo=dataset_repo,
        env_type=env_type,
        env_task=env_task,
        batch_size=batch_size,
        steps=steps,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        run_name=run_name,
        seed=seed,
        policy_path=policy_path,
        wandb_notes=wandb_notes,
        extra_train_args=extra_train_args,
    )
    version_cmd = build_version_prelude_cmd()
    entrypoint_cmd = f"{version_cmd} && {training_cmd}"

    pip_packages = [
        lerobot_req,  # Expect lerobot==0.3.3 by default (base package only)
        # Pin a transformers version compatible with PI0 pretrained weights API surface (embed_tokens present)
        "transformers==4.52.0",
        # Direct env deps
        "dm_control==1.0.14",
        "mujoco==2.3.7",
        "num2words",
        # PushT env specific dep
        "pymunk==6.11.0",
        "gym_pusht",
        "gym_aloha",
    ]

    job_id = client.submit_job(
        entrypoint=entrypoint_cmd,
        runtime_env={
            "pip": pip_packages,
            "env_vars": {
                # Important: DO NOT set CUDA_VISIBLE_DEVICES. Ray sets it per allocation so concurrent jobs each see GPU 0 within their namespace.
                # Avoid setting PYTHONPATH to prevent local path shadowing the pip-installed lerobot package.
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "TOKENIZERS_PARALLELISM": "false",
                "MUJOCO_GL": "egl",
                "DISPLAY": "",
            },
        },
        entrypoint_num_cpus=cpus,
        entrypoint_num_gpus=gpus,
        metadata={
            "job_name": f"{run_name}",
            "description": f"{model_name} training on {dataset_repo}",
        },
    )

    print("Job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Monitor at: {ray_dashboard_url}/jobs/{job_id}")
    return job_id, client


def monitor_job(job_id: str, client: job_submission.JobSubmissionClient) -> None:
    """Monitor job progress and print final logs."""
    import time

    print("\nMonitoring job progress...")
    while True:
        status = client.get_job_status(job_id)
        print(f"Job status: {status}")
        if status in ["SUCCEEDED", "FAILED", "STOPPED"]:
            break
        time.sleep(30)

    logs = client.get_job_logs(job_id)
    print("\nFinal job logs:")
    print(logs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit LeRobot training jobs to a Ray cluster")

    # Ray
    p.add_argument(
        "--ray-url",
        default="http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com",
        help="Ray Dashboard/Jobs API URL (protocol + host).",
    )

    # Workload (manual)
    p.add_argument("--model", required=True, help="Policy model name (e.g., smolvla, diffusion, pi0)")
    p.add_argument("--dataset", required=True, help="HF dataset repo_id")
    p.add_argument("--env-type", required=True)
    p.add_argument("--env-task", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=200_000)

    # WandB
    p.add_argument("--wandb-entity", required=True)
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-notes", default=None, help="Optional WandB run notes")

    # Resources
    p.add_argument("--cpus", type=int, default=32, help="CPUs per job")
    p.add_argument("--gpus", type=float, default=1.0, help="GPUs per job (can be fractional)")

    # Naming
    p.add_argument("--name", default=None, help="Optional base name override for job/run")

    # Batch submission
    p.add_argument("--jobs", type=int, default=1, help="Number of jobs to submit")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. If set and --jobs>1, seeds increment per job.",
    )

    # LeRobot requirement (version or VCS)
    p.add_argument(
        "--lerobot-req",
        default="lerobot==0.3.3",
        help="pip requirement string for lerobot (e.g., 'lerobot==0.3.3' or a VCS URL)",
    )

    # Optional pretrained policy path or Hub repo id (will be passed as --policy.path=...)
    p.add_argument(
        "--policy-path",
        default=None,
        help="Optional pretrained policy local path or Hub repo_id passed to --policy.path for initialization.",
    )

    # Monitoring
    p.add_argument("--wait", action="store_true", help="Block and stream each job until completion")

    # Remainder: any args after "--" are forwarded verbatim to train.py
    p.add_argument("train_args", nargs=argparse.REMAINDER, help="Optional: args after -- are passed to train.py verbatim")

    args = p.parse_args()

    # Strip leading "--" in remainder if present
    if args.train_args and len(args.train_args) > 0 and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]

    return args


if __name__ == "__main__":
    args = parse_args()

    submitted: List[Tuple[str, job_submission.JobSubmissionClient]] = []
    for i in range(args.jobs):
        job_suffix = f"j{i+1}"
        seed_i = None if args.seed is None else args.seed + i
        job = submit_training_job(
            ray_dashboard_url=args.ray_url.rstrip("/"),
            model_name=args.model,
            dataset_repo=args.dataset,
            env_type=args.env_type,
            env_task=args.env_task,
            batch_size=args.batch_size,
            steps=args.steps,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            job_suffix=job_suffix,
            cpus=args.cpus,
            gpus=args.gpus,
            seed=seed_i,
            lerobot_req=args.lerobot_req,
            policy_path=args.policy_path,
            wandb_notes=args.wandb_notes,
            run_base_override=args.name,
            extra_train_args=args.train_args,
        )
        submitted.append(job)

    for job_id, _ in submitted:
        print(f"\nYou can monitor the job at: {args.ray_url.rstrip('/')}/jobs/{job_id}")
        print(f"ray job logs {job_id} -f")

    if args.wait:
        # Monitor sequentially to keep output readable
        for job_id, client in submitted:
            monitor_job(job_id, client)
