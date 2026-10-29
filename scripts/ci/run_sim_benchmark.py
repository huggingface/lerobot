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

"""Run one benchmark in Docker: start the simulator server, then evaluate the policy against it."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

DOCKER = shutil.which("docker")
if DOCKER is None:
    raise FileNotFoundError("Docker is required to run simulator benchmarks")


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--name", help="A benchmarks/benchmarks.yaml entry; needs PyYAML")
    # With --entry, the GPU runner needs only the standard library (no PyYAML).
    selected.add_argument("--entry", help="A resolved benchmark_matrix.py entry, as JSON")
    parser.add_argument("--policy-image", required=True)
    parser.add_argument("--sim-image", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    if args.entry:
        entry = json.loads(args.entry)
    else:
        from benchmark_matrix import load_entries

        entry = next(e for e in load_entries(Path(".")) if e["name"] == args.name)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    # Unique names, so parallel jobs and leftover containers on a persistent runner don't clash.
    run_id = os.environ.get("GITHUB_RUN_ID", "local") + "-" + uuid.uuid4().hex[:8]
    network = f"sim-benchmark-{entry['name']}-{run_id}"
    simulator = f"{network}-server"
    driver = f"{network}-driver"
    training = f"{network}-training"
    run([DOCKER, "network", "create", network])
    try:
        run(
            [
                DOCKER,
                "run",
                "-d",
                "--name",
                simulator,
                "--network",
                network,
                "--network-alias",
                "sim",
                "--gpus",
                "all",
                "--shm-size=4g",
                args.sim_image,
            ]
        )
        # Poll describe() until the server is up: it has a timeout and does not open a session.
        # Connecting fails until the server listens, so keep retrying until the deadline.
        probe = (
            "import time; from lerobot.env_server.client import EnvClient\n"
            "deadline = time.monotonic() + 300\n"
            "while True:\n"
            f"    c = EnvClient('tcp/sim:7448', {entry['name']!r}, handshake_timeout_s=30)\n"
            "    try:\n"
            "        c.describe(); break\n"
            "    except Exception:\n"
            "        if time.monotonic() > deadline: raise\n"
            "        time.sleep(2)\n"
            "    finally:\n"
            "        c.close()\n"
        )
        common = [
            DOCKER,
            "run",
            "--network",
            network,
            "--gpus",
            "all",
            "--shm-size=4g",
            "-e",
            "HF_HOME=/tmp/hf",
            "-e",
            "HF_TOKEN",
            "-e",
            "HF_HUB_DOWNLOAD_TIMEOUT=300",
        ]
        run(common + ["--rm", args.policy_image, "python", "-c", probe])
        if entry["parity"]:
            with (args.artifacts / "transition-parity.log").open("w") as stream:
                subprocess.run(
                    [
                        DOCKER,
                        "exec",
                        simulator,
                        "python",
                        "scripts/ci/check_sim_parity.py",
                        "--config",
                        entry["sim_config"],
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        command = [
            "lerobot-eval",
            f"--policy.path={entry['policy']}",
            f"--policy.pretrained_revision={entry['revision']}",
            "--policy.device=cuda",
            f"--env.type={entry['name']}",
            f"--env.task={entry['task']}",
            "--env.server.endpoint=tcp/sim:7448",
            f"--rename_map={json.dumps(entry['rename_map'])}",
            f"--eval.n_episodes={entry['episodes']}",
            f"--eval.batch_size={entry['batch_size']}",
            f"--eval.max_episode_steps={entry['max_episode_steps']}",
            "--eval.use_async_envs=false",
            "--output_dir=/tmp/eval-artifacts",
        ]
        try:
            run(common + ["--name", driver, args.policy_image] + command)
        finally:
            subprocess.run(
                [DOCKER, "cp", f"{driver}:/tmp/eval-artifacts/.", str(args.artifacts)], check=False
            )
        run(
            [
                sys.executable,
                "scripts/ci/parse_eval_metrics.py",
                "--artifacts-dir",
                str(args.artifacts),
                "--env",
                entry["name"],
                "--task",
                entry["task"],
                "--policy",
                entry["policy"],
                "--policy-revision",
                entry["revision"],
                "--sim-tag",
                entry["sim_tag"],
                *(["--commit", os.environ["GITHUB_SHA"]] if "GITHUB_SHA" in os.environ else []),
                *(
                    ["--max-episode-steps", str(entry["max_episode_steps"])]
                    if entry["max_episode_steps"] is not None
                    else []
                ),
            ]
        )
        if entry["train_smoke"]:
            if entry["name"] != "libero":
                raise ValueError("train_smoke uses the LIBERO dataset; only the libero entry may set it")
            train_command = [
                "lerobot-train",
                "--policy.path=lerobot/smolvla_base",
                "--policy.load_vlm_weights=true",
                "--policy.scheduler_decay_steps=25000",
                "--policy.freeze_vision_encoder=false",
                "--policy.train_expert_only=false",
                "--dataset.repo_id=lerobot/libero",
                "--dataset.episodes=[0]",
                "--dataset.use_imagenet_stats=false",
                "--env.type=libero",
                f"--env.task={entry['task']}",
                "--env.server.endpoint=tcp/sim:7448",
                "--policy.empty_cameras=1",
                "--output_dir=/tmp/train-smoke",
                "--steps=1",
                "--batch_size=1",
                "--env_eval_freq=1",
                "--eval.n_episodes=1",
                "--eval.batch_size=1",
                f"--eval.max_episode_steps={entry['max_episode_steps']}",
                "--eval.use_async_envs=false",
                "--save_freq=1",
                "--policy.push_to_hub=false",
                f"--rename_map={json.dumps(entry['rename_map'])}",
            ]
            try:
                run(common + ["--name", training, args.policy_image] + train_command)
            finally:
                (args.artifacts / "train-smoke").mkdir(exist_ok=True)
                subprocess.run(
                    [DOCKER, "cp", f"{training}:/tmp/train-smoke/.", str(args.artifacts / "train-smoke")],
                    check=False,
                )
    finally:
        with (args.artifacts / "sim-server.log").open("w") as stream:
            subprocess.run([DOCKER, "logs", simulator], stdout=stream, stderr=subprocess.STDOUT, check=False)
        for container in (driver, training, simulator):
            subprocess.run([DOCKER, "rm", "-f", container], check=False, stdout=subprocess.DEVNULL)
        subprocess.run([DOCKER, "network", "rm", network], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
