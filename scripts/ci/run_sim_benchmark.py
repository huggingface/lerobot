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

"""Run the same simulator sidecar and driver locally and in benchmark CI."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_matrix import load_entries

DOCKER = shutil.which("docker")
if DOCKER is None:
    raise FileNotFoundError("Docker is required to run simulator benchmarks")


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--policy-image", required=True)
    parser.add_argument("--sim-image", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    entry = next(e for e in load_entries(Path(".")) if e["name"] == args.name)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    network = f"sim-benchmark-{args.name}"
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
        # Describe is a bounded readiness check; the session is opened only by evaluation.
        probe = (
            "from lerobot.env_server.client import EnvClient; c=EnvClient('tcp/sim:7448', '"
            + entry["name"]
            + "', 300); c.describe(); c.close()"
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
        if entry["name"] in {"libero", "metaworld"}:
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
            "--env.type=sim",
            "--env.endpoint=tcp/sim:7448",
            f"--env.deployment={entry['name']}",
            f"--eval.profile={entry['profile']}",
            f"--eval.n_episodes={entry['episodes']}",
            f"--eval.batch_size={entry['batch_size']}",
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
            ]
        )
        if entry["train_smoke"]:
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
                "--env.type=sim",
                "--env.endpoint=tcp/sim:7448",
                "--env.deployment=libero",
                "--env.profile=benchmarks/profiles/libero.yaml",
                "--policy.empty_cameras=1",
                "--output_dir=/tmp/train-smoke",
                "--steps=1",
                "--batch_size=1",
                "--env_eval_freq=1",
                "--eval.n_episodes=1",
                "--eval.batch_size=1",
                "--eval.use_async_envs=false",
                "--save_freq=1",
                "--policy.push_to_hub=false",
                '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}',
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
