#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import json
from pathlib import Path


def find_missing_workers(completions_dir, world_size):
    """Find workers that are not completed and returns their indices."""
    full = list(range(world_size))

    completed = []
    for path in completions_dir.glob("*"):
        if path.name in [".", ".."]:
            continue
        index = path.name.lstrip("0")
        index = 0 if index == "" else int(index)
        completed.append(index)

    missing_workers = set(full) - set(completed)
    return missing_workers


def find_output_files(slurm_dir, worker_indices):
    """Find output files associated to worker indices, and return tuples
    of (worker index, output file path)
    """
    out_files = []
    for path in slurm_dir.glob("*.out"):
        _, worker_id = path.name.replace(".out", "").split("_")
        worker_id = int(worker_id)
        if worker_id in worker_indices:
            out_files.append((worker_id, path))
    return out_files


def display_error_files(logs_dir, job_name):
    executor_path = Path(logs_dir) / job_name / "executor.json"
    completions_dir = Path(logs_dir) / job_name / "completions"

    with open(executor_path) as f:
        executor = json.load(f)

    missing_workers = find_missing_workers(completions_dir, executor["world_size"])

    for missing in sorted(missing_workers)[::-1]:
        print(missing)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logs-dir",
        type=str,
        help="Path to logs directory for `datatrove`.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="port_droid",
        help="Job name used in slurm, and name of the directory created inside the provided logs directory.",
    )

    args = parser.parse_args()

    display_error_files(**vars(args))


if __name__ == "__main__":
    main()
