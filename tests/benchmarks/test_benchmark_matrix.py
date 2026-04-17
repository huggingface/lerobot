#!/usr/bin/env python

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

import json

from benchmarks.run_benchmark_matrix import (
    PlannedJob,
    compute_gradient_accumulation_steps,
    plan_jobs,
    render_sbatch_script,
    write_manifest,
)


def _one_job(job_list: list[PlannedJob]) -> PlannedJob:
    assert len(job_list) == 1
    return job_list[0]


def test_compute_gradient_accumulation_steps_for_fixed_effective_batch():
    assert compute_gradient_accumulation_steps(
        effective_batch_size=256,
        num_gpus=8,
        microbatch_per_gpu=32,
    ) == 1
    assert compute_gradient_accumulation_steps(
        effective_batch_size=256,
        num_gpus=4,
        microbatch_per_gpu=32,
    ) == 2
    assert compute_gradient_accumulation_steps(
        effective_batch_size=256,
        num_gpus=1,
        microbatch_per_gpu=32,
    ) == 8


def test_plan_jobs_filters_libero_plus_only(tmp_path):
    jobs = plan_jobs(
        output_dir=tmp_path,
        hub_org="lerobot",
        results_repo="lerobot/benchmark-history",
        policies=["pi0", "act"],
        benchmarks=["libero_plus"],
    )

    assert [job.benchmark for job in jobs] == ["libero_plus", "libero_plus"]
    assert [job.policy for job in jobs] == ["pi0", "act"]


def test_plan_jobs_includes_libero_plus_and_robomme(tmp_path):
    jobs = plan_jobs(
        output_dir=tmp_path,
        hub_org="lerobot",
        results_repo="lerobot/benchmark-history",
        policies=["pi0"],
        benchmarks=["libero_plus", "robomme"],
    )

    assert [job.benchmark for job in jobs] == ["libero_plus", "robomme"]
    assert jobs[0].effective_batch_size == 256
    assert jobs[1].effective_batch_size == 256


def test_plan_jobs_sets_expected_gpu_and_accumulation(tmp_path):
    jobs = plan_jobs(
        output_dir=tmp_path,
        hub_org="lerobot",
        results_repo="lerobot/benchmark-history",
        policies=["pi0", "xvla", "act"],
        benchmarks=["robomme"],
    )
    by_policy = {job.policy: job for job in jobs}

    assert by_policy["pi0"].num_gpus == 8
    assert by_policy["pi0"].gradient_accumulation_steps == 1
    assert by_policy["xvla"].num_gpus == 4
    assert by_policy["xvla"].gradient_accumulation_steps == 2
    assert by_policy["act"].num_gpus == 1
    assert by_policy["act"].gradient_accumulation_steps == 8


def test_render_sbatch_script_contains_train_eval_and_publish(tmp_path):
    job = _one_job(
        plan_jobs(
            output_dir=tmp_path,
            hub_org="lerobot",
            results_repo="lerobot/benchmark-history",
            policies=["pi0_fast"],
            benchmarks=["robomme"],
        )
    )

    script = render_sbatch_script(
        job=job,
        output_dir=tmp_path,
        results_repo_id="lerobot/benchmark-history",
        git_commit="deadbeef",
    )

    assert "docker/Dockerfile" not in script
    assert "lerobot-benchmark-robomme:latest" in script
    assert '--dataset.repo_id="lerobot/robomme"' in script
    assert '--env.type="robomme"' in script
    assert "--gradient_accumulation_steps=1" in script
    assert "lerobot-train-tokenizer" in script
    assert "benchmarks/publish_benchmark_result.py" in script


def test_write_manifest_records_job_metadata(tmp_path):
    jobs = plan_jobs(
        output_dir=tmp_path,
        hub_org="lerobot",
        results_repo="lerobot/benchmark-history",
        policies=["pi0"],
        benchmarks=["libero_plus", "robomme"],
    )
    manifest_path = write_manifest(
        output_dir=tmp_path,
        jobs=jobs,
        git_commit="deadbeef",
        hub_org="lerobot",
        results_repo="lerobot/benchmark-history",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_commit"] == "deadbeef"
    assert manifest["results_repo"] == "lerobot/benchmark-history"
    assert [job["benchmark"] for job in manifest["jobs"]] == ["libero_plus", "robomme"]
