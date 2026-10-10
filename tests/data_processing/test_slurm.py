# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lerobot.data_processing.configs import RuntimeConfig
from lerobot.data_processing.runtime import finalize_stage
from lerobot.data_processing.types import Resources
from lerobot.data_processing.worker import run_worker_group
from lerobot.jobs.processing_worker import assigned_shards, main
from lerobot.jobs.slurm import render_slurm, run_slurm

pytest.importorskip("pyarrow")
from lerobot.data_processing.artifacts import ArtifactStore  # noqa: E402
from tests.data_processing.test_runtime import make_plan  # noqa: E402


def test_cpu_worker_package_imports_no_model():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lerobot.jobs.slurm; assert 'torch' not in sys.modules; assert 'openai' not in sys.modules",
        ],
        check=True,
    )


def test_assignment_and_internal_worker_roundtrip(tmp_path, capsys):
    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=7, shard_size=1)
    assert [assigned_shards(7, 3, index) for index in range(3)] == [[0, 3, 6], [1, 4], [2, 5]]
    with pytest.raises(ValueError):
        assigned_shards(7, 3, 3)
    for index in range(3):
        main(
            [
                "worker",
                "--run-uri",
                store.uri,
                "--plan-id",
                plan.plan_id,
                "--groups",
                "3",
                "--worker-index",
                str(index),
            ]
        )
    main(["finalize", "--run-uri", store.uri, "--plan-id", plan.plan_id])
    assert json.loads(capsys.readouterr().out)["completed"] == 7


def runtime(tmp_path):
    cfg = RuntimeConfig(backend="slurm", workers=3)
    cfg.slurm.script_dir = tmp_path / "scripts with spaces"
    return cfg


def test_render_cpu_gpu_and_1024_groups(tmp_path, monkeypatch):
    from lerobot.jobs import slurm

    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=1025, shard_size=1)
    cfg = runtime(tmp_path)
    cfg.workers = 1024
    scripts = render_slurm(store, plan, cfg)
    assert scripts.groups == 1024
    assert scripts.resources.gpus == 0
    assert "--groups 1024" in scripts.worker.read_text()
    assert "processing_worker worker" in scripts.worker.read_text()
    assert render_slurm(store, plan, cfg) == scripts
    monkeypatch.setattr(
        slurm,
        "load_module",
        lambda *args: SimpleNamespace(spec=SimpleNamespace(resources=Resources(cpus=4, gpus=1, memory_gb=8))),
    )
    assert render_slurm(store, plan, cfg).resources.gpus == 1


@pytest.mark.parametrize("gpus", [0, 1])
def test_sbatch_dispatch_dependencies_real_worker_outputs_and_resume(tmp_path, monkeypatch, gpus):
    from lerobot.jobs import slurm

    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=5, shard_size=1)
    cfg = runtime(tmp_path)
    original_loader = slurm.load_module

    def resources(*args):
        from dataclasses import replace

        module = original_loader(*args)
        module.spec = replace(module.spec, resources=Resources(cpus=2, gpus=gpus, memory_gb=8))
        return module

    monkeypatch.setattr(slurm, "load_module", resources)
    monkeypatch.setenv("SBATCH_GRES", "gpu:99")
    commands = []
    real_run = subprocess.run

    def submit(command, **kwargs):
        if command[0] != "sbatch":
            return real_run(command, **kwargs)
        commands.append(command)
        assert "SBATCH_GRES" not in kwargs["env"]
        if "--wait" not in command:
            for index in range(3):
                run_worker_group(store.uri, plan.plan_id, assigned_shards(plan.shards, 3, index), 1, 0)
            return SimpleNamespace(stdout="123;cluster\n")
        finalize_stage(store, plan)
        return SimpleNamespace(stdout="124\n")

    monkeypatch.setattr(slurm.subprocess, "run", submit)
    result = run_slurm(store, plan, cfg)
    assert result.completed == 5
    assert "--array=0-2%3" in commands[0]
    assert "--cpus-per-task=2" in commands[0]
    assert any(arg.startswith("--gpus") for arg in commands[0]) == bool(gpus)
    assert "--dependency=afterok:123" in commands[1]
    assert "--kill-on-invalid-dep=yes" in commands[1]
    assert run_slurm(store, plan, cfg) == result
    assert len(commands) == 2


def test_active_submission_cannot_be_duplicated(tmp_path, monkeypatch):
    from lerobot.jobs import slurm

    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store)
    store.put_json(f"{plan.prefix}/slurm/submission.json", {"array_job_id": "123"})

    def active(command, **kwargs):
        assert command[0] == "squeue"
        return SimpleNamespace(stdout="123_0\n")

    monkeypatch.setattr(slurm.subprocess, "run", active)
    with pytest.raises(RuntimeError, match="still active"):
        run_slurm(store, plan, runtime(tmp_path))


def test_process_local_storage_is_rejected(tmp_path):
    store = ArtifactStore("memory://slurm-local-only")
    plan = make_plan(store)
    with pytest.raises(ValueError, match="process-local"):
        render_slurm(store, plan, runtime(tmp_path))


def test_generated_shell_executes_with_spaces(tmp_path):
    store = ArtifactStore(tmp_path / "run with spaces")
    plan = make_plan(store, size=3, shard_size=1)
    cfg = runtime(tmp_path)
    scripts = render_slurm(store, plan, cfg)
    for index in range(3):
        subprocess.run(
            ["/bin/sh", str(scripts.worker)],
            env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index)},
            check=True,
        )
    result = subprocess.run(["/bin/sh", str(scripts.finalizer)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["completed"] == 3
