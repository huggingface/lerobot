# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import draccus
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.rollout.agent.experiments import ExperimentStore, compare_runs


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", str(workspace)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "initial",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return ExperimentStore(tmp_path / "experiments", workspace)


@pytest.mark.parametrize("candidate_index,policy_type", [(0, "smolvla"), (1, "pi05")])
def test_training_job_uses_saved_recipe_and_argument_vector(store, monkeypatch, candidate_index, policy_type):
    config = json.loads((Path(__file__).parents[2] / "examples/rebot_agent/session.json").read_text())
    candidate = config["candidates"][candidate_index]
    recipe = TrainingRecipe.from_yaml(Path(__file__).parents[2] / candidate["recipe_path"])
    store.create_candidate(candidate["name"], candidate["base_model"], candidate["training"], asdict(recipe))
    calls = []
    monkeypatch.setattr(store, "launch", lambda kind, argv, **kw: calls.append((kind, argv, kw)))
    store.start_training(candidate["name"])
    kind, argv, metadata = calls[0]
    assert kind == "train"
    assert f"--policy.path={candidate['base_model']}" in argv
    assert any(arg.startswith("--dataset.task_recipe=") for arg in argv)
    # Parse every generated training flag without downloading a model or running training.
    parse_args = [arg for arg in argv[3:] if not arg.startswith("--policy.path=")]
    parsed = draccus.parse(TrainPipelineConfig, args=[f"--policy.type={policy_type}", *parse_args])
    assert parsed.dataset.task_recipe["blend"]["subtask"]["weight"] == 0.8
    assert parsed.eval_steps == 1000
    assert parsed.dataset.revision == "93c97807c46535745d0587d4296416bf2d4aa80d"
    assert metadata["metadata"]["candidate"]["base_model"] == candidate["base_model"]


def test_jobs_capture_exit_status_and_logs(store):
    state = store.launch("test", [sys.executable, "-c", "print('job ran'); raise SystemExit(3)"])
    deadline = time.monotonic() + 5
    while state["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
        state = store.status(state["id"])
    assert state["status"] == "failed"
    assert state["returncode"] == 3
    assert "job ran" in state["log_tail"]


def test_candidate_names_cannot_escape_store(store):
    with pytest.raises(ValueError):
        store.create_code_candidate("../escape", "")


def test_unknown_outcomes_remain_in_success_denominator():
    result = compare_runs(
        [{"outcome": "success"}, {"outcome": "unknown"}], [{"outcome": "failure", "intervention_frames": 8}]
    )
    assert result["baseline"]["success_rate"] == 0.5
    assert result["candidate"]["intervened_episodes"] == 1
    assert result["promotion"].startswith("requires")
