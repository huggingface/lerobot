# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Versioned, inspectable subprocess jobs using LeRobot's existing CLIs."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from lerobot.datasets.recipe import TrainingRecipe


class ExperimentStore:
    def __init__(self, root: str | Path, workspace: str | Path):
        self.root = Path(root).resolve()
        self.workspace = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.git = shutil.which("git")
        if self.git is None:
            raise RuntimeError("Git is required to track experiment code")
        self._processes = {}
        self._lock = threading.RLock()

    def _name(self, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
            raise ValueError("Names may contain only letters, digits, underscores, and hyphens")
        return name

    def candidates(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("candidates/*.json"))]

    def create_candidate(self, name, base_model, training, recipe, parent=None):
        """A candidate records the complete training config and recipe, not just a mutable filename."""
        name = self._name(name)
        path = self.root / "candidates" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        recipe = TrainingRecipe.from_dict(recipe)
        if not base_model or not isinstance(training, dict):
            raise ValueError("A base model and training configuration are required")
        if not training.get("dataset", {}).get("repo_id"):
            raise ValueError("training.dataset.repo_id is required")
        training = json.loads(json.dumps(training))
        training.setdefault("dataset", {})["task_recipe"] = asdict(recipe)
        training.setdefault("policy", {})["push_to_hub"] = False
        candidate = {
            "name": name,
            "base_model": base_model,
            "training": training,
            "parent": parent,
            "created_at": time.time(),
        }
        with path.open("x") as stream:
            json.dump(candidate, stream, indent=2)
        return candidate

    def _save(self, directory, state):
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(directory / "job.json")

    def launch(self, kind, argv, *, cwd=None, metadata=None):
        """Launch an argument vector without a shell; capture logs, config, and code provenance."""
        with self._lock:
            job_id = uuid.uuid4().hex
            directory = self.root / "jobs" / job_id
            directory.mkdir(parents=True)
            cwd = Path(cwd or self.workspace).resolve()
            revision = subprocess.run(
                [self.git, "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
            ).stdout.strip()
            patch = subprocess.run(
                [self.git, "diff", "HEAD", "--binary"], cwd=cwd, capture_output=True, check=True
            ).stdout
            (directory / "code.patch").write_bytes(patch)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(cwd / "src") + os.pathsep + env.get("PYTHONPATH", "")
            state = {
                "id": job_id,
                "kind": kind,
                "argv": argv,
                "cwd": str(cwd),
                "revision": revision,
                "status": "running",
                "started_at": time.time(),
                "metadata": metadata or {},
                "log_path": str(directory / "output.log"),
            }
            with (directory / "output.log").open("wb") as log:
                process = subprocess.Popen(
                    argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            state["pid"] = process.pid
            self._processes[job_id] = process
            self._save(directory, state)
            return state

    def start_training(self, candidate_id):
        candidate_id = self._name(candidate_id)
        candidate = json.loads((self.root / "candidates" / f"{candidate_id}.json").read_text())
        training = candidate["training"]
        run_id = uuid.uuid4().hex
        output = self.root / "checkpoints" / candidate_id / run_id
        argv = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--policy.path={candidate['base_model']}",
        ]
        for key, value in training.items():
            if key in ("output_dir", "resume", "job"):
                raise ValueError(f"{key} is owned by the experiment runner")
            if isinstance(value, dict):
                for field, item in value.items():
                    if key == "policy" and field in ("path", "pretrained_path", "type"):
                        raise ValueError("Choose base_model instead of overriding policy path/type")
                    argv.append(f"--{key}.{field}={json.dumps(item) if not isinstance(item, str) else item}")
            else:
                argv.append(f"--{key}={json.dumps(value) if not isinstance(value, str) else value}")
        argv.append(f"--output_dir={output}")
        return self.launch("train", argv, metadata={"candidate": candidate, "output_dir": str(output)})

    def status(self, job_id, tail_chars=4000):
        self._name(job_id)
        with self._lock:
            directory = self.root / "jobs" / job_id
            state = json.loads((directory / "job.json").read_text())
            process = self._processes.get(job_id)
            if process is not None:
                returncode = process.poll()
                if returncode is not None and state["status"] == "running":
                    state.update(
                        status="completed" if returncode == 0 else "failed",
                        returncode=returncode,
                        finished_at=time.time(),
                    )
                    self._save(directory, state)
            elif state["status"] == "running":
                # Never assume that an old PID still names our training process after restart.
                state["status"] = "unknown_after_restart"
            path = directory / "output.log"
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - max(0, min(tail_chars, 16000))))
                state["log_tail"] = log.read().decode(errors="replace")
            return state

    def cancel(self, job_id):
        with self._lock:
            process = self._processes.get(job_id)
            if process is None:
                raise ValueError("Can only cancel a job owned by this service instance")
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            return self.status(job_id)

    def create_code_candidate(self, name, patch):
        """Apply proposed code in a separate checkout; never change a running robot process."""
        name = self._name(name)
        directory = self.root / "code_candidates" / name
        directory.parent.mkdir(exist_ok=True)
        patch_path = directory.parent / f"{name}.patch"
        with patch_path.open("x") as stream:
            stream.write(patch)
        subprocess.run(
            [self.git, "worktree", "add", "--detach", str(directory), "HEAD"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [self.git, "apply", "--check", str(patch_path)],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [self.git, "apply", str(patch_path)], cwd=directory, capture_output=True, text=True, check=True
        )
        return {"name": name, "workspace": str(directory), "status": "unvalidated"}

    def test_code_candidate(self, name):
        directory = self.root / "code_candidates" / self._name(name)
        if not directory.is_dir():
            raise ValueError("Unknown code candidate")
        return self.launch(
            "test",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/rollout_agent",
                "tests/datasets/test_language_task.py",
                "-q",
            ],
            cwd=directory,
        )


def compare_runs(baseline, candidate):
    """Keep unknown outcomes in the denominator and show intervention use alongside success."""

    def metrics(episodes):
        total = len(episodes)
        successes = sum(item["outcome"] == "success" for item in episodes)
        return {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else None,
            "unknown": sum(item["outcome"] == "unknown" for item in episodes),
            "label_sources": dict(Counter(item.get("label_source", "unspecified") for item in episodes)),
            "intervened_episodes": sum(item.get("intervention_frames", 0) > 0 for item in episodes),
            "policy_frames": sum(item.get("policy_frames", 0) for item in episodes),
            "total_frames": sum(item.get("frames", 0) for item in episodes),
        }

    return {
        "baseline": metrics(baseline),
        "candidate": metrics(candidate),
        "promotion": "requires matched physical evaluation with a fixed outcome rubric",
    }
