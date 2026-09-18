# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Local agent tools attached to LeRobot's interactive language rollout entry point."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import sys
import threading
import uuid
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.recipe import TrainingRecipe

from .control import ControlConfig, HybridRuntime
from .experiments import ExperimentStore, compare_runs
from .provider import OpenAIReasoner, public_snapshot
from .robot_io import CartesianTools, RealRobotIO
from .tools import CONTROL_TOOLS, TOOLS

CONTROL_NAMES = {tool["name"] for tool in CONTROL_TOOLS} - {"get_observation", "get_action"}


class AgentHarness:
    def __init__(self, config):
        self.config = config
        self.workspace = Path(config.get("workspace", Path(__file__).resolve().parents[4])).resolve()
        self.root = Path(config["root"]).resolve()
        self.experiments = ExperimentStore(self.root, self.workspace)
        self.policies_path = self.root / "policies.json"
        self.policies = json.loads(self.policies_path.read_text()) if self.policies_path.exists() else {}
        self.runtime = None
        self._thread = None
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._session = {"status": "idle"}
        self._operator_paused = True
        self._session_stop = threading.Event()
        self._agent_history = deque(maxlen=20)
        self._automatic = None
        for candidate in config.get("candidates", []):
            path = self.root / "candidates" / f"{candidate['name']}.json"
            if not path.exists():
                recipe = asdict(TrainingRecipe.from_yaml(self.workspace / candidate["recipe_path"]))
                self.experiments.create_candidate(
                    candidate["name"], candidate["base_model"], candidate["training"], recipe
                )
        provider = config.get("supervisor")
        if provider:
            provider = dict(provider)
            if prompt_path := provider.pop("prompt_path", None):
                provider["prompt"] = (self.workspace / prompt_path).read_text()
            self._automatic = OpenAIReasoner(**provider)
            threading.Thread(target=self._agent_loop, daemon=True).start()

    def status(self):
        with self._lock:
            return {
                "session": dict(self._session),
                "runtime": self.runtime.status() if self.runtime else None,
                "policies": dict(self.policies),
                "operator_paused": self._operator_paused,
            }

    def call(self, name, arguments=None, *, actor="external_agent", revision=None, observed_at=None):
        args = arguments or {}
        if name == "status":
            return self.status()
        if name == "get_observation":
            return public_snapshot(self.runtime.snapshot()) if self.runtime else None
        if name == "get_action":
            snapshot = self.runtime.snapshot() if self.runtime else None
            return snapshot.get("proposal") if snapshot else None
        if name in CONTROL_NAMES:
            runtime = self.runtime
            if runtime is None or self._session["status"] != "running":
                raise ValueError("No active real-robot session")
            if actor == "supervisor" and self._operator_paused and name != "pause":
                raise ValueError("Operator paused; await an explicit operator resume")
            if name == "pause" and actor != "supervisor":
                self._operator_paused = True
            snapshot = runtime.snapshot()
            if revision is None and snapshot is not None:
                revision = snapshot["revision"]
            if observed_at is None and snapshot is not None:
                observed_at = snapshot["observed_at"]
            command = runtime.submit(name, args, actor=actor, revision=revision, observed_at=observed_at)
            if not command["done"].wait(timeout=10):
                # Do not replay a physical command after a transport timeout.
                return {
                    "call_id": command["id"],
                    "status": "pending",
                    "message": "Inspect status; do not retry motion",
                }
            result = command["result"]
            if "error" not in result:
                if name in ("set_task", "steer", "resume_policy") and actor != "supervisor":
                    self._operator_paused = False
                if name == "finish_episode":
                    self._operator_paused = True
            return result
        if name == "list_jobs":
            return [
                self.experiments.status(path.parent.name, tail_chars=0)
                for path in sorted((self.root / "jobs").glob("*/job.json"))
            ]
        if name == "read_code":
            path = (self.workspace / args["path"]).resolve()
            if not path.is_relative_to(self.workspace):
                raise ValueError("Source path must be inside the configured workspace")
            start = max(1, int(args.get("start_line", 1)))
            count = max(1, min(300, int(args.get("max_lines", 150))))
            with path.open() as file:
                lines = list(islice(file, start - 1, start - 1 + count))
            return {"path": str(path), "start_line": start, "text": "".join(lines)}
        if name == "get_run_trace":
            run = self.experiments._name(args["run_id"])
            count = max(1, min(200, int(args.get("max_events", 50))))
            with (self.root / "runs" / run / "events.jsonl").open() as file:
                return [json.loads(line) for line in deque(file, maxlen=count)]
        if name == "list_candidates":
            return self.experiments.candidates()
        if name == "create_candidate":
            return self.experiments.create_candidate(
                args["name"],
                args["base_model"],
                json.loads(args["training_json"]),
                json.loads(args["recipe_json"]),
                args.get("parent"),
            )
        if name == "start_training":
            if self._thread and self._thread.is_alive():
                raise ValueError("Stop the physical rollout before starting training on this host")
            return self.experiments.start_training(args["candidate_id"])
        if name == "job_status":
            return self.experiments.status(args["job_id"])
        if name == "cancel_job":
            return self.experiments.cancel(args["job_id"])
        if name == "register_policy":
            key = self.experiments._name(args["name"])
            config = PreTrainedConfig.from_pretrained(args["checkpoint"])
            expected_dim = self.config.get("action_dim")
            action = (config.output_features or {}).get("action")
            if expected_dim is not None and (action is None or action.shape[-1] != expected_dim):
                raise ValueError("Checkpoint action features do not match the configured robot")
            with self._lock:
                if key in self.policies:
                    raise ValueError("Use a new policy name to preserve checkpoint provenance")
                self.policies[key] = {"checkpoint": args["checkpoint"], "type": config.type}
                self.policies_path.write_text(json.dumps(self.policies, indent=2))
            return self.policies[key]
        if name == "start_rollout":
            with self._lock:
                if any("REPLACE_" in arg for arg in self.config.get("robot_args", [])):
                    raise ValueError("Configure the actual arm ports and camera IDs in agent_config first")
                if self._thread and self._thread.is_alive():
                    raise ValueError("Stop the current rollout before selecting another policy")
                if actor == "supervisor":
                    raise ValueError("An operator starts each physical session after resetting the scene")
                policy = self.policies[args["policy_id"]]
                run_id = uuid.uuid4().hex
                self._session = {"status": "loading", "run_id": run_id, "policy_id": args["policy_id"]}
                self._operator_paused = False
                self._session_stop.clear()
                self._thread = threading.Thread(
                    target=self._run_rollout, args=(policy, args["instruction"], run_id), daemon=True
                )
                self._thread.start()
                return dict(self._session)
        if name == "stop_rollout":
            self._session_stop.set()
            if self.runtime:
                self.runtime.stop()
            if self._thread:
                self._thread.join(timeout=10)
            return self.status()
        if name == "build_dagger_dataset":
            if self._thread and self._thread.is_alive():
                raise ValueError("Finalize the rollout before reading correction data")
            spec = json.loads(args["spec_json"])
            path = self.root / f"aggregation-{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(spec, indent=2))
            return self.experiments.launch(
                "aggregate",
                [sys.executable, "-m", "lerobot.rollout.agent.dagger", "--spec", str(path)],
                metadata=spec,
            )
        if name == "compare_runs":

            def read(run):
                run = self.experiments._name(run)
                return [
                    json.loads(line)
                    for line in (self.root / "runs" / run / "episodes.jsonl").read_text().splitlines()
                ]

            return compare_runs(read(args["baseline_run"]), read(args["candidate_run"]))
        if name == "create_code_candidate":
            return self.experiments.create_code_candidate(args["name"], args["patch"])
        if name == "test_code_candidate":
            return self.experiments.test_code_candidate(args["name"])
        raise ValueError(f"Unknown tool: {name}")

    def _run_rollout(self, policy, instruction, run_id):
        robot_io = None
        try:
            control = ControlConfig(**self.config.get("control", {}))
            args = [*self.config["robot_args"], f"--policy.path={policy['checkpoint']}"]
            run_root = self.root / "runs" / run_id
            robot_io = RealRobotIO(args, root=run_root, repo_id=f"local/{run_id}", fps=control.fps)
            if not control.joint_limits:
                control.joint_limits = robot_io.limits()
            ik = CartesianTools(self.config["ik"]) if self.config.get("ik") else None
            runtime = HybridRuntime(
                predict=robot_io.predict,
                observe=robot_io.observation_provider,
                execute=robot_io.execute,
                config=control,
                record=robot_io.record,
                finish=robot_io.finish,
                event=robot_io.event,
                solve_ik=ik,
            )
            with self._lock:
                self.runtime = runtime
                self._session["status"] = "running"
            (run_root / "session.json").write_text(
                json.dumps(
                    {
                        "policy": policy,
                        "config": self.config,
                        "control": asdict(control),
                        "instruction": instruction,
                    },
                    indent=2,
                )
            )
            runtime.set_task(instruction)
            runtime.state.mode = "action"
            if self._session_stop.is_set() or self._shutdown.is_set():
                runtime.stop()
            runtime.run()
            self._session["status"] = "completed"
        except Exception as exc:
            self._session.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if self.runtime:
                self.runtime.stop()
            if robot_io:
                robot_io.close()

    def _agent_loop(self):
        while not self._shutdown.wait(1):
            try:
                # Preserve explicit operator reset/pause boundaries. Tools for experiments
                # remain available to the external agent even when autonomous polling is idle.
                if not self.runtime or self._operator_paused or self._session["status"] != "running":
                    continue
                snapshot = self.runtime.snapshot()
                if snapshot is None:
                    continue
                calls = self._automatic.decide(
                    snapshot,
                    TOOLS,
                    {
                        "status": self.status(),
                        "recent_tools": list(self._agent_history),
                        "joint_limits": self.runtime.config.joint_limits,
                        "ik": self.config.get("ik", {}),
                    },
                )
                for call in calls:
                    result = self.call(
                        call["name"],
                        call["arguments"],
                        actor="supervisor",
                        revision=snapshot["revision"],
                        observed_at=snapshot["observed_at"],
                    )
                    self._agent_history.append({"call": call, "result": result})
            except Exception as exc:
                self._agent_history.append({"error": f"{type(exc).__name__}: {exc}"})

    def close(self):
        self._shutdown.set()
        self.call("stop_rollout")


def make_server(harness, port, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reply(self, code, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def do_GET(self):
            if not self._authorized():
                return self._reply(401, {"error": "Authentication required"})
            if self.path == "/tools":
                return self._reply(200, TOOLS)
            return self._reply(200, harness.status())

        def do_POST(self):
            if not self._authorized():
                return self._reply(401, {"error": "Authentication required"})
            if self.path != "/call":
                return self._reply(404, {"error": "Unknown endpoint"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                result = harness.call(
                    request["name"],
                    request.get("arguments", {}),
                    revision=request.get("revision"),
                    observed_at=request.get("observed_at"),
                )
                return self._reply(200, {"result": result})
            except Exception as exc:
                return self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_config", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.agent_config).read_text())
    harness = AgentHarness(config)
    token = secrets.token_urlsafe(32)
    token_path = harness.root / "tool_token"
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(token)
    server = make_server(harness, int(config.get("port", 8767)), token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"LeRobot tools: http://127.0.0.1:{server.server_port}; token file: {token_path}")
    print(
        "/rollout POLICY INSTRUCTION | /train CANDIDATE | /pause | /resume | /finish OUTCOME EVIDENCE | /stop | /quit"
    )
    try:
        if args.headless:
            while not harness._shutdown.wait(1):
                pass
        else:
            while True:
                line = input("robot> ").strip()
                if line == "/quit":
                    break
                try:
                    if line.startswith("/rollout "):
                        _, policy, instruction = line.split(" ", 2)
                        result = harness.call(
                            "start_rollout",
                            {"policy_id": policy, "instruction": instruction},
                            actor="operator",
                        )
                    elif line.startswith("/train "):
                        result = harness.call(
                            "start_training", {"candidate_id": line.split(maxsplit=1)[1]}, actor="operator"
                        )
                    elif line.startswith("/finish "):
                        _, outcome, evidence = line.split(" ", 2)
                        result = harness.call(
                            "finish_episode", {"outcome": outcome, "evidence": evidence}, actor="operator"
                        )
                    elif line in ("/pause", "/resume", "/stop", "/status"):
                        name = {
                            "/pause": "pause",
                            "/resume": "resume_policy",
                            "/stop": "stop_rollout",
                            "/status": "status",
                        }[line]
                        result = harness.call(name, actor="operator")
                    elif line:
                        result = harness.call("set_task", {"instruction": line}, actor="operator")
                    else:
                        continue
                    print(json.dumps(result, indent=2))
                except Exception as exc:
                    print(f"{type(exc).__name__}: {exc}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        harness.close()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
