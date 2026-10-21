"""Temporary, pinned camera-map/prompt comparison for PR 4860."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

PR_HEAD = "d3bbc288b93e2b8eaa61f63cc52c4f8498c70407"
PR_BASE = "200ee53596d464bd28f6595cdb31be69a2b5e379"
MODEL = "lerobot/smolvla_robocasa"
MODEL_REVISION = "f7af588d11e30b0069f200792fb285904b74c80b"
TASKS = ["CloseFridge", "OpenDrawer", "TurnOnSinkFaucet"]
SEEDS = [500, 501, 502]
MAPS = {
    "old": {
        "observation.images.robot0_agentview_left": "observation.images.camera1",
        "observation.images.robot0_eye_in_hand": "observation.images.camera2",
        "observation.images.robot0_agentview_right": "observation.images.camera3",
    },
    "fixed": {
        "observation.images.robot0_agentview_left": "observation.images.camera1",
        "observation.images.robot0_agentview_right": "observation.images.camera2",
        "observation.images.robot0_eye_in_hand": "observation.images.camera3",
    },
}


def episode_main():
    import numpy as np

    import lerobot
    from lerobot.envs.robocasa import RoboCasaEnv
    from lerobot.scripts.lerobot_eval import main

    expected = Path.cwd() / "src/lerobot/__init__.py"
    assert Path(lerobot.__file__).resolve() == expected.resolve(), lerobot.__file__
    original_reset = RoboCasaEnv.reset
    trace_file = Path(os.environ["EPISODE_DIR"]) / "reset_trace.jsonl"

    def traced_reset(self, *args, **kwargs):
        obs, info = original_reset(self, *args, **kwargs)
        original_prompt = self.task_description
        if os.environ["PROMPT_MODE"] == "task_name":
            self.task_description = self.task
        hashes = {}
        for key, value in {**obs.get("pixels", {}), "agent_pos": obs.get("agent_pos")}.items():
            if value is not None:
                array = np.ascontiguousarray(value)
                hashes[key] = {
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
        with trace_file.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "task": self.task,
                        "seed": kwargs.get("seed"),
                        "original_prompt": original_prompt,
                        "policy_prompt": self.task_description,
                        "observations": hashes,
                    }
                )
                + "\n"
            )
        return obs, info

    RoboCasaEnv.reset = traced_reset
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    main()


def compare_main():
    from huggingface_hub import snapshot_download

    map_mode, prompt_mode, output = sys.argv[1:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(Path("scripts/ci/robocasa_4860_source.json").read_text())
    for name, digest in source_manifest["source_files"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)
    assert MAPS["fixed"] == json.loads((Path(model_path) / "train_config.json").read_text())["rename_map"]
    manifest = {
        "pr_head": PR_HEAD,
        "pr_merge_base": PR_BASE,
        "code_head": source_manifest["code_head"],
        "source_changes": source_manifest["source_changes"],
        "source_tree": source_manifest["source_tree"],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_path": model_path,
        "map_mode": map_mode,
        "rename_map": MAPS[map_mode],
        "prompt_mode": prompt_mode,
        "split": "target",
        "obj_registries": ["lightwheel"],
        "tasks": TASKS,
        "seeds": SEEDS,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "mujoco", "numpy", "gymnasium")
        },
        "gpu": subprocess.check_output(
            [
                shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "simulator_commits": {
            name: subprocess.check_output(
                [shutil.which("git") or "/usr/bin/git", "-C", str(Path.home() / name), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            for name in ("robocasa", "robosuite")
        },
    }
    assert not manifest["source_changes"], "The benchmark must use the PR's unchanged src/"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    records = []
    for task in TASKS:
        for seed in SEEDS:
            episode_dir = output / f"{task}-{seed}"
            episode_dir.mkdir()
            command = [
                sys.executable,
                __file__,
                "--episode",
                f"--policy.path={model_path}",
                "--policy.device=cuda",
                "--env.type=robocasa",
                f"--env.task={task}",
                "--env.split=target",
                "--env.max_parallel_tasks=1",
                "--eval.batch_size=1",
                "--eval.n_episodes=1",
                "--eval.use_async_envs=false",
                f"--seed={seed}",
                f"--rename_map={json.dumps(MAPS[map_mode])}",
                f"--output_dir={episode_dir}",
            ]
            (episode_dir / "command.json").write_text(json.dumps(command, indent=2))
            env = {
                **os.environ,
                "PROMPT_MODE": prompt_mode,
                "EPISODE_DIR": str(episode_dir),
                "PYTHONHASHSEED": str(seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
            print(f"START {map_mode}/{prompt_mode} {task} seed={seed}", flush=True)
            with (episode_dir / "eval.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {"task": task, "seed": seed, "returncode": result.returncode}
            try:
                assert result.returncode == 0, f"eval exited {result.returncode}"
                info = json.loads((episode_dir / "eval_info.json").read_text())
                assert info["overall"]["n_episodes"] == 1, info["overall"]
                assert info["overall"]["pc_success"] in (0.0, 100.0), info["overall"]
                successes = info["per_task"][0]["metrics"]["successes"]
                assert len(successes) == 1, successes
                record.update(success=bool(successes[0]), overall=info["overall"])
            except Exception:
                record["error"] = traceback.format_exc()
                print((episode_dir / "eval.log").read_text()[-15000:], flush=True)
            records.append(record)
            summary = {
                "manifest": manifest,
                "episodes": records,
                "n_completed": sum("success" in r for r in records),
                "n_success": sum(r.get("success", False) for r in records),
                "n_errors": sum("error" in r for r in records),
            }
            (output / "results.json").write_text(json.dumps(summary, indent=2))
            print("RESULT " + json.dumps(record), flush=True)
    if summary["n_errors"] or summary["n_completed"] != len(TASKS) * len(SEEDS):
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "--episode":
        episode_main()
    else:
        compare_main()
