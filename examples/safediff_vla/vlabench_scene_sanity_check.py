#!/usr/bin/env python
"""VLABench environment sanity cross-check (DIAGNOSTIC ONLY -- no training, no architecture or
checkpoint changes, no performance evaluation).

Context: another dry run ("A") reported empty scenes, floating objects, and odd camera views
when constructing the FULL VLABench task suite. Before starting a full benchmark here, this
script checks scene validity for 3 tasks -- `select_poker` (the existing known-good control used
by every prior eval script on this branch), `select_fruit`, `select_toy` -- via a single `reset()`
each, and additionally cross-checks `select_poker` under TWO different env-construction paths:

  1. "isolated" -- the exact pattern every existing eval script / `tests/envs/test_vlabench.py`
     uses: `make_env(VLABenchEnv(task=X), n_envs=1, use_async_envs=False)` in its own fresh
     process, building exactly one VLABench task per process.
  2. "multitask" -- `lerobot.envs.vlabench.create_vlabench_envs`'s documented comma-separated
     multi-task construction (`task="select_fruit,select_toy,select_poker"`), all three built and
     RESET IN THAT ORDER inside the SAME process -- i.e. by the time `select_poker` is reset here,
     two other VLABench/MuJoCo tasks have already been built and reset in this process. This is
     the one variable a real full-suite driver introduces that the known-good single-task eval
     scripts never exercise (see `VLABenchEnv.__init__`'s own docstring on deferring env
     construction "to avoid inheriting stale GPU/EGL contexts" -- written with `AsyncVectorEnv`'s
     per-task subprocess isolation in mind; a same-process multi-task sequence gets none of that
     isolation).

Each mode is run in its OWN fresh subprocess (`uv run python -m ... --mode ...`) so the "isolated"
baseline is never contaminated by anything this script itself did earlier, and the "multitask"
run's only difference from "isolated" is genuinely the multi-task construction path, not leftover
process state from a previous invocation of this very script.

For each task, records: task name, instantiated task class, object count, object names, target
object, object world xyz, robot base xyz, table/workspace xyz, observation camera name, video
camera name, and saves the first rendered frame (all three camera channels) as PNGs for visual
inspection. Runs validity checks: object(s) actually present, target present, objects resting
within a reasonable band of the table/workspace surface (not floating/penetrating), and camera
name resolution actually found named cameras (not silently falling back to a positional guess).

Usage (single subprocess, called by `run_all` below or directly):
    uv run python examples/safediff_vla/vlabench_scene_sanity_check.py --mode isolated --task select_poker --seed 1000 --out-dir outputs/eval/vlabench_scene_sanity_check
    uv run python examples/safediff_vla/vlabench_scene_sanity_check.py --mode multitask --tasks select_fruit,select_toy,select_poker --seed 1000 --out-dir outputs/eval/vlabench_scene_sanity_check

Orchestrated end-to-end (recommended):
    uv run python examples/safediff_vla/vlabench_scene_sanity_check.py --mode all --out-dir outputs/eval/vlabench_scene_sanity_check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

# Reasonable object-height allowance above/below the workspace's own configured z-range for the
# "floating/penetration" heuristic -- generous on purpose (this is a coarse sanity flag, not a
# precise contact check): real objects (cups, cards, fruit, toys) are a few cm tall at most, and
# the workspace z-range itself already spans table-surface to comfortably-above-table.
Z_SLACK_BELOW = 0.05
Z_SLACK_ABOVE = 0.5


def _entity_xyz(entity: Any, physics: Any) -> list[float] | None:
    try:
        return np.asarray(entity.get_xpos(physics), dtype=float).tolist()
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic only
        logger.warning("get_xpos failed for entity %r: %s", entity, exc)
        return None


def _camera_names(physics: Any) -> list[str]:
    try:
        return [physics.model.camera(i).name for i in range(physics.model.ncam)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to enumerate camera names: %s", exc)
        return []


def inspect_task(raw_env, task_name: str, out_dir: Path, tag: str) -> dict[str, Any]:
    """`raw_env`: the underlying `VLABenchEnv` (gym.Env, `lerobot.envs.vlabench` impl) AFTER
    `reset()` has already been called on its owning vector env. `tag`: unique filename prefix for
    saved frames (distinguishes isolated vs multitask runs, and which task in a multitask run)."""
    env = raw_env._env
    physics = env.physics
    task = env.task

    task_class = f"{type(task).__module__}.{type(task).__name__}"
    entities = dict(task.entities)
    object_names = list(entities.keys())
    object_count = len(object_names)
    object_world_xyz = {name: _entity_xyz(ent, physics) for name, ent in entities.items()}

    target_entity = getattr(task, "target_entity", None)
    target_list = target_entity if isinstance(target_entity, list) else ([target_entity] if target_entity else [])
    target_present = all(t in entities for t in target_list) if target_list else False

    robot_base_xyz = np.asarray(raw_env._robot_base_xyz, dtype=float).tolist() if raw_env._robot_base_xyz is not None else None

    table_entity = entities.get("table")
    table_xyz = _entity_xyz(table_entity, physics) if table_entity is not None else None
    workspace_bounds = list(getattr(task, "workspace", []))  # [minx, maxx, miny, maxy, minz, maxz]

    camera_names_all = _camera_names(physics)
    cam_order = raw_env._camera_order if raw_env._camera_order is not None else [0, 1, 2]
    channel_names = ["image", "second_image", "wrist_image"]

    def cam_name(idx: int | None) -> str | None:
        if idx is None or idx >= len(camera_names_all):
            return None
        return camera_names_all[idx]

    resolved_cameras = {ch: {"index": idx, "name": cam_name(idx)} for ch, idx in zip(channel_names, cam_order, strict=True)}
    camera_resolution_used_fallback = cam_order == [0, 1, 2] and not any(
        s in "".join(camera_names_all).lower() for s in ("wrist", "forward", "front", "right")
    )

    frame = raw_env.render()  # same camera as obs["pixels"]["image"] -- see VLABenchEnv.render()
    obs = raw_env._get_obs()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = {}
    import cv2

    for ch in channel_names:
        arr = obs["pixels"][ch]
        path = frames_dir / f"{tag}_{ch}.png"
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        saved_frames[ch] = str(path)
    assert np.array_equal(frame, obs["pixels"]["image"]), "render() must match obs['pixels']['image'] exactly"

    # ---- validity checks ----
    objects_present = object_count > 0 and all(v is not None for v in object_world_xyz.values())
    if workspace_bounds and len(workspace_bounds) == 6:
        _, _, _, _, z_min, z_max = workspace_bounds
    else:
        z_min, z_max = None, None
    floating_or_penetrating: dict[str, bool] = {}
    if z_min is not None:
        for name, xyz in object_world_xyz.items():
            if name == "table" or xyz is None:
                continue
            z = xyz[2]
            floating_or_penetrating[name] = not (z_min - Z_SLACK_BELOW <= z <= z_max + Z_SLACK_ABOVE)
    any_floating_or_penetrating = any(floating_or_penetrating.values()) if floating_or_penetrating else None

    result = {
        "task_name": task_name,
        "tag": tag,
        "task_class": task_class,
        "object_count": object_count,
        "object_names": object_names,
        "target_entity": target_entity,
        "object_world_xyz": object_world_xyz,
        "robot_base_xyz": robot_base_xyz,
        "table_xyz": table_xyz,
        "workspace_bounds": workspace_bounds,
        "camera_names_all": camera_names_all,
        "resolved_cameras": resolved_cameras,
        "camera_resolution_used_positional_fallback": camera_resolution_used_fallback,
        "video_camera": resolved_cameras.get("image"),  # render()/video uses the same camera as "image"
        "saved_frames": saved_frames,
        "validity": {
            "objects_present": objects_present,
            "target_present": target_present,
            "floating_or_penetrating_per_object": floating_or_penetrating,
            "any_floating_or_penetrating": any_floating_or_penetrating,
            "camera_used_positional_fallback": camera_resolution_used_fallback,
        },
    }
    return result


def run_isolated(task: str, seed: int, out_dir: Path) -> dict[str, Any]:
    from lerobot.envs import make_env
    from lerobot.envs.configs import VLABenchEnv

    envs = make_env(VLABenchEnv(task=task), n_envs=1, use_async_envs=False)
    vec_env = envs[task][0]
    try:
        vec_env.reset(seed=[seed])
        raw_env = vec_env.envs[0]
        return inspect_task(raw_env, task, out_dir, tag=f"isolated_{task}_seed{seed}")
    finally:
        vec_env.close()


def run_multitask(tasks: list[str], seed: int, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Builds all `tasks` via the SAME multi-task construction call (comma-separated `task=` --
    see `create_vlabench_envs`'s docstring), then resets them IN LIST ORDER in this one process --
    the only thing a real full-suite driver does differently from the isolated single-task path."""
    from lerobot.envs import make_env
    from lerobot.envs.configs import VLABenchEnv

    envs = make_env(VLABenchEnv(task=",".join(tasks)), n_envs=1, use_async_envs=False)
    results: dict[str, Any] = {}
    try:
        for task in tasks:
            vec_env = envs[task][0]
            vec_env.reset(seed=[seed])
            raw_env = vec_env.envs[0]
            results[task] = inspect_task(raw_env, task, out_dir, tag=f"multitask_{task}_seed{seed}")
    finally:
        for task in tasks:
            with contextlib.suppress(Exception):
                envs[task][0].close()
    return results


def _run_subprocess(mode: str, out_dir: Path, task: str | None = None, tasks: str | None = None, seed: int = 1000) -> dict[str, Any]:
    """Runs this script's own `--mode isolated`/`--mode multitask` in a fresh `uv run python`
    subprocess (own interpreter, own process) so results never carry over leftover state between
    runs -- see module docstring for why that isolation matters here specifically."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--seed", str(seed), "--out-dir", str(out_dir)]
    if task:
        cmd += ["--task", task]
    if tasks:
        cmd += ["--tasks", tasks]
    logger.info("=== subprocess: %s ===", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    if proc.returncode != 0:
        logger.error("subprocess FAILED (mode=%s task=%s tasks=%s):\nSTDOUT(tail):\n%s\nSTDERR(tail):\n%s",
                      mode, task, tasks, proc.stdout[-4000:], proc.stderr[-4000:])
        return {"error": True, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    # The subprocess writes its own result JSON; load it back.
    result_path = out_dir / f"result_{mode}_{task or tasks.replace(',', '-')}_seed{seed}.json"
    return json.loads(result_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["isolated", "multitask", "all"], required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--tasks", default=None, help="comma-separated, e.g. select_fruit,select_toy,select_poker")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="outputs/eval/vlabench_scene_sanity_check")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "isolated":
        assert args.task, "--task required for --mode isolated"
        result = run_isolated(args.task, args.seed, out_dir)
        (out_dir / f"result_isolated_{args.task}_seed{args.seed}.json").write_text(json.dumps(result, indent=2, default=str))
        logger.info("wrote result_isolated_%s_seed%d.json", args.task, args.seed)
        return

    if args.mode == "multitask":
        assert args.tasks, "--tasks required for --mode multitask"
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        result = run_multitask(tasks, args.seed, out_dir)
        (out_dir / f"result_multitask_{'-'.join(tasks)}_seed{args.seed}.json").write_text(json.dumps(result, indent=2, default=str))
        logger.info("wrote result_multitask_%s_seed%d.json", "-".join(tasks), args.seed)
        return

    # --mode all: orchestrate every subprocess and write a combined report.
    seed = args.seed
    report: dict[str, Any] = {"seed": seed, "isolated": {}, "multitask": {}}
    for task in ("select_poker", "select_fruit", "select_toy"):
        report["isolated"][task] = _run_subprocess("isolated", out_dir, task=task, seed=seed)
    report["multitask"] = _run_subprocess("multitask", out_dir, tasks="select_fruit,select_toy,select_poker", seed=seed)

    (out_dir / "combined_report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote %s", out_dir / "combined_report.json")


if __name__ == "__main__":
    main()
