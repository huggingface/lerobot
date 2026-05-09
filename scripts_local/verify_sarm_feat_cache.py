"""Identity + speedup verifier for the SARMRewardProcessorStep ring-buf CLIP cache.

Feeds an offline LeRobotDataset through SARMRewardProcessorStep.__call__ as if
each frame were a live env step. Two runs per ep:
  - cache OFF (original behaviour: re-encode whole window every eval)
  - cache ON  (encode once on push, lookup features on eval)

Per env step: blocks on the async future right after __call__ so info.sarm_progress
is fresh (deterministic). Records the sequence and compares.

Speedup is measured per run (wall clock for the same number of frames).

Sweeps eval_every_n_steps in {1, 2, 5}.
"""
from __future__ import annotations
import argparse
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.processor.reward_model.sarm import SARMRewardConfig, SARMRewardProcessorStep


def make_proc(args, *, feat_cache_enabled: bool, eval_every_n: int) -> SARMRewardProcessorStep:
    cfg = SARMRewardConfig(
        type=args.sarm_type,
        pretrained_path=args.pretrained,
        device=args.device,
        task=args.task,
        head_mode=args.head_mode,
        reward_mode="dense",  # any; we only inspect info.sarm_progress
        success_threshold=2.0,  # avoid early termination
        stats_dataset_repo_id=args.stats,
        eval_every_n_steps=eval_every_n,
        feat_cache_enabled=feat_cache_enabled,
    )
    proc = SARMRewardProcessorStep(config=cfg, terminate_on_success=False)
    return proc


def run_episode(
    proc: SARMRewardProcessorStep,
    frames: list[dict[str, torch.Tensor]],
) -> list[float]:
    """Feed frames through proc.__call__, blocking on the async future after each
    call so info.sarm_progress is the fresh value (deterministic comparison)."""
    proc.reset()
    progresses: list[float] = []
    for i, obs in enumerate(frames):
        is_last = i == len(frames) - 1
        transition: EnvTransition = {
            TransitionKey.OBSERVATION: obs,
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: is_last,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
        }
        out = proc(transition)
        # Block on pending eval → flush future into _last_progress so the next
        # info read reflects this step's submission.
        proc._drain_pending()
        prog = out.get(TransitionKey.INFO, {}).get("sarm_progress", proc._last_progress)
        # If no submission this step (eval_every_n_steps gating), prog is the
        # previous value — that's expected. We compare sequences anyway.
        progresses.append(float(prog))
    return progresses


def collect_episode_frames(ds: LeRobotDataset, ep_idx: int, image_keys: list[str], state_key: str) -> list[dict]:
    ep = ds.meta.episodes[ep_idx]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    out = []
    for i in range(start, end):
        sample = ds[i]
        obs = {}
        for k in image_keys:
            v = sample[k]
            if isinstance(v, torch.Tensor):
                obs[k] = v
            else:
                obs[k] = torch.as_tensor(v)
        if state_key in sample:
            v = sample[state_key]
            obs[state_key] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        out.append(obs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--task", default="Two-stage assembly")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--sarm-type", default="sarm_ext")
    ap.add_argument("--head-mode", default="sparse")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", default="0,1,2", help="comma-separated ep indices")
    ap.add_argument("--ks", default="1,2,5", help="comma-separated eval_every_n_steps values")
    args = ap.parse_args()

    ds = LeRobotDataset(repo_id=args.dataset)
    ks = [int(x) for x in args.ks.split(",")]
    ep_ids = [int(x) for x in args.episodes.split(",")]

    # Probe one proc to discover image_keys + state_key.
    probe = make_proc(args, feat_cache_enabled=False, eval_every_n=1)
    image_keys = probe._image_keys
    state_key = probe._state_key
    print(f"[probe] image_keys={image_keys}  state_key={state_key}")
    del probe

    rows: list[dict] = []
    for k in ks:
        for ep_idx in ep_ids:
            frames = collect_episode_frames(ds, ep_idx, image_keys, state_key)
            n = len(frames)
            print(f"\n=== ep={ep_idx}  K={k}  frames={n} ===")

            for label, cache_on in (("OFF", False), ("ON ", True)):
                proc = make_proc(args, feat_cache_enabled=cache_on, eval_every_n=k)
                # Warmup first call (CLIP+SARM compile/cache)
                _ = run_episode(proc, frames[: min(n, 12)])
                proc.reset()
                t0 = time.perf_counter()
                progs = run_episode(proc, frames)
                t1 = time.perf_counter()
                wall = t1 - t0
                rows.append(dict(ep=ep_idx, k=k, cache=label.strip(), wall=wall, progresses=progs))
                print(f"  cache={label}  wall={wall:6.2f}s  fps={n/wall:6.1f}  prog[0,n//2,-1]={progs[0]:.4f}, {progs[n//2]:.4f}, {progs[-1]:.4f}")

            # compare cache OFF vs ON
            off_row = next(r for r in rows if r["ep"] == ep_idx and r["k"] == k and r["cache"] == "OFF")
            on_row = next(r for r in rows if r["ep"] == ep_idx and r["k"] == k and r["cache"] == "ON")
            off = np.array(off_row["progresses"])
            on = np.array(on_row["progresses"])
            mad = float(np.max(np.abs(off - on)))
            l2 = float(np.linalg.norm(off - on))
            speedup = off_row["wall"] / on_row["wall"]
            print(f"  IDENTITY: max|Δ|={mad:.2e}  L2={l2:.2e}  SPEEDUP={speedup:.2f}×")

    # summary
    print("\n=== summary ===")
    for k in ks:
        offs = [r for r in rows if r["k"] == k and r["cache"] == "OFF"]
        ons  = [r for r in rows if r["k"] == k and r["cache"] == "ON"]
        wo = sum(r["wall"] for r in offs)
        wn = sum(r["wall"] for r in ons)
        mad = max(
            float(np.max(np.abs(np.array(o["progresses"]) - np.array(n["progresses"]))))
            for o, n in zip(offs, ons, strict=True)
        )
        print(f"  K={k}: cache OFF wall={wo:.2f}s  ON wall={wn:.2f}s  speedup={wo/wn:.2f}×  max|Δ|={mad:.2e}")


if __name__ == "__main__":
    main()
