"""SARM eval harness for sim_assemble.

Ported from lerobot-panda/scripts/viz_sarm_progress.py with metric aggregation
aligned to our 4-stage gates (panda gates + per-stage plateau accuracy).

Runs SARM sync on every frame (return_stages=True) against an offline
LeRobotDataset. Emits:
    - per-ep PNG (progress curve + stage probs + sample frames)
    - metrics.json (per-ep + per-bucket + overall)
    - summary.md (human-readable gate-check table)

Usage (examples):
    uv run python -m lerobot.policies.sarm.eval_sarm_sim_assemble \
        --dataset domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2_val \
        --pretrained outputs/sim_assemble_sarm_multistage_iterB/checkpoints/last/pretrained_model \
        --task "Two-stage assembly" \
        --stats domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2 \
        --out outputs/sarm_eval/iterB_val \
        --label iterB_val

Gates (panda + ours):
    SuccTerm≥0.9: ≥95% on full-success eps
    FailTerm≥0.9: 0% on failure (partial / 0-of-4) eps
    FailMax≥0.5:  0% on 0-of-4 eps (strict)
    lin_mad ≤0.25 on full-success eps
    mean_mid ≥0.25 on full-success eps
    plateau k/4 ep: peak ∈ [bp[k]-0.10, bp[k]+0.10]
    stage argmax acc ≥0.80 (on GT-labeled frames)
    monotonicity ≥0.85 on full-success
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.reward_model.sarm import SARMRewardConfig, SARMRewardProcessorStep
from lerobot.policies.sarm.sarm_utils import (
    temporal_proportions_to_breakpoints,
)


# ------------------------------------------------------------------
# Per-episode SARM inference
# ------------------------------------------------------------------

def run_episode(
    step: SARMRewardProcessorStep,
    ds: LeRobotDataset,
    a: int,
    b: int,
    head_mode: str = "sparse",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Walk [a, b), return (progress, stage_probs_matrix, frames_RGB_list)."""
    step.reset()
    model = step._model
    preprocess = step._preprocess
    center_idx = step._center_idx

    ep_len = b - a
    progresses = np.zeros(ep_len)
    stage_probs_mat = None
    frames = []

    for k, i in enumerate(range(a, b)):
        f = ds[i]
        state_tensor = f[step._state_key]
        obs = {
            step._image_key: f[step._image_key],
            step._state_key: state_tensor,
        }
        step._push_obs_to_buffer(obs)
        img_snap, state_snap = step._snapshot_buffers()
        image_stack, state_stack = step._build_window_from_snapshot(img_snap, state_snap)
        batch = {
            step._image_key: image_stack,
            "task": step.config.task,
            "index": 0,
            "episode_index": 0,
        }
        if state_stack is not None:
            batch[step._state_key] = state_stack
        with torch.inference_mode():
            processed = preprocess(batch)
            prog, stage_probs = model.calculate_rewards(
                text_embeddings=processed["text_features"],
                video_embeddings=processed["video_features"],
                state_features=processed.get("state_features"),
                lengths=processed.get("lengths"),
                frame_index=center_idx,
                return_all_frames=False,
                return_stages=True,
                head_mode=head_mode,
            )
        if isinstance(prog, torch.Tensor):
            prog = float(prog.detach().cpu().reshape(-1)[0].item())
        else:
            prog = float(np.asarray(prog).reshape(-1)[0])
        progresses[k] = max(0.0, min(1.0, prog))

        if isinstance(stage_probs, torch.Tensor):
            stage_probs = stage_probs.detach().cpu().numpy()
        stage_probs = np.asarray(stage_probs)
        if stage_probs.ndim == 3:
            sp = stage_probs[0, center_idx]
        elif stage_probs.ndim == 2:
            sp = stage_probs[center_idx]
        else:
            sp = stage_probs
        if stage_probs_mat is None:
            stage_probs_mat = np.zeros((ep_len, len(sp)))
        stage_probs_mat[k] = sp

        img = f[step._image_key].cpu().numpy()
        if img.dtype != np.uint8:
            img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.transpose(1, 2, 0)
        frames.append(img)

    return progresses, stage_probs_mat, frames


# ------------------------------------------------------------------
# Per-episode metrics
# ------------------------------------------------------------------

@dataclass
class EpMetrics:
    ep: int
    ep_len: int
    n_stages_gt: int
    bucket: str
    terminal: float
    max_prog: float
    mean_mid: float
    mean_terminal: float
    lin_mad: float
    monotonicity: float
    stage_argmax_acc: float | None
    peak_within_plateau: bool | None
    gt_breakpoint: float | None


def _compute_linear_gt(ep_len: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, ep_len) if ep_len > 1 else np.array([0.0])


def _mean_absolute_deviation(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - ref)))


def _monotonicity(pred: np.ndarray) -> float:
    if len(pred) < 2:
        return 1.0
    smoothed = pd.Series(pred).rolling(window=5, min_periods=1, center=True).mean().values
    deltas = np.diff(smoothed)
    return float(np.mean(deltas >= -1e-3))


def _stage_argmax_accuracy(
    stage_probs: np.ndarray,
    names: list[str] | None,
    starts: list[int] | None,
    ends: list[int] | None,
    stage_name_to_global_idx: dict[str, int],
) -> float | None:
    """Compute frame-wise argmax accuracy against GT stage labels.

    Returns None if no GT stages labeled (e.g. 0/4 ep).
    """
    if names is None or not names:
        return None
    ep_len = stage_probs.shape[0]
    gt = np.full(ep_len, -1, dtype=int)
    for name, s, e in zip(names, starts, ends):
        g = stage_name_to_global_idx.get(name)
        if g is None:
            continue
        gt[s : min(e + 1, ep_len)] = g
    mask = gt >= 0
    if not np.any(mask):
        return None
    pred = np.argmax(stage_probs, axis=1)
    return float(np.mean(pred[mask] == gt[mask]))


def compute_ep_metrics(
    ep: int,
    ep_len: int,
    progs: np.ndarray,
    stage_probs: np.ndarray,
    n_stages_gt: int,
    names: list[str] | None,
    starts: list[int] | None,
    ends: list[int] | None,
    breakpoints: np.ndarray,
    stage_name_to_global_idx: dict[str, int],
    total_stages_gt: int,
) -> EpMetrics:
    is_full = n_stages_gt == total_stages_gt
    bucket = f"{n_stages_gt}-of-{total_stages_gt}" if not is_full else "full"

    terminal = float(progs[-1])
    max_prog = float(progs.max())
    mid_idx = ep_len // 2
    mean_mid = float(progs[max(0, mid_idx - 3) : mid_idx + 4].mean())
    mean_terminal = float(progs[max(0, ep_len - 10) :].mean())

    # lin_mad: meaningful only for full-success (linear 0→1 expected)
    if is_full:
        gt = _compute_linear_gt(ep_len)
        lin_mad = _mean_absolute_deviation(progs, gt)
    else:
        lin_mad = float("nan")

    monotonicity = _monotonicity(progs)

    stage_acc = _stage_argmax_accuracy(
        stage_probs, names, starts, ends, stage_name_to_global_idx
    )

    # plateau check: k-of-N partial ep's peak should be near breakpoints[k].
    # Skip if model has fewer stages than the dataset annotation (e.g. single-stage
    # model on 4-stage-labeled dataset — no per-stage breakdown is meaningful).
    plateau_ok: bool | None
    gt_bp: float | None
    if is_full or n_stages_gt >= len(breakpoints):
        plateau_ok = None
        gt_bp = None
    else:
        gt_bp = float(breakpoints[n_stages_gt])
        plateau_ok = bool(gt_bp - 0.10 <= max_prog <= gt_bp + 0.10)

    return EpMetrics(
        ep=ep,
        ep_len=ep_len,
        n_stages_gt=n_stages_gt,
        bucket=bucket,
        terminal=terminal,
        max_prog=max_prog,
        mean_mid=mean_mid,
        mean_terminal=mean_terminal,
        lin_mad=lin_mad,
        monotonicity=monotonicity,
        stage_argmax_acc=stage_acc,
        peak_within_plateau=plateau_ok,
        gt_breakpoint=gt_bp,
    )


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_episode(
    progs: np.ndarray,
    stage_probs: np.ndarray,
    frames: list[np.ndarray],
    stage_names: list[str],
    title: str,
    out_path: str,
    draw_gt: bool,
    gt_stage_boundaries: list[tuple[str, int, int]] | None = None,
    breakpoints: np.ndarray | None = None,
) -> None:
    ep_len = len(progs)
    xs = np.arange(ep_len)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1.3, 1.3], hspace=0.35)

    ax_p = fig.add_subplot(gs[0, 0])
    ax_p.fill_between(xs, progs, alpha=0.25, color="tab:cyan")
    ax_p.plot(xs, progs, label="Predicted", color="tab:cyan", lw=2)
    if draw_gt:
        gt = _compute_linear_gt(ep_len)
        ax_p.plot(xs, gt, label="Linear GT (reference)", color="tab:green",
                  linestyle="--", lw=1.5)
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints[1:-1], start=1):
            ax_p.axhline(bp, color="gray", linestyle=":", alpha=0.4, lw=0.8)
    ax_p.axhline(0.9, color="red", linestyle=":", alpha=0.5, lw=1)
    ax_p.set_ylim(-0.05, 1.08)
    ax_p.set_xlim(0, max(ep_len - 1, 1))
    ax_p.set_ylabel("Progress")
    ax_p.set_title(title)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper left", fontsize=9)

    ax_s = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(ep_len)
    colors = plt.cm.tab10.colors
    for i, name in enumerate(stage_names):
        ax_s.fill_between(xs, bottom, bottom + stage_probs[:, i],
                          label=name, color=colors[i % len(colors)], alpha=0.8)
        bottom = bottom + stage_probs[:, i]
    if gt_stage_boundaries:
        for name, s, e in gt_stage_boundaries:
            ax_s.axvline(s, color="black", linestyle="--", alpha=0.3, lw=0.7)
    ax_s.set_ylim(0, 1.05)
    ax_s.set_xlim(0, max(ep_len - 1, 1))
    ax_s.set_ylabel("Stage Probability")
    ax_s.set_xlabel("Frame")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper left", fontsize=8)

    ax_f = fig.add_subplot(gs[2, 0])
    ax_f.axis("off")
    n_samples = min(8, ep_len)
    if n_samples > 0:
        idxs = np.linspace(0, ep_len - 1, n_samples).astype(int)
        strip = np.concatenate([frames[i] for i in idxs], axis=1)
        ax_f.imshow(strip)
        H = frames[0].shape[0]
        W = frames[0].shape[1]
        for j, i in enumerate(idxs):
            x_center = (j + 0.5) * W
            ax_f.text(x_center, H + H * 0.09, f"Frame {int(i)}",
                      ha="center", va="top", fontsize=9)
    ax_f.set_title("Sample Frames", fontsize=11, pad=6)

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Dataset annotation loader
# ------------------------------------------------------------------

def _load_annotations(ds_root: Path) -> dict[int, tuple]:
    meta_dir = ds_root / "meta" / "episodes"
    paths = sorted(glob.glob(str(meta_dir / "chunk-*" / "file-*.parquet")))
    if not paths:
        return {}
    df = pd.concat([pd.read_parquet(p) for p in paths]).sort_values(
        "episode_index"
    ).reset_index(drop=True)
    out: dict[int, tuple] = {}
    if "sparse_subtask_names" not in df.columns:
        return out
    for _, row in df.iterrows():
        ep = int(row["episode_index"])
        names = row.get("sparse_subtask_names")
        starts = row.get("sparse_subtask_start_frames")
        ends = row.get("sparse_subtask_end_frames")
        if names is None or (hasattr(names, "__len__") and len(names) == 0):
            out[ep] = (None, None, None)
        else:
            out[ep] = (
                list(names),
                [int(s) for s in starts],
                [int(e) for e in ends],
            )
    return out


# ------------------------------------------------------------------
# Aggregation + summary
# ------------------------------------------------------------------

@dataclass
class Gates:
    succ_terminal_min_rate: float = 0.95  # fraction of full eps with term≥0.9
    fail_terminal_max_rate: float = 0.0
    fail_max_at_0of4_max_rate: float = 0.0
    lin_mad_max: float = 0.25
    mean_mid_min: float = 0.25
    plateau_tolerance: float = 0.10
    stage_argmax_min: float = 0.80
    monotonicity_min: float = 0.85


def aggregate(metrics: list[EpMetrics], gates: Gates) -> dict:
    by_bucket: dict[str, list[EpMetrics]] = {}
    for m in metrics:
        by_bucket.setdefault(m.bucket, []).append(m)

    bucket_rows = []
    for bucket, ms in sorted(by_bucket.items()):
        terms = np.array([x.terminal for x in ms])
        maxes = np.array([x.max_prog for x in ms])
        bucket_rows.append({
            "bucket": bucket,
            "n_eps": len(ms),
            "mean_term": float(np.mean(terms)),
            "mean_max": float(np.mean(maxes)),
            "rate_term_ge_0.9": float(np.mean(terms >= 0.9)),
            "rate_max_ge_0.9": float(np.mean(maxes >= 0.9)),
            "rate_max_ge_0.5": float(np.mean(maxes >= 0.5)),
            "mean_lin_mad": float(np.nanmean([x.lin_mad for x in ms])),
            "mean_mid": float(np.mean([x.mean_mid for x in ms])),
            "mean_monotonicity": float(np.mean([x.monotonicity for x in ms])),
            "mean_stage_argmax_acc": float(
                np.nanmean([x.stage_argmax_acc if x.stage_argmax_acc is not None
                            else np.nan for x in ms])
            ),
            "plateau_ok_rate": float(
                np.mean([x.peak_within_plateau for x in ms
                         if x.peak_within_plateau is not None])
            ) if any(x.peak_within_plateau is not None for x in ms) else float("nan"),
        })

    # gate evaluation (key metrics)
    full_metrics = by_bucket.get("full", [])
    gates_result = {}
    if full_metrics:
        terms = np.array([x.terminal for x in full_metrics])
        lin_mads = np.array([x.lin_mad for x in full_metrics])
        mean_mids = np.array([x.mean_mid for x in full_metrics])
        monos = np.array([x.monotonicity for x in full_metrics])
        gates_result["succ_term_rate"] = {
            "value": float(np.mean(terms >= 0.9)),
            "gate": gates.succ_terminal_min_rate,
            "pass": bool(np.mean(terms >= 0.9) >= gates.succ_terminal_min_rate),
        }
        gates_result["lin_mad"] = {
            "value": float(np.nanmean(lin_mads)),
            "gate": gates.lin_mad_max,
            "pass": bool(np.nanmean(lin_mads) <= gates.lin_mad_max),
        }
        gates_result["mean_mid"] = {
            "value": float(np.mean(mean_mids)),
            "gate": gates.mean_mid_min,
            "pass": bool(np.mean(mean_mids) >= gates.mean_mid_min),
        }
        gates_result["monotonicity"] = {
            "value": float(np.mean(monos)),
            "gate": gates.monotonicity_min,
            "pass": bool(np.mean(monos) >= gates.monotonicity_min),
        }

    fail_metrics = [m for m in metrics if m.bucket != "full"]
    if fail_metrics:
        terms = np.array([x.terminal for x in fail_metrics])
        gates_result["fail_term_rate"] = {
            "value": float(np.mean(terms >= 0.9)),
            "gate": gates.fail_terminal_max_rate,
            "pass": bool(np.mean(terms >= 0.9) <= gates.fail_terminal_max_rate),
        }

    zero_of_k = [m for m in metrics if m.n_stages_gt == 0]
    if zero_of_k:
        maxes = np.array([x.max_prog for x in zero_of_k])
        gates_result["zero_max_ge_0.5"] = {
            "value": float(np.mean(maxes >= 0.5)),
            "gate": gates.fail_max_at_0of4_max_rate,
            "pass": bool(np.mean(maxes >= 0.5) <= gates.fail_max_at_0of4_max_rate),
        }

    plateau_eps = [m for m in metrics if m.peak_within_plateau is not None]
    if plateau_eps:
        gates_result["plateau_ok_rate"] = {
            "value": float(np.mean([m.peak_within_plateau for m in plateau_eps])),
            "gate": 0.8,
            "pass": bool(
                np.mean([m.peak_within_plateau for m in plateau_eps]) >= 0.8
            ),
        }

    return {
        "per_bucket": bucket_rows,
        "gates": gates_result,
        "n_eps_total": len(metrics),
    }


def write_summary_md(
    out_path: Path,
    agg: dict,
    config_header: dict,
) -> None:
    lines = [
        "# SARM eval summary",
        "",
        "## Config",
        "",
    ]
    for k, v in config_header.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Gates", "", "| gate | value | threshold | pass |", "|---|---|---|---|"]
    for name, g in agg.get("gates", {}).items():
        lines.append(f"| {name} | {g['value']:.3f} | {g['gate']:.3f} | "
                     f"{'✅' if g['pass'] else '❌'} |")
    lines += ["", "## Per-bucket", "", "| bucket | n | term | max | term≥0.9 | max≥0.9 | max≥0.5 | lin_mad | mean_mid | stage_argmax | plateau_ok |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg.get("per_bucket", []):
        lines.append(
            f"| {r['bucket']} | {r['n_eps']} | {r['mean_term']:.2f} | "
            f"{r['mean_max']:.2f} | {r['rate_term_ge_0.9']:.2f} | "
            f"{r['rate_max_ge_0.9']:.2f} | {r['rate_max_ge_0.5']:.2f} | "
            f"{r['mean_lin_mad']:.3f} | {r['mean_mid']:.3f} | "
            f"{r['mean_stage_argmax_acc']:.3f} | {r['plateau_ok_rate']:.2f} |"
        )
    out_path.write_text("\n".join(lines))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--task", default="Two-stage assembly")
    parser.add_argument("--stats", default=None)
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--head-mode", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--out", default="outputs/sarm_eval")
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-eps", type=int, default=None)
    parser.add_argument("--gt", action="store_true",
                        help="Overlay linear GT reference on full-success plots")
    args = parser.parse_args()

    ds = LeRobotDataset(repo_id=args.dataset)
    cfg = SARMRewardConfig(
        type="sarm",
        pretrained_path=args.pretrained,
        stats_dataset_repo_id=args.stats,
        device=args.device,
        task=args.task,
        head_mode=args.head_mode,
        reward_mode="dense",
    )
    step = SARMRewardProcessorStep(config=cfg, terminate_on_success=False)
    names_attr = f"{args.head_mode}_subtask_names"
    props_attr = f"{args.head_mode}_temporal_proportions"
    stage_names = list(
        getattr(step._model.config, names_attr, None) or ["task"]
    )
    name_to_idx = {n: i for i, n in enumerate(stage_names)}

    # Compute breakpoints from the model's own temporal proportions for selected head.
    props = getattr(step._model.config, props_attr, None) or [1.0]
    bp = np.array(temporal_proportions_to_breakpoints(props, len(props)))
    breakpoints = bp

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-ep GT annotations from the dataset shards (for stage_argmax + n_stages_gt).
    annots = _load_annotations(Path(ds.root))

    n_eps = ds.num_episodes if args.max_eps is None else min(args.max_eps, ds.num_episodes)
    print(f"dataset={args.dataset} eps={n_eps} stages={stage_names}")
    print(f"breakpoints={bp.tolist()}")

    # Derive total GT stages from dataset annotations (max n_stages observed).
    max_stages_gt = max((len(v[0]) for v in annots.values() if v[0] is not None), default=len(stage_names))

    all_metrics: list[EpMetrics] = []
    per_ep_rows = []

    for ep in range(n_eps):
        m = ds.meta.episodes[ep]
        a, b = int(m["dataset_from_index"]), int(m["dataset_to_index"])
        progs, stage_probs, frames = run_episode(step, ds, a, b, head_mode=args.head_mode)

        names, starts, ends = annots.get(ep, (None, None, None))
        n_stages_gt = 0 if names is None else len(names)

        is_full = n_stages_gt == max_stages_gt
        em = compute_ep_metrics(
            ep=ep,
            ep_len=b - a,
            progs=progs,
            stage_probs=stage_probs,
            n_stages_gt=n_stages_gt,
            names=names,
            starts=starts,
            ends=ends,
            breakpoints=breakpoints,
            stage_name_to_global_idx=name_to_idx,
            total_stages_gt=max_stages_gt,
        )
        all_metrics.append(em)

        gt_boundaries = None
        if names is not None:
            gt_boundaries = list(zip(names, starts, ends))

        title = (f"{args.label} ep{ep:02d}  len={b - a}  {em.bucket}  "
                 f"term={em.terminal:.2f}  max={em.max_prog:.2f}  "
                 f"lin_mad={em.lin_mad:.2f}")
        out_path = out_dir / f"{args.label}_ep{ep:02d}_{em.bucket}.png"
        plot_episode(
            progs=progs,
            stage_probs=stage_probs,
            frames=frames,
            stage_names=stage_names,
            title=title,
            out_path=str(out_path),
            draw_gt=bool(args.gt and is_full),
            gt_stage_boundaries=gt_boundaries,
            breakpoints=breakpoints,
        )
        per_ep_rows.append(vars(em))
        print(f"  ep {ep} ({em.bucket}): term={em.terminal:.3f} "
              f"max={em.max_prog:.3f} plateau_ok={em.peak_within_plateau}")

    agg = aggregate(all_metrics, Gates())
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(
        {
            "dataset": args.dataset,
            "pretrained": args.pretrained,
            "task": args.task,
            "stats": args.stats,
            "per_ep": per_ep_rows,
            "summary": agg,
        },
        indent=2,
        default=float,
    ))

    write_summary_md(
        out_dir / "summary.md",
        agg,
        {"dataset": args.dataset,
         "ckpt": args.pretrained,
         "task": args.task,
         "stats": args.stats},
    )

    print(f"\nWrote: {out_dir}")
    print(f"  {metrics_path}")
    print(f"  {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
