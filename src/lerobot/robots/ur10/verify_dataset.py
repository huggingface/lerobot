#!/usr/bin/env python

"""
End-to-end UR10 demo-dataset verification.

Loads a recorded dataset once, runs a configurable set of checks against every frame in a
single pass, and prints a unified report. Each check is a self-contained class in
`dataset_checks.py`; this file is just the orchestrator + CLI.

Usage:
    # all checks, with train-config-aware schema and state-bounds verification
    python -m lerobot.robots.ur10.verify_dataset \
        --repo_id local/ur10_usb_insertion \
        --train_config src/lerobot/rl/ur10_train_3cams.json \
        --penalty -0.02

    # subset
    python -m lerobot.robots.ur10.verify_dataset \
        --repo_id local/ur10_usb_insertion \
        --checks gripper_penalty,gripper_activity,finite

    # robots whose state layout differs from UR10's default (e.g. panda sim has state[-4])
    python -m lerobot.robots.ur10.verify_dataset \
        --repo_id local/panda_sim_usb_insertion_demos \
        --gripper_state_index -4

Exit code:
    0 = all checks PASS / SKIP / INFO-only
    1 = at least one BLOCKER check FAILed
    2 = no BLOCKER failures, but some checks emitted WARN
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.ur10.dataset_checks import (
    ActionBoundsCheck,
    Check,
    CheckResult,
    DatasetStatsEmitter,
    EpisodeLengthCheck,
    FiniteCheck,
    GripperActivityCheck,
    GripperPenaltyCheck,
    SchemaCheck,
    SEVERITY_BLOCKER,
    SEVERITY_INFO,
    SEVERITY_WARN,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    StateBoundsCheck,
    StationaryFramesCheck,
    TimestampMonotonicityCheck,
    load_train_config,
)


def _build_checks(args, train_config: dict | None) -> list[Check]:
    """Construct every check instance based on CLI args. Order is the report order."""
    checks: list[Check] = [
        SchemaCheck(train_config=train_config),
        FiniteCheck(),
        ActionBoundsCheck(action_continuous_dims=args.action_continuous_dims),
        StateBoundsCheck(train_config=train_config),
        EpisodeLengthCheck(),
        GripperPenaltyCheck(
            penalty=args.penalty,
            gripper_state_index=args.gripper_state_index,
            gripper_action_index=args.gripper_action_index,
        ),
        GripperActivityCheck(
            gripper_state_index=args.gripper_state_index,
            gripper_action_index=args.gripper_action_index,
            min_toggles_per_episode=args.min_toggles_per_episode,
        ),
        StationaryFramesCheck(
            movement_threshold_m=args.stationary_movement_threshold_m,
            run_threshold_frames=args.stationary_run_threshold_frames,
        ),
        TimestampMonotonicityCheck(),
        DatasetStatsEmitter(percentile=args.stats_percentile),
    ]
    return checks


def _filter_checks(checks: list[Check], selection: str) -> list[Check]:
    if selection == "all":
        return checks
    requested = {s.strip() for s in selection.split(",") if s.strip()}
    unknown = requested - {c.name for c in checks}
    if unknown:
        print(f"WARNING: unknown check name(s) ignored: {sorted(unknown)}")
    return [c for c in checks if c.name in requested]


def _color(status: str) -> str:
    # Plain markers — keep terminal output portable.
    return {
        STATUS_PASS: "PASS",
        STATUS_WARN: "WARN",
        STATUS_FAIL: "FAIL",
        STATUS_SKIP: "SKIP",
    }.get(status, status)


def _print_header(ds: LeRobotDataset) -> None:
    print(f"  total frames:    {len(ds)}")
    print(f"  total episodes:  {ds.num_episodes}")
    feats = ds.features

    def _sh(k):
        e = feats.get(k)
        if e is None:
            return "MISSING"
        return tuple(e.get("shape", ())) if isinstance(e, dict) else getattr(e, "shape", "?")

    print(f"  observation.state shape:  {_sh('observation.state')}")
    print(f"  action shape:             {_sh('action')}")
    print(f"  has discrete_penalty:     {'complementary_info.discrete_penalty' in feats}")
    print()


def _print_check_result(idx: int, total: int, res: CheckResult, verbose: bool) -> None:
    status = _color(res.status)
    sev = res.severity.ljust(7)
    name = res.name.ljust(24)
    line = f"  [{idx:>2}/{total}] {name} {sev} {status}: {res.summary}"
    print(line)
    if verbose and res.details:
        for d in res.details[:10]:
            print(f"           ↳ {d}")
        if len(res.details) > 10:
            print(f"           ↳ ... and {len(res.details) - 10} more")


def _print_summary(results: list[CheckResult]) -> None:
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r.name.ljust(26)} {r.severity.ljust(7)} {_color(r.status):4s}  {r.summary}")
    blockers = [r for r in results if r.severity == SEVERITY_BLOCKER and r.status == STATUS_FAIL]
    warns = [r for r in results if r.status == STATUS_WARN]
    print()
    if blockers:
        print(f"OVERALL: FAIL ({len(blockers)} blocker, {len(warns)} warn)")
    elif warns:
        print(f"OVERALL: PASS WITH WARNINGS ({len(warns)} warn)")
    else:
        print("OVERALL: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a UR10 (or compatible) HIL-SERL demo dataset against a battery of checks",
    )
    parser.add_argument("--repo_id", required=True, type=str)
    parser.add_argument("--root", default=None, type=str, help="Optional dataset root")
    parser.add_argument(
        "--train_config",
        default=None,
        type=str,
        help="Path to ur10_train_*.json (enables schema + state_bounds)",
    )
    parser.add_argument("--penalty", type=float, default=-0.02)
    parser.add_argument("--gripper_state_index", type=int, default=-1)
    parser.add_argument("--gripper_action_index", type=int, default=-1)
    parser.add_argument("--action_continuous_dims", type=int, default=3)
    parser.add_argument("--min_toggles_per_episode", type=int, default=2)
    parser.add_argument(
        "--stationary_movement_threshold_m",
        type=float,
        default=1e-4,
        help="Per-frame TCP xyz delta below this is 'stationary'",
    )
    parser.add_argument(
        "--stationary_run_threshold_frames",
        type=int,
        default=20,
        help="Stationary runs ≥ this many frames are flagged",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default="all",
        help="Comma-separated check names, or 'all' (default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, audit only the first N frames",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print first 10 detail lines per check",
    )
    parser.add_argument(
        "--stats_percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile-clipped range for `dataset_stats` emitter. p99 = 0.5%/99.5%% range, "
            "clips single-frame contact spikes that would otherwise inflate normalization."
        ),
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id, root=args.root)
    _print_header(ds)

    train_config = None
    if args.train_config:
        print(f"Loading train_config: {args.train_config}")
        train_config = load_train_config(args.train_config)

    all_checks = _build_checks(args, train_config)
    checks = _filter_checks(all_checks, args.checks)

    # Setup phase — checks that can't run on this dataset (e.g. state_bounds without
    # train_config) are filtered out here.
    ctx: dict[str, Any] = {}
    active: list[Check] = []
    for c in checks:
        if c.setup(ds, ctx):
            active.append(c)
        else:
            # Manufacture a SKIP result so it appears in the summary.
            res = c.finalize()
            res.status = STATUS_SKIP
            res.summary = res.summary or "preconditions not met"
            ctx.setdefault("skipped_results", []).append(res)
    skipped_results: list[CheckResult] = ctx.get("skipped_results", [])

    n_frames = len(ds) if args.limit is None else min(len(ds), args.limit)
    print(f"Running {len(active)} check(s) on {n_frames} frame(s)")
    print()

    t0 = time.perf_counter()
    for i in range(n_frames):
        sample = ds[i]
        episode_idx = int(_to_int(sample.get("episode_index", 0)))
        for c in active:
            c.process_frame(i, episode_idx, sample)
    t_elapsed = time.perf_counter() - t0

    # Finalize and print.
    results = [c.finalize() for c in active] + skipped_results
    # Re-order to match `_build_checks` order for stable summary output.
    name_order = [c.name for c in checks]
    results.sort(key=lambda r: name_order.index(r.name) if r.name in name_order else len(name_order))

    print(f"Audited {n_frames} frame(s) in {t_elapsed:.1f}s")
    print()
    print("Check results:")
    for i, r in enumerate(results, start=1):
        _print_check_result(i, len(results), r, args.verbose)

    _print_summary(results)

    blocker_failed = any(
        r.severity == SEVERITY_BLOCKER and r.status == STATUS_FAIL for r in results
    )
    any_warn = any(r.status == STATUS_WARN for r in results)
    if blocker_failed:
        return 1
    if any_warn:
        return 2
    return 0


def _to_int(v: Any) -> int:
    try:
        return int(v.item()) if hasattr(v, "item") else int(v)
    except (TypeError, ValueError):
        return int(v[0]) if hasattr(v, "__getitem__") else 0


if __name__ == "__main__":
    sys.exit(main())
