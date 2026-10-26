#!/usr/bin/env python
"""B-track Stage 1: multi-task `LocalGraspRefiner`-only adaptation of `temporal_decoder_grounded_
grasp_v2_5`. Question this answers: can `LocalGraspRefiner` (see `local_grasp_refiner.py`) learn
local grasp dynamics shared across object categories, not just `select_poker` (the only task it has
been closed-loop-evaluated on so far -- see `eval_grounded_grasp_v2_5_5k_dev.py`)?

Init checkpoint: `outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/010000` --
verified byte-identical to `outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/015000`
on every one of the 618 weight tensors OUTSIDE `local_grasp_refiner.*` (the global modules were
frozen for that whole 10k run, so no drift occurred -- see this script's own preflight check
below), plus its own 6 trained `local_grasp_refiner.*` tensors. So this checkpoint already IS
"v2 15k global modules + v2.5 10k LocalGraspRefiner" -- no separate checkpoint merge step needed.

NOT changed from that 10k run or from `train_grounded_grasp_v2_5_5k.py` (per the B-track
instructions: architecture/config/loss/threshold stay fixed): every `SafeDiffVLAConfig` field
below is copied verbatim, including `local_grasp_freeze_global_modules=True` (so `get_optim_params()`
still contains ONLY `LocalGraspRefiner`'s own parameters -- backbone, `TargetQueryExtractor`,
`VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder` all stay frozen, exactly
as in every prior run on this architecture) and every loss/threshold/radius/horizon field.

What IS new here (data only):
  1. The dataset is restricted to the 3500 episodes (500 each) belonging to
     select_fruit/select_toy/select_drink/select_book/select_ingredient/select_mahjong/select_poker
     (see `_multitask_common.py` for the substring rules verified against the real dataset --
     mutually exclusive, 500/500/500/500/500/500/500, zero overlap). Every other "primitive:"/
     "composite:" task in `lerobot/vlabench_unified` (paintings, vases, condiments, multi-step
     composites, etc.) is excluded -- not requested, and not comparable single-object pick tasks.
  2. Task-balanced sampling: `lerobot_train.make_dataloaders` is monkeypatched (same
     non-invasive-monkeypatch pattern this branch already uses for `VLABenchEnv.step`/`.reset` in
     every rollout eval script) to swap in `TaskBalancedFrameSampler` for the TRAIN dataloader only
     -- draws each of the 20000 (5000 steps * batch_size=4) training frames by picking one of the 7
     tasks UNIFORMLY AT RANDOM, then a uniformly random frame from that task's own
     `EpisodeAwareSampler`-backed pool (so within-task frame order/epoch-shuffling still goes
     through the library's own well-tested sampler -- only the ACROSS-task selection is custom).
     This is deliberately NOT proportional to each task's frame count (75626 down to 36333 frames
     -- see the audit below) specifically so the larger tasks (fruit/toy) cannot dominate training
     the way plain uniform-frame sampling would let them. The eval dataloader (unused here --
     `eval_steps=0`, same as every prior run) is left on the library's own default construction.
     The realized per-task draw counts are logged after training (never assumed equal to the
     1/7 target) to `task_balanced_sampling_log.json`.

Preceded by `multitask_grasp_supervision_audit.py`, which records (pure parquet audit, no model
forward) each task's episode/frame counts, open->close transition counts, valid K=10 local-window
sample counts, and local-loss mask coverage BEFORE this run -- run it first if
`outputs/eval/safediff_vla_grounded_grasp_multitask_supervision_audit/per_task_summary.json`
does not already exist.

5k steps only, single checkpoint at the end (save_freq=steps) -- this is Stage 1 only. Stage 2
(unfreezing global modules) is explicitly out of scope per the B-track instructions.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_5_multitask_stage1_5k.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID, TASK_NAMES, build_episode_task_map  # noqa: E402

import lerobot.scripts.lerobot_train as lerobot_train_module  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SPLIT = 0.1  # same genuine held-out split as every prior run on this branch
V2_5_10K_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/010000/pretrained_model"
OUT_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k")


def preflight_check_init_checkpoint(pretrained_path: str) -> None:
    """Confirms this script's docstring claim (init checkpoint == v2 15k global weights + v2.5 10k
    LocalGraspRefiner) instead of just asserting it -- fails loudly instead of silently training
    from an unexpected init if the referenced checkpoints ever change on disk."""
    from safetensors.torch import load_file

    v2_path = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/015000/pretrained_model/model.safetensors"
    init_path = f"{pretrained_path}/model.safetensors"
    if not Path(v2_path).exists() or not Path(init_path).exists():
        logger.warning("preflight: skipping init-checkpoint identity check (one of the files is missing)")
        return
    v2_state = load_file(v2_path)
    init_state = load_file(init_path)
    common = set(v2_state) & set(init_state)
    n_diff = sum(1 for k in common if not torch.equal(v2_state[k], init_state[k]))
    local_only = [k for k in init_state if k not in v2_state]
    logger.info(
        "preflight: init checkpoint has %d weights in common with v2-15k (%d differ) + %d "
        "local_grasp_refiner-only weights", len(common), n_diff, len(local_only),
    )
    if n_diff:
        raise RuntimeError(
            f"{n_diff} weights outside local_grasp_refiner differ between v2-15k and the init "
            "checkpoint -- global modules may not actually be frozen/unchanged as this script's "
            "docstring assumes. Aborting rather than training from an unverified init."
        )


class TaskBalancedFrameSampler(torch.utils.data.Sampler[int]):
    """Draws `num_samples` dataset-relative frame indices: each draw picks one of `task_names`
    UNIFORMLY (not weighted by task size), then the next frame from that task's own
    `EpisodeAwareSampler` (each such per-task sampler shuffles/epoch-advances independently,
    reusing the library's own resume-safe permutation logic -- only which task supplies the next
    frame is custom here). Every draw is appended to `self.draw_log` (task name, in yield order) so
    the realized batch composition can be reported exactly, not assumed."""

    def __init__(
        self,
        per_task_samplers: dict[str, EpisodeAwareSampler],
        num_samples: int,
        seed: int,
    ) -> None:
        self.per_task_samplers = per_task_samplers
        self.task_names = sorted(per_task_samplers)
        self.num_samples = num_samples
        self.seed = seed
        self.draw_log: list[str] = []

    @staticmethod
    def _infinite(sampler: EpisodeAwareSampler):
        while True:
            yield from sampler  # each `iter(sampler)` call advances that sampler's own epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        task_iters = {name: self._infinite(s) for name, s in self.per_task_samplers.items()}
        self.draw_log = []
        for _ in range(self.num_samples):
            task = self.task_names[int(rng.integers(len(self.task_names)))]
            idx = next(task_iters[task])
            self.draw_log.append(task)
            yield idx

    def __len__(self) -> int:
        return self.num_samples


_sampling_state: dict = {}


def build_task_balanced_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
    """Monkeypatch target for `lerobot_train_module.make_dataloaders`. Builds the EVAL dataloader
    via the library's own unmodified function (unused here, `eval_steps=0`, but kept for parity/
    future use), then replaces only the TRAIN dataloader with one driven by
    `TaskBalancedFrameSampler`."""
    _, eval_dataloader = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS(cfg, dataset, eval_dataset, step, parallel_dims)

    episode_task = build_episode_task_map()
    train_episodes = set(dataset.episodes) if dataset.episodes is not None else set(range(dataset.num_episodes))
    from_idx = dataset.meta.episodes["dataset_from_index"]
    to_idx = dataset.meta.episodes["dataset_to_index"]

    per_task_episode_indices: dict[str, list[int]] = {name: [] for name in TASK_NAMES}
    for ep_idx in train_episodes:
        task = episode_task.get(ep_idx)
        if task is not None:
            per_task_episode_indices[task].append(ep_idx)

    empty = [t for t, eps in per_task_episode_indices.items() if not eps]
    if empty:
        raise RuntimeError(f"no training episodes left for task(s) {empty} after the eval_split held-out cut")

    per_task_frame_counts = {}
    per_task_samplers: dict[str, EpisodeAwareSampler] = {}
    for i, (task, eps) in enumerate(per_task_episode_indices.items()):
        sampler = EpisodeAwareSampler(
            from_idx, to_idx, episode_indices_to_use=eps, shuffle=True,
            seed=(cfg.seed or 0) + i + 1, absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        per_task_samplers[task] = sampler
        per_task_frame_counts[task] = len(sampler)

    num_samples = cfg.steps * cfg.batch_size
    sampler = TaskBalancedFrameSampler(per_task_samplers, num_samples=num_samples, seed=cfg.seed or 0)

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    device_type = parallel_dims.device_type
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
    )

    _sampling_state["sampler"] = sampler
    _sampling_state["per_task_episode_counts"] = {t: len(eps) for t, eps in per_task_episode_indices.items()}
    _sampling_state["per_task_frame_pool_counts"] = per_task_frame_counts
    return dataloader, eval_dataloader


def write_sampling_log(out_dir: Path, batch_size: int) -> None:
    sampler: TaskBalancedFrameSampler = _sampling_state["sampler"]
    draw_log = sampler.draw_log
    n_draws = len(draw_log)
    per_task_sampled = {t: draw_log.count(t) for t in TASK_NAMES}

    n_batches = n_draws // batch_size
    per_task_batches_containing = {t: 0 for t in TASK_NAMES}
    batch_task_composition: list[dict[str, int]] = []
    for b in range(n_batches):
        chunk = draw_log[b * batch_size : (b + 1) * batch_size]
        counts: dict[str, int] = {}
        for t in chunk:
            counts[t] = counts.get(t, 0) + 1
        batch_task_composition.append(counts)
        for t in counts:
            per_task_batches_containing[t] += 1

    log = {
        "n_draws": n_draws,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "target_fraction_per_task": 1.0 / len(TASK_NAMES),
        "per_task_train_episode_counts": _sampling_state["per_task_episode_counts"],
        "per_task_frame_pool_counts": _sampling_state["per_task_frame_pool_counts"],
        "per_task_sampled_frame_counts": per_task_sampled,
        "per_task_realized_fraction": {t: c / n_draws for t, c in per_task_sampled.items()},
        "per_task_n_batches_containing_task": per_task_batches_containing,
        "batch_task_composition": batch_task_composition,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_balanced_sampling_log.json").write_text(json.dumps(log, indent=2))
    logger.info("wrote %s", out_dir / "task_balanced_sampling_log.json")
    logger.info("=== realized per-task sampling fractions (target: %.4f each) ===", 1.0 / len(TASK_NAMES))
    for t in TASK_NAMES:
        logger.info(
            "%-20s sampled=%6d (%.4f) in %d/%d batches",
            t, per_task_sampled[t], per_task_sampled[t] / n_draws,
            per_task_batches_containing[t], n_batches,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--skip-preflight-check", action="store_true")
    args = parser.parse_args()
    if args.steps > 5000:
        raise ValueError("Stage 1 is 5k steps only -- do not extend without a separate decision.")
    out_dir = Path(args.output_dir)

    audit_path = Path("outputs/eval/safediff_vla_grounded_grasp_multitask_supervision_audit/per_task_summary.json")
    if not audit_path.exists():
        raise RuntimeError(
            f"{audit_path} not found -- run examples/safediff_vla/multitask_grasp_supervision_audit.py first."
        )

    if not args.skip_preflight_check:
        preflight_check_init_checkpoint(V2_5_10K_CHECKPOINT)

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2_5",
        pretrained_path=V2_5_10K_CHECKPOINT,
        action_horizon=50,
        execute_horizon=50,
        use_temporal_ensembling=False,
        backbone_name="lerobot/smolvla_vlabench",
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        freeze_backbone=True,
        freeze_vision_encoder=True,
        use_lora=False,
        decoder_hidden_dim=512,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_ffn_dim=1024,
        decoder_dropout=0.1,
        use_backbone_domain_adapter=False,
        backbone_action_conversion_semantics="per_step",
        lambda_smooth=0.0,
        lambda_target=1.0,
        target_head_hidden_dim=256,
        grounded_grasp_gripper_open_threshold=0.5,
        grounded_grasp_close_threshold_m=0.03,
        local_grasp_radius_m=0.10,
        local_grasp_max_delta_xyz_m=0.03,
        local_grasp_max_delta_rot_rad=0.05,
        local_grasp_action_horizon=5,
        local_grasp_execute_horizon=1,
        local_grasp_window_k=10,
        lambda_local_grasp=1.0,
        local_grasp_hidden_dim=256,
        local_grasp_freeze_global_modules=True,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    episode_task = build_episode_task_map()
    union_episodes = sorted(episode_task)
    logger.info("training dataset restricted to %d episodes across %d tasks", len(union_episodes), len(TASK_NAMES))

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, episodes=union_episodes, eval_split=EVAL_SPLIT),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=out_dir,
        job_name="safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k",
        resume=False,
        seed=1000,
        num_workers=8,
        batch_size=4,
        steps=args.steps,
        log_freq=100,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.steps,
        use_policy_training_preset=True,
    )

    lerobot_train_module._ORIGINAL_MAKE_DATALOADERS = lerobot_train_module.make_dataloaders
    lerobot_train_module.make_dataloaders = build_task_balanced_dataloaders
    try:
        lerobot_train_module.train(cfg)
    finally:
        lerobot_train_module.make_dataloaders = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS
        del lerobot_train_module._ORIGINAL_MAKE_DATALOADERS

    if "sampler" in _sampling_state:
        write_sampling_log(out_dir, cfg.batch_size)


if __name__ == "__main__":
    main()
