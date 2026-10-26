#!/usr/bin/env python
"""LIBERO / LIBERO-Plus FINAL external-benchmark training: SmolVLA vs SafeDiff-VLA Final.

"SafeDiff-VLA Final" = the VLABench-final architecture kept EXACTLY as-is (no new architecture):
TemporalActionDecoder + SinCos/PadFix + TargetQueryExtractor + VisualGroundingCrossAttention +
TargetPointHead + LocalGraspRefiner + Phase-aware loss (`architecture=
"temporal_decoder_grounded_grasp_v2_5"`, `local_grasp_use_phase_aware_loss=True`) -- every
architecture/loss-coefficient/threshold field below is copied verbatim from
`examples/safediff_vla/train_grounded_grasp_v2_5_phase_aware_loss_100k.py` (the established
VLABench-final recipe), EXCEPT the 3 fields that must change to even run on LIBERO's domain and
7-D-state constraint:
  - `backbone_name=None` (fresh SmolVLM2 base weights, not the VLABench-domain
    `lerobot/smolvla_vlabench` checkpoint -- wrong action/vision domain for LIBERO)
  - `pretrained_path` unset (no LIBERO "Stage 2" checkpoint exists to resume from; this trains
    the whole non-backbone stack from scratch, unlike the 100k script's LocalGraspRefiner-only
    continuation)
  - `local_grasp_freeze_global_modules=False` (only meaningful as True when resuming a
    already-good Stage-2 checkpoint; from scratch, TargetQueryExtractor/VisualGroundingCrossAttention/
    TargetPointHead/TemporalActionDecoder must train jointly with LocalGraspRefiner)
Two LIBERO-scoped adapters are applied for the SafeDiff-VLA Final branch only (see
`../libero_smoke_test/adapter.py`, never touching global `action_semantics` or
`utils.find_grasp_target`):
  1. 8-D -> 7-D state collapse (LIBERO's real 8-D state vs the architecture's 7-D requirement).
  2. LIBERO gripper polarity/threshold for grasp-transition detection -- verified via
    `../libero_smoke_test/grasp_transition_audit.py` (36/40 real episodes across many tasks show
    a clean detected transition; 6/6 spot-checked against real dataset video land exactly on the
    gripper closing on the target object) before this script was written.
SmolVLA needs neither adapter and is otherwise identical between the two models: same dataset,
same 40-task episode pool, same task-balanced sampler, same batch size, same training steps, same
image/language inputs (both read directly from the dataset's own native features -- no rename).

action_horizon=50 kept fixed (matches the established SmolVLA/SafeDiff-VLA baseline recipe).
LIBERO fps=10 -> 50 steps = 5.0s physical horizon. LIBERO-Plus fps=20 -> 50 steps = 2.5s physical
horizon. Both logged explicitly at startup, not silently assumed equal.

Usage:
    uv run python examples/safediff_vla/libero_final/train_final.py \
        --dataset libero --policy smolvla --steps 50000
    uv run python examples/safediff_vla/libero_final/train_final.py \
        --dataset libero --policy safediff_vla_final --steps 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "libero_smoke_test"))

from episode_task_map import build_episode_task_map  # noqa: E402

from adapter import (  # noqa: E402
    install_libero_grasp_target_patch,
    install_libero_gripper_collapse_training_patch,
)

import lerobot.scripts.lerobot_train as lerobot_train_module  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

REPO_IDS = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus"}
FPS = {"libero": 10, "libero_plus": 20}
ACTION_HORIZON = 50
CHECKPOINTS = (10_000, 25_000, 50_000)


class TaskBalancedFrameSampler(torch.utils.data.Sampler[int]):
    """Copy of `train_grounded_grasp_v2_5_phase_aware_loss_100k.py`'s own sampler (not imported,
    per that script's own stated convention of keeping each experiment script self-contained),
    generalized from VLABench's named tasks to LIBERO's 40 `task_index` values."""

    def __init__(self, per_task_samplers: dict[int, EpisodeAwareSampler], num_samples: int, seed: int) -> None:
        self.per_task_samplers = per_task_samplers
        self.task_ids = sorted(per_task_samplers)
        self.num_samples = num_samples
        self.seed = seed
        self.draw_log: list[int] = []

    @staticmethod
    def _infinite(sampler: EpisodeAwareSampler):
        while True:
            yield from sampler

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        task_iters = {t: self._infinite(s) for t, s in self.per_task_samplers.items()}
        self.draw_log = []
        for _ in range(self.num_samples):
            task = self.task_ids[int(rng.integers(len(self.task_ids)))]
            idx = next(task_iters[task])
            self.draw_log.append(task)
            yield idx

    def __len__(self) -> int:
        return self.num_samples


def build_task_balanced_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
    _, eval_dataloader = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS(cfg, dataset, eval_dataset, step, parallel_dims)

    episode_task = build_episode_task_map(_CURRENT_DATASET_KEY[0])
    from_idx = dataset.meta.episodes["dataset_from_index"]
    to_idx = dataset.meta.episodes["dataset_to_index"]

    per_task_episode_indices: dict[int, list[int]] = {}
    for ep_idx in range(dataset.num_episodes):
        task = episode_task.get(ep_idx)
        if task is not None:
            per_task_episode_indices.setdefault(task, []).append(ep_idx)

    per_task_samplers: dict[int, EpisodeAwareSampler] = {}
    for i, (task, eps) in enumerate(per_task_episode_indices.items()):
        per_task_samplers[task] = EpisodeAwareSampler(
            from_idx, to_idx, episode_indices_to_use=eps, shuffle=True,
            seed=(cfg.seed or 0) + i + 1, absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )

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
    print(f"[final] task-balanced sampler over {len(per_task_samplers)} tasks, "
          f"{sum(len(v) for v in per_task_episode_indices.values())} episodes total")
    return dataloader, eval_dataloader


_CURRENT_DATASET_KEY = [None]  # set in main(); read by build_task_balanced_dataloaders


def build_smolvla_config() -> SmolVLAConfig:
    return SmolVLAConfig(
        device="cuda",
        push_to_hub=False,
        load_vlm_weights=True,
        chunk_size=ACTION_HORIZON,
        n_action_steps=ACTION_HORIZON,
        optimizer_lr=1e-4,
    )


def build_safediff_vla_final_config(steps: int) -> SafeDiffVLAConfig:
    return SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2_5",
        action_horizon=ACTION_HORIZON,
        execute_horizon=ACTION_HORIZON,
        use_temporal_ensembling=False,
        backbone_name=None,  # LIBERO domain -- not lerobot/smolvla_vlabench
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
        # Ignored at runtime once install_libero_grasp_target_patch() is active (LIBERO's own
        # fixed threshold/polarity is used instead) -- kept in (0,1) only to satisfy
        # SafeDiffVLAConfig's own validation, unchanged from the VLABench recipe.
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
        local_grasp_freeze_global_modules=False,  # training from scratch, not resuming Stage 2
        local_grasp_use_correction_head=False,
        local_grasp_use_phase_aware_loss=True,
        lambda_local_terminal_xyz=2.0,
        lambda_local_approach_xyz=1.0,
        lambda_local_grip_weighted=2.0,
        local_grasp_grip_close_weight=3.0,
        local_grasp_approach_weight_min=0.1,
        local_grasp_approach_weight_max=1.0,
        local_grasp_huber_delta=1.0,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=min(30_000, steps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--policy", choices=("smolvla", "safediff_vla_final"), required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--output-root", default="outputs/train/libero_final")
    args = parser.parse_args()

    _CURRENT_DATASET_KEY[0] = args.dataset
    fps = FPS[args.dataset]
    physical_horizon_s = ACTION_HORIZON / fps
    print(f"[final] dataset={args.dataset} fps={fps} action_horizon={ACTION_HORIZON} "
          f"-> physical_horizon={physical_horizon_s:.1f}s")
    print(f"[final] policy={args.policy} steps={args.steps} batch_size={args.batch_size} "
          f"checkpoints_requested={CHECKPOINTS}")

    if args.policy == "safediff_vla_final":
        install_libero_gripper_collapse_training_patch()
        install_libero_grasp_target_patch()
        policy_cfg = build_safediff_vla_final_config(args.steps)
    else:
        policy_cfg = build_smolvla_config()

    run_name = f"{args.dataset}_{args.policy}_final"
    output_dir = Path(args.output_root) / run_name

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_IDS[args.dataset]),  # all 40 tasks, all episodes
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=run_name,
        resume=False,
        seed=1000,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=100,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
    )

    lerobot_train_module._ORIGINAL_MAKE_DATALOADERS = lerobot_train_module.make_dataloaders
    lerobot_train_module.make_dataloaders = build_task_balanced_dataloaders
    try:
        lerobot_train_module.train(cfg)
    finally:
        lerobot_train_module.make_dataloaders = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS
        del lerobot_train_module._ORIGINAL_MAKE_DATALOADERS


if __name__ == "__main__":
    main()
