#!/usr/bin/env python
"""Long-convergence run for the grasp-phase-aware `LocalGraspRefiner` loss (see
`train_grounded_grasp_v2_5_phase_aware_loss_5k.py`'s own docstring for the full loss description --
UNCHANGED here, same 3 terms/coefficients/thresholds). The 5k pilot showed the loss demonstrably
changes LOCAL_GRASP behavior (terminal distance / lateral precision / close timing all showed
movement in two independent 5k-scale checks) but did not yet translate into a stable end-to-end
success-rate improvement at 5k steps. This run tests whether a much longer training horizon does.

Init: the SAME Stage 2 baseline checkpoint every prior experiment on this branch used
(`STAGE2_5K_CHECKPOINT` below) -- NOT resumed from the previous phase-aware 5k checkpoint, so this
run is a clean, directly-comparable continuation of the exact same starting point, not a
continuation of the 5k pilot's own (already-diverged) weights.

Trainable: `LocalGraspRefiner` ONLY. Frozen: backbone, `TargetQueryExtractor`,
`VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder` -- identical to the 5k
pilot's freeze pattern, verified byte-for-byte in postflight below.

Coefficients/thresholds: IDENTICAL to the 5k pilot (`lambda_local_terminal_xyz=2.0`,
`lambda_local_approach_xyz=1.0`, `lambda_local_grip_weighted=2.0`, `local_grasp_grip_close_weight=
3.0`, approach weight schedule 0.1->1.0, huber delta=1.0) -- explicitly NOT tuned for this run, per
instructions ("coefficient, architecture, threshold는 변경하지 마세요"). `scheduler_warmup_steps=
1_000`/`scheduler_decay_steps=30_000` are ALSO carried over verbatim from every prior 5k-scale
script on this branch (not re-tuned for the new 100k horizon either, same "don't touch the recipe"
principle) -- this means the LR schedule reaches its decay floor (1e-5) at step 30_000 and stays
there for the remaining 70k steps, a deliberate consequence of holding the recipe fixed rather than
an oversight.

Single continuous `train()` call for the full 100_000 steps (no resume-based staging): this
trivially keeps the optimizer/scheduler state continuous for the whole run (a single Python process
never stops), rather than relying on this codebase's CLI-driven `--resume`/`--config_path`
mechanism (which reads real `sys.argv`, not a program-built `TrainPipelineConfig`, and so doesn't
compose cleanly with this script's existing non-CLI construction pattern). `save_freq=5_000` is the
largest common divisor of the 5 checkpoints actually requested (10k/25k/50k/75k/100k), so exactly
those 5 land as an (unlabeled) subset of the 20 saved checkpoints -- the other 15 are harmless
disk-space spillover (~1GB each, plentiful free space), not something later scripts need to touch.

Health monitoring (divergence/NaN/degenerate gripper) happens externally, by tailing this run's
stdout/log while it executes in the background -- not built into this script, so a crash/kill here
never leaves training-loop state half-mutated.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_5_phase_aware_loss_100k.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID, STAGE2_TASK_NAMES, build_episode_task_map  # noqa: E402

import lerobot.scripts.lerobot_train as lerobot_train_module  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SPLIT = 0.1
STAGE2_5K_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/checkpoints/005000/pretrained_model"
OUT_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k")
REQUESTED_CHECKPOINTS = (10_000, 25_000, 50_000, 75_000, 100_000)

FROZEN_MODULE_PREFIXES = (
    "backbone.", "decoder.", "target_query_extractor.", "visual_grounding_cross_attn.", "target_point_head.",
)
TRAINABLE_MODULE_PREFIXES = ("local_grasp_refiner.",)


def preflight_check_init_checkpoint() -> None:
    ckpt = Path(STAGE2_5K_CHECKPOINT) / "model.safetensors"
    if not ckpt.exists():
        raise RuntimeError(f"Stage 2 init checkpoint not found at {ckpt} -- run Stage 2 first.")
    logger.info("preflight: found init checkpoint %s", ckpt)


def postflight_check_trained_checkpoint(init_path: str, final_path: str) -> None:
    from safetensors.torch import load_file

    init_state = load_file(f"{init_path}/model.safetensors")
    final_state = load_file(f"{final_path}/model.safetensors")
    common = set(init_state) & set(final_state)
    changed = {k for k in common if not torch.equal(init_state[k], final_state[k])}

    frozen_changed = [k for k in changed if k.startswith(FROZEN_MODULE_PREFIXES)]
    if frozen_changed:
        raise RuntimeError(
            f"{len(frozen_changed)} weights in supposedly-frozen modules changed during training: {frozen_changed[:10]}"
        )
    for prefix in TRAINABLE_MODULE_PREFIXES:
        n_keys = sum(1 for k in common if k.startswith(prefix))
        n_changed = sum(1 for k in changed if k.startswith(prefix))
        logger.info("postflight: %-24s %d/%d weights changed", prefix, n_changed, n_keys)
        if n_keys > 0 and n_changed == 0:
            raise RuntimeError(f"Module '{prefix}' was supposed to train but NONE of its {n_keys} weights changed.")
    logger.info(
        "postflight: %d weights confirmed frozen (backbone/decoder/target_query_extractor/"
        "visual_grounding_cross_attn/target_point_head); local_grasp_refiner confirmed to have moved.",
        len(common) - len(changed),
    )


class TaskBalancedFrameSampler(torch.utils.data.Sampler[int]):
    """Copy of every prior stage's sampler (not imported, so those scripts stay untouched)."""

    def __init__(self, per_task_samplers: dict[str, EpisodeAwareSampler], num_samples: int, seed: int) -> None:
        self.per_task_samplers = per_task_samplers
        self.task_names = sorted(per_task_samplers)
        self.num_samples = num_samples
        self.seed = seed
        self.draw_log: list[str] = []

    @staticmethod
    def _infinite(sampler: EpisodeAwareSampler):
        while True:
            yield from sampler

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
    _, eval_dataloader = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS(cfg, dataset, eval_dataset, step, parallel_dims)

    episode_task = build_episode_task_map()
    train_episodes = set(dataset.episodes) if dataset.episodes is not None else set(range(dataset.num_episodes))
    from_idx = dataset.meta.episodes["dataset_from_index"]
    to_idx = dataset.meta.episodes["dataset_to_index"]

    per_task_episode_indices: dict[str, list[int]] = {name: [] for name in STAGE2_TASK_NAMES}
    for ep_idx in train_episodes:
        task = episode_task.get(ep_idx)
        if task in per_task_episode_indices:
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
    per_task_sampled = {t: draw_log.count(t) for t in STAGE2_TASK_NAMES}
    log = {
        "n_draws": n_draws,
        "batch_size": batch_size,
        "per_task_train_episode_counts": _sampling_state["per_task_episode_counts"],
        "per_task_frame_pool_counts": _sampling_state["per_task_frame_pool_counts"],
        "per_task_sampled_frame_counts": per_task_sampled,
        "per_task_realized_fraction": {t: c / n_draws for t, c in per_task_sampled.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_balanced_sampling_log.json").write_text(json.dumps(log, indent=2))
    logger.info("wrote %s", out_dir / "task_balanced_sampling_log.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--skip-postflight-check", action="store_true")
    args = parser.parse_args()
    if args.steps > 100_000:
        raise ValueError("This run is capped at 100k steps -- do not extend without a separate decision.")
    out_dir = Path(args.output_dir)

    if not args.skip_preflight_check:
        preflight_check_init_checkpoint()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2_5",
        pretrained_path=STAGE2_5K_CHECKPOINT,
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
        local_grasp_use_correction_head=False,
        # ---- this experiment's subject -- IDENTICAL to the 5k pilot, not re-tuned ----
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
        scheduler_decay_steps=30_000,
    )

    episode_task = build_episode_task_map()
    union_episodes = sorted(ep for ep, t in episode_task.items() if t in STAGE2_TASK_NAMES)
    logger.info("training dataset restricted to %d episodes across %d tasks (select_ingredient excluded)", len(union_episodes), len(STAGE2_TASK_NAMES))
    logger.info("requested checkpoints: %s (subset of every save_freq=%d multiple)", REQUESTED_CHECKPOINTS, args.save_freq)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, episodes=union_episodes, eval_split=EVAL_SPLIT),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=out_dir,
        job_name="safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k",
        resume=False,
        seed=1000,
        num_workers=8,
        batch_size=4,
        steps=args.steps,
        log_freq=args.log_freq,
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

    if "sampler" in _sampling_state:
        write_sampling_log(out_dir, cfg.batch_size)

    if not args.skip_postflight_check:
        final_ckpt = out_dir / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
        postflight_check_trained_checkpoint(STAGE2_5K_CHECKPOINT, str(final_ckpt))


if __name__ == "__main__":
    main()
