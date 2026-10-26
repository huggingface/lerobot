#!/usr/bin/env python
"""B-track Stage 2: multi-task adaptation of v2's grounding/localization branch
(`TargetQueryExtractor`, `VisualGroundingCrossAttention`, `TargetPointHead`), together with
`LocalGraspRefiner` (kept trainable, continuing Stage 1's adaptation), while
`TemporalActionDecoder` and the SmolVLA backbone stay frozen. Question: Stage 1 showed
`LocalGraspRefiner` reduces EE-to-object distance after LOCAL_GRASP entry in 21/21 rollouts across
6 valid task categories (`select_ingredient` excluded -- see `_multitask_common.STAGE2_TASK_NAMES`'s
docstring: confirmed train/eval scene-distribution mismatch for that task, not a model-quality
issue), but task/grasp SUCCESS barely moved, and for tasks with poor localization (LOCAL_GRASP
entering far from the real object) the refiner has nothing to recover from. Stage 2 asks: does
letting the GROUNDING pipeline itself adapt across tasks (not just the short-horizon local
controller) fix that upstream localization gap and translate into real success?

Init checkpoint: `outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k/checkpoints/
005000` (Stage 1's own output).

Trainable (verified after construction AND after training, not just asserted):
    TargetQueryExtractor, VisualGroundingCrossAttention, TargetPointHead, LocalGraspRefiner
Frozen:
    SmolVLA backbone (`freeze_backbone=True`/`freeze_vision_encoder=True`, same as every prior
    script on this branch), TemporalActionDecoder (NOT covered by any existing config toggle --
    `local_grasp_freeze_global_modules=False` leaves ALL FOUR non-backbone modules trainable at
    construction, see `local_grasp_freeze_global_modules`'s only use site in
    `modeling_safediff_vla.py`, so this script monkeypatches `lerobot_train.make_policy` --
    same non-invasive per-script monkeypatch pattern as Stage 1's dataloader swap -- to call
    `policy.decoder.requires_grad_(False)` immediately after construction, before the optimizer is
    built from `get_optim_params()` (`(p for p in self.parameters() if p.requires_grad)` --
    verified in `pretrained.py`/`modeling_safediff_vla.py`). No config/modeling/architecture file is
    touched.)

Task set / task-balanced sampling: IDENTICAL mechanism to Stage 1 (`TaskBalancedFrameSampler`,
copied rather than imported so Stage 1's own script is never touched), restricted to the 6-task
`STAGE2_TASK_NAMES` (select_ingredient excluded -- see above).

5k steps only, single checkpoint at the end. Every other hyperparameter (optimizer/scheduler/
action_horizon/local_grasp_* thresholds/lambda weights) copied verbatim from Stage 1 -- no
threshold sweep, no architecture change, no backbone/decoder unfreeze, no 20k extension.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_5_multitask_stage2_5k.py
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
STAGE1_5K_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k/checkpoints/005000/pretrained_model"
OUT_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k")

# Same 618 "global" keys Stage 1 verified frozen. Stage 2 additionally trains
# target_query_extractor/visual_grounding_cross_attn/target_point_head, so only backbone + decoder
# (the *actually* frozen ones this stage) are expected to stay byte-identical.
FROZEN_MODULE_PREFIXES = ("backbone.", "decoder.")
TRAINABLE_MODULE_PREFIXES = (
    "target_query_extractor.", "visual_grounding_cross_attn.", "target_point_head.", "local_grasp_refiner.",
)


def preflight_check_init_checkpoint(pretrained_path: str) -> None:
    """Confirms Stage 1's own global-module weights (everything outside local_grasp_refiner) are
    still byte-identical to v2-15k before Stage 2 starts training on top of them."""
    from safetensors.torch import load_file

    v2_path = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/015000/pretrained_model/model.safetensors"
    init_path = f"{pretrained_path}/model.safetensors"
    if not Path(v2_path).exists() or not Path(init_path).exists():
        logger.warning("preflight: skipping init-checkpoint identity check (one of the files is missing)")
        return
    v2_state = load_file(v2_path)
    init_state = load_file(init_path)
    common = set(v2_state) & set(init_state)
    diffs = [k for k in common if not torch.equal(v2_state[k], init_state[k])]
    non_local_diffs = [k for k in diffs if not k.startswith("local_grasp_refiner.")]
    logger.info(
        "preflight: init checkpoint (Stage 1 5k) vs v2-15k: %d/%d common weights differ (%d local_grasp_refiner, %d other)",
        len(diffs), len(common), len(diffs) - len(non_local_diffs), len(non_local_diffs),
    )
    if non_local_diffs:
        raise RuntimeError(
            f"{len(non_local_diffs)} non-local_grasp_refiner weights differ between v2-15k and the "
            f"Stage 1 checkpoint (expected 0, since Stage 1 froze everything but local_grasp_refiner): "
            f"{non_local_diffs[:10]}. Aborting rather than starting Stage 2 from an unverified init."
        )


def postflight_check_trained_checkpoint(init_path: str, final_path: str) -> None:
    """After training: backbone/decoder must be byte-identical to init; the 4 Stage-2-trainable
    modules must have moved. Fails loudly rather than silently reporting an unverified freeze."""
    from safetensors.torch import load_file

    init_state = load_file(f"{init_path}/model.safetensors")
    final_state = load_file(f"{final_path}/model.safetensors")
    common = set(init_state) & set(final_state)
    changed = {k for k in common if not torch.equal(init_state[k], final_state[k])}

    frozen_changed = [k for k in changed if k.startswith(FROZEN_MODULE_PREFIXES)]
    if frozen_changed:
        raise RuntimeError(
            f"{len(frozen_changed)} weights in supposedly-frozen modules (backbone/decoder) changed "
            f"during Stage 2 training: {frozen_changed[:10]}"
        )
    for prefix in TRAINABLE_MODULE_PREFIXES:
        n_keys = sum(1 for k in common if k.startswith(prefix))
        n_changed = sum(1 for k in changed if k.startswith(prefix))
        logger.info("postflight: %-32s %d/%d weights changed", prefix, n_changed, n_keys)
        if n_keys > 0 and n_changed == 0:
            raise RuntimeError(f"Module '{prefix}' was supposed to train but NONE of its {n_keys} weights changed.")
    logger.info("postflight: backbone/decoder confirmed frozen (0 changes); all 4 trainable modules confirmed to have moved.")


class TaskBalancedFrameSampler(torch.utils.data.Sampler[int]):
    """Copy of Stage 1's sampler (not imported, so Stage 1's own script stays untouched) -- see
    `train_grounded_grasp_v2_5_multitask_stage1_5k.py` for the full rationale."""

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


def build_policy_with_stage2_freeze(*args, **kwargs):
    """Monkeypatch target for `lerobot_train_module.make_policy`. Constructs the policy exactly as
    the library would (with `local_grasp_freeze_global_modules=False`, so target_query_extractor/
    visual_grounding_cross_attn/target_point_head/decoder are all left trainable at construction --
    see `modeling_safediff_vla.py`'s only use site of that flag), then re-freezes ONLY the decoder
    (TemporalActionDecoder) -- backbone freezing is already handled by the existing
    `freeze_backbone`/`freeze_vision_encoder` config fields, unrelated to this flag."""
    policy = lerobot_train_module._ORIGINAL_MAKE_POLICY(*args, **kwargs)
    if hasattr(policy, "decoder"):
        n_before = sum(1 for p in policy.decoder.parameters() if p.requires_grad)
        policy.decoder.requires_grad_(False)
        logger.info("Stage 2 freeze: policy.decoder re-frozen (%d params were trainable, now 0)", n_before)
    return policy


def write_sampling_log(out_dir: Path, batch_size: int) -> None:
    sampler: TaskBalancedFrameSampler = _sampling_state["sampler"]
    draw_log = sampler.draw_log
    n_draws = len(draw_log)
    per_task_sampled = {t: draw_log.count(t) for t in STAGE2_TASK_NAMES}

    n_batches = n_draws // batch_size
    per_task_batches_containing = {t: 0 for t in STAGE2_TASK_NAMES}
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
        "target_fraction_per_task": 1.0 / len(STAGE2_TASK_NAMES),
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
    logger.info("=== realized per-task sampling fractions (target: %.4f each) ===", 1.0 / len(STAGE2_TASK_NAMES))
    for t in STAGE2_TASK_NAMES:
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
    parser.add_argument("--skip-postflight-check", action="store_true")
    args = parser.parse_args()
    if args.steps > 5000:
        raise ValueError("Stage 2 is 5k steps only -- do not extend without a separate decision.")
    out_dir = Path(args.output_dir)

    if not args.skip_preflight_check:
        preflight_check_init_checkpoint(STAGE1_5K_CHECKPOINT)

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2_5",
        pretrained_path=STAGE1_5K_CHECKPOINT,
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
        local_grasp_freeze_global_modules=False,  # Stage 2: unfreeze grounding branch (see docstring)
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    episode_task = build_episode_task_map()
    union_episodes = sorted(ep for ep, t in episode_task.items() if t in STAGE2_TASK_NAMES)
    logger.info("training dataset restricted to %d episodes across %d tasks (select_ingredient excluded)", len(union_episodes), len(STAGE2_TASK_NAMES))

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, episodes=union_episodes, eval_split=EVAL_SPLIT),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=out_dir,
        job_name="safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k",
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
    lerobot_train_module._ORIGINAL_MAKE_POLICY = lerobot_train_module.make_policy
    lerobot_train_module.make_policy = build_policy_with_stage2_freeze
    try:
        lerobot_train_module.train(cfg)
    finally:
        lerobot_train_module.make_dataloaders = lerobot_train_module._ORIGINAL_MAKE_DATALOADERS
        del lerobot_train_module._ORIGINAL_MAKE_DATALOADERS
        lerobot_train_module.make_policy = lerobot_train_module._ORIGINAL_MAKE_POLICY
        del lerobot_train_module._ORIGINAL_MAKE_POLICY

    if "sampler" in _sampling_state:
        write_sampling_log(out_dir, cfg.batch_size)

    if not args.skip_postflight_check:
        final_ckpt = out_dir / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
        postflight_check_trained_checkpoint(STAGE1_5K_CHECKPOINT, str(final_ckpt))


if __name__ == "__main__":
    main()
