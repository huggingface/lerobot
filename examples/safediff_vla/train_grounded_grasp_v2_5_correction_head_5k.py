#!/usr/bin/env python
"""First experiment for `LocalGraspCorrectionHead` (see `grasp_correction_head.py`'s module
docstring and the manual 18-episode review, `local_grasp_correction_manual_review.py`, that
motivated it and set its two data-derived config values below).

Init checkpoint: the existing multitask Stage 2 v2.5 checkpoint (`outputs/train/
safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/checkpoints/005000/pretrained_model`) --
the SAME checkpoint `grasp_point_bias_diagnostic.py`/`trajectory_shape_diagnostic.py`/the manual
review all already analyzed, so this run and its eventual eval are directly comparable to that
existing body of diagnostics with nothing else changing underneath.

Trainable: ONLY `local_grasp_correction_head` (new, random init -- absent from the Stage 2
checkpoint, so non-strict `from_pretrained` reports it "missing" and it keeps its fresh init; this
is expected, not an error, same as every prior `*_5k.py` script's own pretrained-path init).
Frozen (verified post-training, not just asserted): backbone, `TargetQueryExtractor`,
`VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder`
(`local_grasp_freeze_global_modules=True`) AND `LocalGraspRefiner` itself
(`local_grasp_correction_freeze_refiner=True`, this pilot's explicit ask: isolate whether the new
head alone learns a real spatial-anchor correction before ever letting it co-adapt with the
refiner's own already-checkpointed control policy).

`local_grasp_max_grasp_correction_m=0.16` and `local_grasp_correction_outlier_clip_m=0.20` are NOT
arbitrary -- see the manual review's A_FINE_GRASP_ERROR-class residual distribution (p90 ~15.6cm,
max 16.4cm, n=4 -- SMALL SAMPLE) and its B_WRONG_OBJECT-class observations (21.7cm/22.5cm) that the
clip sits between.

Task set / task-balanced sampling: identical mechanism to Stage 2 (`TaskBalancedFrameSampler`,
copied rather than imported so that script stays untouched), restricted to the same 6-task
`STAGE2_TASK_NAMES`. No monkeypatch of `make_policy` needed this time (unlike Stage 2) --
`local_grasp_freeze_global_modules=True` already freezes all 4 v2 modules directly; only the
dataloader needs the same task-balancing monkeypatch as Stage 2.

5k steps only, single checkpoint at the end.

RESULT (this first run, see `eval_grounded_grasp_v2_5_correction_head_5k.py`'s comparison output
at `outputs/eval/safediff_vla_grounded_grasp_v2_5_correction_head_5k/comparison.json`): task
success on the same 18 dev episodes dropped 4/18 (stage2_baseline) -> 1/18 (this checkpoint).
Root cause, NOT "the design is wrong": `loss_grasp_correction` stayed ~0.000 for all 5000 logged
steps, and the actual learned correction magnitude on rollout is tiny (mean 1.2cm, max 2.7cm,
n=349 LOCAL_GRASP steps across all 18 episodes) -- nowhere near the ~13-16cm gap the manual review
(`local_grasp_correction_manual_review.py`) found necessary. This is a TRAINING-SUPERVISION-SOURCE
mismatch: `gt_target_xyz - target_xyz` measured on OFFLINE i.i.d. TRAINING SET frames is small by
construction, because `target_point_head` was already extensively trained (v2 15k + Stage1 5k +
Stage2 5k) to fit exactly those same labels -- the large 7-18cm bias this design exists to fix is a
ROLLOUT-TIME/closed-loop/out-of-distribution phenomenon (camera framing drift, compounding control
error near the object) that offline training frames never exhibit, so there was almost no genuine
signal for this head to learn from. Before extending this experiment: get supervision that reflects
the LOCAL_GRASP-time distribution (e.g. on-policy rollout frames with a privileged-state grasp
target, not only offline dataset frames) -- do not just scale up steps/lr on the current signal.
Separately, because LOCAL_GRASP replans every step, even this tiny correction measurably changes
full closed-loop trajectories (confirmed: the SAME frozen `target_point_head` produces different
`predicted_target` error magnitudes for the identical task/seed between this checkpoint's rollout
and the reused Stage 2 baseline rollout -- only possible via chaotic closed-loop divergence, not a
difference in the frozen head's own output) -- with n=3 episodes/task, the 4->1 success comparison
is likely dominated by that noise, not a reliable measurement of the correction's own effect.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_5_correction_head_5k.py
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
OUT_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_5_correction_head_5k")

# From the manual review (see module docstring) -- not picked without that data.
MAX_GRASP_CORRECTION_M = 0.16
CORRECTION_OUTLIER_CLIP_M = 0.20

FROZEN_MODULE_PREFIXES = (
    "backbone.", "decoder.", "target_query_extractor.", "visual_grounding_cross_attn.",
    "target_point_head.", "local_grasp_refiner.",
)
TRAINABLE_MODULE_PREFIXES = ("local_grasp_correction_head.",)


def preflight_check_init_checkpoint() -> None:
    """Confirms the Stage 2 checkpoint (everything this run initializes from) is what it claims to
    be before spending 5k steps on top of it -- just existence/loadability, not a weight diff (
    unlike Stage 2's own preflight, there is no "expected-frozen-since" reference to diff against:
    every module here, including `local_grasp_refiner`, is ALLOWED to have moved during Stage 2)."""
    ckpt = Path(STAGE2_5K_CHECKPOINT) / "model.safetensors"
    if not ckpt.exists():
        raise RuntimeError(f"Stage 2 init checkpoint not found at {ckpt} -- run Stage 2 first.")
    logger.info("preflight: found init checkpoint %s", ckpt)


def postflight_check_trained_checkpoint(init_path: str, final_path: str) -> None:
    """After training: everything except `local_grasp_correction_head` must be byte-identical to
    the Stage 2 init; the new head must have moved. Fails loudly rather than silently reporting an
    unverified freeze."""
    from safetensors.torch import load_file

    init_state = load_file(f"{init_path}/model.safetensors")
    final_state = load_file(f"{final_path}/model.safetensors")
    common = set(init_state) & set(final_state)
    changed = {k for k in common if not torch.equal(init_state[k], final_state[k])}
    new_keys = sorted(set(final_state) - set(init_state))

    frozen_changed = [k for k in changed if k.startswith(FROZEN_MODULE_PREFIXES)]
    if frozen_changed:
        raise RuntimeError(
            f"{len(frozen_changed)} weights in supposedly-frozen modules changed during training: {frozen_changed[:10]}"
        )
    logger.info("postflight: %d weights confirmed frozen across %d prefixes", len(common) - len(changed), len(FROZEN_MODULE_PREFIXES))
    logger.info("postflight: %d NEW keys present (local_grasp_correction_head, absent from Stage 2 init): %s", len(new_keys), new_keys[:5])
    head_keys = [k for k in final_state if k.startswith(TRAINABLE_MODULE_PREFIXES)]
    if not head_keys:
        raise RuntimeError("local_grasp_correction_head has zero parameters in the final checkpoint -- did it build at all?")
    logger.info("postflight: local_grasp_correction_head has %d parameter tensors in the final checkpoint (new module, moved from its own random init by construction).", len(head_keys))


class TaskBalancedFrameSampler(torch.utils.data.Sampler[int]):
    """Copy of Stage 1/2's sampler (not imported, so those scripts stay untouched)."""

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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--skip-postflight-check", action="store_true")
    args = parser.parse_args()
    if args.steps > 5000:
        raise ValueError("This pilot is 5k steps only -- do not extend without a separate decision.")
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
        local_grasp_freeze_global_modules=True,  # freeze all 4 v2 modules (Stage 2 left them trainable)
        local_grasp_use_correction_head=True,
        local_grasp_max_grasp_correction_m=MAX_GRASP_CORRECTION_M,
        local_grasp_correction_outlier_clip_m=CORRECTION_OUTLIER_CLIP_M,
        local_grasp_correction_hidden_dim=256,
        lambda_grasp_correction=1.0,
        local_grasp_correction_freeze_refiner=True,  # this pilot's explicit ask: head-only trainable
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
        job_name="safediff_vla_grounded_grasp_v2_5_correction_head_5k",
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

    if not args.skip_postflight_check:
        final_ckpt = out_dir / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
        postflight_check_trained_checkpoint(STAGE2_5K_CHECKPOINT, str(final_ckpt))


if __name__ == "__main__":
    main()
