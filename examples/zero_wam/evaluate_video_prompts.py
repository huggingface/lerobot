#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare correct, contrasting and absent video prompts on held-out streamed clips.

This is an offline flow-matching diagnostic, not a task-success or robot rollout benchmark.
Target text and demonstration captions are empty. Each noise draw is prepared once, including
MCP targets, then reused across all three conditions. Prompt shape and caption embeddings must
match, so changing captions or token counts cannot masquerade as changing video content.

Export DCP checkpoints first with ``lerobot-convert-dcp --checkpoint_dir=...``. The cases JSONL
declares source episode IDs, pool IDs, dataset revision and held-out split. See zero_wam.mdx.
"""

import argparse
import copy
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.zero_wam.configuration_zero_wam import ZeroWAMConfig
from lerobot.policies.zero_wam.modeling_zero_wam import ZeroWAMPolicy, get_mesh_id
from lerobot.policies.zero_wam.prompt_cache import resolve_prompt_path
from lerobot.policies.zero_wam.prompt_pool import ICLPromptPools
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION


@dataclass(frozen=True)
class PromptCase:
    target_episode: int
    target_task: str
    correct_episode: int
    contrasting_episode: int
    correct_latent: str
    contrasting_latent: str


def load_cases(
    cases_path: Path, pools_path: Path, splits_path: Path, repo_id: str, revision: str, split: str
) -> dict[int, PromptCase]:
    """Resolve explicit held-out-target / training-prompt pairs without sampling training pools."""
    splits = json.loads(splits_path.read_text())
    if splits.get("revision") != revision or split not in ("validation", "test"):
        raise ValueError("Cases require the dataset's pinned revision and a validation/test split")
    train, heldout = set(splits["train"]), set(splits[split])
    if train & (set(splits["validation"]) | set(splits["test"])) or set(splits["validation"]) & set(
        splits["test"]
    ):
        raise ValueError("Training and held-out episode IDs must be disjoint")
    ICLPromptPools(pools_path, "train")  # Reuse dataset/split/member integrity checks.
    paths: dict[tuple[str, int], set[str]] = {}
    pool_datasets = set()
    for line in pools_path.read_text().splitlines():
        if not line.strip():
            continue
        pool = json.loads(line)
        if pool["split"] != "train":
            continue
        pool_datasets.add(pool["dataset"])
        for member in pool["members"]:
            key = pool["pool_id"], member["episode_index"]
            paths.setdefault(key, set()).add(resolve_prompt_path(member["latent_path"], pools_path))
    cases = {}
    for line in cases_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("dataset_repo_id") != repo_id
            or row.get("revision") != revision
            or row.get("split") != split
        ):
            raise ValueError("Every case must declare the same source dataset, revision and held-out split")
        if row.get("pool_dataset") not in pool_datasets:
            raise ValueError("Case pool_dataset must match the encoded demonstration pools")
        target, correct, contrasting = (
            row[k] for k in ("target_episode", "correct_prompt_episode", "contrasting_prompt_episode")
        )
        if any(type(e) is not int for e in [target, correct, contrasting]):
            raise ValueError("Case episode IDs must be integers")
        if (
            target not in heldout
            or correct not in train
            or contrasting not in train
            or correct == contrasting
        ):
            raise ValueError("Use a held-out target and two distinct training prompt episodes")
        if target in cases:
            raise ValueError("Use one prompt comparison per target episode in a cases file")
        if not isinstance(row.get("target_task"), str) or not row["target_task"]:
            raise ValueError("Each case requires its exact target task label")
        resolved = []
        for kind, episode in [("correct", correct), ("contrasting", contrasting)]:
            candidates = paths.get((row[f"{kind}_pool_id"], episode), set())
            if len(candidates) != 1:
                raise ValueError("Each declared pool/episode must resolve to one encoded prompt segment")
            path = next(iter(candidates))
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            resolved.append(path)
        cases[target] = PromptCase(target, row["target_task"], correct, contrasting, *resolved)
    if not cases:
        raise ValueError("No diagnostic cases were supplied")
    return cases


@torch.no_grad()
def compare_video_prompts(
    policy: ZeroWAMPolicy,
    batch: dict,
    correct: str,
    contrasting: str,
    *,
    seed: int,
    chunk_size: int = 2,
    window_size: int = 16,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Use one normalized target clip and one fixed noising draw for a three-condition comparison.

    Only demonstration context changes. Calls to the policy's preparation/loss helpers keep the
    exact training loss and masks. This diagnostic is intended for a fresh, offline policy.
    """
    if batch[ACTION].shape[0] != 1:
        raise ValueError("Compare one target clip at a time")
    if (
        not 1 <= chunk_size <= policy.config.max_frame_chunk_size
        or not 1 <= window_size <= policy.config.max_attn_window
    ):
        raise ValueError("Diagnostic chunk/window sizes must be within the policy's training limits")
    device = torch.device(policy.config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The offline diagnostic supports CPU or CUDA")
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    overrides = {
        "drop_icl": 1.0,
        "droptext_target": 1.0,
        "train_frame_chunk_size": chunk_size,
        "train_attn_window": window_size,
    }
    previous = {name: getattr(policy.config, name) for name in overrides}
    was_training = policy.training
    velocities, conditions = {}, {}
    try:
        policy.eval()
        for name, value in overrides.items():
            setattr(policy.config, name, value)
        with torch.random.fork_rng(devices=devices):
            # Lazy checkpoint/text loading can consume RNG on the first call. Warm it before
            # setting the diagnostic seed, while still preserving the caller's RNG state.
            policy._ensure_frozen_modules()
            policy._encode_text("")
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            prepared = policy._prepare_training_sample(batch, 0)
            assert prepared["icl_latent_dict"] is None
            empty = prepared["text_emb"]
            contexts = {}
            for name, path in [("correct", correct), ("contrasting", contrasting)]:
                latent, text = policy._load_icl_latent(path)
                if text is None or not torch.equal(text.to(empty), empty):
                    raise ValueError(
                        "Video-only diagnostics require the exact padded empty-caption embedding"
                    )
                if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != policy.config.in_channels:
                    raise ValueError(
                        "Prompt latents must have shape [1, policy channels, frames, height, width]"
                    )
                contexts[name] = latent
            if contexts["correct"].shape != contexts["contrasting"].shape:
                raise ValueError("Correct and contrasting prompts must have the same latent shape")
            for name in ["correct", "contrasting", "absent"]:
                inputs = copy.deepcopy(prepared)
                if name != "absent":
                    latent = contexts[name]
                    _, _, frames, height, width = latent.shape
                    pf, ph, pw = policy.config.patch_size
                    grid = get_mesh_id(
                        frames // pf, height // ph, width // pw, t=0, h_shift=policy.config.icl_rope_h
                    ).to(device)
                    inputs["icl_latent_dict"] = {
                        "latent": latent,
                        "timesteps": latent.new_zeros((1, frames)),
                        "grid_id": grid[None],
                    }
                    inputs["text_emb"] = torch.cat([empty, empty], dim=1)
                    inputs["encoder_seq_ids"] = torch.cat(
                        [
                            torch.zeros(empty.shape[1], device=device, dtype=torch.int),
                            torch.ones(empty.shape[1], device=device, dtype=torch.int),
                        ]
                    )
                captured = []

                def capture(_module, _args, output, *, inputs=inputs, captured=captured):
                    # The action expert predicts flow velocity, not denoised robot actions.
                    pred = output[1].detach().float()
                    mask = inputs["action_dict"]["actions_mask"].clone()
                    mask[:, :, :1] = False  # Omit the artificial history frame from sensitivity metrics.
                    frames = mask.shape[-3]
                    pred = pred.reshape(1, frames, -1, pred.shape[-1]).permute(0, 3, 1, 2).unsqueeze(-1)
                    captured.append(pred[mask].cpu())

                hook = policy.transformer.register_forward_hook(capture)
                try:
                    with (
                        torch.autocast(device_type=device.type, dtype=amp_dtype)
                        if amp_dtype
                        else nullcontext()
                    ):
                        loss, parts = policy._sample_loss(inputs)
                finally:
                    hook.remove()
                if (
                    len(captured) != 1
                    or not captured[0].numel()
                    or not torch.isfinite(captured[0]).all()
                    or not torch.isfinite(loss)
                    or not all(torch.isfinite(v) for v in parts.values())
                ):
                    raise ValueError(
                        "Diagnostic forward must produce finite losses and one action prediction"
                    )
                velocities[name] = captured[0]
                conditions[name] = {
                    "total_loss": float(loss),
                    **{key: float(value) for key, value in parts.items()},
                    "icl_video_tokens": 0
                    if name == "absent"
                    else inputs["icl_latent_dict"]["grid_id"].shape[-1],
                }
    finally:
        for name, value in previous.items():
            setattr(policy.config, name, value)
        policy.train(was_training)
    changes = {
        f"correct_vs_{name}_action_velocity_rmse": float(
            (velocities["correct"] - velocities[name]).square().mean().sqrt()
        )
        for name in ["contrasting", "absent"]
    }
    return {
        "seed": seed,
        "chunk_size": chunk_size,
        "window_size": window_size,
        "target_and_demo_text": "empty",
        "prepared_noise_reused": True,
        "conditions": conditions,
        "future_action_velocity_elements": velocities["correct"].numel(),
        **changes,
        "scope": "Flow-matching loss and action-velocity sensitivity; not denoised action error or task success.",
    }


def run_comparisons(
    policy,
    preprocessor,
    dataset,
    cases,
    *,
    clips_per_episode,
    seeds,
    max_scanned,
    rename_map,
    chunk_size,
    window_size,
    amp_dtype,
    output,
):
    """Compare bounded, unpadded native streaming samples; write each completed noise draw immediately."""
    tasks = {int(row["episode_index"]): set(row["tasks"]) for row in dataset.meta.episodes}
    for case in cases.values():
        if (
            case.target_task not in tasks[case.target_episode]
            or case.target_task not in tasks[case.correct_episode]
        ):
            raise ValueError("Target and correct prompt must share the declared dataset task")
        if case.target_task in tasks[case.contrasting_episode]:
            raise ValueError("Contrasting prompt must come from a different dataset task")
    counts = dict.fromkeys(cases, 0)
    seen = set()
    records = []
    iterator = iter(dataset)
    try:
        for sample in islice(iterator, max_scanned):
            episode, frame = int(sample["episode_index"]), int(sample["frame_index"])
            if episode not in cases or counts[episode] >= clips_per_episode or (episode, frame) in seen:
                continue
            if any(torch.as_tensor(value).any() for key, value in sample.items() if key.endswith("_is_pad")):
                continue
            batch = lerobot_collate_fn([sample])
            for camera in dataset.meta.camera_keys:
                if batch[camera].dtype == torch.uint8:
                    batch[camera] = batch[camera].float() / 255.0
            batch = preprocessor(rename_batch_keys(batch, rename_map))
            case = cases[episode]
            for seed in seeds:
                result = compare_video_prompts(
                    policy,
                    batch,
                    case.correct_latent,
                    case.contrasting_latent,
                    seed=seed,
                    chunk_size=chunk_size,
                    window_size=window_size,
                    amp_dtype=amp_dtype,
                )
                record = {
                    "target_episode": episode,
                    "target_frame": frame,
                    "target_task": case.target_task,
                    "correct_prompt_episode": case.correct_episode,
                    "contrasting_prompt_episode": case.contrasting_episode,
                    "correct_latent": case.correct_latent,
                    "contrasting_latent": case.contrasting_latent,
                    **result,
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                records.append(record)
            seen.add((episode, frame))
            counts[episode] += 1
            if all(count >= clips_per_episode for count in counts.values()):
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if not all(count >= clips_per_episode for count in counts.values()):
        raise ValueError(f"Not enough complete clips before max-scanned limit: {counts}")
    return {
        "status": "completed_offline_loss_diagnostic",
        "clips_per_episode": counts,
        "target_clips": len(seen),
        "noise_draws": len(records),
        "conditions_per_draw": 3,
        "scope": "Offline conditional loss/sensitivity only; no robot or success evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy", type=Path, required=True, help="Local exported native checkpoint with processors"
    )
    parser.add_argument(
        "--train-config", type=Path, required=True, help="Reuse pinned dataset/streaming settings"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prompt-pools", type=Path, required=True)
    parser.add_argument("--episode-splits", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-embedding-cache", type=Path)
    parser.add_argument("--clips-per-episode", type=int, default=1)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-scanned-clips", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run the exported-checkpoint diagnostic in one process")
    if (
        args.clips_per_episode < 1
        or args.max_scanned_clips < 1
        or len(set(args.noise_seeds)) != len(args.noise_seeds)
    ):
        parser.error("Use positive clip limits and distinct noise seeds")
    if not args.policy.is_dir() or not list(args.policy.glob("model*.safetensors")):
        parser.error("Use an exported local checkpoint; convert DCP first with lerobot-convert-dcp")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".summary.json").exists():
        parser.error(f"Refusing to overwrite diagnostic results: {args.output}")
    cfg = TrainPipelineConfig.from_pretrained(args.train_config)
    if not cfg.dataset.streaming or not cfg.dataset.revision:
        parser.error("Reuse a pinned LeRobot streaming dataset configuration")
    cases = load_cases(
        args.cases,
        args.prompt_pools,
        args.episode_splits,
        cfg.dataset.repo_id,
        cfg.dataset.revision,
        args.split,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy, cli_overrides=[f"--device={args.device}"])
    if not isinstance(policy_cfg, ZeroWAMConfig):
        parser.error("This diagnostic requires a Zero-WAM policy")
    if args.text_embedding_cache:
        policy_cfg.text_embedding_cache_path = str(args.text_embedding_cache)
    cfg.policy = policy_cfg
    cfg.dataset.episodes = sorted(cases)
    cfg.dataset.image_transforms.enable = False
    cfg.num_workers = 0
    dataset = make_dataset(cfg)
    policy = ZeroWAMPolicy.from_pretrained(args.policy, config=policy_cfg, strict=True)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    amp_dtype = (
        torch.bfloat16 if args.amp == "bfloat16" and torch.device(args.device).type == "cuda" else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        summary = run_comparisons(
            policy,
            preprocessor,
            dataset,
            cases,
            clips_per_episode=args.clips_per_episode,
            seeds=args.noise_seeds,
            max_scanned=args.max_scanned_clips,
            rename_map=cfg.rename_map,
            chunk_size=args.chunk_size,
            window_size=args.window_size,
            amp_dtype=amp_dtype,
            output=output,
        )
    summary.update(
        policy=str(args.policy),
        dataset=cfg.dataset.repo_id,
        revision=cfg.dataset.revision,
        split=args.split,
        image_augmentation=False,
        results=str(args.output),
        cases=str(args.cases),
        prompt_pools=str(args.prompt_pools),
        device=args.device,
        noise_seeds=args.noise_seeds,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        amp=str(amp_dtype) if amp_dtype is not None else "none",
        input_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "cases": args.cases,
                "prompt_pools": args.prompt_pools,
                "episode_splits": args.episode_splits,
                "train_config": args.train_config,
                "policy_config": args.policy / "config.json",
            }.items()
        },
    )
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
