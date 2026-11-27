# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 1 of CAE-FastWAM: train the action VAE on the action chunks of a `LeRobotDataset`.

`lerobot-train-tokenizer --tokenizer_type=cae_action_vae` runs `train_cae_action_vae_from_config`. The
objective and the defaults are those of `FastWAM/scripts/train_action_ae.py` and
`FastWAM/src/fastwam/models/action_ae.py` of https://github.com/AMD-AGI/faster-action-models (Apache-2.0,
Copyright (c) 2026 Advanced Micro Devices, Inc.): masked MSE reconstruction, a 0.1-weighted velocity
term, a free-bits KL with a linear warmup, Adam, gradient clipping and EMA weights.

The output directory holds `config.json` (architecture, normalization mode, suite name), `model.safetensors`
(EMA weights of `encoder.*` and `decoder.*`), `stats.json` (the dataset statistics the actions were
normalized with, in the `task_suite_stats` layout) and `metadata.json` (training options and metrics).
`--policy.action_vae_path` points CAE-FastWAM at such a directory or Hub repo.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .modeling_cae_fastwam import CAEActionDecoder, CAEActionEncoder

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

logger = logging.getLogger(__name__)

ACTION_VAE_TYPE = "cae_action_vae"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
STATS_NAME = "stats.json"
METADATA_NAME = "metadata.json"


@dataclass
class CAEActionVAETrainingConfig:
    """Options of the CAE action VAE (`--cae.*` of `lerobot-train-tokenizer --tokenizer_type=cae_action_vae`).

    The defaults are the released LIBERO action VAE's (8 latent tokens of width 16 for a 32-step chunk).
    Keep `num_tokens` and `latent_dim` to warm-start the action expert from `amd/CAE-FastWAM-lerobot`.
    """

    # Latent tokens per chunk (`action_horizon` must be a multiple) and their width.
    num_tokens: int = 8
    latent_dim: int = 16
    encoder_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    # Variational encoder (mean and log-variance) with a free-bits KL, as the released VAE.
    variational: bool = True
    beta_kl: float = 1e-4
    free_bits: float = 0.1
    kl_warmup_epochs: int = 100
    velocity_weight: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    seed: int = 1000
    # Name of the dataset's entry in the policy's `task_suites` / `task_suite_stats` (default: the repo id).
    suite_name: str | None = None
    # Training device (default: CUDA when available, else CPU).
    device: str | None = None


class CAEActionVAE(nn.Module):
    """The action encoder and decoder of CAE-FastWAM (`CAEActionEncoder`, `CAEActionDecoder`) and the stage 1 loss."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        num_tokens: int,
        latent_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        num_suites: int,
        variational: bool,
    ):
        super().__init__()
        self.encoder = CAEActionEncoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            num_suites=num_suites,
            variational=variational,
        )
        self.decoder = CAEActionDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            num_suites=num_suites,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CAEActionVAE:
        """Build the VAE described by a saved `config.json`."""
        return cls(
            action_dim=config["action_dim"],
            action_horizon=config["action_horizon"],
            num_tokens=config["num_tokens"],
            latent_dim=config["latent_dim"],
            encoder_hidden_dim=config["encoder_hidden_dim"],
            decoder_hidden_dim=config["decoder_hidden_dim"],
            num_suites=len(config["task_suites"]),
            variational=config["variational"],
        )

    def forward(self, actions: Tensor, suite_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Encode (a posterior sample for a variational encoder) and decode `[B, horizon, action_dim]` actions."""
        encoded = self.encoder(actions, suite_ids)
        return {"recon": self.decoder(encoded["z"], suite_ids), **encoded}

    def compute_loss(
        self,
        actions: Tensor,
        action_is_pad: Tensor | None = None,
        suite_ids: Tensor | None = None,
        beta_kl: float = 0.0,
        free_bits: float = 0.0,
        velocity_weight: float = 0.1,
    ) -> dict[str, Tensor]:
        """Masked MSE reconstruction + `velocity_weight` x velocity MSE + `beta_kl` x free-bits KL."""
        output = self(actions, suite_ids)
        recon = output["recon"]
        error = functional.mse_loss(recon, actions, reduction="none").mean(dim=-1)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(error.dtype)
            error = (error * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            error = error.mean(1)
        loss_recon = error.mean()
        loss_vel = functional.mse_loss(recon[:, 1:] - recon[:, :-1], actions[:, 1:] - actions[:, :-1])
        loss = loss_recon + velocity_weight * loss_vel
        loss_kl = torch.zeros((), device=actions.device)
        if "mu" in output and beta_kl > 0:
            kl_per_dim = -0.5 * (1 + output["log_var"] - output["mu"].pow(2) - output["log_var"].exp())
            if free_bits > 0:
                kl_per_dim = kl_per_dim.clamp(min=free_bits)
            loss_kl = kl_per_dim.mean()
            loss = loss + beta_kl * loss_kl
        return {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_vel": loss_vel.detach(),
            "loss_kl": loss_kl.detach(),
        }


def _episode_actions(dataset: LeRobotDataset, max_episodes: int | None) -> list[np.ndarray]:
    """The `[length, action_dim]` actions of each selected episode, in episode order."""
    try:
        table = dataset.hf_dataset.select_columns([ACTION, "episode_index", "frame_index"]).with_format(
            "numpy"
        )
        actions = np.asarray(table[ACTION], dtype=np.float32)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        frame_index = np.asarray(table["frame_index"]).reshape(-1)
    except AttributeError:  # storage formats without an `hf_dataset`
        rows = [dataset.get_raw_item(i) for i in range(dataset.num_frames)]
        actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows])
        episode_index = np.array([int(row["episode_index"]) for row in rows])
        frame_index = np.array([int(row["frame_index"]) for row in rows])
    order = np.lexsort((frame_index, episode_index))
    actions, episode_index = actions[order], episode_index[order]
    starts = np.unique(episode_index, return_index=True)[1]
    ends = np.append(starts[1:], len(episode_index))
    if max_episodes is not None:
        starts, ends = starts[:max_episodes], ends[:max_episodes]
    return [actions[start:end] for start, end in zip(starts, ends, strict=True)]


def build_action_chunks(
    dataset: LeRobotDataset,
    action_horizon: int,
    *,
    max_episodes: int | None = None,
    sample_fraction: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Every `action_horizon`-step action chunk of the dataset, as `lerobot-train` builds them.

    A chunk starts at every frame. Steps past the end of the episode repeat its last action and are marked
    in `action_is_pad`, like LeRobot's `delta_timestamps` padding. With `sample_fraction` < 1, that
    fraction of each episode's chunks is drawn at random.

    Returns:
        tuple[np.ndarray, np.ndarray]: Chunks `[N, action_horizon, action_dim]` (float32) and
        `action_is_pad` `[N, action_horizon]` (bool).
    """
    rng = np.random.default_rng(seed)
    offsets = np.arange(action_horizon)
    chunks, pads = [], []
    for actions in _episode_actions(dataset, max_episodes):
        length = len(actions)
        starts = np.arange(length)
        if sample_fraction < 1.0:
            starts = np.sort(rng.choice(length, size=max(1, int(length * sample_fraction)), replace=False))
        steps = starts[:, None] + offsets[None, :]
        chunks.append(actions[np.minimum(steps, length - 1)])
        pads.append(steps >= length)
    return np.concatenate(chunks), np.concatenate(pads)


def normalize_action_chunks(chunks: np.ndarray, stats: dict[str, Any], mode: NormalizationMode) -> np.ndarray:
    """Normalize actions with LeRobot's `NormalizerProcessorStep` (the policy's preprocessor step)."""
    if mode == NormalizationMode.IDENTITY:
        return chunks
    from lerobot.processor import NormalizerProcessorStep, TransitionKey, create_transition

    norm_map: dict[FeatureType, NormalizationMode] = {FeatureType.ACTION: mode}
    step = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(chunks.shape[-1],))},
        norm_map=norm_map,
        stats={ACTION: stats},
    )
    normalized = step(create_transition(action=torch.from_numpy(np.ascontiguousarray(chunks))))[
        TransitionKey.ACTION
    ]
    if not isinstance(normalized, Tensor):
        raise TypeError(f"Expected a normalized action tensor, got {type(normalized).__name__}.")
    return normalized.numpy()


@torch.no_grad()
def evaluate_action_vae(
    vae: CAEActionVAE,
    chunks: Tensor,
    action_is_pad: Tensor,
    suite_ids: Tensor | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Reconstruction error of normalized chunks over the valid (non-padding) steps.

    `mse` decodes the posterior mean (deterministic), `mse_sample` a posterior sample (what the policy's
    training targets are). Velocity MSE and per-dimension MSE use the posterior mean.
    """
    vae.eval()
    sums = {"mse": 0.0, "mse_sample": 0.0, "vel_mse": 0.0}
    per_dim = torch.zeros(chunks.shape[-1], dtype=torch.float64)
    valid_steps = valid_vel = 0
    for start in range(0, len(chunks), batch_size):
        actions = chunks[start : start + batch_size]
        valid = ~action_is_pad[start : start + batch_size]
        ids = None if suite_ids is None else suite_ids[start : start + batch_size]
        encoded = vae.encoder(actions, ids)
        latents = encoded.get("mu", encoded["z"])
        recon = vae.decoder(latents, ids)
        recon_sample = vae.decoder(encoded["z"], ids)
        error = (recon - actions).pow(2)
        sums["mse"] += error.mean(-1)[valid].double().sum().item()
        sums["mse_sample"] += (recon_sample - actions).pow(2).mean(-1)[valid].double().sum().item()
        per_dim += error[valid].double().sum(0).cpu()
        valid_pair = valid[:, 1:] & valid[:, :-1]
        vel_error = ((recon[:, 1:] - recon[:, :-1]) - (actions[:, 1:] - actions[:, :-1])).pow(2).mean(-1)
        sums["vel_mse"] += vel_error[valid_pair].double().sum().item()
        valid_steps += int(valid.sum())
        valid_vel += int(valid_pair.sum())
    return {
        "mse": sums["mse"] / max(valid_steps, 1),
        "mse_sample": sums["mse_sample"] / max(valid_steps, 1),
        "vel_mse": sums["vel_mse"] / max(valid_vel, 1),
        "per_dim_mse": (per_dim / max(valid_steps, 1)).tolist(),
        "num_chunks": len(chunks),
    }


def train_cae_action_vae(
    chunks: np.ndarray,
    action_is_pad: np.ndarray,
    cfg: CAEActionVAETrainingConfig,
    *,
    num_suites: int = 1,
    suite_ids: np.ndarray | None = None,
) -> tuple[CAEActionVAE, dict[str, Any]]:
    """Train the action VAE on normalized chunks; returns the EMA model and the metrics."""
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    _, action_horizon, action_dim = chunks.shape
    vae = CAEActionVAE(
        action_dim=action_dim,
        action_horizon=action_horizon,
        num_tokens=cfg.num_tokens,
        latent_dim=cfg.latent_dim,
        encoder_hidden_dim=cfg.encoder_hidden_dim,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        num_suites=num_suites,
        variational=cfg.variational,
    ).to(device)
    ema = copy.deepcopy(vae).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    actions = torch.as_tensor(chunks, dtype=torch.float32, device=device)
    is_pad = torch.as_tensor(action_is_pad, dtype=torch.bool, device=device)
    if num_suites > 0:
        ids_np = np.zeros(len(chunks), dtype=np.int64) if suite_ids is None else suite_ids
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
    else:
        ids = None
    if len(actions) < cfg.batch_size:
        raise ValueError(f"Only {len(actions)} chunks for a batch size of {cfg.batch_size}.")
    generator = torch.Generator().manual_seed(cfg.seed)
    steps_per_epoch = len(actions) // cfg.batch_size  # the last partial batch is dropped
    history = []
    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        beta = cfg.beta_kl if cfg.variational else 0.0
        if cfg.variational and cfg.kl_warmup_epochs > 0:
            beta *= min(1.0, epoch / cfg.kl_warmup_epochs)
        order = torch.randperm(len(actions), generator=generator).to(device)
        totals = dict.fromkeys(("loss", "loss_recon", "loss_vel", "loss_kl"), 0.0)
        for step in range(steps_per_epoch):
            batch = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            losses = vae.compute_loss(
                actions[batch],
                is_pad[batch],
                None if ids is None else ids[batch],
                beta_kl=beta,
                free_bits=cfg.free_bits,
                velocity_weight=cfg.velocity_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            with torch.no_grad():
                for ema_param, param in zip(ema.parameters(), vae.parameters(), strict=True):
                    ema_param.lerp_(param, 1 - cfg.ema_decay)
            for key in totals:
                totals[key] += losses[key].item()
        epoch_losses = {key: value / steps_per_epoch for key, value in totals.items()}
        history.append({"epoch": epoch, "beta_kl": beta, **epoch_losses})
        logger.info(
            "epoch %d/%d: loss %.6f, recon %.6f, vel %.6f, kl %.4f",
            epoch,
            cfg.epochs,
            *(epoch_losses[key] for key in ("loss", "loss_recon", "loss_vel", "loss_kl")),
        )
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in vae.named_buffers():
        ema_buffers[name].copy_(buffer)
    torch.manual_seed(cfg.seed)
    metrics = evaluate_action_vae(ema, actions, is_pad, ids)
    if not all(math.isfinite(metrics[key]) for key in ("mse", "mse_sample", "vel_mse")):
        raise RuntimeError(f"The action VAE diverged: {metrics}")
    logger.info("EMA reconstruction (normalized actions): %s", {k: metrics[k] for k in ("mse", "vel_mse")})
    return ema.cpu(), {"final": metrics, "history": history, "steps": steps_per_epoch * cfg.epochs}


def _json_stats(stats: dict[str, Any]) -> dict[str, list[float]]:
    return {name: np.asarray(value, dtype=np.float64).reshape(-1).tolist() for name, value in stats.items()}


def save_cae_action_vae(
    output_dir: str | Path,
    vae: CAEActionVAE,
    config: dict[str, Any],
    stats: dict[str, dict[str, dict[str, list[float]]]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write `config.json`, `model.safetensors`, `stats.json` (and `metadata.json`) to `output_dir`."""
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / CONFIG_NAME).write_text(json.dumps({"type": ACTION_VAE_TYPE, **config}, indent=2) + "\n")
    state_dict = {key: value.detach().cpu().contiguous() for key, value in vae.state_dict().items()}
    save_file(state_dict, str(output / WEIGHTS_NAME), metadata={"format": "pt"})
    (output / STATS_NAME).write_text(json.dumps(stats, indent=2) + "\n")
    if metadata is not None:
        (output / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def _resolve_file(path: str | Path, filename: str, revision: str | None = None) -> Path:
    local = Path(path)
    if local.is_dir():
        return local / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(path), filename=filename, revision=revision))


def load_cae_action_vae_config(
    path: str | Path, revision: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the `config.json` and `stats.json` of a trained action VAE (local directory or Hub repo)."""
    config = json.loads(_resolve_file(path, CONFIG_NAME, revision).read_text())
    if config.get("type") != ACTION_VAE_TYPE:
        raise ValueError(f"{path} is not a CAE-FastWAM action VAE (config type {config.get('type')!r}).")
    stats = json.loads(_resolve_file(path, STATS_NAME, revision).read_text())
    return config, stats


def load_cae_action_vae_state_dict(path: str | Path, revision: str | None = None) -> dict[str, Tensor]:
    """The `encoder.*` / `decoder.*` weights of a trained action VAE (local directory or Hub repo)."""
    from safetensors.torch import load_file

    return load_file(str(_resolve_file(path, WEIGHTS_NAME, revision)))


def train_cae_action_vae_from_config(cfg: Any) -> Path:
    """`lerobot-train-tokenizer --tokenizer_type=cae_action_vae`: fit the action VAE on `cfg.repo_id`.

    Uses the script's `repo_id`, `root`, `action_horizon`, `max_episodes`, `sample_fraction`,
    `normalization_mode`, `output_dir` and `push_to_hub` / `hub_repo_id` / `hub_private` options, and the
    `cae` options (`CAEActionVAETrainingConfig`). Every action dimension is encoded; FAST's options
    (`encoded_dims`, relative actions, `vocab_size`, `scale`) do not apply.
    """
    from lerobot.datasets import LeRobotDataset

    vae_cfg: CAEActionVAETrainingConfig = cfg.cae
    if cfg.use_relative_transform:
        raise ValueError(
            "The CAE action VAE encodes absolute actions; `use_relative_transform` is not supported."
        )
    if cfg.action_horizon % vae_cfg.num_tokens != 0:
        raise ValueError(
            f"`action_horizon` ({cfg.action_horizon}) must be a multiple of `cae.num_tokens` ({vae_cfg.num_tokens}); "
            "CAE-FastWAM's released checkpoint uses `--action_horizon=32`."
        )
    norm_mode = NormalizationMode(cfg.normalization_mode)
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root, download_videos=False)
    stats = dataset.meta.stats
    if norm_mode != NormalizationMode.IDENTITY and (stats is None or ACTION not in stats):
        raise ValueError(f"{cfg.repo_id} has no action statistics for {norm_mode.value} normalization.")
    chunks, action_is_pad = build_action_chunks(
        dataset,
        cfg.action_horizon,
        max_episodes=cfg.max_episodes,
        sample_fraction=cfg.sample_fraction,
        seed=vae_cfg.seed,
    )
    logger.info(
        "%s: %d action chunks of shape %s (%.2f%% padding steps)",
        cfg.repo_id,
        len(chunks),
        tuple(chunks.shape[1:]),
        100 * float(action_is_pad.mean()),
    )
    normalized = normalize_action_chunks(chunks, stats[ACTION] if stats else {}, norm_mode)
    vae, metrics = train_cae_action_vae(normalized, action_is_pad, vae_cfg)

    suite = vae_cfg.suite_name or cfg.repo_id
    config = {
        "action_dim": int(chunks.shape[2]),
        "action_horizon": int(cfg.action_horizon),
        "num_tokens": vae_cfg.num_tokens,
        "latent_dim": vae_cfg.latent_dim,
        "encoder_hidden_dim": vae_cfg.encoder_hidden_dim,
        "decoder_hidden_dim": vae_cfg.decoder_hidden_dim,
        "variational": vae_cfg.variational,
        "task_suites": [suite],
        "normalization_mode": norm_mode.value,
    }
    suite_stats = {key: _json_stats(stats[key]) for key in (ACTION, OBS_STATE) if stats and key in stats}
    metadata = {
        "repo_id": cfg.repo_id,
        "num_episodes": dataset.num_episodes if cfg.max_episodes is None else cfg.max_episodes,
        "num_training_chunks": len(chunks),
        "sample_fraction": cfg.sample_fraction,
        "training": asdict(vae_cfg),
        "metrics": metrics,
    }
    output_dir = Path(cfg.output_dir or f"cae_action_vae_{cfg.repo_id.replace('/', '_')}")
    save_cae_action_vae(output_dir, vae, config, {suite: suite_stats}, metadata)
    logger.info("Saved the CAE action VAE to %s", output_dir)

    if cfg.push_to_hub:
        from huggingface_hub import HfApi

        hub_repo_id = cfg.hub_repo_id or output_dir.name
        api = HfApi()
        api.create_repo(hub_repo_id, private=cfg.hub_private, exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo_id,
            folder_path=output_dir,
            commit_message=f"Upload CAE-FastWAM action VAE trained on {cfg.repo_id}",
        )
        logger.info("Pushed the CAE action VAE to https://huggingface.co/%s", hub_repo_id)
    return output_dir
