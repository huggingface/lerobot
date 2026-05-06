"""BC pretraining for an SAC actor with optional RABC weighting.

Trains *only* the SAC actor (encoder + policy network + mean/std heads)
to imitate expert (obs, action) pairs from a relabeled offline dataset.
Critic, temperature, and discrete-critic heads are left untouched.
The resulting checkpoint can then be passed to HIL-SERL via
`policy.pretrained_path` so SAC starts from a demo-imitating actor
instead of a randomly initialized one.

Usage:
    uv run python -m lerobot.scripts.bc_pretrain_sac \
        --config_path src/lerobot/rl/sim_assembling_sarm_hilserl_rabc_v3_train.json \
        --pretrain.steps=5000 \
        --pretrain.batch_size=256 \
        --pretrain.output_dir=outputs/bc_pretrain_v1
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy, TanhMultivariateNormalDiag
from lerobot.rl.buffer import ReplayBuffer
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class BCPretrainOpts:
    steps: int = 5000
    batch_size: int = 256
    lr: float = 3e-4
    log_freq: int = 50
    save_every: int = 1000
    output_dir: str = "outputs/bc_pretrain_v1"
    drop_idle_threshold: float = 0.0  # if >0, drop frames where ||action[:4]||_inf < this


@dataclass
class BCPretrainPipelineConfig(TrainRLServerPipelineConfig):
    pretrain: BCPretrainOpts = field(default_factory=BCPretrainOpts)


def _build_offline_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
    drop_idle_threshold: float = 0.0,
) -> ReplayBuffer:
    offline_dataset = make_dataset(cfg)
    stride = int(getattr(cfg, "offline_dataset_stride", 1))
    return ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        stride=stride,
        drop_idle_threshold=drop_idle_threshold,
    )


def bc_loss_step(
    policy: SACPolicy,
    batch: dict,
    rabc_provider,
    encoder_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    obs = batch["state"]
    actions = batch[ACTION]

    obs_enc = policy.actor.encoder(obs, cache=encoder_features, detach=False)
    outputs = policy.actor.network(obs_enc)
    means = policy.actor.mean_layer(outputs)
    if policy.actor.fixed_std is None:
        log_std = policy.actor.std_layer(outputs)
        std = torch.exp(log_std).clamp(policy.actor.std_min, policy.actor.std_max)
    else:
        std = policy.actor.fixed_std.expand_as(means)

    dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
    target = actions[..., : policy.actor.action_dim]
    if policy.actor.use_tanh_squash:
        target = target.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    log_prob = dist.log_prob(target)

    info = {"log_prob_mean": float(log_prob.detach().mean().item())}

    if rabc_provider is not None:
        # Thread original dataset index from complementary_info so RABCWeights
        # can look up real per-frame progress instead of falling back to uniform.
        rabc_batch = dict(batch)
        comp = batch.get("complementary_info") or {}
        if "dataset_index" in comp and "index" not in rabc_batch:
            rabc_batch["index"] = comp["dataset_index"]
        weights, w_stats = rabc_provider.compute_batch_weights(rabc_batch)
        actor_loss = -(weights * log_prob).sum() / (weights.sum() + 1e-6)
        info.update({f"rabc/{k}": v for k, v in w_stats.items()})
    else:
        actor_loss = -log_prob.mean()

    # NOTE: Earlier attempts to add a supervised CE loss on the discrete critic
    # (gripper head) regressed eval performance — uniform CE collapsed to the
    # majority no-op class, balanced CE memorized but failed OOD, and a
    # uniform-random gripper at eval beat both trained variants. We therefore
    # leave the discrete head untrained during BC pretrain and rely on online
    # SAC (or stochastic gripper sampling at eval) to learn it.
    return actor_loss, info


@parser.wrap()
def main(cfg: BCPretrainPipelineConfig) -> None:
    init_logging()
    cfg.validate()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage_device = cfg.policy.storage_device or "cpu"

    logging.info("Building offline replay buffer from dataset: %s", cfg.dataset.repo_id)
    buf = _build_offline_buffer(
        cfg,
        device=device,
        storage_device=storage_device,
        drop_idle_threshold=cfg.pretrain.drop_idle_threshold,
    )
    logging.info("Offline buffer size: %d", len(buf))

    logging.info("Instantiating SAC policy")
    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.train()

    # Optional pretrained init.
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    if pretrained_path:
        logging.info("Loading pretrained SAC actor weights from %s", pretrained_path)

    # Set up RABC if configured.
    rabc_provider = None
    if getattr(cfg.policy, "bc_use_rabc", False) and getattr(cfg.policy, "bc_rabc_progress_path", None):
        from lerobot.utils.rabc import RABCWeights

        rabc_provider = RABCWeights(
            progress_path=cfg.policy.bc_rabc_progress_path,
            chunk_size=int(cfg.policy.bc_rabc_chunk_size),
            head_mode=str(cfg.policy.bc_rabc_head_mode),
            kappa=float(cfg.policy.bc_rabc_kappa),
            device=device,
        )
        logging.info("RABC weighting enabled: %s", rabc_provider.get_stats())

    actor_params = list(policy.actor.parameters())
    train_params = list(actor_params)
    optimizer = Adam(train_params, lr=cfg.pretrain.lr)

    iterator = buf.get_iterator(batch_size=cfg.pretrain.batch_size, async_prefetch=False, queue_size=2)

    output_dir = Path(cfg.pretrain.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(cfg.pretrain.steps), desc="bc_pretrain")
    for step in pbar:
        batch = next(iterator)
        loss, info = bc_loss_step(policy, batch, rabc_provider)
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(train_params, max_norm=10.0).item()
        optimizer.step()

        if step % cfg.pretrain.log_freq == 0:
            log_msg = f"step={step} loss={loss.item():.4f} log_prob={info['log_prob_mean']:.3f} grad_norm={gn:.2f}"
            if "discrete_ce_loss" in info:
                log_msg += f" disc_ce={info['discrete_ce_loss']:.3f} disc_acc={info['discrete_acc']:.3f}"
            if "rabc/raw_mean_weight" in info:
                log_msg += (
                    f" rabc_w={info['rabc/raw_mean_weight']:.3f}"
                    f" zero={info.get('rabc/num_zero_weight', 0)}"
                    f" full={info.get('rabc/num_full_weight', 0)}"
                )
            tqdm.write(log_msg)

        if (step + 1) % cfg.pretrain.save_every == 0:
            ckpt = output_dir / f"step_{step + 1}"
            ckpt.mkdir(exist_ok=True)
            policy.save_pretrained(str(ckpt))

    final_dir = output_dir / "last"
    final_dir.mkdir(exist_ok=True)
    policy.save_pretrained(str(final_dir))
    logging.info("Saved final BC-pretrained SAC policy to %s", final_dir)


if __name__ == "__main__":
    main()
