import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def setup_distributed():
    """
    Initialize the distributed environment for torchrun
    """
    # torchrun sets these environment variables
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logging.info("RANK or WORLD_SIZE not found in environment variables")
        return False, 0, 1, None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    logging.info(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    logging.info(f"Process group initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    return True, rank, world_size, device


def cleanup_distributed():
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_metrics(metrics_dict, device):
    """
    Gather metrics from all processes and compute their average
    """
    result = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            result[key] = tensor.item() / dist.get_world_size()
    return result


def gather_rewards(rewards, device):
    """
    Gather rewards from all processes for RL training
    Returns a tensor containing all rewards from all processes
    """
    local_size = torch.tensor([rewards.shape[0]], device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)

    max_size = max(size.item() for size in size_list)

    # Pad rewards tensor to max_size
    if rewards.shape[0] < max_size:
        padding = torch.zeros(max_size - rewards.shape[0], *rewards.shape[1:], device=device)
        rewards = torch.cat([rewards, padding], dim=0)

    # Gather all rewards
    tensor_list = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, rewards)

    # Remove padding and concatenate
    all_rewards = []
    for i, tensor in enumerate(tensor_list):
        all_rewards.append(tensor[: size_list[i].item()])

    return torch.cat(all_rewards, dim=0)


def create_distributed_envs(env_cfg, world_size, rank):
    """
    Create environments for distributed RL training
    Each process gets its own set of environments
    """
    # Set different seeds for each process's environments
    env_seed = (env_cfg.seed if hasattr(env_cfg, "seed") and env_cfg.seed is not None else 0) + rank * 1000

    # Create environments for this process
    env = make_env(
        env_cfg,
        n_envs=env_cfg.n_envs_per_process if hasattr(env_cfg, "n_envs_per_process") else 1,
        start_seed=env_seed,
    )

    return env


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Entry point for distributed training - compatible with torchrun
    """
    # Initialize distributed environment (required for torchrun)
    is_distributed, rank, world_size, device = setup_distributed()
    if not is_distributed:
        logging.warning("Not running in distributed mode.")
        rank = 0
        world_size = 1
        device = get_safe_torch_device(cfg.policy.device, log=True)

    # Configure logging for each process
    if rank != 0:
        # Disable verbose logging for non-master processes
        logging.getLogger().setLevel(logging.WARNING)

    try:
        cfg.validate()
        if rank == 0:
            logging.info(pformat(cfg.to_dict()))

        # Initialize wandb only on the main process
        wandb_logger = None
        if rank == 0 and cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        elif rank == 0:
            logging.info("Logs will be saved locally.")

        # Set seed with rank-specific offset to ensure different batch sampling
        if cfg.seed is not None:
            set_seed(cfg.seed + rank)

        # Configure device settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        if rank == 0:
            logging.info("Creating dataset")
        dataset = make_dataset(cfg)

        # Create environment used for evaluating checkpoints during training on simulation data.
        # For distributed setting, only the main process needs to handle evaluation
        eval_env = None
        if rank == 0 and cfg.eval_freq > 0 and cfg.env is not None:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

        if rank == 0:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
        policy = policy.to(device)

        # Important: Create optimizer BEFORE wrapping model in DDP
        # This ensures get_optim_params() works correctly
        if rank == 0:
            logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

        # Now wrap the policy with DDP and enable unused parameter detection
        if is_distributed:
            # Enable find_unused_parameters to handle parameters not used in forward pass
            policy = DDP(policy, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            if rank == 0:
                logging.info("DDP enabled with unused parameter detection")

        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0  # number of policy updates (forward + backward + optim)

        if cfg.resume:
            # Only the main process loads the checkpoint, then broadcast to all
            map_location = {"cuda:0": f"cuda:{rank}"}
            if rank == 0:
                state_dict = torch.load(cfg.checkpoint_path, map_location=map_location)
                step = state_dict["step"]
            else:
                state_dict = None

            if is_distributed:
                # Broadcast step from rank 0 to all other processes
                step_tensor = torch.tensor([step], device=device)
                dist.broadcast(step_tensor, src=0)
                step = step_tensor.item()

                # Broadcast model parameters
                if rank == 0:
                    model_state_dict = state_dict["model"]
                else:
                    model_state_dict = policy.module.state_dict()

                for param_name, param in model_state_dict.items():
                    dist.broadcast(param, src=0)

                policy.module.load_state_dict(model_state_dict)
            else:
                # Non-distributed mode
                policy.load_state_dict(state_dict["model"])

            # Optimizer and scheduler are initialized per process
            if rank == 0:
                optimizer.load_state_dict(state_dict["optimizer"])
                if lr_scheduler is not None and "lr_scheduler" in state_dict:
                    lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        if rank == 0:
            num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in policy.parameters())

            logging.info(f"Output dir: {cfg.output_dir}")
            if cfg.env is not None:
                logging.info(f"{cfg.env.task=}")
            logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
            logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
            logging.info(f"{dataset.num_episodes=}")
            logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
            logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
            logging.info(f"Distributed training on {world_size} GPUs")

        # Create distributed sampler to ensure different data on each GPU
        if is_distributed:
            if hasattr(cfg.policy, "drop_n_last_frames"):
                episodic_sampler = EpisodeAwareSampler(
                    dataset.episode_data_index,
                    drop_n_last_frames=cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                # Wrap the episodic sampler with DistributedSampler
                sampler = DistributedSampler(
                    episodic_sampler,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed if cfg.seed is not None else 0,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed if cfg.seed is not None else 0,
                )
            shuffle = False
        else:
            # For non-distributed case, use the original sampler logic
            if hasattr(cfg.policy, "drop_n_last_frames"):
                shuffle = False
                sampler = EpisodeAwareSampler(
                    dataset.episode_data_index,
                    drop_n_last_frames=cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
            else:
                shuffle = True
                sampler = None

        # Calculate per-GPU batch size
        batch_size = cfg.batch_size // world_size if is_distributed else cfg.batch_size

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        dl_iter = cycle(dataloader)

        policy.train()

        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        train_tracker = MetricsTracker(
            batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
        )

        # Create a lock for multi-process optimizer updates if needed
        optimizer_lock = None

        if rank == 0:
            logging.info("Start distributed offline training on a fixed dataset")

        # Track the start time for interval timing
        interval_start_time = time.time()
        last_log_step = step

        for _ in range(step, cfg.steps):
            # Add barrier to synchronize processes before each step
            if is_distributed:
                dist.barrier()

            # Set epoch for the sampler to ensure proper shuffling
            if is_distributed:
                sampler.set_epoch(step)

            start_time = time.perf_counter()
            batch = next(dl_iter)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
                lock=optimizer_lock,
            )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # Synchronize metrics across processes for logging
            if is_log_step:
                # Calculate interval time
                interval_time = time.time() - interval_start_time
                local_steps = step - last_log_step

                # Calculate global steps across all GPUs - for DDP, each GPU does the same steps
                # but processes different data, so total steps = local steps
                global_steps = local_steps

                # Calculate total batches processed - each GPU processes its own batches
                # so total batches = local steps * world_size
                total_batches = local_steps * world_size

                # Calculate total samples - each batch has batch_size samples per GPU
                total_samples = total_batches * batch_size

                # Calculate rates
                samples_per_second = total_samples / interval_time
                batches_per_second = total_batches / interval_time
                steps_per_second = global_steps / interval_time

                # Gather metrics from all processes
                metrics_dict = train_tracker.to_dict()
                if is_distributed:
                    global_metrics = all_gather_metrics(metrics_dict, device)
                else:
                    global_metrics = metrics_dict

                if rank == 0:
                    # Manually update the metrics for reporting
                    # Create a new tracker to hold the aggregated metrics
                    agg_metrics = {}
                    for key in train_metrics:
                        agg_metrics[key] = AverageMeter(train_metrics[key].name, train_metrics[key].fmt)
                        if key in global_metrics:
                            agg_metrics[key].avg = global_metrics[key]

                    agg_tracker = MetricsTracker(
                        batch_size, dataset.num_frames, dataset.num_episodes, agg_metrics, initial_step=step
                    )

                    # Log comprehensive throughput information
                    gpu_info = f"{world_size} GPU{'s' if world_size > 1 else ''}"
                    throughput = f"Throughput: {samples_per_second:.1f} samples/sec ({batches_per_second:.1f} batches/sec)"
                    time_info = f"Time: {interval_time:.2f}s for {global_steps} steps"
                    batch_info = f"Batch size: {batch_size}/GPU, {batch_size * world_size} total"

                    logging.info(f"[{gpu_info} | {throughput} | {time_info} | {batch_info}] {agg_tracker}")

                    if wandb_logger:
                        wandb_log_dict = global_metrics
                        # Add timing information to wandb log
                        wandb_log_dict["interval_time"] = interval_time
                        wandb_log_dict["samples_per_second"] = samples_per_second
                        wandb_log_dict["batches_per_second"] = batches_per_second
                        wandb_log_dict["steps_per_second"] = steps_per_second
                        wandb_log_dict["total_gpus"] = world_size
                        wandb_log_dict["total_batch_size"] = batch_size * world_size
                        if output_dict:
                            wandb_log_dict.update(output_dict)
                        wandb_logger.log_dict(wandb_log_dict, step)

                train_tracker.reset_averages()

                # Reset interval timer and counters
                interval_start_time = time.time()
                last_log_step = step

            # Only the main process handles checkpoint saving
            if rank == 0 and cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                # Unwrap the policy if using DDP
                save_policy = policy.module if is_distributed else policy
                save_checkpoint(checkpoint_dir, step, cfg, save_policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Only the main process handles evaluation
            if rank == 0 and cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_start_time = time.time()
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    # Unwrap the policy if using DDP
                    eval_policy_model = policy.module if is_distributed else policy
                    eval_info = eval_policy(
                        eval_env,
                        eval_policy_model,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )
                eval_time = time.time() - eval_start_time

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(f"[Eval time: {eval_time:.2f}s] {eval_tracker}")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_log_dict["eval_time"] = eval_time
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

        # Cleanup
        if eval_env:
            eval_env.close()

        if is_distributed:
            cleanup_distributed()

        if rank == 0:
            logging.info("End of distributed training")

    except Exception as e:
        logging.error(f"Error during training: {e}")
        if is_distributed:
            cleanup_distributed()
        raise


if __name__ == "__main__":
    init_logging()
    train()
