import logging
import os
import shutil
from pprint import pformat

import torch
from termcolor import colored

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.rl.learner import (
    use_threads,
)
from lerobot.robots import so100_follower, so101_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)

from .configs import ACFQLTrainRLServerPipelineConfig as TrainRLServerPipelineConfig
from .learner import handle_resume_logic, initialize_offline_replay_buffer


@parser.wrap()
def edit_dataset_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    edit_dataset(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def edit_dataset(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)  # noqa: F841
    else:
        wandb_logger = None  # noqa: F841
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Extract all configuration variables at the beginning
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device_offline_replay_buffer = get_safe_torch_device(
        try_device=cfg.policy.storage_device_offline_replay_buffer
    )
    fps = cfg.env.fps

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"edit_dataset_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device_offline_replay_buffer,
        )

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        logging.info(
            f"Saving offline replay buffer to {dataset_offline_dir} with repo id {cfg.dataset.repo_id}"
        )
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )


if __name__ == "__main__":
    register_third_party_plugins()
    edit_dataset_cli()
    logging.info("[LEARNER] main finished")
