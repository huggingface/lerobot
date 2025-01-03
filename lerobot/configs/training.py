import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
from deepdiff import DeepDiff

from lerobot.common import envs
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.eval import EvalConfig
from lerobot.configs.policies import PretrainedConfig


@dataclass
class OfflineConfig:
    steps: int = 100_000


@dataclass
class OnlineConfig:
    """
    The online training loop looks something like:

    ```python
    for i in range(steps):
        do_online_rollout_and_update_online_buffer()
        for j in range(steps_between_rollouts):
            batch = next(dataloader_with_offline_and_online_data)
            loss = policy(batch)
            loss.backward()
            optimizer.step()
    ```

    Note that the online training loop adopts most of the options from the offline loop unless specified
    otherwise.
    """

    steps: int = 0
    # How many episodes to collect at once when we reach the online rollout part of the training loop.
    rollout_n_episodes: int = 1
    # The number of environments to use in the gym.vector.VectorEnv. This ends up also being the batch size for
    # the policy. Ideally you should set this to by an even divisor of rollout_n_episodes.
    rollout_batch_size: int = 1
    # How many optimization steps (forward, backward, optimizer step) to do between running rollouts.
    steps_between_rollouts: int | None = None
    # The proportion of online samples (vs offline samples) to include in the online training batches.
    sampling_ratio: float = 0.5
    # First seed to use for the online rollout environment. Seeds for subsequent rollouts are incremented by 1.
    env_seed: int | None = None
    # Sets the maximum number of frames that are stored in the online buffer for online training. The buffer is
    # FIFO.
    buffer_capacity: int | None = None
    # The minimum number of frames to have in the online buffer before commencing online training.
    # If buffer_seed_size > rollout_n_episodes, the rollout will be run multiple times until the
    # seed size condition is satisfied.
    buffer_seed_size: int = 0
    # Whether to run the online rollouts asynchronously. This means we can run the online training steps in
    # parallel with the rollouts. This might be advised if your GPU has the bandwidth to handle training
    # + eval + environment rendering simultaneously.
    do_rollout_async: bool = False


@dataclass
class TrainPipelineConfig:
    policy: PretrainedConfig
    dataset: DatasetConfig
    env: envs.EnvConfig = field(default_factory=envs.RealEnv)
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    device: str = "cuda"  # | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        if not self.job_name:
            self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.dir = Path("outputs/train") / train_dir

        if self.online.steps > 0 and isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("Online training with LeRobotMultiDataset is not implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        # If we are resuming a run, we need to check that a checkpoint exists in the log directory, and we need
        # to check for any differences between the provided config and the checkpoint's config.
        checkpoint_cfg_path = self.dir / "checkpoints/last/config.yaml"
        if self.resume:
            if not checkpoint_cfg_path.exists():
                raise RuntimeError(
                    f"You have set resume=True, but there is no model checkpoint in {self.dir}"
                )

            # Get the configuration file from the last checkpoint.
            checkpoint_cfg = self.from_checkpoint(checkpoint_cfg_path)

            # # Check for differences between the checkpoint configuration and provided configuration.
            # # Hack to resolve the delta_timestamps ahead of time in order to properly diff.
            # resolve_delta_timestamps(cfg)
            diff = DeepDiff(checkpoint_cfg, self)
            # Ignore the `resume` and parameters.
            if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
                del diff["values_changed"]["root['resume']"]
            # Log a warning about differences between the checkpoint configuration and the provided
            # configuration.
            if len(diff) > 0:
                logging.warning(
                    "At least one difference was detected between the checkpoint configuration and "
                    f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                    "takes precedence.",
                )
            # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
            self = checkpoint_cfg
            self.resume = True

        elif checkpoint_cfg_path.exists():
            raise RuntimeError(
                f"The configured output directory {checkpoint_cfg_path} already exists. If "
                "you meant to resume training, please use `resume=true` in your command or yaml configuration."
            )

    @classmethod
    def from_checkpoint(cls, config_path: Path):
        with open(config_path) as f:
            cfg = draccus.load(cls, f)
        return cfg
