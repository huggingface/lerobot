import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

import draccus

from lerobot.common import envs
from lerobot.common.policies.utils import get_pretrained_policy_path
from lerobot.configs.policies import PretrainedConfig


@dataclass
class EvalConfig:
    n_episodes: int = 50
    episode_length: int | None = None
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    pretrained_policy_path: Path
    eval: EvalConfig
    env: envs.EnvConfig
    policy: PretrainedConfig | None = None
    dir: Path | None = None
    job_name: str | None = None
    device: str = "cuda"  # | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False
    seed: int | None = 1000

    def __post_init__(self):
        # TODO(aliberts, rcadene): move this logic out of the config
        from time import sleep

        sleep(1)
        self.resolve_policy_name_or_path()
        cfg_path = self.pretrained_policy_path / "config.json"
        with open(cfg_path) as f:
            policy_cfg = draccus.load(PretrainedConfig, f)

        if self.policy is not None:
            # override policy config
            replace(policy_cfg, self.policy)

        self.policy = policy_cfg

        if not self.job_name:
            self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.dir = Path("outputs/eval") / eval_dir

    def resolve_policy_name_or_path(self):
        self.pretrained_policy_path = get_pretrained_policy_path(self.pretrained_policy_path)
