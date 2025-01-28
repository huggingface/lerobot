import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common import envs, policies  # noqa: F401
from lerobot.common.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    # TODO(rcadene, aliberts): By default, use device and use_amp values from policy checkpoint.
    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False
    seed: int | None = 1000

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

            # When no device or use_amp are given, use the one from training config.
            if self.device is None or self.use_amp is None:
                train_cfg = TrainPipelineConfig.from_pretrained(policy_path)
                if self.device is None:
                    self.device = train_cfg.device
                if self.use_amp is None:
                    self.use_amp = train_cfg.use_amp

            # Automatically switch to available device if necessary
            if not is_torch_device_available(self.device):
                auto_device = auto_select_torch_device()
                logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
                self.device = auto_device

            # Automatically deactivate AMP if necessary
            if self.use_amp and not is_amp_available(self.device):
                logging.warning(
                    f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
                )
                self.use_amp = False

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

        if self.device is None:
            raise ValueError("Set one of the following device: cuda, cpu or mps")
        elif self.device == "cuda" and self.use_amp is None:
            raise ValueError("Set 'use_amp' to True or False.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
