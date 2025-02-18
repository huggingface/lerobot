import logging
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import torch
import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    auto_select_torch_device,
    init_logging,
    is_amp_available,
    is_torch_device_available,
)
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class SaveInferenceConfig:
    dataset: DatasetConfig
    # Delete the output directory if it exists already.
    force_override: bool = False

    batch_size: int = 16
    num_workers: int = 4

    policy: PreTrainedConfig | None = None
    # TODO(rcadene, aliberts): By default, use device and use_amp values from policy checkpoint.
    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool | None = None

    output_dir: str | Path | None = None

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

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


@parser.wrap()
def save_inference(cfg: SaveInferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)

    output_dir = cfg.output_dir
    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_save_inference_")

    elif Path(output_dir).exists() and cfg.force_override:
        shutil.rmtree(cfg.output_dir)

    output_dir = Path(output_dir)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        pin_memory=cfg.device != "cpu",
        drop_last=False,
    )

    policy.train()

    episode_indices = []
    frame_indices = []
    feats = {}

    for batch in tqdm.tqdm(dataloader):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type=cfg.device) if cfg.use_amp else nullcontext():
            _, output_dict = policy.forward(batch)

        batch_size = batch["episode_index"].shape[0]
        episode_indices.append(batch["episode_index"])
        frame_indices.append(batch["frame_index"])

        for key, value in output_dict.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
                print(f"Skipping {key}")
                continue

            if key not in feats:
                feats[key] = []

            feats[key].append(value)

    episode_indices = torch.cat(episode_indices).cpu()
    frame_indices = torch.cat(frame_indices).cpu()

    # TODO(rcadene): use collate?
    for key, value in feats.items():
        if isinstance(value[0], (float, int)):
            feats[key] = torch.tensor(value)
        elif isinstance(value[0], torch.Tensor):
            feats[key] = torch.cat(value, dim=0).cpu()
        elif isinstance(value[0], str):
            pass
        else:
            raise NotImplementedError(f"{key}: {value}")

    # Find unique episode indices
    unique_episodes = torch.unique(episode_indices)

    for episode in unique_episodes:
        ep_feats = {}
        for key in feats:
            ep_feats[key] = feats[key][episode_indices == episode]

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ep_feats, output_dir / f"output_features_episode_{episode}.pth")

    print(f"Features can be found in: {output_dir}")


if __name__ == "__main__":
    save_inference()
