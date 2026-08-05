import logging
from pathlib import Path
from termcolor import colored
import torchvision

from lerobot.configs.train import TrainPipelineConfig

class TensorBoardLogger:
    """A helper class to log object using tensorboard."""

    def __init__(self, cfg: TrainPipelineConfig):
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = cfg.output_dir / "tensorboard"
        self.env_fps = cfg.env.fps if cfg.env else None
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logging.info(colored(f"Logs will be synced with tensorboard in {self.log_dir}", "blue", attrs=["bold"]))

    def log_policy(self, checkpoint_dir: Path):
        """TensorBoard doesn't natively support logging artifacts like weights/configs as wandb does."""
        pass

    def log_dict(self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        for k, v in d.items():
            if custom_step_key is not None and k == custom_step_key:
                continue

            if not isinstance(v, (int, float)):
                continue

            step_val = step if custom_step_key is None else d[custom_step_key]
            self.writer.add_scalar(f"{mode}/{k}", v, step_val)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        try:
            # read_video returns (vframes, aframes, info)
            # vframes shape is (T, H, W, C)
            vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            # TensorBoard expects (N, T, C, H, W)
            vframes = vframes.unsqueeze(0).permute(0, 1, 4, 2, 3)
            self.writer.add_video(f"{mode}/video", vframes, step, fps=self.env_fps or 4)
        except Exception as e:
            logging.warning(f"Failed to log video to TensorBoard: {e}")
