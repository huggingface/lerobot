import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import DictConfig


def get_wandb_run_id_from_filesystem(checkpoint_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(checkpoint_dir / "../wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


class ExperimentTracker(Protocol):
    def init(self, *args, **kwargs): ...

    def log_model(self, save_dir: Path, model_name: str, *args, **kwargs) -> None: ...

    def log_data(self, data: dict[str, Any], step: int | None, *args, **kwargs) -> None: ...

    def log_video(self, video_path: str, fps: int, format: str, mode: str, *args, **kwargs) -> None: ...
    @property
    def experiment_url(self) -> str | None: ...

    @property
    def tracker_name(self) -> str: ...


class WandB:
    def __init__(self):
        os.environ["WANDB_SILENT"] = "true"
        import wandb

        self._wandb = wandb

    def init(self, *args, **kwargs):
        checkpoints_dir = kwargs.get("checkpoints_dir", None)
        resume = kwargs.get("resume", False)

        wandb_run_id = None
        if resume:
            wandb_run_id = get_wandb_run_id_from_filesystem(checkpoints_dir)

        self._wandb.init(
            id=wandb_run_id,
            project=kwargs.get("project", None),
            entity=kwargs.get("entity", None),
            name=kwargs.get("job_name", None),
            notes=kwargs.get("notes", None),
            tags=kwargs.get("tags", None),
            dir=kwargs.get("log_dir", None),
            config=kwargs.get("config", None),
            save_code=kwargs.get("save_code", False),
            job_type=kwargs.get("job_type", None),
            resume="must" if resume else None,
        )

    def log_model(self, save_dir: Path, model_name: str, *args, **kwargs):
        artifact = self._wandb.Artifact(model_name, type="model")
        artifact.add_file(str(save_dir / SAFETENSORS_SINGLE_FILE))
        self._wandb.save(artifact)

    def log_data(self, data: dict[str, Any], step: int | None, *args, **kwargs):
        self._wandb.log(data, step=step)

    def log_video(self, video_path: str, fps: int, format: str, mode: str, step: int, *args, **kwargs):
        video = self._wandb.Video(data_or_path=video_path, fps=fps, format=format)
        self.log_data({f"{mode}/video": video}, step=step)

    @property
    def experiment_url(self) -> str | None:
        if self._wandb.run:
            return self._wandb.run.get_url()

    @property
    def tracker_name(self) -> str:
        return "wandb"


def experiment_tracker_factory(cfg: DictConfig) -> ExperimentTracker | None:
    if cfg.get("wandb"):
        return WandB()
    else:
        return None
