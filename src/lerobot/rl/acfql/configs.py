from dataclasses import dataclass

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainRLServerPipelineConfig


@dataclass(kw_only=True)
class ACFQLTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    online_dataset: DatasetConfig | None = None
