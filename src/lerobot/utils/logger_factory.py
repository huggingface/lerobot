# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import List, Optional

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.experiment_logger import ExperimentLogger
from lerobot.utils.wandb_utils import WandBLogger


def create_experiment_loggers(cfg: TrainPipelineConfig) -> Optional[ExperimentLogger]:
    """Create experiment loggers based on configuration.
    
    Args:
        cfg: Training pipeline configuration
        
    Returns:
        ExperimentLogger instance, CompositeLogger for multiple backends, or None if no logging is enabled
    """
    loggers = []
    
    # Create WandB logger if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        try:
            loggers.append(WandBLogger(cfg))
        except Exception as e:
            logging.warning(f"Failed to initialize WandB logger: {e}")
    
    # Create MLflow logger if enabled
    if cfg.mlflow.enable:
        try:
            # Lazy import to avoid dependency issues
            from lerobot.utils.mlflow_utils import MLflowLogger
            loggers.append(MLflowLogger(cfg))
        except Exception as e:
            logging.warning(f"Failed to initialize MLflow logger: {e}")
    
    # Return appropriate logger based on count
    if len(loggers) == 0:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return CompositeLogger(loggers)


class CompositeLogger(ExperimentLogger):
    """Logger that forwards calls to multiple backend loggers."""
    
    def __init__(self, loggers: List[ExperimentLogger]):
        self.loggers = [logger for logger in loggers if logger is not None]
    
    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        for logger in self.loggers:
            logger.log_dict(d, step, mode, custom_step_key)            
    
    def log_policy(self, checkpoint_dir):
        for logger in self.loggers:
            logger.log_policy(checkpoint_dir)            
    
    def log_video(self, video_path, step, mode="train"):
        for logger in self.loggers:
            logger.log_video(video_path, step, mode)            
    
    def finish(self):
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                logging.warning(f"Failed to finish logger {type(logger).__name__}: {e}")
