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
from pathlib import Path
from typing import Any, Dict

import torch
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.experiment_logger import ExperimentLogger

import mlflow

class MLflowLogger(ExperimentLogger):
    """MLflow experiment logger implementation."""
    
    def __init__(self, cfg: TrainPipelineConfig):
        
        self.cfg = cfg.mlflow
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        
        mlflow.start_run()
        
        # Log configuration
        mlflow.log_params(self._flatten_config(cfg.to_dict()))
        
        # Log tags
        if self.cfg.tags:
            mlflow.set_tags(self.cfg.tags)
        
        # Add automatic tags
        auto_tags = {
            "policy_type": cfg.policy.type if cfg.policy else "none",
            "dataset": cfg.dataset.repo_id if cfg.dataset else "none",
            "env_type": cfg.env.type if cfg.env else "none",
        }
        mlflow.set_tags(auto_tags)
        
        print(colored("Logs will be synced with MLflow.", "blue", attrs=["bold"]))
        run_info = mlflow.active_run()
        if run_info:
            tracking_uri = mlflow.get_tracking_uri()
            run_id = run_info.info.run_id
            experiment_id = run_info.info.experiment_id
            url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
            logging.info(f"Track this run --> {colored(url, 'yellow', attrs=['bold'])}")
    
    def _flatten_config(self, config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration dictionary for MLflow params."""
        flat_dict = {}
        for key, value in config_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(self._flatten_config(value, new_key))
            elif isinstance(value, (list, tuple)):
                flat_dict[new_key] = str(value)
            elif value is not None:
                flat_dict[new_key] = value
        return flat_dict
    
    def log_dict(
        self, 
        d: Dict[str, Any], 
        step: int | None = None, 
        mode: str = "train", 
        custom_step_key: str | None = None
    ) -> None:
        """Log a dictionary of metrics to MLflow."""
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Filter out non-numeric values and convert tensors
        metrics_to_log = {}
        for key, value in d.items():
            if isinstance(value, (int, float)):
                metrics_to_log[f"{mode}/{key}"] = value
            elif isinstance(value, torch.Tensor):
                # Handle PyTorch tensors by converting to scalar
                try:
                    if value.numel() == 1:
                        metrics_to_log[f"{mode}/{key}"] = value.item()
                    else:
                        logging.warning(
                            f'MLflow logging of key "{key}" was ignored as tensor has {value.numel()} elements (expected 1).'
                        )
                except (RuntimeError, ValueError) as e:
                    logging.warning(
                        f'MLflow logging of key "{key}" was ignored due to tensor conversion error: {e}'
                    )
            elif isinstance(value, str):
                # Log string values as tags instead of metrics
                mlflow.set_tag(f"{mode}/{key}", value)
            else:
                logging.warning(
                    f'MLflow logging of key "{key}" was ignored as its type "{type(value)}" is not supported.'
                )
        
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=step)
    
    def log_policy(self, checkpoint_dir: Path) -> None:
        """Log policy checkpoint as MLflow artifact."""
        if self.cfg.disable_artifact:
            return
        
        try:
            # Log the entire checkpoint directory
            mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")
            
            # Also log the model specifically if it exists
            model_path = checkpoint_dir / "pretrained_model" / "model.safetensors"
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
        except Exception as e:
            logging.warning(f"Failed to log policy checkpoint to MLflow: {e}")
    
    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Log video file to MLflow."""
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}")
        
        try:
            # Log video as artifact
            mlflow.log_artifact(video_path, artifact_path=f"{mode}/videos")
            
            # Also log a metric indicating video was logged
            mlflow.log_metric(f"{mode}/video_logged", 1.0, step=step)
        except Exception as e:
            logging.warning(f"Failed to log video to MLflow: {e}")
    
    def finish(self) -> None:
        """End the MLflow run."""
        try:
            mlflow.end_run()
        except Exception as e:
            logging.warning(f"Failed to end MLflow run: {e}")
