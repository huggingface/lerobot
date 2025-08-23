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

import pytest
from unittest.mock import Mock, patch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.logger_factory import create_experiment_loggers, CompositeLogger
from lerobot.utils.experiment_logger import ExperimentLogger


@pytest.fixture
def sample_config() -> TrainPipelineConfig:
    """Create a sample training configuration."""
    cfg = TrainPipelineConfig(
        dataset=Mock(repo_id="test/dataset"),
        policy=Mock(type="test_policy"),
        env=Mock(type="test_env", fps=30),
        output_dir="/tmp/test",
        job_name="test_job",
        steps=1000,
        log_freq=100,
        save_freq=200,
        eval_freq=300,
        save_checkpoint=True,
        seed=42
    )
    return cfg


def test_create_experiment_loggers_no_loggers_enabled(sample_config: TrainPipelineConfig) -> None:
    """Test that None is returned when no loggers are enabled."""
    sample_config.wandb.enable = False
    sample_config.mlflow.enable = False
    
    logger = create_experiment_loggers(sample_config)
    
    assert logger is None


@patch('lerobot.utils.logger_factory.WandBLogger')
def test_create_experiment_loggers_wandb_only(mock_wandb_logger: Mock, sample_config: TrainPipelineConfig) -> None:
    """Test creating only WandB logger."""
    sample_config.wandb.enable = True
    sample_config.wandb.project = "test_project"
    sample_config.mlflow.enable = False
    
    mock_logger_instance = Mock()
    mock_wandb_logger.return_value = mock_logger_instance
    
    logger = create_experiment_loggers(sample_config)
    
    assert logger == mock_logger_instance
    mock_wandb_logger.assert_called_once_with(sample_config)


@patch('lerobot.utils.logger_factory.WandBLogger')
@patch('lerobot.utils.mlflow_utils.MLflowLogger')
def test_create_experiment_loggers_mlflow_only(mock_mlflow_logger: Mock, mock_wandb_logger: Mock, sample_config: TrainPipelineConfig) -> None:
    """Test creating only MLflow logger."""
    sample_config.wandb.enable = False
    sample_config.mlflow.enable = True
    
    mock_logger_instance = Mock()
    mock_mlflow_logger.return_value = mock_logger_instance
    
    logger = create_experiment_loggers(sample_config)
    
    assert logger == mock_logger_instance
    mock_mlflow_logger.assert_called_once_with(sample_config)
    mock_wandb_logger.assert_not_called()


@patch('lerobot.utils.logger_factory.WandBLogger')
@patch('lerobot.utils.mlflow_utils.MLflowLogger')
def test_create_experiment_loggers_both_enabled(mock_mlflow_logger: Mock, mock_wandb_logger: Mock, sample_config: TrainPipelineConfig) -> None:
    """Test creating both loggers returns CompositeLogger."""
    sample_config.wandb.enable = True
    sample_config.wandb.project = "test_project"
    sample_config.mlflow.enable = True
    
    mock_wandb_instance = Mock()
    mock_mlflow_instance = Mock()
    mock_wandb_logger.return_value = mock_wandb_instance
    mock_mlflow_logger.return_value = mock_mlflow_instance
    
    logger = create_experiment_loggers(sample_config)
    
    assert isinstance(logger, CompositeLogger)
    assert len(logger.loggers) == 2
    assert mock_wandb_instance in logger.loggers
    assert mock_mlflow_instance in logger.loggers


@patch('lerobot.utils.logger_factory.WandBLogger')
def test_create_experiment_loggers_wandb_init_failure(mock_wandb_logger: Mock, sample_config: TrainPipelineConfig) -> None:
    """Test handling of WandB logger initialization failure."""
    sample_config.wandb.enable = True
    sample_config.wandb.project = "test_project"
    sample_config.mlflow.enable = False
    
    mock_wandb_logger.side_effect = Exception("WandB init failed")
    
    logger = create_experiment_loggers(sample_config)
    
    assert logger is None


def test_composite_logger_log_dict() -> None:
    """Test CompositeLogger forwards log_dict calls to all loggers."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = Mock(spec=ExperimentLogger)
    
    composite = CompositeLogger([logger1, logger2])
    
    test_data = {"loss": 0.5}
    composite.log_dict(test_data, step=100, mode="train")
    
    logger1.log_dict.assert_called_once_with(test_data, 100, "train", None)
    logger2.log_dict.assert_called_once_with(test_data, 100, "train", None)


def test_composite_logger_log_policy() -> None:
    """Test CompositeLogger forwards log_policy calls to all loggers."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = Mock(spec=ExperimentLogger)
    
    composite = CompositeLogger([logger1, logger2])
    
    checkpoint_dir = "/tmp/checkpoint"
    composite.log_policy(checkpoint_dir)
    
    logger1.log_policy.assert_called_once_with(checkpoint_dir)
    logger2.log_policy.assert_called_once_with(checkpoint_dir)


def test_composite_logger_log_video() -> None:
    """Test CompositeLogger forwards log_video calls to all loggers."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = Mock(spec=ExperimentLogger)
    
    composite = CompositeLogger([logger1, logger2])
    
    video_path = "/tmp/video.mp4"
    composite.log_video(video_path, step=50, mode="eval")
    
    logger1.log_video.assert_called_once_with(video_path, 50, "eval")
    logger2.log_video.assert_called_once_with(video_path, 50, "eval")


def test_composite_logger_finish() -> None:
    """Test CompositeLogger forwards finish calls to all loggers."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = Mock(spec=ExperimentLogger)
    
    composite = CompositeLogger([logger1, logger2])
    
    composite.finish()
    
    logger1.finish.assert_called_once()
    logger2.finish.assert_called_once()


def test_composite_logger_finish_with_exception() -> None:
    """Test CompositeLogger handles exceptions during finish gracefully."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = Mock(spec=ExperimentLogger)
    
    # Make logger1 raise an exception during finish
    logger1.finish.side_effect = Exception("Finish failed")
    
    composite = CompositeLogger([logger1, logger2])
    
    # Should not raise exception
    composite.finish()
    
    logger1.finish.assert_called_once()
    logger2.finish.assert_called_once()


def test_composite_logger_filters_none_loggers() -> None:
    """Test CompositeLogger filters out None loggers."""
    logger1 = Mock(spec=ExperimentLogger)
    logger2 = None
    logger3 = Mock(spec=ExperimentLogger)
    
    composite = CompositeLogger([logger1, logger2, logger3])
    
    assert len(composite.loggers) == 2
    assert logger1 in composite.loggers
    assert logger3 in composite.loggers
    assert logger2 not in composite.loggers
