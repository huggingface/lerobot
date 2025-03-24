import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.utils.data import Dataset

from lerobot.common.policies.hilserl.classifier.configuration_classifier import (
    ClassifierConfig,
)
from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier
from lerobot.scripts.train_hilserl_classifier import (
    create_balanced_sampler,
    train,
    train_epoch,
    validate,
)


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.meta = MagicMock()
        self.meta.stats = {}

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def make_dummy_model():
    model_config = ClassifierConfig(
        num_classes=2,
        model_name="hf-tiny-model-private/tiny-random-ResNetModel",
        num_cameras=1,
    )
    model = Classifier(config=model_config)
    return model


def test_create_balanced_sampler():
    # Mock dataset with imbalanced classes
    data = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 0},
        {"label": 1},
        {"label": 1},
        {"label": 1},
        {"label": 1},
    ]
    dataset = MockDataset(data)
    cfg = MagicMock()
    cfg.training.label_key = "label"

    sampler = create_balanced_sampler(dataset, cfg)

    # Get weights from the sampler
    weights = sampler.weights.float()

    # Check that samples have appropriate weights
    labels = [item["label"] for item in data]
    class_counts = torch.tensor([labels.count(0), labels.count(1)], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    expected_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float32)

    # Test that the weights are correct
    assert torch.allclose(weights, expected_weights)


def test_train_epoch():
    model = make_dummy_model()
    # Mock components
    model.train = MagicMock()

    train_loader = [
        {
            "image": torch.rand(2, 3, 224, 224),
            "label": torch.tensor([0.0, 1.0]),
        }
    ]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = MagicMock()
    grad_scaler = MagicMock()
    device = torch.device("cpu")
    logger = MagicMock()
    step = 0
    cfg = MagicMock()
    cfg.training.image_keys = ["image"]
    cfg.training.label_key = "label"
    cfg.training.use_amp = False

    # Call the function under test
    train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        grad_scaler,
        device,
        logger,
        step,
        cfg,
    )

    # Check that model.train() was called
    model.train.assert_called_once()

    # Check that optimizer.zero_grad() was called
    optimizer.zero_grad.assert_called()

    # Check that logger.log_dict was called
    logger.log_dict.assert_called()


def test_validate():
    model = make_dummy_model()

    # Mock components
    model.eval = MagicMock()
    val_loader = [
        {
            "image": torch.rand(2, 3, 224, 224),
            "label": torch.tensor([0.0, 1.0]),
        }
    ]
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    logger = MagicMock()
    cfg = MagicMock()
    cfg.training.image_keys = ["image"]
    cfg.training.label_key = "label"
    cfg.training.use_amp = False

    # Call validate
    accuracy, eval_info = validate(model, val_loader, criterion, device, logger, cfg)

    # Check that model.eval() was called
    model.eval.assert_called_once()

    # Check accuracy/eval_info are calculated and of the correct type
    assert isinstance(accuracy, float)
    assert isinstance(eval_info, dict)


def test_train_epoch_multiple_cameras():
    model_config = ClassifierConfig(
        num_classes=2,
        model_name="hf-tiny-model-private/tiny-random-ResNetModel",
        num_cameras=2,
    )
    model = Classifier(config=model_config)

    # Mock components
    model.train = MagicMock()

    train_loader = [
        {
            "image_1": torch.rand(2, 3, 224, 224),
            "image_2": torch.rand(2, 3, 224, 224),
            "label": torch.tensor([0.0, 1.0]),
        }
    ]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = MagicMock()
    grad_scaler = MagicMock()
    device = torch.device("cpu")
    logger = MagicMock()
    step = 0
    cfg = MagicMock()
    cfg.training.image_keys = ["image_1", "image_2"]
    cfg.training.label_key = "label"
    cfg.training.use_amp = False

    # Call the function under test
    train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        grad_scaler,
        device,
        logger,
        step,
        cfg,
    )

    # Check that model.train() was called
    model.train.assert_called_once()

    # Check that optimizer.zero_grad() was called
    optimizer.zero_grad.assert_called()

    # Check that logger.log_dict was called
    logger.log_dict.assert_called()


@pytest.mark.parametrize("resume", [True, False])
@patch("lerobot.scripts.train_hilserl_classifier.init_hydra_config")
@patch("lerobot.scripts.train_hilserl_classifier.Logger.get_last_checkpoint_dir")
@patch("lerobot.scripts.train_hilserl_classifier.Logger.get_last_pretrained_model_dir")
@patch("lerobot.scripts.train_hilserl_classifier.Logger")
@patch("lerobot.scripts.train_hilserl_classifier.LeRobotDataset")
@patch("lerobot.scripts.train_hilserl_classifier.get_model")
def test_resume_function(
    mock_get_model,
    mock_dataset,
    mock_logger,
    mock_get_last_pretrained_model_dir,
    mock_get_last_checkpoint_dir,
    mock_init_hydra_config,
    resume,
):
    # Initialize Hydra
    test_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(test_file_dir, "..", "lerobot", "configs", "policy"))
    assert os.path.exists(config_dir), f"Config directory does not exist at {config_dir}"

    with initialize_config_dir(config_dir=config_dir, job_name="test_app", version_base="1.2"):
        cfg = compose(
            config_name="hilserl_classifier",
            overrides=[
                "device=cpu",
                "seed=42",
                f"output_dir={tempfile.mkdtemp()}",
                "wandb.enable=False",
                f"resume={resume}",
                "dataset_repo_id=dataset_repo_id",
                "train_split_proportion=0.8",
                "training.num_workers=0",
                "training.batch_size=2",
                "training.image_keys=[image]",
                "training.label_key=label",
                "training.use_amp=False",
                "training.num_epochs=1",
                "eval.batch_size=2",
            ],
        )

    # Mock the init_hydra_config function to return cfg
    mock_init_hydra_config.return_value = cfg

    # Mock dataset
    dataset = MockDataset([{"image": torch.rand(3, 224, 224), "label": i % 2} for i in range(10)])
    mock_dataset.return_value = dataset

    # Mock checkpoint handling
    mock_checkpoint_dir = MagicMock(spec=Path)
    mock_checkpoint_dir.exists.return_value = resume  # Only exists if resuming
    mock_get_last_checkpoint_dir.return_value = mock_checkpoint_dir
    mock_get_last_pretrained_model_dir.return_value = Path(tempfile.mkdtemp())

    # Mock logger
    logger = MagicMock()
    resumed_step = 1000
    if resume:
        logger.load_last_training_state.return_value = resumed_step
    else:
        logger.load_last_training_state.return_value = 0
    mock_logger.return_value = logger

    # Instantiate the model and set make_policy to return it
    model = make_dummy_model()
    mock_get_model.return_value = model

    # Call train
    train(cfg)

    # Check that checkpoint handling methods were called
    if resume:
        mock_get_last_checkpoint_dir.assert_called_once_with(Path(cfg.output_dir))
        mock_get_last_pretrained_model_dir.assert_called_once_with(Path(cfg.output_dir))
        mock_checkpoint_dir.exists.assert_called_once()
        logger.load_last_training_state.assert_called_once()
    else:
        mock_get_last_checkpoint_dir.assert_not_called()
        mock_get_last_pretrained_model_dir.assert_not_called()
        mock_checkpoint_dir.exists.assert_not_called()
        logger.load_last_training_state.assert_not_called()

    # Collect the steps from logger.log_dict calls
    train_log_calls = logger.log_dict.call_args_list

    # Extract the steps used in the train logging
    steps = []
    for call in train_log_calls:
        mode = call.kwargs.get("mode", call.args[2] if len(call.args) > 2 else None)
        if mode == "train":
            step = call.kwargs.get("step", call.args[1] if len(call.args) > 1 else None)
            steps.append(step)

    expected_start_step = resumed_step if resume else 0

    # Calculate expected_steps
    train_size = int(cfg.train_split_proportion * len(dataset))
    batch_size = cfg.training.batch_size
    num_batches = (train_size + batch_size - 1) // batch_size

    expected_steps = [expected_start_step + i for i in range(num_batches)]

    assert steps == expected_steps, f"Expected steps {expected_steps}, got {steps}"
