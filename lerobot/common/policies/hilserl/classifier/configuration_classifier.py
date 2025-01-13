import json
import os
from dataclasses import asdict, dataclass


@dataclass
class ClassifierConfig:
    """Configuration for the Classifier model."""

    num_classes: int = 2
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    model_name: str = "microsoft/resnet-50"
    device: str = "cpu"
    model_type: str = "cnn"  # "transformer" or "cnn"
    num_cameras: int = 2

    def save_pretrained(self, save_dir):
        """Save config to json file."""
        os.makedirs(save_dir, exist_ok=True)

        # Convert to dict and save as JSON
        config_dict = asdict(self)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        """Load config from json file."""
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")

        with open(config_file) as f:
            config_dict = json.load(f)

        return cls(**config_dict)
