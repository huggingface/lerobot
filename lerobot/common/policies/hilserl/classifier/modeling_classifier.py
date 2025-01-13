import logging
from typing import Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from .configuration_classifier import ClassifierConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self, logits: Tensor, probabilities: Optional[Tensor] = None, hidden_states: Optional[Tensor] = None
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class Classifier(
    nn.Module,
    PyTorchModelHubMixin,
    # Add Hub metadata
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "vision-classifier"],
):
    """Image classifier built on top of a pre-trained encoder."""

    # Add name attribute for factory
    name = "classifier"

    def __init__(self, config: ClassifierConfig):
        from transformers import AutoImageProcessor, AutoModel

        super().__init__()
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # Extract vision model if we're given a multimodal model
        if hasattr(encoder, "vision_model"):
            logging.info("Multimodal model detected - using vision encoder only")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # Model type from config
        self.is_cnn = self.config.model_type == "cnn"

        # For CNNs, initialize backbone
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()
        self._build_classifier_head()

    def _setup_cnn_backbone(self):
        """Set up CNN encoder"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # Last channel dimension
        else:
            raise ValueError("Unsupported CNN architecture")

        self.encoder = self.encoder.to(self.config.device)

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        if self.is_cnn:
            input_dim = self.feature_dim
        else:  # Transformer models
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * self.config.num_cameras, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1 if self.config.num_classes == 2 else self.config.num_classes),
        )
        self.classifier_head = self.classifier_head.to(self.config.device)

    def _get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the appropriate output from the encoder."""
        # Process images with the processor (handles resizing and normalization)
        processed = self.processor(
            images=x,  # LeRobotDataset already provides proper tensor format
            return_tensors="pt",
        )
        processed = processed["pixel_values"].to(x.device)

        with torch.no_grad():
            if self.is_cnn:
                # The HF ResNet applies pooling internally
                outputs = self.encoder(processed)
                # Get pooled output directly
                features = outputs.pooler_output

                if features.dim() > 2:
                    features = features.squeeze(-1).squeeze(-1)
                return features
            else:  # Transformer models
                outputs = self.encoder(processed)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                return outputs.last_hidden_state[:, 0, :]

    def forward(self, xs: torch.Tensor) -> ClassifierOutput:
        """Forward pass of the classifier."""
        # For training, we expect input to be a tensor directly from LeRobotDataset
        encoder_outputs = torch.hstack([self._get_encoder_output(x) for x in xs])
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def predict_reward(self, x):
        if self.config.num_classes == 2:
            return (self.forward(x).probabilities > 0.5).float()
        else:
            return torch.argmax(self.forward(x).probabilities, dim=1)
