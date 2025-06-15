# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import Tensor, nn

from lerobot.common.constants import OBS_IMAGE, REWARD
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
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


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, H, W, C] or [H, W, C] if no batch
        Returns:
            Output tensor of shape [B, C*F] or [C*F] if no batch
        """

        features = features.last_hidden_state

        original_shape = features.shape
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Add batch dim

        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, H, W, C, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum H,W

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        # Remove batch dim
        if len(original_shape) == 3:
            output = output.squeeze(0)

        return output


class Classifier(PreTrainedPolicy):
    """Image classifier built on top of a pre-trained encoder."""

    name = "reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # Initialize normalization (standardized with the policy framework)
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Set up encoder
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

        # Extract image keys from input_features
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        if self.is_cnn:
            self.encoders = nn.ModuleDict()
            for image_key in self.image_keys:
                encoder = self._create_single_encoder()
                self.encoders[image_key] = encoder

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

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_single_encoder(self):
        encoder = nn.Sequential(
            self.encoder,
            SpatialLearnedEmbeddings(
                height=4,
                width=4,
                channel=self.feature_dim,
                num_features=self.config.image_embedding_pooling_dim,
            ),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.feature_dim * self.config.image_embedding_pooling_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.Tanh(),
        )

        return encoder

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        if self.is_cnn:
            input_dim = self.config.latent_dim
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
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """Extract the appropriate output from the encoder."""
        with torch.no_grad():
            if self.is_cnn:
                # The HF ResNet applies pooling internally
                outputs = self.encoders[image_key](x)
                return outputs
            else:  # Transformer models
                outputs = self.encoder(x)
                return outputs.last_hidden_state[:, 0, :]

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor]:
        """Extract image tensors and label tensors from batch."""
        # Check for both OBS_IMAGE and OBS_IMAGES prefixes
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        labels = batch[REWARD]

        return images, labels

    def predict(self, xs: list) -> ClassifierOutput:
        """Forward pass of the classifier for inference."""
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
        )
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard forward pass for training compatible with train.py."""
        # Normalize inputs if needed
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract images and labels
        images, labels = self.extract_images_and_labels(batch)

        # Get predictions
        outputs = self.predict(images)

        # Calculate loss
        if self.config.num_classes == 2:
            # Binary classification
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # Multi-class classification
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        # Calculate accuracy for logging
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # Return loss and metrics for logging
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch, threshold=0.5):
        """Eval method. Returns predicted reward with the decision threshold as argument."""
        # Check for both OBS_IMAGE and OBS_IMAGES prefixes
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract images from batch dict
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        if self.config.num_classes == 2:
            probs = self.predict(images).probabilities
            logging.debug(f"Predicted reward images: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images).probabilities, dim=1)

    def get_optim_params(self):
        """Return optimizer parameters for the policy."""
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        raise NotImplementedError("Reward classifiers do not select actions")

    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        pass
