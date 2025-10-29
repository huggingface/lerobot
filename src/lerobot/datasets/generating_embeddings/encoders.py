#!/usr/bin/env python

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

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEncoder:
    """Base class for image encoders."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of images."""
        raise NotImplementedError


class DinoV2Encoder(ImageEncoder):
    """DinoV2 image encoder.

    DinoV2 is a self-supervised vision transformer that produces high-quality image embeddings.
    Supports multiple model sizes (ViT-S/14, ViT-B/14, ViT-L/14).
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda", batch_size: int = 32):
        super().__init__(device)
        self.batch_size = batch_size
        self.model_name = model_name
        logger.info(f"Loading DinoV2 model: {model_name}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)  # nosec B614
        self.model = self.model.to(self.device)
        self.model.eval()

        # DinoV2 preprocessing
        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of images."""
        embeddings = []

        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i : i + self.batch_size]
                # Convert numpy arrays to PIL Images and apply transforms
                pil_images = [Image.fromarray(img.astype(np.uint8)) for img in batch_images]
                tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)

                # Get embeddings
                batch_embeddings = self.model(tensors).cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.concatenate(embeddings, axis=0)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension based on model size."""
        if "vits14" in self.model_name:
            return 384  # DinoV2 ViT-S/14
        elif "vitb14" in self.model_name:
            return 768  # DinoV2 ViT-B/14
        elif "vitl14" in self.model_name:
            return 1024  # DinoV2 ViT-L/14
        else:
            return 768  # Default to ViT-B/14


class LanguageEncoder:
    """Base class for language encoders."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        raise NotImplementedError


class MiniLMEncoder(LanguageEncoder):
    """MiniLM language encoder.

    MiniLM is a lightweight sentence transformer model that produces high-quality text embeddings.
    Supports L6 and L12 model sizes.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device: str = "cuda"):
        super().__init__(device)
        self.model_name = model_name
        logger.info(f"Loading MiniLM model: {model_name}")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        with torch.inference_mode():
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

            return embeddings.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return 384  # Both MiniLM-L6 and L12 output 384-dim embeddings
