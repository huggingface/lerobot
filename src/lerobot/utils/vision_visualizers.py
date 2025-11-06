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

"""Vision feature visualizers for debugging and analysis."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class VisionVisualizer(Protocol):
    """Protocol for vision feature visualizers.

    Any visualizer that implements __call__ with this signature can be used.
    """

    def __call__(self, image: np.ndarray | torch.Tensor, camera_name: str) -> None:
        """Visualize features from an image.

        Args:
            image: RGB image as numpy array (H, W, 3) or torch tensor (C, H, W) in [0, 255] or [0, 1]
            camera_name: Name of the camera for logging
        """
        ...


class DINOv2Visualizer:
    """Visualizer for DINOv2/v3 features.

    Extracts and visualizes:
    - PCA of patch tokens (spatial feature map)
    - Attention maps (optional)
    - CLS token features

    Supports both DINOv2 and DINOv3 models from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vit-base-pretrain-lvd1689m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        visualize_attention: bool = False,
        log_to_rerun: bool = True,
    ):
        """Initialize DINOv2/v3 visualizer.

        Args:
            model_name: HuggingFace model name
                DINOv3 ViT: facebook/dinov3-vit-{small/base/large/huge}-pretrain-{lvd1689m/sat493m}
                DINOv3 ConvNeXt: facebook/dinov3-convnext-{tiny/small/base/large}-pretrain-{lvd1689m/sat493m}
                DINOv2: facebook/dinov2-{small/base/large/giant}
            device: Device to run model on
            visualize_attention: Whether to visualize attention maps (ViT only)
            log_to_rerun: Whether to log to rerun (if available)
        """
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.visualize_attention = visualize_attention
        self.log_to_rerun = log_to_rerun and RERUN_AVAILABLE
        self.model_name = model_name

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Get model config
        self.is_vit = hasattr(self.model.config, 'patch_size')
        if self.is_vit:
            self.patch_size = self.model.config.patch_size
            self.hidden_size = self.model.config.hidden_size
        else:
            # ConvNeXt models don't have patch_size, will extract spatial features differently
            self.patch_size = None
            self.hidden_size = self.model.config.hidden_sizes[-1] if hasattr(self.model.config, 'hidden_sizes') else self.model.config.num_channels

        # PCA components (will be fit on first batch)
        self.pca_components = None
        self.pca_mean = None

    @torch.no_grad()
    def __call__(self, image: np.ndarray | torch.Tensor, camera_name: str = "camera") -> dict:
        """Extract and visualize DINOv2 features.

        Args:
            image: RGB image as numpy array (H, W, 3) or torch tensor (C, H, W)
            camera_name: Name of the camera for logging

        Returns:
            Dictionary with extracted features and visualizations
        """
        # Convert to PIL for processor
        if isinstance(image, torch.Tensor):
            # Assume (C, H, W) format
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = image.permute(1, 2, 0).cpu().numpy()

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Process and extract features
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_attentions=self.visualize_attention)

        # Get patch tokens (exclude CLS token)
        last_hidden_state = outputs.last_hidden_state  # (1, num_patches + 1, hidden_size)
        cls_token = last_hidden_state[:, 0]  # (1, hidden_size)
        patch_tokens = last_hidden_state[:, 1:]  # (1, num_patches, hidden_size)

        # Reshape patch tokens to spatial grid
        num_patches = patch_tokens.shape[1]
        grid_size = int(np.sqrt(num_patches))
        assert grid_size * grid_size == num_patches, "Number of patches must be a perfect square"

        # (1, num_patches, hidden_size) -> (1, hidden_size, grid_size, grid_size)
        spatial_features = patch_tokens.reshape(1, grid_size, grid_size, self.hidden_size)
        spatial_features = spatial_features.permute(0, 3, 1, 2)

        # Create PCA visualization (reduce hidden_size to 3 for RGB)
        pca_vis = self._create_pca_visualization(spatial_features, grid_size)

        # Resize PCA visualization to match input image size
        h, w = pil_image.size[1], pil_image.size[0]  # PIL is (W, H)
        pca_vis_resized = F.interpolate(
            pca_vis.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        result = {
            "cls_token": cls_token.cpu(),
            "patch_tokens": patch_tokens.cpu(),
            "spatial_features": spatial_features.cpu(),
            "pca_visualization": pca_vis_resized.cpu(),
        }

        # Visualize attention if requested
        if self.visualize_attention and outputs.attentions is not None:
            attention_maps = outputs.attentions[-1]  # Last layer attention
            result["attention"] = attention_maps.cpu()

            # Create attention visualization (average over heads, focus on CLS token)
            attn_vis = self._create_attention_visualization(attention_maps, grid_size)
            attn_vis_resized = F.interpolate(
                attn_vis.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            result["attention_visualization"] = attn_vis_resized.cpu()

        # Log to rerun
        if self.log_to_rerun:
            self._log_to_rerun(image, result, camera_name)

        return result

    def _create_pca_visualization(self, spatial_features: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Create PCA visualization of spatial features.

        Args:
            spatial_features: (1, hidden_size, grid_size, grid_size)
            grid_size: Size of the spatial grid

        Returns:
            RGB visualization (3, grid_size, grid_size) in [0, 1]
        """
        # Reshape to (num_patches, hidden_size)
        features = spatial_features.squeeze(0).permute(1, 2, 0).reshape(-1, self.hidden_size)

        # Initialize PCA components if needed (simple PCA on first call)
        if self.pca_components is None:
            # Compute mean and top 3 principal components
            self.pca_mean = features.mean(dim=0, keepdim=True)
            centered = features - self.pca_mean

            # SVD for PCA
            u, _, _ = torch.svd(centered.T @ centered)
            self.pca_components = u[:, :3]  # (hidden_size, 3)

        # Project features to 3D
        centered = features - self.pca_mean
        pca_features = centered @ self.pca_components  # (num_patches, 3)

        # Normalize to [0, 1] for RGB visualization
        pca_features = pca_features.reshape(grid_size, grid_size, 3)
        for i in range(3):
            channel = pca_features[..., i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                pca_features[..., i] = (channel - min_val) / (max_val - min_val)

        # Convert to (3, grid_size, grid_size)
        pca_vis = pca_features.permute(2, 0, 1)

        return pca_vis

    def _create_attention_visualization(self, attention_maps: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Create attention visualization (CLS token attention to patches).

        Args:
            attention_maps: (1, num_heads, num_tokens, num_tokens)
            grid_size: Size of the spatial grid

        Returns:
            Attention heatmap (grid_size, grid_size) in [0, 1]
        """
        # Average over heads and get CLS token attention to patches
        attn = attention_maps.mean(dim=1)  # (1, num_tokens, num_tokens)
        cls_attn = attn[0, 0, 1:]  # (num_patches,) - CLS attention to patches

        # Reshape to spatial grid
        attn_map = cls_attn.reshape(grid_size, grid_size)

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        return attn_map

    def _log_to_rerun(self, original_image: np.ndarray, features: dict, camera_name: str):
        """Log visualizations to rerun.

        Args:
            original_image: Original RGB image
            features: Dictionary with extracted features
            camera_name: Name of the camera
        """
        if not RERUN_AVAILABLE:
            return

        prefix = f"vision_features/{camera_name}"

        # Log original image
        rr.log(f"{prefix}/original", rr.Image(original_image))

        # Log PCA visualization
        pca_vis = features["pca_visualization"].permute(1, 2, 0).numpy()  # (H, W, 3)
        pca_vis = (pca_vis * 255).astype(np.uint8)
        rr.log(f"{prefix}/dino_pca", rr.Image(pca_vis))

        # Log attention if available
        if "attention_visualization" in features:
            attn_vis = features["attention_visualization"].numpy()  # (H, W)
            # Convert to RGB heatmap
            attn_rgb = self._heatmap_to_rgb(attn_vis)
            rr.log(f"{prefix}/dino_attention", rr.Image(attn_rgb))

        # Log CLS token as time series (useful for tracking)
        cls_token = features["cls_token"].squeeze().numpy()
        for i in range(min(16, len(cls_token))):  # Log first 16 dims
            rr.log(f"{prefix}/cls_token/dim_{i}", rr.Scalar(float(cls_token[i])))

    @staticmethod
    def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
        """Convert heatmap to RGB using jet colormap.

        Args:
            heatmap: (H, W) array in [0, 1]

        Returns:
            RGB image (H, W, 3) in [0, 255]
        """
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('jet')
        colored = cmap(heatmap)  # (H, W, 4) RGBA
        rgb = (colored[..., :3] * 255).astype(np.uint8)
        return rgb


class IdentityVisualizer:
    """No-op visualizer that does nothing."""

    def __call__(self, image: np.ndarray | torch.Tensor, camera_name: str = "camera") -> None:
        """Do nothing."""
        pass


def make_vision_visualizer(
    visualizer_type: str | None = None,
    **kwargs
) -> VisionVisualizer:
    """Factory function for vision visualizers.

    Args:
        visualizer_type: Type of visualizer ("dinov2", "none", or None)
            Note: "dinov2" supports both DINOv2 and DINOv3 models via the model_name parameter
        **kwargs: Additional arguments passed to visualizer constructor
            For DINOv2Visualizer:
                - model_name: str (default: "facebook/dinov3-vit-base-pretrain-lvd1689m")
                - device: str (default: "cuda" if available else "cpu")
                - visualize_attention: bool (default: False)
                - log_to_rerun: bool (default: True)

    Returns:
        Vision visualizer instance

    Examples:
        >>> # DINOv3 ViT
        >>> viz = make_vision_visualizer("dinov2", model_name="facebook/dinov3-vit-base-pretrain-lvd1689m")
        >>> # DINOv3 ConvNeXt
        >>> viz = make_vision_visualizer("dinov2", model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        >>> # DINOv2
        >>> viz = make_vision_visualizer("dinov2", model_name="facebook/dinov2-base")
    """
    if visualizer_type is None or visualizer_type == "none":
        return IdentityVisualizer()
    elif visualizer_type == "dinov2":
        return DINOv2Visualizer(**kwargs)
    else:
        raise ValueError(f"Unknown visualizer type: {visualizer_type}")
