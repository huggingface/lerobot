#!/usr/bin/env python

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

"""Visualization utilities for RTC debug information."""

import torch


class RTCDebugVisualizer:
    """Visualizer for RTC debug information.

    This class provides methods to visualize debug information collected by the Tracker,
    including corrections, errors, weights, and guidance weights over denoising steps.
    """

    @staticmethod
    def plot_waypoints(
        axes,
        tensor,
        start_from: int = 0,
        color: str = "blue",
        label: str = "",
        alpha: float = 0.7,
        linewidth: float = 2,
        marker: str | None = None,
        markersize: int = 4,
    ):
        """Plot trajectories across multiple dimensions.

        This function plots a tensor's values across time for multiple dimensions,
        with each dimension plotted on a separate axis.

        Args:
            axes: Array of matplotlib axes (one for each dimension).
            tensor: The tensor to plot (can be torch.Tensor or numpy array).
                   Shape should be (time_steps, num_dims) or (batch, time_steps, num_dims).
            start_from: Starting index for the x-axis.
            color: Color for the plot lines.
            label: Label for the plot legend.
            alpha: Transparency level for the plot.
            linewidth: Width of the plot lines.
            marker: Marker style for data points (e.g., 'o', 's', '^').
            markersize: Size of the markers.
        """
        import numpy as np

        # Handle None tensor
        if tensor is None:
            return

        # Convert tensor to numpy if needed
        tensor_np = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

        # Handle different tensor shapes
        if tensor_np.ndim == 3:
            # If batch dimension present, take first batch
            tensor_np = tensor_np[0]
        elif tensor_np.ndim == 1:
            # If 1D, reshape to (time_steps, 1)
            tensor_np = tensor_np.reshape(-1, 1)

        # Get dimensions
        time_steps, num_dims = tensor_np.shape

        # Create x-axis indices
        x_indices = np.arange(start_from, start_from + time_steps)

        # Plot each dimension on its corresponding axis
        num_axes = len(axes) if hasattr(axes, "__len__") else 1
        for dim_idx in range(min(num_dims, num_axes)):
            ax = axes[dim_idx] if hasattr(axes, "__len__") else axes

            # Plot the trajectory
            if marker:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # Only show label once
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                )
            else:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # Only show label once
                    alpha=alpha,
                    linewidth=linewidth,
                )

            # Add grid and labels if not already present
            if not ax.xaxis.get_label().get_text():
                ax.set_xlabel("Step", fontsize=10)
            if not ax.yaxis.get_label().get_text():
                ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)
