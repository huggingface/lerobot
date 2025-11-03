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

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor

from lerobot.policies.rtc.debug_handler import Tracker


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
        import torch

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

            # Add legend if label provided and this is the first dimension
            if label and dim_idx == 0:
                ax.legend(loc="best", fontsize=8)

    @staticmethod
    def plot_debug_summary(
        tracker: Tracker,
        save_path: str | None = None,
        show: bool = False,
        figsize: tuple[int, int] = (16, 12),
    ) -> Figure:
        """Create a comprehensive summary plot of debug information.

        Args:
            tracker (Tracker): Tracker with recorded steps.
            save_path (str | None): Path to save the figure. If None, figure is not saved.
            show (bool): Whether to display the figure.
            figsize (tuple[int, int]): Figure size in inches (width, height).

        Returns:
            Figure: The matplotlib figure object.
        """
        if not tracker.enabled or len(tracker) == 0:
            print("Tracker is disabled or has no recorded steps.")
            return None

        steps = tracker.get_all_steps()
        num_steps = len(steps)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f"RTC Debug Summary ({num_steps} steps)", fontsize=16, fontweight="bold")

        # Plot 1: Correction norms over steps
        ax = axes[0, 0]
        correction_norms = [step.correction.norm().item() for step in steps if step.correction is not None]
        if correction_norms:
            ax.plot(correction_norms, marker="o", linewidth=2, markersize=4)
            ax.set_xlabel("Step Index", fontsize=12)
            ax.set_ylabel("Correction Norm", fontsize=12)
            ax.set_title("Correction Magnitude Over Steps", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot 2: Error norms over steps
        ax = axes[0, 1]
        error_norms = [step.err.norm().item() for step in steps if step.err is not None]
        if error_norms:
            ax.plot(error_norms, marker="o", linewidth=2, markersize=4, color="orange")
            ax.set_xlabel("Step Index", fontsize=12)
            ax.set_ylabel("Error Norm", fontsize=12)
            ax.set_title("Error Magnitude Over Steps", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot 3: Guidance weights over steps
        ax = axes[1, 0]
        guidance_weights = [
            step.guidance_weight.item() if isinstance(step.guidance_weight, Tensor) else step.guidance_weight
            for step in steps
            if step.guidance_weight is not None
        ]
        if guidance_weights:
            ax.plot(guidance_weights, marker="o", linewidth=2, markersize=4, color="green")
            ax.set_xlabel("Step Index", fontsize=12)
            ax.set_ylabel("Guidance Weight", fontsize=12)
            ax.set_title("Guidance Weight Over Steps", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot 4: Time parameter over steps
        ax = axes[1, 1]
        times = [
            step.time.item() if isinstance(step.time, Tensor) else step.time
            for step in steps
            if step.time is not None
        ]
        if times:
            ax.plot(times, marker="o", linewidth=2, markersize=4, color="purple")
            ax.set_xlabel("Step Index", fontsize=12)
            ax.set_ylabel("Time Parameter", fontsize=12)
            ax.set_title("Time Parameter Over Steps", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot 5: Correction vs Error relationship
        ax = axes[2, 0]
        if correction_norms and error_norms:
            ax.scatter(error_norms, correction_norms, alpha=0.6, s=50)
            ax.set_xlabel("Error Norm", fontsize=12)
            ax.set_ylabel("Correction Norm", fontsize=12)
            ax.set_title("Correction vs Error", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot 6: Prefix attention weights visualization (last step)
        ax = axes[2, 1]
        last_step = steps[-1]
        if last_step.weights is not None:
            weights = last_step.weights.squeeze().cpu().numpy()
            ax.plot(weights, linewidth=2, marker="o", markersize=4, color="red")
            ax.set_xlabel("Time Index", fontsize=12)
            ax.set_ylabel("Weight Value", fontsize=12)
            ax.set_title("Prefix Attention Weights (Last Step)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Debug summary saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def plot_correction_heatmap(
        tracker: Tracker,
        save_path: str | None = None,
        show: bool = False,
        figsize: tuple[int, int] = (14, 8),
        max_dims: int = 6,
    ) -> Figure:
        """Create a heatmap showing correction values across steps and action dimensions.

        Args:
            tracker (Tracker): Tracker with recorded steps.
            save_path (str | None): Path to save the figure.
            show (bool): Whether to display the figure.
            figsize (tuple[int, int]): Figure size in inches.
            max_dims (int): Maximum number of action dimensions to visualize.

        Returns:
            Figure: The matplotlib figure object.
        """
        if not tracker.enabled or len(tracker) == 0:
            print("Tracker is disabled or has no recorded steps.")
            return None

        steps = tracker.get_all_steps()

        # Collect corrections across steps (shape: [num_steps, time, action_dim])
        corrections = [step.correction for step in steps if step.correction is not None]
        if not corrections:
            print("No corrections found in debug steps.")
            return None

        # Stack corrections: [num_steps, time, action_dim]
        # Take mean over time dimension and limit action dims
        corrections_stacked = torch.stack(corrections)  # [num_steps, batch, time, action_dim]
        corrections_mean = corrections_stacked.mean(dim=(1, 2))  # [num_steps, action_dim]

        # Limit to max_dims
        corrections_mean = corrections_mean[:, :max_dims].cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corrections_mean.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")

        ax.set_xlabel("Step Index", fontsize=12)
        ax.set_ylabel("Action Dimension", fontsize=12)
        ax.set_title("Correction Values Heatmap (averaged over time)", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correction Value", fontsize=12)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Correction heatmap saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def plot_step_by_step_comparison(
        tracker: Tracker,
        step_idx: int = -1,
        save_path: str | None = None,
        show: bool = False,
        figsize: tuple[int, int] = (18, 10),
        max_dims: int = 6,
    ) -> Figure:
        """Plot detailed comparison for a single denoising step.

        Args:
            tracker (Tracker): Tracker with recorded steps.
            step_idx (int): Step index to visualize (-1 for last step).
            save_path (str | None): Path to save the figure.
            show (bool): Whether to display the figure.
            figsize (tuple[int, int]): Figure size in inches.
            max_dims (int): Maximum number of action dimensions to visualize.

        Returns:
            Figure: The matplotlib figure object.
        """
        if not tracker.enabled or len(tracker) == 0:
            print("Tracker is disabled or has no recorded steps.")
            return None

        steps = tracker.get_all_steps()
        step = steps[step_idx]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f"Detailed Step Analysis (Step {step.step_idx})",
            fontsize=16,
            fontweight="bold",
        )

        # Get tensors and squeeze batch dimension
        x_t = step.x_t.squeeze(0).cpu().numpy() if step.x_t is not None else None
        v_t = step.v_t.squeeze(0).cpu().numpy() if step.v_t is not None else None
        x1_t = step.x1_t.squeeze(0).cpu().numpy() if step.x1_t is not None else None
        correction = step.correction.squeeze(0).cpu().numpy() if step.correction is not None else None
        err = step.err.squeeze(0).cpu().numpy() if step.err is not None else None
        weights = step.weights.squeeze().cpu().numpy() if step.weights is not None else None

        # Limit to max_dims
        num_dims = min(max_dims, x_t.shape[1] if x_t is not None else 0)

        # Plot 1: x_t (current state)
        ax = axes[0, 0]
        if x_t is not None:
            for dim in range(num_dims):
                ax.plot(x_t[:, dim], label=f"Dim {dim}", alpha=0.7)
            ax.set_title("x_t (Current State)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 2: v_t (velocity)
        ax = axes[0, 1]
        if v_t is not None:
            for dim in range(num_dims):
                ax.plot(v_t[:, dim], label=f"Dim {dim}", alpha=0.7)
            ax.set_title("v_t (Velocity)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 3: x1_t (predicted state)
        ax = axes[0, 2]
        if x1_t is not None:
            for dim in range(num_dims):
                ax.plot(x1_t[:, dim], label=f"Dim {dim}", alpha=0.7)
            ax.set_title("x1_t (Predicted State)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 4: correction
        ax = axes[1, 0]
        if correction is not None:
            for dim in range(num_dims):
                ax.plot(correction[:, dim], label=f"Dim {dim}", alpha=0.7)
            ax.set_title("Correction", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 5: error
        ax = axes[1, 1]
        if err is not None:
            for dim in range(num_dims):
                ax.plot(err[:, dim], label=f"Dim {dim}", alpha=0.7)
            ax.set_title("Error (Weighted)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 6: prefix weights
        ax = axes[1, 2]
        if weights is not None:
            ax.plot(weights, linewidth=2, marker="o", markersize=4, color="red")
            ax.set_title("Prefix Attention Weights", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Weight Value")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Step-by-step comparison saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
