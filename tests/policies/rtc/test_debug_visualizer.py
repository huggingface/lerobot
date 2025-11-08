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

"""Tests for RTC debug visualizer module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer

# ====================== Fixtures ======================


@pytest.fixture
def mock_axes():
    """Create mock matplotlib axes."""
    axes = []
    for _ in range(6):
        ax = MagicMock()
        ax.xaxis.get_label.return_value.get_text.return_value = ""
        ax.yaxis.get_label.return_value.get_text.return_value = ""
        axes.append(ax)
    return axes


@pytest.fixture
def sample_tensor_2d():
    """Create a 2D sample tensor (time_steps, num_dims)."""
    return torch.randn(50, 6)


@pytest.fixture
def sample_tensor_3d():
    """Create a 3D sample tensor (batch, time_steps, num_dims)."""
    return torch.randn(1, 50, 6)


@pytest.fixture
def sample_numpy_2d():
    """Create a 2D numpy array."""
    return np.random.randn(50, 6)


# ====================== Basic Plotting Tests ======================


def test_plot_waypoints_with_2d_tensor(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with 2D tensor."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d)

    # Should call plot on each axis (6 dimensions)
    for ax in mock_axes:
        ax.plot.assert_called_once()


def test_plot_waypoints_with_3d_tensor(mock_axes, sample_tensor_3d):
    """Test plot_waypoints with 3D tensor (batch dimension)."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_3d)

    # Should still plot 6 dimensions (batch dimension removed)
    for ax in mock_axes:
        ax.plot.assert_called_once()


def test_plot_waypoints_with_numpy_array(mock_axes, sample_numpy_2d):
    """Test plot_waypoints with numpy array."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_numpy_2d)

    # Should work with numpy arrays
    for ax in mock_axes:
        ax.plot.assert_called_once()


def test_plot_waypoints_with_none_tensor(mock_axes):
    """Test plot_waypoints returns early when tensor is None."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, None)

    # Should not call plot on any axis
    for ax in mock_axes:
        ax.plot.assert_not_called()


# ====================== Parameter Tests ======================


def test_plot_waypoints_with_custom_color(mock_axes, sample_tensor_2d):
    """Test plot_waypoints uses custom color."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, color="red")

    # Check that color was passed to plot
    for ax in mock_axes:
        call_kwargs = ax.plot.call_args[1]
        assert call_kwargs["color"] == "red"


def test_plot_waypoints_with_custom_label(mock_axes, sample_tensor_2d):
    """Test plot_waypoints uses custom label."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, label="test_label")

    # First axis should have label, others should not
    first_ax_kwargs = mock_axes[0].plot.call_args[1]
    assert first_ax_kwargs["label"] == "test_label"

    # Other axes should have empty label
    for ax in mock_axes[1:]:
        call_kwargs = ax.plot.call_args[1]
        assert call_kwargs["label"] == ""


def test_plot_waypoints_with_custom_alpha(mock_axes, sample_tensor_2d):
    """Test plot_waypoints uses custom alpha."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, alpha=0.5)

    for ax in mock_axes:
        call_kwargs = ax.plot.call_args[1]
        assert call_kwargs["alpha"] == 0.5


def test_plot_waypoints_with_custom_linewidth(mock_axes, sample_tensor_2d):
    """Test plot_waypoints uses custom linewidth."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, linewidth=3)

    for ax in mock_axes:
        call_kwargs = ax.plot.call_args[1]
        assert call_kwargs["linewidth"] == 3


def test_plot_waypoints_with_marker(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with marker style."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, marker="o", markersize=5)

    for ax in mock_axes:
        call_kwargs = ax.plot.call_args[1]
        assert call_kwargs["marker"] == "o"
        assert call_kwargs["markersize"] == 5


def test_plot_waypoints_without_marker(mock_axes, sample_tensor_2d):
    """Test plot_waypoints without marker (default)."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, marker=None)

    # Marker should not be in kwargs when None
    for ax in mock_axes:
        call_kwargs = ax.plot.call_args[1]
        assert "marker" not in call_kwargs
        assert "markersize" not in call_kwargs


# ====================== start_from Parameter Tests ======================


def test_plot_waypoints_with_start_from_zero(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with start_from=0."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, start_from=0)

    # X indices should start from 0
    for ax in mock_axes:
        call_args = ax.plot.call_args[0]
        x_indices = call_args[0]
        assert x_indices[0] == 0
        assert len(x_indices) == 50


def test_plot_waypoints_with_start_from_nonzero(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with start_from > 0."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, start_from=10)

    # X indices should start from 10
    for ax in mock_axes:
        call_args = ax.plot.call_args[0]
        x_indices = call_args[0]
        assert x_indices[0] == 10
        assert x_indices[-1] == 59  # 10 + 50 - 1


# ====================== Tensor Shape Tests ======================


def test_plot_waypoints_with_1d_tensor(mock_axes):
    """Test plot_waypoints with 1D tensor."""
    tensor_1d = torch.randn(50)
    RTCDebugVisualizer.plot_waypoints(mock_axes, tensor_1d)

    # Should reshape to (50, 1) and plot on first axis only
    mock_axes[0].plot.assert_called_once()
    for ax in mock_axes[1:]:
        ax.plot.assert_not_called()


def test_plot_waypoints_with_fewer_dims_than_axes(sample_tensor_2d):
    """Test plot_waypoints when tensor has fewer dims than axes."""
    # Create tensor with only 3 dimensions
    tensor_3d = sample_tensor_2d[:, :3]

    # Create 6 axes but tensor only has 3 dims
    mock_axes = [MagicMock() for _ in range(6)]
    for ax in mock_axes:
        ax.xaxis.get_label.return_value.get_text.return_value = ""
        ax.yaxis.get_label.return_value.get_text.return_value = ""

    RTCDebugVisualizer.plot_waypoints(mock_axes, tensor_3d)

    # Should only plot on first 3 axes
    for i in range(3):
        mock_axes[i].plot.assert_called_once()
    for i in range(3, 6):
        mock_axes[i].plot.assert_not_called()


# ====================== Axis Labeling Tests ======================


def test_plot_waypoints_sets_xlabel(mock_axes, sample_tensor_2d):
    """Test plot_waypoints sets x-axis label."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d)

    for ax in mock_axes:
        ax.set_xlabel.assert_called_once_with("Step", fontsize=10)


def test_plot_waypoints_sets_ylabel(mock_axes, sample_tensor_2d):
    """Test plot_waypoints sets y-axis label."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d)

    for i, ax in enumerate(mock_axes):
        ax.set_ylabel.assert_called_once_with(f"Dim {i}", fontsize=10)


def test_plot_waypoints_skips_label_if_exists(sample_tensor_2d):
    """Test plot_waypoints doesn't set labels if they already exist."""
    mock_axes_with_labels = []
    for _ in range(6):
        ax = MagicMock()
        # Simulate existing labels
        ax.xaxis.get_label.return_value.get_text.return_value = "Existing X Label"
        ax.yaxis.get_label.return_value.get_text.return_value = "Existing Y Label"
        mock_axes_with_labels.append(ax)

    RTCDebugVisualizer.plot_waypoints(mock_axes_with_labels, sample_tensor_2d)

    # Should not set labels when they already exist
    for ax in mock_axes_with_labels:
        ax.set_xlabel.assert_not_called()
        ax.set_ylabel.assert_not_called()


# ====================== Grid Tests ======================


def test_plot_waypoints_enables_grid(mock_axes, sample_tensor_2d):
    """Test plot_waypoints enables grid on all axes."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d)

    for ax in mock_axes:
        ax.grid.assert_called_once_with(True, alpha=0.3)


# ====================== Legend Tests ======================


def test_plot_waypoints_adds_legend_with_label(mock_axes, sample_tensor_2d):
    """Test plot_waypoints adds legend when label is provided."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, label="test_label")

    # Should add legend to first axis only
    mock_axes[0].legend.assert_called_once_with(loc="best", fontsize=8)

    # Should not add legend to other axes
    for ax in mock_axes[1:]:
        ax.legend.assert_not_called()


def test_plot_waypoints_no_legend_without_label(mock_axes, sample_tensor_2d):
    """Test plot_waypoints doesn't add legend when no label provided."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, label="")

    # Should not add legend to any axis
    for ax in mock_axes:
        ax.legend.assert_not_called()


# ====================== Data Correctness Tests ======================


def test_plot_waypoints_plots_correct_data(mock_axes, sample_tensor_2d):
    """Test plot_waypoints plots correct tensor values."""
    RTCDebugVisualizer.plot_waypoints(mock_axes, sample_tensor_2d, start_from=0)

    # Check first axis to verify data correctness
    call_args = mock_axes[0].plot.call_args[0]
    x_indices = call_args[0]
    y_values = call_args[1]

    # X indices should be 0 to 49
    np.testing.assert_array_equal(x_indices, np.arange(50))

    # Y values should match first dimension of tensor
    expected_y = sample_tensor_2d[:, 0].cpu().numpy()
    np.testing.assert_array_almost_equal(y_values, expected_y)


def test_plot_waypoints_handles_gpu_tensor(mock_axes):
    """Test plot_waypoints handles GPU tensors (if CUDA available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensor_gpu = torch.randn(50, 6, device="cuda")
    RTCDebugVisualizer.plot_waypoints(mock_axes, tensor_gpu)

    # Should successfully plot without errors
    for ax in mock_axes:
        ax.plot.assert_called_once()


# ====================== Edge Cases Tests ======================


def test_plot_waypoints_with_empty_tensor(mock_axes):
    """Test plot_waypoints with empty tensor."""
    empty_tensor = torch.empty(0, 6)
    RTCDebugVisualizer.plot_waypoints(mock_axes, empty_tensor)

    # Should plot empty data
    for ax in mock_axes:
        call_args = ax.plot.call_args[0]
        x_indices = call_args[0]
        assert len(x_indices) == 0


def test_plot_waypoints_with_single_timestep(mock_axes):
    """Test plot_waypoints with single timestep."""
    single_step_tensor = torch.randn(1, 6)
    RTCDebugVisualizer.plot_waypoints(mock_axes, single_step_tensor)

    # Should plot single point
    for ax in mock_axes:
        call_args = ax.plot.call_args[0]
        x_indices = call_args[0]
        assert len(x_indices) == 1


def test_plot_waypoints_with_very_large_tensor(mock_axes):
    """Test plot_waypoints with very large tensor."""
    large_tensor = torch.randn(10000, 6)
    RTCDebugVisualizer.plot_waypoints(mock_axes, large_tensor)

    # Should handle large tensors
    for ax in mock_axes:
        call_args = ax.plot.call_args[0]
        x_indices = call_args[0]
        assert len(x_indices) == 10000


# ====================== Multiple Calls Tests ======================


def test_plot_waypoints_multiple_calls_on_same_axes(mock_axes, sample_tensor_2d):
    """Test multiple plot_waypoints calls on same axes."""
    tensor1 = sample_tensor_2d
    tensor2 = torch.randn(50, 6)

    RTCDebugVisualizer.plot_waypoints(mock_axes, tensor1, color="blue", label="Series 1")
    RTCDebugVisualizer.plot_waypoints(mock_axes, tensor2, color="red", label="Series 2")

    # Each axis should have been called twice
    for ax in mock_axes:
        assert ax.plot.call_count == 2


# ====================== Integration Tests ======================


def test_plot_waypoints_typical_usage(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with typical usage pattern."""
    RTCDebugVisualizer.plot_waypoints(
        mock_axes, sample_tensor_2d, start_from=0, color="blue", label="Trajectory", alpha=0.7, linewidth=2
    )

    # Verify all expected calls were made
    for ax in mock_axes:
        ax.plot.assert_called_once()
        ax.set_xlabel.assert_called_once()
        ax.set_ylabel.assert_called_once()
        ax.grid.assert_called_once()

    # First axis should have legend
    mock_axes[0].legend.assert_called_once()


def test_plot_waypoints_with_all_parameters(mock_axes, sample_tensor_2d):
    """Test plot_waypoints with all parameters specified."""
    RTCDebugVisualizer.plot_waypoints(
        axes=mock_axes,
        tensor=sample_tensor_2d,
        start_from=10,
        color="green",
        label="Full Test",
        alpha=0.8,
        linewidth=3,
        marker="o",
        markersize=6,
    )

    # Check first axis for all parameters
    call_kwargs = mock_axes[0].plot.call_args[1]
    assert call_kwargs["color"] == "green"
    assert call_kwargs["label"] == "Full Test"
    assert call_kwargs["alpha"] == 0.8
    assert call_kwargs["linewidth"] == 3
    assert call_kwargs["marker"] == "o"
    assert call_kwargs["markersize"] == 6
