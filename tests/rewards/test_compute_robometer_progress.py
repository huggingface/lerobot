import numpy as np
import pytest
import torch

from lerobot.rewards.robometer.compute_rabc_weights import (
    _build_inference_indices,
    _build_subsample_indices,
    _interpolate_dense,
    _limit_trajectory_indices,
    _sample_scalar,
)


def test_frame_step_indices_match_upstream_integer_truncation():
    indices = _build_subsample_indices(num_frames=3, num_subsampled_frames=4)

    assert indices[0].tolist() == [0, 0, 0, 0]
    assert indices[1].tolist() == [0, 0, 0, 1]
    assert indices[2].tolist() == [0, 0, 1, 2]


def test_inference_indices_respect_requested_fps_and_keep_last_frame():
    indices = _build_inference_indices(num_frames=65, dataset_fps=30, inference_fps=3)

    assert indices.tolist() == [0, 10, 20, 30, 40, 50, 60, 64]


def test_inference_indices_use_every_frame_at_native_rate():
    indices = _build_inference_indices(num_frames=5, dataset_fps=30, inference_fps=30)

    assert indices.tolist() == [0, 1, 2, 3, 4]


def test_full_trajectory_limit_is_uniform_and_keeps_endpoints():
    indices = np.arange(7, dtype=np.int64)

    limited = _limit_trajectory_indices(indices, max_frames=4)

    assert limited.tolist() == [0, 2, 4, 6]


def test_dense_interpolation_restores_original_frame_count():
    dense = _interpolate_dense(
        np.array([0, 2, 4]),
        np.array([0.0, 0.5, 1.0]),
        num_frames=5,
    )

    assert dense == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_sample_scalar_handles_dataset_tensor_and_array_fields():
    sample = {
        "mc_return": torch.tensor([-0.25]),
        "intervention": np.array([True]),
    }

    assert _sample_scalar(sample, "mc_return", float("nan")) == pytest.approx(-0.25)
    assert _sample_scalar(sample, "intervention", False) is True
    assert np.isnan(_sample_scalar(sample, "missing", float("nan")))
