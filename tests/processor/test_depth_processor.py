#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the depth-to-point-cloud processor step."""

import pytest
import torch

from lerobot.configs import PipelineFeatureType
from lerobot.processor.depth_processor import OBS_POINTCLOUD, DepthToPointCloudStep


@pytest.fixture
def scene():
    """A fronto-parallel plane 1.5 m away, seen by a 16x16 pinhole camera."""
    size, focal = 16, 8.0
    centre = (size - 1) / 2
    intrinsics = torch.tensor([[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]])
    depth = torch.full((size, size), 1.5)
    return size, intrinsics, depth


def test_writes_a_point_cloud_of_the_requested_size(scene):
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(num_points=64).observation(
        {"observation.images.top_depth": depth, "observation.intrinsics.top": intrinsics}
    )
    assert out[OBS_POINTCLOUD].shape == (64, 3)
    assert torch.allclose(out[OBS_POINTCLOUD][:, 2], torch.full((64,), 1.5))


def test_handles_a_batch(scene):
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(num_points=64).observation(
        {
            "observation.images.top_depth": depth.expand(5, size, size),
            "observation.intrinsics.top": intrinsics.expand(5, 3, 3),
        }
    )
    assert out[OBS_POINTCLOUD].shape == (5, 64, 3)


def test_accepts_channel_first_depth(scene):
    """Loaders often hand over (1, H, W) to match the image layout."""
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(num_points=32).observation(
        {"observation.images.top_depth": depth[None], "observation.intrinsics.top": intrinsics}
    )
    assert out[OBS_POINTCLOUD].shape == (32, 3)


def test_excludes_no_return_pixels(scene):
    """Depth sensors report 0 for 'no return'; those are points at the origin."""
    size, intrinsics, depth = scene
    holed = depth.clone()
    holed[:8] = 0.0
    out = DepthToPointCloudStep(num_points=64).observation(
        {"observation.images.top_depth": holed, "observation.intrinsics.top": intrinsics}
    )
    assert torch.all(out[OBS_POINTCLOUD][:, 2] > 1.0)


def test_depth_scale_converts_millimetres(scene):
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(num_points=16, depth_scale=1e-3).observation(
        {
            "observation.images.top_depth": torch.full((size, size), 1500.0),
            "observation.intrinsics.top": intrinsics,
        }
    )
    assert torch.allclose(out[OBS_POINTCLOUD][:, 2], torch.full((16,), 1.5))


def test_accepts_the_bare_suffix_form_too():
    """Some datasets store `{cam}_depth` without nesting under images."""
    size, focal = 16, 8.0
    centre = (size - 1) / 2
    intrinsics = torch.tensor([[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]])
    out = DepthToPointCloudStep(num_points=32).observation(
        {"top_depth": torch.full((size, size), 1.5), "observation.intrinsics.top": intrinsics}
    )
    assert out[OBS_POINTCLOUD].shape == (32, 3)


def test_refuses_to_fuse_two_cameras_in_camera_frame(scene):
    """Clouds from different cameras live in different frames.

    Concatenating them without extrinsics is meaningless, and silently doing it
    would produce a plausible-looking tensor that is geometric nonsense.
    """
    size, intrinsics, depth = scene
    with pytest.raises(ValueError, match="frame='camera'"):
        DepthToPointCloudStep(num_points=8).observation(
            {
                "observation.images.a_depth": depth,
                "observation.intrinsics.a": intrinsics,
                "observation.images.b_depth": depth,
                "observation.intrinsics.b": intrinsics,
            }
        )


def test_world_frame_requires_extrinsics(scene):
    size, intrinsics, depth = scene
    with pytest.raises(KeyError, match="extrinsics"):
        DepthToPointCloudStep(num_points=8, frame="world").observation(
            {"observation.images.top_depth": depth, "observation.intrinsics.top": intrinsics}
        )


def test_world_frame_fuses_multiple_cameras(scene):
    size, intrinsics, depth = scene
    first, second = torch.eye(4), torch.eye(4)
    second[:3, 3] = torch.tensor([0.5, 0.0, 0.0])
    out = DepthToPointCloudStep(num_points=128, frame="world").observation(
        {
            "observation.images.a_depth": depth,
            "observation.intrinsics.a": intrinsics,
            "observation.extrinsics.a": first,
            "observation.images.b_depth": depth,
            "observation.intrinsics.b": intrinsics,
            "observation.extrinsics.b": second,
        }
    )
    xs = out[OBS_POINTCLOUD][:, 0]
    assert out[OBS_POINTCLOUD].shape == (128, 3)
    assert (xs.max() - xs.min()).item() > 0.4  # the cameras are 0.5 m apart


def test_colour_adds_three_channels(scene):
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(num_points=32, with_colour=True).observation(
        {
            "observation.images.top_depth": depth,
            "observation.intrinsics.top": intrinsics,
            "observation.images.top": torch.rand(3, size, size),
        }
    )
    assert out[OBS_POINTCLOUD].shape == (32, 6)


def test_workspace_crop_normalises_into_unit_range(scene):
    size, intrinsics, depth = scene
    out = DepthToPointCloudStep(
        num_points=32, workspace_centre=(0.0, 0.0, 1.5), workspace_extent=0.4
    ).observation({"observation.images.top_depth": depth, "observation.intrinsics.top": intrinsics})
    assert out[OBS_POINTCLOUD].abs().max().item() <= 1.001


def test_passthrough_when_there_is_no_depth():
    out = DepthToPointCloudStep().observation({"observation.state": torch.zeros(4)})
    assert OBS_POINTCLOUD not in out


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="frame must be"):
        DepthToPointCloudStep(frame="nonsense")
    with pytest.raises(ValueError, match="must be set together"):
        DepthToPointCloudStep(workspace_centre=(0.0, 0.0, 0.0))


def test_transform_features_declares_the_new_key():
    features = {PipelineFeatureType.OBSERVATION: {}, PipelineFeatureType.ACTION: {}}
    out = DepthToPointCloudStep(num_points=512, with_colour=True).transform_features(features)
    assert out[PipelineFeatureType.OBSERVATION][OBS_POINTCLOUD].shape == (512, 6)


# --- calibration supplied by configuration -----------------------------------
#
# Every dataset recorded before this PR has depth but no intrinsics, because
# LeRobot discarded them at capture time. That is the common case, not an edge
# case, so it gets an error that names the way out and a way out that works.


def _depth_only_observation(h: int = 8, w: int = 8) -> dict:
    return {"observation.images.top_depth": torch.full((h, w), 0.5)}


def test_missing_intrinsics_names_the_config_option():
    step = DepthToPointCloudStep(num_points=16)
    with pytest.raises(KeyError) as excinfo:
        step.observation(_depth_only_observation())
    message = str(excinfo.value)
    assert "observation.intrinsics.top" in message
    assert "intrinsics={'top': (fx, fy, cx, cy)}" in message
    assert "get_intrinsics" in message


def test_intrinsics_can_be_supplied_as_config():
    step = DepthToPointCloudStep(num_points=16, intrinsics={"top": (10.0, 10.0, 4.0, 4.0)})
    out = step.observation(_depth_only_observation())
    assert out["observation.pointcloud"].shape == (16, 3)
    assert torch.isfinite(out["observation.pointcloud"]).all()


def test_config_intrinsics_accept_a_full_matrix():
    matrix = [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
    from_matrix = DepthToPointCloudStep(num_points=16, seed=0, intrinsics={"top": matrix})
    from_tuple = DepthToPointCloudStep(num_points=16, seed=0, intrinsics={"top": (10.0, 10.0, 4.0, 4.0)})
    a = from_matrix.observation(_depth_only_observation())["observation.pointcloud"]
    b = from_tuple.observation(_depth_only_observation())["observation.pointcloud"]
    torch.testing.assert_close(a, b)


def test_observation_intrinsics_win_over_config():
    """A dataset recorded after this PR needs no config, and must not be overridden."""
    observation = _depth_only_observation()
    observation["observation.intrinsics.top"] = torch.tensor(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
    )
    from_observation = DepthToPointCloudStep(num_points=16, seed=0, intrinsics={"top": (1.0, 1.0, 0.0, 0.0)})
    matching = DepthToPointCloudStep(num_points=16, seed=0, intrinsics={"top": (10.0, 10.0, 4.0, 4.0)})
    a = from_observation.observation(dict(observation))["observation.pointcloud"]
    b = matching.observation(_depth_only_observation())["observation.pointcloud"]
    torch.testing.assert_close(a, b)


def test_bad_config_intrinsics_are_rejected():
    step = DepthToPointCloudStep(num_points=16, intrinsics={"top": (1.0, 2.0)})
    with pytest.raises(ValueError, match="fx, fy, cx, cy"):
        step.observation(_depth_only_observation())


def test_missing_extrinsics_names_the_config_option():
    step = DepthToPointCloudStep(num_points=16, frame="world", intrinsics={"top": (10.0, 10.0, 4.0, 4.0)})
    with pytest.raises(KeyError) as excinfo:
        step.observation(_depth_only_observation())
    message = str(excinfo.value)
    assert "observation.extrinsics.top" in message
    assert "frame='camera'" in message


def test_extrinsics_can_be_supplied_as_config():
    step = DepthToPointCloudStep(
        num_points=16,
        frame="world",
        intrinsics={"top": (10.0, 10.0, 4.0, 4.0)},
        extrinsics={"top": torch.eye(4).tolist()},
    )
    out = step.observation(_depth_only_observation())
    assert out["observation.pointcloud"].shape == (16, 3)


def test_config_round_trips_through_json():
    """get_config feeds a saved pipeline, so tensors and tuples must be plain lists."""
    import json

    step = DepthToPointCloudStep(
        num_points=16,
        frame="world",
        intrinsics={"top": (10.0, 10.0, 4.0, 4.0)},
        extrinsics={"top": torch.eye(4)},
    )
    config = json.loads(json.dumps(step.get_config()))
    assert config["intrinsics"]["top"] == [10.0, 10.0, 4.0, 4.0]
    assert config["extrinsics"]["top"][0] == [1.0, 0.0, 0.0, 0.0]

    rebuilt = DepthToPointCloudStep(**config)
    assert rebuilt.observation(_depth_only_observation())["observation.pointcloud"].shape == (16, 3)


def test_seed_makes_sampling_reproducible():
    """`seed` was accepted and serialised while doing nothing: the generator it
    named was never built, so every run subsampled differently and a "seeded"
    pipeline was not reproducible at all."""
    observation = _depth_only_observation(16, 16)
    intrinsics = {"top": (10.0, 10.0, 8.0, 8.0)}

    a = DepthToPointCloudStep(num_points=32, seed=7, intrinsics=intrinsics)
    b = DepthToPointCloudStep(num_points=32, seed=7, intrinsics=intrinsics)
    c = DepthToPointCloudStep(num_points=32, seed=8, intrinsics=intrinsics)

    first_a = a.observation(dict(observation))["observation.pointcloud"]
    first_b = b.observation(dict(observation))["observation.pointcloud"]
    first_c = c.observation(dict(observation))["observation.pointcloud"]
    torch.testing.assert_close(first_a, first_b)
    assert not torch.allclose(first_a, first_c), "different seeds gave identical samples"

    # The generator advances, so consecutive frames differ but stay in step.
    torch.testing.assert_close(
        a.observation(dict(observation))["observation.pointcloud"],
        b.observation(dict(observation))["observation.pointcloud"],
    )


def test_millimetre_depth_read_as_metres_raises():
    """The default wiring of LeRobotDataset to this step is off by 1000x.

    The dataset dequantises to millimetres by default and `depth_scale` defaults
    to metres, so the crop rejects every point and the sampler returns zeros --
    a policy trains on empty clouds and nothing reports a fault.
    """
    observation = {"observation.images.top_depth": torch.full((8, 8), 500.0)}  # mm
    step = DepthToPointCloudStep(
        num_points=16,
        intrinsics={"top": (10.0, 10.0, 4.0, 4.0)},
        workspace_centre=(0.0, 0.0, 0.5),
        workspace_extent=0.6,
    )
    with pytest.raises(ValueError, match="millimetres"):
        step.observation(dict(observation))


def test_millimetre_depth_works_once_scaled():
    observation = {"observation.images.top_depth": torch.full((8, 8), 500.0)}
    step = DepthToPointCloudStep(
        num_points=16,
        depth_scale=1e-3,
        intrinsics={"top": (10.0, 10.0, 4.0, 4.0)},
        workspace_centre=(0.0, 0.0, 0.5),
        workspace_extent=0.6,
    )
    cloud = step.observation(dict(observation))["observation.pointcloud"]
    assert cloud.shape == (16, 3)
    assert cloud.abs().max() <= 1.0 + 1e-6, "cropped points should normalise into [-1, 1]"
    assert cloud.abs().sum() > 0, "cloud collapsed to zeros"


def test_plausible_metric_depth_does_not_trip_the_guard():
    """A real scene up to a few metres must pass, whatever max_depth is set to."""
    observation = {"observation.images.top_depth": torch.full((8, 8), 2.5)}
    step = DepthToPointCloudStep(num_points=16, max_depth=1.0, intrinsics={"top": (10.0, 10.0, 4.0, 4.0)})
    cloud = step.observation(dict(observation))["observation.pointcloud"]
    assert cloud.shape == (16, 3)
