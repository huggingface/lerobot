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
import sys
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_dataset_viz as viz
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


@pytest.fixture
def logged_images(monkeypatch):
    rr = pytest.importorskip("rerun", reason="rerun-sdk is required (install lerobot[viz])")
    images = {}
    monkeypatch.setattr(rr, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(rr, "set_time", lambda *args, **kwargs: None)
    monkeypatch.setattr(rr, "log", lambda key, entity: images.__setitem__(key, entity))
    return images


def make_image_dataset(height: int, width: int):
    """Build one local RGB and depth frame without video files or Hub access."""

    class ImageDataset(list):
        pass

    rgb = torch.from_numpy(np.random.default_rng(0).random((3, height, width), dtype=np.float32))
    dataset = ImageDataset(
        [
            {
                "index": torch.tensor(0),
                "timestamp": torch.tensor(0.0),
                "rgb": rgb,
                "depth": torch.full((1, height, width), 2.0),
            }
        ]
    )
    dataset.repo_id = "test/images"
    dataset.depth_output_unit = "m"
    dataset.meta = SimpleNamespace(camera_keys=["rgb", "depth"], depth_keys=["depth"], stats={})
    dataset.features = {}
    return dataset


@pytest.mark.parametrize(
    "shape,image_max_size,expected_size",
    [
        ((60, 100), None, (100, 60)),
        ((60, 100), 40, (40, 24)),
        ((100, 60), 40, (24, 40)),
        ((60, 100), 200, (100, 60)),
        ((60, 100), 1, (1, 1)),
        ((4, 100), 1, (1, 1)),
    ],
)
def test_rgb_size_controls_preserve_depth_and_source(logged_images, shape, image_max_size, expected_size):
    dataset = make_image_dataset(*shape)
    source_rgb = dataset[0]["rgb"].clone()
    visualize_dataset(dataset, episode_index=0, image_max_size=image_max_size)

    rgb_format = logged_images["rgb"].format.as_arrow_array()[0].as_py()
    assert (rgb_format["width"], rgb_format["height"]) == expected_size
    if image_max_size is None:
        rgb_bytes = bytes(logged_images["rgb"].buffer.as_arrow_array()[0].as_py())
        np.testing.assert_array_equal(
            np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(*shape, 3),
            viz.to_hwc_uint8_numpy(source_rgb),
        )
    depth = logged_images["depth"]
    depth_format = depth.format.as_arrow_array()[0].as_py()
    assert (depth_format["height"], depth_format["width"]) == shape
    np.testing.assert_array_equal(
        np.frombuffer(bytes(depth.buffer.as_arrow_array()[0].as_py()), dtype=np.float32),
        np.full(shape[0] * shape[1], 2.0, dtype=np.float32),
    )
    torch.testing.assert_close(dataset[0]["rgb"], source_rgb)


def test_jpeg_quality_changes_encoded_images(logged_images):
    dataset = make_image_dataset(60, 100)
    visualize_dataset(dataset, episode_index=0, display_compressed_images=True)
    default_jpeg = bytes(logged_images["rgb"].blob.as_arrow_array()[0].as_py())

    rr = pytest.importorskip("rerun")
    previous_default = rr.Image(viz.to_hwc_uint8_numpy(dataset[0]["rgb"])).compress()
    assert default_jpeg == bytes(previous_default.blob.as_arrow_array()[0].as_py())

    visualize_dataset(
        dataset, episode_index=0, image_max_size=40, display_compressed_images=True, jpeg_quality=25
    )
    smaller_jpeg = bytes(logged_images["rgb"].blob.as_arrow_array()[0].as_py())
    with Image.open(BytesIO(smaller_jpeg)) as resized, Image.open(BytesIO(default_jpeg)) as original:
        assert resized.size == (40, 24)
        assert resized.quantization != original.quantization
    assert len(smaller_jpeg) < len(default_jpeg)


@pytest.mark.parametrize("jpeg_quality", [0, 100])
def test_jpeg_quality_boundaries(logged_images, jpeg_quality):
    dataset = make_image_dataset(60, 100)
    visualize_dataset(dataset, episode_index=0, display_compressed_images=True, jpeg_quality=jpeg_quality)
    with Image.open(BytesIO(bytes(logged_images["rgb"].blob.as_arrow_array()[0].as_py()))) as image:
        assert image.size == (100, 60)
        image.load()


@pytest.mark.parametrize(
    "options,message",
    [
        ({"image_max_size": 0}, "image_max_size must be greater than 0"),
        ({"image_max_size": -1}, "image_max_size must be greater than 0"),
        ({"jpeg_quality": -1}, "jpeg_quality must be between 0 and 100"),
        ({"jpeg_quality": 101}, "jpeg_quality must be between 0 and 100"),
    ],
)
def test_invalid_image_options_fail_before_loading(monkeypatch, capsys, options, message):
    with pytest.raises(ValueError, match=message):
        visualize_dataset(None, episode_index=0, **options)

    def unexpected_dataset_load(*args, **kwargs):
        pytest.fail("Invalid image options must be rejected before loading a dataset")

    monkeypatch.setattr(viz, "LeRobotDataset", unexpected_dataset_load)
    argv = ["lerobot-dataset-viz", "--repo-id", "test/images", "--episode-index", "0"]
    for name, value in options.items():
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        viz.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_forwards_image_controls(monkeypatch):
    dataset = make_image_dataset(60, 100)
    received = {}
    monkeypatch.setattr(viz, "LeRobotDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(viz, "visualize_dataset", lambda dataset, **kwargs: received.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-dataset-viz",
            "--repo-id",
            "test/images",
            "--episode-index",
            "0",
            "--image-max-size",
            "40",
            "--display-compressed-images",
            "--jpeg-quality",
            "25",
        ],
    )
    viz.main()
    assert received["image_max_size"] == 40
    assert received["jpeg_quality"] == 25
    assert received["display_compressed_images"] is True


def test_foxglove_warns_about_rerun_image_controls(monkeypatch, caplog):
    monkeypatch.setattr(viz, "LeRobotDataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(viz, "visualize_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-dataset-viz",
            "--repo-id",
            "test/images",
            "--episode-index",
            "0",
            "--display-mode",
            "foxglove",
            "--image-max-size",
            "40",
            "--jpeg-quality",
            "25",
        ],
    )
    viz.main()
    assert "ignored" in caplog.text
    assert "--image-max-size" in caplog.text
    assert "--jpeg-quality" in caplog.text


@pytest.mark.parametrize("image_max_size,jpeg_quality", [(None, 95), (40, 25)])
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory, image_max_size, jpeg_quality):
    rr = pytest.importorskip("rerun", reason="rerun-sdk is required (install lerobot[viz])")
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(
        root=root,
        total_episodes=1,
        total_frames=2,
        use_videos=False,
        camera_features={
            "observation.images.camera": {"shape": (60, 100, 3), "names": ["height", "width", "channels"]}
        },
    )
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
        display_compressed_images=True,
        image_max_size=image_max_size,
        jpeg_quality=jpeg_quality,
    )
    rr.get_global_data_recording().flush()
    assert rrd_path.stat().st_size > 0
