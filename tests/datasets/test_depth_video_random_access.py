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

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import av  # noqa: E402

from lerobot.configs import DepthEncoderConfig
from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames_pyav

pytestmark = pytest.mark.dataset


@pytest.mark.skipif(get_codec("hevc") is None, reason="HEVC encoder unavailable")
@pytest.mark.parametrize("streaming_encoding", [False, True])
@pytest.mark.parametrize("custom_quantizer", [False, True])
@pytest.mark.parametrize("reader", ["random", "streaming"])
def test_recorded_depth_random_access(
    tmp_path: Path, streaming_encoding: bool, custom_quantizer: bool, reader: str
) -> None:
    key = "observation.images.depth"
    encoder = (
        DepthEncoderConfig(crf=0, depth_min=0.05, depth_max=8, use_log=False)
        if custom_quantizer
        else DepthEncoderConfig()
    )
    dataset = LeRobotDataset.create(
        "local/depth-seek",
        root=tmp_path / "dataset",
        fps=10,
        features={
            key: {
                "dtype": "video",
                "shape": (32, 48, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
        },
        depth_encoder=encoder,
        encoder_threads=1,
        streaming_encoding=streaming_encoding,
    )
    quantizer = {
        "depth_min": encoder.depth_min,
        "depth_max": encoder.depth_max,
        "shift": encoder.shift,
        "use_log": encoder.use_log,
    }
    expected = []
    for episode, length in enumerate((3, 5, 4)):
        for frame in range(length):
            depth = np.linspace(0.5, 1.5, 32 * 48, dtype=np.float32).reshape(32, 48, 1)
            depth = depth + episode + frame / 10
            expected.append(quantize_depth(depth, **quantizer, video_backend=None))
            dataset.add_frame({key: depth, "task": "move"})
        dataset.save_episode()
    dataset.finalize()
    dataset = LeRobotDataset(dataset.repo_id, root=dataset.root, video_backend="pyav")
    video_path = dataset.root / dataset.meta.get_video_file_path(0, key)
    with av.open(str(video_path)) as container:
        sequential = [frame.to_ndarray(format="gray12le") for frame in container.decode(video=0)]
    assert len(sequential) == 12
    np.testing.assert_array_equal(sequential, expected)

    if reader == "random":
        # Starting at a CRA in an open GOP can silently corrupt a later frame even
        # when its timestamp matches. Test the pixel oracle before single-frame seeks.
        for indices in ([2, 4], *([i] for i in range(12))):
            actual = decode_video_frames_pyav(
                video_path, [i / 10 for i in indices], tolerance_s=1e-4, is_depth=True
            )
            np.testing.assert_array_equal(actual[:, 0].numpy(), np.stack([sequential[i] for i in indices]))
    else:
        streaming = StreamingLeRobotDataset(
            dataset.repo_id,
            root=dataset.root,
            video_backend="pyav",
            shuffle=False,
            decode_threads=1,
            decoded_queue_size=1,
        )
        rows = list(streaming)
        assert sorted(int(row["index"]) for row in rows) == list(range(12))
        for row in rows:
            index = int(row["index"])
            depth = dequantize_depth(torch.from_numpy(sequential[index].copy()), **quantizer)
            np.testing.assert_array_equal(row[key].numpy(), depth.numpy())
