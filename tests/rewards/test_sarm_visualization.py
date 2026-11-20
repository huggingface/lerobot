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

import numpy as np
from matplotlib.axes import Axes

from lerobot.rewards.sarm import compute_rabc_weights


def test_visualize_episode_aligns_thumbnails_with_episode_predictions(tmp_path, monkeypatch):
    display_frame_indices = np.array([0, 2, 5, 7, 9])
    frames = np.stack(
        [np.full((2, 3, 3), fill_value=thumbnail_idx, dtype=np.uint8) for thumbnail_idx in range(5)]
    )
    progress_preds = np.arange(10, dtype=np.float32) / 10
    stage_preds = np.eye(3, dtype=np.float32)[np.arange(10) % 3]
    stage_labels = ["stage zero", "stage one", "stage two"]

    annotations = []
    displayed_images = []

    def capture_text(_axes, _x, _y, text, **_kwargs):
        annotations.append(text)

    def capture_image(_axes, image, *_args, **_kwargs):
        displayed_images.append(image)

    monkeypatch.setattr(Axes, "text", capture_text)
    monkeypatch.setattr(Axes, "imshow", capture_image)
    monkeypatch.setattr(compute_rabc_weights.plt, "savefig", lambda *_args, **_kwargs: None)

    compute_rabc_weights.visualize_episode(
        frames=frames,
        progress_preds=progress_preds,
        stage_preds=stage_preds,
        title="test task",
        output_path=tmp_path / "visualization.png",
        stage_labels=stage_labels,
        display_frame_indices=display_frame_indices,
    )

    frame_annotations = [annotation for annotation in annotations if annotation.startswith("Frame ")]
    assert frame_annotations == [
        "Frame 0\n0.00\nstage zero",
        "Frame 2\n0.20\nstage two",
        "Frame 5\n0.50\nstage two",
        "Frame 7\n0.70\nstage one",
        "Frame 9\n0.90\nstage zero",
    ]
    assert len(displayed_images) == 1
    np.testing.assert_array_equal(displayed_images[0], np.concatenate(frames, axis=1))
