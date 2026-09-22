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

import re
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from lerobot.data_processing.sarm_annotations.subtask_annotation import (
    Subtask,
    SubtaskAnnotation,
    Timestamp,
    VideoAnnotator,
    compute_temporal_proportions,
)


@pytest.fixture
def qwen3vl_processor():
    """Build a real processor locally, without downloading a tokenizer or model weights."""
    special_tokens = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    words = ["[UNK]", "[PAD]", "seconds>", *special_tokens]
    words.extend(f"<{i / 10:.1f}" for i in range(500))
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special_tokens,
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen3VLVideoProcessor(do_resize=False),
    )


@pytest.mark.parametrize(
    "frame_indices,source_fps,total_frames,duration,expected_slots,expected_first,expected_last",
    [
        ([*range(23), *range(24, 47)], 1.0, 47, 46.6, 46, "0.5", "45.5"),
        ([0, 3, 6, 9, 12, 15], 3.0, 18, 6.0, 6, "0.5", "4.5"),
        ([0, 2, 4, 6, 8], 1.0, 9, 9.0, 6, "1.0", "8.0"),
    ],
    ids=["46-second-clip", "source-frame-indices", "odd-frame-padding"],
)
def test_annotate_preserves_video_timing(
    monkeypatch,
    tmp_path,
    qwen3vl_processor,
    frame_indices,
    source_fps,
    total_frames,
    duration,
    expected_slots,
    expected_first,
    expected_last,
):
    vision_process = pytest.importorskip("qwen_vl_utils.vision_process")
    frames = torch.zeros(len(frame_indices), 3, 32, 32, dtype=torch.uint8)

    def fetch_video(_video, return_video_metadata=False, **_kwargs):
        metadata = {"fps": source_fps, "total_num_frames": total_frames, "frames_indices": frame_indices}
        return ((frames, metadata) if return_video_metadata else frames), 1.0

    # Keep the real process_vision_info and processor; replace only video I/O and generation.
    monkeypatch.setattr(vision_process, "fetch_video", fetch_video)
    annotation = make_annotation([("task", 0, int(duration))])
    processor = Mock(wraps=qwen3vl_processor)
    processor.apply_chat_template.return_value = "<|vision_start|><|video_pad|><|vision_end|>"
    processor.batch_decode.return_value = [annotation.model_dump_json()]
    model = Mock()
    model.generate.side_effect = lambda **kwargs: torch.cat(
        [kwargs["input_ids"], torch.zeros((1, 1), dtype=torch.long)], dim=1
    )
    annotator = VideoAnnotator(["task"], device="cpu", model=model, processor=processor)
    clip = tmp_path / "episode.mp4"
    clip.touch()
    extract = Mock(return_value=clip)
    monkeypatch.setattr(annotator, "extract_episode_segment", extract)
    source = tmp_path / "source.mp4"

    result = annotator.annotate(
        source, fps=30, start_timestamp=12.0, end_timestamp=12.0 + duration, max_retries=1
    )

    assert result == annotation
    extract.assert_called_once_with(source, 12.0, 12.0 + duration, 1)
    model.generate.assert_called_once()
    inputs = model.generate.call_args.kwargs
    grid = inputs["video_grid_thw"][0]
    assert int(grid[0]) * qwen3vl_processor.video_processor.temporal_patch_size == expected_slots
    prompt = qwen3vl_processor.batch_decode(inputs["input_ids"])[0]
    timestamps = re.findall(r"<(\d+\.\d) seconds>", prompt)
    assert len(timestamps) == int(grid[0])
    assert timestamps[0] == expected_first
    assert timestamps[-1] == expected_last
    assert not clip.exists()


def make_annotation(subtasks: list[tuple[str, int, int]]) -> SubtaskAnnotation:
    """Helper to create SubtaskAnnotation from list of (name, start_sec, end_sec)."""
    return SubtaskAnnotation(
        subtasks=[
            Subtask(
                name=name,
                timestamps=Timestamp(
                    start=f"{start // 60:02d}:{start % 60:02d}", end=f"{end // 60:02d}:{end % 60:02d}"
                ),
            )
            for name, start, end in subtasks
        ]
    )


class TestComputeTemporalProportions:
    """Tests for compute_temporal_proportions (SARM Paper Formula 1).

    Formula: ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)

    Key insight: This averages the PROPORTION of each subtask within each trajectory,
    giving equal weight to all trajectories regardless of absolute length.
    """

    def test_basic_two_trajectories_equal_proportions(self):
        """Test with two trajectories that have equal proportions."""
        # Both trajectories: subtask1 = 50%, subtask2 = 50%
        # Traj 1: T=100s, subtask1=50s, subtask2=50s
        # Traj 2: T=200s, subtask1=100s, subtask2=100s
        annotations = {
            0: make_annotation([("subtask1", 0, 50), ("subtask2", 50, 100)]),
            1: make_annotation([("subtask1", 0, 100), ("subtask2", 100, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        # Both should be 0.5
        assert abs(result["subtask1"] - 0.5) < 1e-6
        assert abs(result["subtask2"] - 0.5) < 1e-6

    def test_paper_example_different_from_avg_durations(self):
        """Test that compute_temporal_proportions differs from naive average duration approach.

        This is the key test showing the difference between:
        - Paper formula: average of (L_i,k / T_i)
        - Naive approach: mean(L_i,k) / sum(mean(L_i,j))
        """
        # Episode 1: T=100s, subtask1=80s, subtask2=20s (proportions: 0.8, 0.2)
        # Episode 2: T=200s, subtask1=40s, subtask2=160s (proportions: 0.2, 0.8)
        annotations = {
            0: make_annotation([("subtask1", 0, 80), ("subtask2", 80, 100)]),
            1: make_annotation([("subtask1", 0, 40), ("subtask2", 40, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        # Paper formula:
        # ᾱ_1 = (1/2) × (80/100 + 40/200) = (1/2) × (0.8 + 0.2) = 0.5
        # ᾱ_2 = (1/2) × (20/100 + 160/200) = (1/2) × (0.2 + 0.8) = 0.5
        assert abs(result["subtask1"] - 0.5) < 1e-6
        assert abs(result["subtask2"] - 0.5) < 1e-6

    def test_single_trajectory(self):
        """Test with a single trajectory."""
        # T=100s, reach=30s, grasp=20s, lift=50s
        annotations = {
            0: make_annotation([("reach", 0, 30), ("grasp", 30, 50), ("lift", 50, 100)]),
        }

        result = compute_temporal_proportions(annotations)

        assert abs(result["reach"] - 0.3) < 1e-6
        assert abs(result["grasp"] - 0.2) < 1e-6
        assert abs(result["lift"] - 0.5) < 1e-6

    def test_sum_to_one(self):
        """Test that proportions always sum to 1."""
        # Three episodes with varying proportions
        annotations = {
            0: make_annotation([("a", 0, 10), ("b", 10, 50), ("c", 50, 100)]),  # 0.1, 0.4, 0.5
            1: make_annotation([("a", 0, 20), ("b", 20, 70), ("c", 70, 100)]),  # 0.2, 0.5, 0.3
            2: make_annotation([("a", 0, 30), ("b", 30, 90), ("c", 90, 100)]),  # 0.3, 0.6, 0.1
        }

        result = compute_temporal_proportions(annotations)

        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_annotations_returns_empty(self):
        """Test that empty annotations returns empty dict."""
        result = compute_temporal_proportions({})
        assert result == {}

    def test_uniform_proportions(self):
        """Test with uniform proportions across subtasks."""
        # Each subtask takes 25% of each episode
        annotations = {
            0: make_annotation([("a", 0, 25), ("b", 25, 50), ("c", 50, 75), ("d", 75, 100)]),
            1: make_annotation([("a", 0, 50), ("b", 50, 100), ("c", 100, 150), ("d", 150, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        for name in ["a", "b", "c", "d"]:
            assert abs(result[name] - 0.25) < 1e-6
