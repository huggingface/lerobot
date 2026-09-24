# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
import math

import draccus
import numpy as np
import pytest
import torch

from examples.rebot_agent.prepare_matched_training import paired_manifests, prepare_pair
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.steering_commands import SteeringCommands
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig  # noqa: F401


def reviewed_manifest():
    segments = []
    for ep in (2, 5, 9, 90):
        segments.append(
            {
                "episode_index": ep,
                "start_frame": 10,
                "end_frame": 16,
                "review": {"verdict": "accepted", "reviewer": "test"},
                "commands": [
                    {"style": "subtask", "text": "pick up the tape", "evidence": "reviewed clip"},
                    {
                        "style": "point",
                        "text": "reach to",
                        "evidence": "reviewed track",
                        "camera": "observation.images.base",
                        "image_size": [640, 480],
                        "points_by_frame": {str(f): [[f, 200]] for f in range(10, 16)},
                    },
                    {"style": "motion", "text": "close the left gripper", "evidence": "measured state"},
                ],
            }
        )
    uncovered = copy.deepcopy(segments[0])
    uncovered.update(start_frame=20, end_frame=23)
    uncovered["commands"] = uncovered["commands"][2:]
    segments.append(uncovered)
    return {
        "version": 1,
        "source": {
            "repo_id": "pepijn223/rebot_diverse_picking_100_annotated",
            "revision": "93c97807c46535745d0587d4296416bf2d4aa80d",
        },
        "segments": segments,
    }


@pytest.mark.parametrize("deterministic", [False, True])
def test_paired_routes_and_action_masks_remain_equal_across_repeated_visits(deterministic):
    original = reviewed_manifest()
    saved = copy.deepcopy(original)
    control, steering, excluded = paired_manifests(original)
    assert original == saved
    a, b = SteeringCommands(control), SteeringCommands(steering)
    assert set(a.episodes) == set(b.episodes) == {2, 5, 9}
    assert len(excluded) == 1 and excluded[0]["start_frame"] == 20
    assert len(control["segments"][0]["commands"]) == 3
    assert steering["segments"][0] == original["segments"][0]

    def collect(index):
        np.random.seed(31)
        rows = []
        for visit in range(300):
            frame = 10 + visit % 6
            sample = {
                "index": 1000 + frame,
                "episode_index": 2,
                "frame_index": frame,
                "task": "overall task",
                "action": torch.arange(8).reshape(4, 2),
                "action_is_pad": torch.tensor([False, False, False, True]),
            }
            result = index.sample(sample, 0.2, deterministic=deterministic, action_offsets=[0, 1, 2, 3])
            assert result["action"] is sample["action"]
            rows.append((result["task"] == "overall task", result["action_is_pad"].tolist()))
        return rows, np.random.random()

    left, right = collect(a), collect(b)
    assert left == right  # Includes the next RNG draw after all command substitutions.
    if not deterministic:
        assert 40 < sum(task for task, _ in left[0]) < 85
    assert any(mask != [False, False, False, True] for _, mask in left[0])


def test_pair_configs_change_only_language_manifest_and_required_styles(tmp_path):
    control, steering, plan = prepare_pair(reviewed_manifest(), tmp_path, [5], batch_size=2)
    assert plan["training_episodes"] == [2, 9]
    assert plan["development_episodes"] == [5]
    assert plan["final_heldout_episodes"] == list(range(90, 100))
    assert plan["training_profile"]["annotated_frames"] == 12
    assert plan["comparison_checkpoint_steps"] == [2, 4, 8, 16]
    configs = copy.deepcopy(plan["configs"])
    for config in configs.values():
        parsed = draccus.decode(TrainPipelineConfig, config)
        assert parsed.policy.steering_coordinate_format == "native_points_v1"
        assert parsed.policy.scheduler_warmup_steps == 1
        assert parsed.policy.scheduler_decay_steps == plan["steps"]
        episodes = config["dataset"]["episodes"]
        tail = math.ceil(len(episodes) * config["dataset"]["eval_split"])
        assert episodes[:-tail] == [2, 9] and episodes[-tail:] == [5]
        assert config["num_workers"] == 0
        assert config["dataset"]["steering_style_weights"] is None
        config.pop("output_dir")
        config["dataset"].pop("steering_manifest")
        config["dataset"].pop("steering_required_styles")
    assert configs["semantic_control"] == configs["steerable"]
    assert SteeringCommands(control).annotation_profile()["missing_styles"] == [
        "combination",
        "motion",
        "point",
        "trace",
    ]
    assert "trace" in SteeringCommands(steering).annotation_profile()["missing_styles"]


@pytest.mark.parametrize("development", [[], [90], [5, 5], [2, 5, 9], [10]])
def test_invalid_development_split_is_rejected(tmp_path, development):
    with pytest.raises(ValueError):
        prepare_pair(reviewed_manifest(), tmp_path, development)


def test_different_source_is_rejected(tmp_path):
    data = reviewed_manifest()
    data["source"]["revision"] = "different-recording"
    with pytest.raises(ValueError, match="Manifest source"):
        prepare_pair(data, tmp_path, [5])
