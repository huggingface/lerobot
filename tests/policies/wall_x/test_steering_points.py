# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Coordinate conventions at the saved WALL-X processor boundary."""

import draccus
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.wall_x.constant import WALL_X_PROMPT_SEGMENTS
from lerobot.policies.wall_x.processor_wall_x import (
    WallXPromptProcessorStep,
    WallXTokenizerStep,
    make_wall_x_pre_post_processors,
)
from lerobot.policies.wall_x.utils import process_steering_points
from lerobot.processor import (
    DataProcessorPipeline,
    RenderRuntimeMessagesStep,
    RenderTrainingMessagesStep,
    batch_to_transition,
)
from lerobot.utils.steering import render_steering_command

DIMENSIONS = {
    "observation.images.base": (480, 640, 196, 252),
    "observation.images.left_wrist": (720, 1280, 140, 252),
}


def command(style="point", camera="base", size=(640, 480), points=None):
    return render_steering_command(
        {
            "style": style,
            "text": "move the right gripper along" if style == "trace" else "pick here and place there",
            "camera": "observation.images." + camera,
            "image_size": size,
            "points": points or [[290, 178], [326, 155]],
        }
    )


@pytest.mark.parametrize("style", ["point", "trace", "combination"])
def test_native_points_use_named_camera_and_preserve_order(style):
    text = command(style)
    segments = [[{"text": text, "target": True}, {"text": "suffix", "target": False}]]
    legacy, _ = WallXTokenizerStep._texts_and_target_spans(
        segments, DIMENSIONS["observation.images.left_wrist"]
    )
    assert legacy == [text + "suffix"]
    native, spans = WallXTokenizerStep._texts_and_target_spans(
        segments,
        DIMENSIONS["observation.images.left_wrist"],
        steering_coordinate_format="native_points_v1",
        dimensions_by_camera=DIMENSIONS,
    )
    expected = (
        "In base view (252x196 pixels), "
        + ("move the right gripper along" if style == "trace" else "pick here and place there")
        + ": <point>[114, 73]</point>, <point>[128, 63]</point>."
    )
    assert native == [expected + "suffix"]
    assert native[0][slice(*spans[0][0])] == expected


def test_each_camera_and_edge_point_uses_its_own_dimensions():
    text = command(camera="left_wrist", size=(1280, 720), points=[[0, 0], [1279, 719]])
    assert process_steering_points(text, DIMENSIONS) == (
        "In left_wrist view (252x140 pixels), pick here and place there: "
        "<point>[0, 0]</point>, <point>[251, 139]</point>."
    )
    assert process_steering_points("open the left gripper", DIMENSIONS) == "open the left gripper"


@pytest.mark.parametrize(
    "text,error",
    [
        (command(camera="unknown"), "absent"),
        (command(size=(800, 600)), "source dimensions"),
        (command().replace("[290, 178]", "[640, 178]"), "outside"),
        (command().replace("[290, 178]", "[-1, 178]"), "Malformed"),
        (command().replace("[290, 178]", "[2.5, 178]"), "Malformed"),
        (command().replace("[290, 178], [326, 155]", "<point>[290, 178]</point>"), "Malformed"),
    ],
)
def test_native_conversion_rejects_ambiguous_or_invalid_geometry(text, error):
    with pytest.raises(ValueError, match=error):
        process_steering_points(text, DIMENSIONS)


@pytest.mark.parametrize("mode", ["original_pixels", "native_points_v1"])
def test_saved_policy_and_tokenizer_retain_the_coordinate_contract(tmp_path, mode):
    config = WallXConfig(device="cpu", steering_coordinate_format=mode)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    config._save_pretrained(tmp_path)
    restored = WallXConfig.from_pretrained(tmp_path)
    assert restored.steering_coordinate_format == mode
    preprocessor, _ = make_wall_x_pre_post_processors(restored)
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, WallXTokenizerStep))
    assert WallXTokenizerStep(**tokenizer.get_config()).steering_coordinate_format == mode
    preprocessor.save_pretrained(tmp_path / "processor", config_filename="processor.json")
    reloaded = DataProcessorPipeline.from_pretrained(tmp_path / "processor", config_filename="processor.json")
    saved_tokenizer = next(step for step in reloaded.steps if isinstance(step, WallXTokenizerStep))
    assert saved_tokenizer.steering_coordinate_format == mode
    legacy_config = draccus.encode(config)
    del legacy_config["steering_coordinate_format"]
    assert draccus.decode(WallXConfig, legacy_config).steering_coordinate_format == "original_pixels"


def test_training_and_runtime_produce_identical_native_steering_prompts():
    recipe = {"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]}
    runtime = RenderRuntimeMessagesStep(recipe)
    training = RenderTrainingMessagesStep(recipe)
    prompt = WallXPromptProcessorStep(image_keys=list(DIMENSIONS), chunk_size=32)
    results = []
    for with_actions in [True, False]:
        batch = {"task": [command()], "observation.state": torch.zeros(1, 14)}
        if with_actions:
            batch["action"] = torch.zeros(1, 32, 14)
        transition = prompt(training(runtime(batch_to_transition(batch))))
        segments = transition[TransitionKey.COMPLEMENTARY_DATA][WALL_X_PROMPT_SEGMENTS]
        texts, _ = WallXTokenizerStep._texts_and_target_spans(
            segments,
            DIMENSIONS["observation.images.left_wrist"],
            steering_coordinate_format="native_points_v1",
            dimensions_by_camera=DIMENSIONS,
        )
        results.append(texts)
    assert results[0] == results[1]
    assert "<point>[114, 73]</point>" in results[0][0]
    assert results[0][0].count("<|action|>") == 32


def test_unknown_format_is_not_silently_accepted():
    with pytest.raises(ValueError, match="steering_coordinate_format"):
        WallXConfig(steering_coordinate_format="typo")
