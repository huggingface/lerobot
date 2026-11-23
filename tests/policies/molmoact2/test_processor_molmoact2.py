# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""MolmoAct2 pre- and post-processing with local tiny processor files (see `conftest.py`)."""

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[molmoact2])")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.molmoact2 import processor_molmoact2  # noqa: E402
from lerobot.policies.molmoact2.modeling_molmoact2 import _get_molmo2_config  # noqa: E402
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    _ACTION_END_TOKEN_ID,
    _ACTION_START_TOKEN_ID,
    _ACTION_TOKEN_START_ID,
    _NUM_ACTION_TOKENS,
    _PROCESSOR_CONFIG,
    EOS_TOKEN_ID,
    MolmoAct2ClipActionsProcessorStep,
    MolmoAct2PackInputsProcessorStep,
    UniversalActionProcessor,
    _build_discrete_state_string,
    _build_robot_text,
    _load_molmoact2_processor,
    _normalization_stats,
    _normalize_question_text,
    decode_discrete_action_chunk,
    make_molmoact2_pre_post_processors,
    make_molmoact2_pre_post_processors_from_pretrained,
    select_constrained_discrete_action_token,
)
from lerobot.processor import (  # noqa: E402
    JointAffineProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

QUANTILES = {FeatureType.STATE: NormalizationMode.QUANTILES, FeatureType.ACTION: NormalizationMode.QUANTILES}


@pytest.fixture
def dataset_meta(sizes):
    """Dimension names of the tiny features; the last state and action dims are grippers."""
    state_names = [f"joint_{index}" for index in range(sizes.state_dim - 1)] + ["gripper"]
    action_names = [f"axis_{index}" for index in range(sizes.action_dim - 1)] + ["gripper"]
    return SimpleNamespace(names={OBS_STATE: state_names, ACTION: action_names})


def _dataset_stats(sizes) -> dict[str, dict[str, torch.Tensor]]:
    """Joints in [-2, 2], a gripper in [0, 1], and a constant last joint (0.5) in the state."""
    stats = {}
    for key, dim in ((OBS_STATE, sizes.state_dim), (ACTION, sizes.action_dim)):
        low, high = torch.full((dim,), -2.0), torch.full((dim,), 2.0)
        low[-1], high[-1] = 0.0, 1.0
        stats[key] = {"min": low, "max": high, "q01": low.clone(), "q99": high.clone()}
    for name in ("min", "max", "q01", "q99"):
        stats[OBS_STATE][name][-2] = 0.5
    return stats


def _raw_batch(sizes, with_action: bool = True) -> dict:
    generator = torch.Generator().manual_seed(0)
    batch = {camera: torch.rand(2, 3, 32, 32, generator=generator) for camera in sizes.cameras}
    batch[OBS_STATE] = torch.rand(2, sizes.state_dim, generator=generator) * 4 - 2
    batch["task"] = ["Pick up the cube.", "open the drawer"]
    if with_action:
        batch[ACTION] = torch.rand(2, sizes.chunk_size, sizes.action_dim, generator=generator) * 4 - 2
    return batch


def _transition(
    sizes, state: torch.Tensor, action: torch.Tensor | None = None, task="Pick up the cube."
) -> dict:
    generator = torch.Generator().manual_seed(0)
    observation = {camera: torch.rand(len(state), 3, 32, 32, generator=generator) for camera in sizes.cameras}
    observation[sizes.cameras[1]] = (observation[sizes.cameras[1]] * 255).to(
        torch.uint8
    )  # float and uint8 cameras
    return {
        TransitionKey.OBSERVATION: {**observation, OBS_STATE: state},
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {"task": task},
    }


def test_build_robot_text_discretizes_the_state_into_256_bins():
    state = np.array([-1.0, 0.0, 1.0, np.nan, np.inf, -np.inf])

    prompt = _build_robot_text("pick", _build_discrete_state_string(state), "franka arm", "joint position", 2)

    bins = "".join(f"<state_{index}>" for index in (0, 128, 255, 128, 255, 0))
    assert prompt.startswith("Image 1<|image|>Image 2<|image|><|im_start|>user\nThe task is to pick. ")
    assert f"<state_start>{bins}<state_end>" in prompt
    assert (
        "<setup_start>franka arm<setup_end>" in prompt
        and "<control_start>joint position<control_end>" in prompt
    )
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n<action_output>")
    assert _build_robot_text("pick", "", "", "", 1).startswith("<|image|><|im_start|>")


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Instruction: Pick up the cube, please!", "pick up the cube, please"),
        ("The task is to open drawer. Then close it.", "open drawer; then close it"),
        ('  "Put the  bowl on the plate."  ', "put the bowl on the plate"),
    ],
)
def test_normalize_question_text_matches_the_training_prompt_style(task, expected):
    assert _normalize_question_text(task) == expected


def test_pack_step_builds_left_padded_model_inputs(sizes, tokenizer_dir):
    step = MolmoAct2PackInputsProcessorStep(
        tokenizer_name=str(tokenizer_dir),
        processor_config=sizes.processor_config,
        action_mode="continuous",
        image_keys=list(sizes.cameras),
        action_dim=sizes.action_dim,
    )
    tokenizer = step._processor.tokenizer
    assert tokenizer.padding_side == "left" and step._processor.image_processor.crop_mode == "resize"
    state = torch.tensor([[-1.0] * sizes.state_dim, [1.0] * sizes.state_dim])

    complementary = step(_transition(sizes, state, task=["Pick up the cube.", "go"]))[
        TransitionKey.COMPLEMENTARY_DATA
    ]

    expected_keys = {"input_ids", "attention_mask", "position_ids", "mm_token_type_ids", "pixel_values"}
    expected_keys |= {"encoder_attention_mask"}
    expected_keys |= {"image_token_pooling", "image_grids", "action_dim_is_pad", "task"}
    assert complementary.keys() == expected_keys
    assert complementary["task"] == ["pick up the cube", "go"]
    input_ids, attention_mask = complementary["input_ids"], complementary["attention_mask"]
    assert input_ids.dtype == torch.long and input_ids.shape[1] % 8 == 0
    # Left padding: every prompt ends in the last column; positions count valid tokens only.
    assert attention_mask[:, -1].all() and not attention_mask[1, 0]
    for row in range(2):
        valid = attention_mask[row].bool()
        assert torch.equal(complementary["position_ids"][row, valid], torch.arange(int(valid.sum())))
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, attention_mask[0].bool()].tolist())
    assert tokens[0] == "<|im_end|>" and tokens[-1] == "<action_output>"  # BOS, then the prompt
    image = ["<im_start>", "<im_patch>", "<im_end>"]
    assert tokens.count("<im_patch>") == 2 and "".join(tokens).count("".join(image)) == 2
    assert tokens.count("<state_0>") == sizes.state_dim
    image_ids = tokenizer.convert_tokens_to_ids(image)
    assert torch.equal(
        complementary["mm_token_type_ids"].bool(), torch.isin(input_ids, torch.tensor(image_ids))
    )
    # One crop per camera and sample, in the order of the image tokens.
    assert complementary["pixel_values"].shape == (4, 4, 14 * 14 * 3)
    assert complementary["image_grids"].tolist() == [[1, 1, 0, 0]] * 4
    assert complementary["image_token_pooling"].dtype == torch.long
    assert complementary["action_dim_is_pad"].tolist() == [[False] * sizes.action_dim + [True] * 25] * 2
    # Without "both" training, the action expert attends to the whole prompt.
    assert torch.equal(complementary["encoder_attention_mask"], attention_mask.bool())


@require_cuda
def test_pack_step_processes_cuda_images_on_the_gpu(sizes, tokenizer_dir):
    step = MolmoAct2PackInputsProcessorStep(
        tokenizer_name=str(tokenizer_dir),
        processor_config=sizes.processor_config,
        action_mode="continuous",
        image_keys=list(sizes.cameras),
    )
    transition = _transition(sizes, torch.zeros(2, sizes.state_dim))
    observation = transition[TransitionKey.OBSERVATION]
    on_gpu = {key: value.cuda() for key, value in observation.items()}

    expected = step(transition)[TransitionKey.COMPLEMENTARY_DATA]
    got = step({**transition, TransitionKey.OBSERVATION: on_gpu})[TransitionKey.COMPLEMENTARY_DATA]

    # The images are resized and patchified where they are, without a copy to the CPU and back.
    assert got["pixel_values"].is_cuda and got["image_token_pooling"].is_cuda
    # GPU and CPU bilinear resizes can round a pixel to the neighboring uint8 level (2/255 once normalized).
    torch.testing.assert_close(
        got["pixel_values"].cpu(), expected["pixel_values"], rtol=0, atol=2 / 255 + 1e-6
    )
    assert torch.equal(got["input_ids"], expected["input_ids"])


def test_pack_step_labels_the_fast_action_span(sizes, tokenizer_dir, fast_dir):
    step = MolmoAct2PackInputsProcessorStep(
        tokenizer_name=str(tokenizer_dir),
        processor_config=sizes.processor_config,
        action_mode="both",
        action_tokenizer_name=str(fast_dir),
    )
    action = torch.linspace(-1.5, 1.5, 2 * sizes.chunk_size * 3).view(2, sizes.chunk_size, 3)

    transition = step(_transition(sizes, torch.zeros(2, sizes.state_dim), action))

    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    clipped = action.clamp(-1, 1)
    assert torch.equal(transition[TransitionKey.ACTION][..., :3], clipped)
    assert torch.count_nonzero(transition[TransitionKey.ACTION][..., 3:]) == 0
    assert transition[TransitionKey.ACTION].shape == (2, sizes.chunk_size, sizes.max_action_dim)
    assert complementary["action_dim_is_pad"].tolist() == [[False] * 3 + [True] * 29] * 2
    tokenizer = step._processor.tokenizer
    fast = UniversalActionProcessor.from_pretrained(str(fast_dir))
    for row, fast_tokens in enumerate(fast(clipped.numpy())):
        labels, input_ids = complementary["labels"][row], complementary["input_ids"][row]
        labeled = labels != -100
        # The span `<action_start> <action_k>... <action_end>` and the `<|im_end|>` after it, at the end of the row.
        span = [*(f"<action_{token}>" for token in fast_tokens), "<action_end>", "<|im_end|>"]
        assert tokenizer.convert_ids_to_tokens(labels[labeled].tolist()) == ["<action_start>", *span]
        assert torch.equal(labels[labeled], input_ids[labeled]) and labeled[-len(span) - 1 :].all()
        assert labeled.sum() == len(span) + 1
        # The action expert sees neither `<|im_end|>` (also the BOS) nor the action span.
        visible = (
            complementary["attention_mask"][row].bool() & ~labeled & (input_ids != tokenizer.eos_token_id)
        )
        assert torch.equal(complementary["encoder_attention_mask"][row], visible)
        assert not visible[input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>")].any()


def test_pack_step_shuffles_the_cameras_only_in_training(sizes, tokenizer_dir):
    step = MolmoAct2PackInputsProcessorStep(
        tokenizer_name=str(tokenizer_dir),
        processor_config=sizes.processor_config,
        action_mode="continuous",
        image_keys=list(sizes.cameras),
        random_camera_order="episode",
    )

    def episode_with_order(order: list[int]) -> int:
        seeds = range(100)
        return next(
            e
            for e in seeds
            if torch.randperm(2, generator=torch.Generator().manual_seed(e)).tolist() == order
        )

    transition = _transition(sizes, torch.zeros(2, sizes.state_dim))
    observation = transition[TransitionKey.OBSERVATION]
    observation[sizes.cameras[0]], observation[sizes.cameras[1]] = (
        torch.zeros(2, 3, 32, 32),
        torch.ones(2, 3, 32, 32),
    )
    episodes = [episode_with_order([1, 0]), episode_with_order([0, 1])]
    transition[TransitionKey.COMPLEMENTARY_DATA]["episode_index"] = torch.tensor(episodes)

    def camera_order(transition) -> list[list[int]]:
        pixel_values = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["pixel_values"]
        # Crops are ordered by sample, then by prompt image; the second camera is white.
        return (pixel_values.view(2, 2, -1).mean(-1) > 0).long().tolist()

    assert camera_order(transition) == [[0, 1], [0, 1]]
    training = {**transition, TransitionKey.ACTION: torch.zeros(2, sizes.chunk_size, sizes.action_dim)}
    # Seeded by the episode: the same episode always gets the same order.
    assert camera_order(training) == [[1, 0], [0, 1]] == camera_order(training)


def test_pack_step_rejects_incomplete_inputs(sizes, tokenizer_dir):
    step = MolmoAct2PackInputsProcessorStep(
        tokenizer_name=str(tokenizer_dir), processor_config=sizes.processor_config, action_mode="continuous"
    )
    transition = _transition(sizes, torch.zeros(2, sizes.state_dim))

    with pytest.raises(ValueError, match="D <= 32"):
        step({**transition, TransitionKey.ACTION: torch.zeros(2, sizes.chunk_size, 33)})
    with pytest.raises(ValueError, match="expected 2 tasks"):
        step({**transition, TransitionKey.COMPLEMENTARY_DATA: {"task": ["a", "b", "c"]}})
    observation = transition[TransitionKey.OBSERVATION]
    with pytest.raises(ValueError, match=OBS_STATE):
        step({**transition, TransitionKey.OBSERVATION: {sizes.cameras[0]: observation[sizes.cameras[0]]}})
    step.image_keys = ["observation.images.missing"]
    with pytest.raises(ValueError, match="observation.images.missing"):
        step(transition)


def _normalize(normalizer_stats, unnormalizer_stats, state, action):
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state.shape[-1],)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action.shape[-1],)),
    }
    normalizer = NormalizerProcessorStep(features=features, norm_map=QUANTILES, stats=normalizer_stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=QUANTILES, stats=unnormalizer_stats)
    normalized = normalizer({TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: action})
    restored = unnormalizer({TransitionKey.ACTION: normalized[TransitionKey.ACTION]})[TransitionKey.ACTION]
    return normalized[TransitionKey.OBSERVATION][OBS_STATE], normalized[TransitionKey.ACTION], restored


def test_normalization_stats_pass_the_gripper_through_and_zero_constant_dims(dataset_meta, sizes):
    stats = _dataset_stats(sizes)

    normalizer_stats, unnormalizer_stats = _normalization_stats(stats, dataset_meta.names)

    state = torch.tensor([[-2.0, 0.0, 2.0, 1.0, -1.0, 0.0, 0.5, 0.3]])
    action = torch.tensor([[-2.0, 0.0, 2.0, 1.0, -1.0, 0.0, 0.7]])
    normalized_state, normalized_action, restored_action = _normalize(
        normalizer_stats, unnormalizer_stats, state, action
    )
    # Joints map [q01, q99] to [-1, 1], the constant joint to 0, and the gripper ([0, 1]) passes through.
    torch.testing.assert_close(normalized_state, torch.tensor([[-1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0, 0.3]]))
    torch.testing.assert_close(normalized_action, torch.tensor([[-1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.7]]))
    torch.testing.assert_close(restored_action, action)
    # The unnormalizer keeps the constant value; the input stats are not modified.
    assert unnormalizer_stats[OBS_STATE]["q01"][-2] == 0.5 and normalizer_stats[OBS_STATE]["q01"][-2] == -0.5
    assert torch.equal(stats[OBS_STATE]["q01"], _dataset_stats(sizes)[OBS_STATE]["q01"])
    assert isinstance(normalizer_stats[ACTION]["q01"], torch.Tensor)


@pytest.mark.parametrize(
    ("normalize_gripper", "gripper_max", "passthrough", "warns"),
    [
        (None, 1.0, True, False),
        (None, 2.0, False, False),
        (False, 2.0, True, True),
        (True, 1.0, False, False),
    ],
)
def test_normalization_stats_gripper_rule(normalize_gripper, gripper_max, passthrough, warns, caplog):
    stats = {
        ACTION: {"min": [-2.0, 0.0], "max": [2.0, gripper_max], "q01": [-2.0, 0.0], "q99": [2.0, gripper_max]}
    }

    with caplog.at_level(logging.WARNING):
        normalizer_stats, _ = _normalization_stats(stats, {ACTION: ["x", "Gripper"]}, normalize_gripper)

    expected = [-1.0, 1.0] if passthrough else [0.0, gripper_max]
    assert [normalizer_stats[ACTION]["q01"][1], normalizer_stats[ACTION]["q99"][1]] == expected
    assert normalizer_stats[ACTION]["q01"][0] == -2.0  # lists stay lists
    assert ("will be clipped" in caplog.text) == warns


def test_normalization_stats_honour_a_mask_unless_normalize_gripper(caplog):
    stats = {
        ACTION: {
            "min": [-2.0, -0.5],
            "max": [2.0, 0.5],
            "q01": [-2.0, -0.5],
            "q99": [2.0, 0.5],
            "mask": [True, False],
        }
    }

    normalizer_stats, unnormalizer_stats = _normalization_stats(stats, {})
    normalized_stats, _ = _normalization_stats(stats, {}, normalize_gripper=True)

    # No names are needed with a mask; it is dropped from the outputs.
    assert normalizer_stats[ACTION]["q99"] == [2.0, 1.0] and unnormalizer_stats[ACTION]["q01"] == [-2.0, -1.0]
    assert "mask" not in normalizer_stats[ACTION] and "mask" not in unnormalizer_stats[ACTION]
    assert normalized_stats[ACTION]["q99"] == [2.0, 0.5]
    # Without a mask or names, every dimension is normalized, with a warning.
    del stats[ACTION]["mask"]
    with caplog.at_level(logging.WARNING):
        unnamed_stats, _ = _normalization_stats(stats, {})
    assert unnamed_stats[ACTION]["q99"] == [2.0, 0.5] and "can't find the gripper" in caplog.text


def test_clip_step_clips_normalized_actions():
    action = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])

    clipped = MolmoAct2ClipActionsProcessorStep()({TransitionKey.ACTION: action})[TransitionKey.ACTION]

    assert clipped.tolist() == [[-1.0, -0.5, 0.5, 1.0]]


class _CharacterBPE:
    """Stands in for the FAST BPE tokenizer: action bin `k` expands to `k % 3 + 1` characters."""

    @staticmethod
    def decode(bins: list[int]) -> str:
        return "".join("c" * (bin_index % 3 + 1) for bin_index in bins)


def _decode_constrained(logits: torch.Tensor, fast, num_coefficients: int) -> list[list[int]]:
    """Run `select_constrained_discrete_action_token` on fixed logits until every row emits EOS."""
    generated: list[list[int]] = [[] for _ in range(logits.shape[0])]
    while not all(tokens and tokens[-1] == EOS_TOKEN_ID for tokens in generated):
        next_tokens = select_constrained_discrete_action_token(
            logits, generated, action_tokenizer=fast, target_coefficients=num_coefficients
        )
        for tokens, token in zip(generated, next_tokens.tolist(), strict=True):
            if not tokens or tokens[-1] != EOS_TOKEN_ID:
                tokens.append(token)
    return generated


def test_constrained_fast_decoding_writes_exactly_one_chunk_per_row():
    fast = SimpleNamespace(bpe_tokenizer=_CharacterBPE())
    logits = torch.zeros(2, _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS)
    logits[:, EOS_TOKEN_ID] = 100.0  # never chosen before the chunk is complete
    for bin_index, logit in ((2, 3.0), (1, 2.0), (0, 1.0)):  # row 0 prefers 3, then 2, then 1 characters
        logits[0, _ACTION_TOKEN_START_ID + bin_index] = logit
    for bin_index, logit in ((4, 3.0), (0, 1.0)):  # row 1 prefers 2, then 1 character
        logits[1, _ACTION_TOKEN_START_ID + bin_index] = logit

    generated = _decode_constrained(logits, fast, num_coefficients=7)

    def action(bin_index: int) -> int:
        return _ACTION_TOKEN_START_ID + bin_index

    # 3 + 3 + 1 and 2 + 2 + 2 + 1 characters: the preferred token is skipped when it would overshoot.
    span_end = [_ACTION_END_TOKEN_ID, EOS_TOKEN_ID]
    assert generated[0] == [_ACTION_START_TOKEN_ID, action(2), action(2), action(0), *span_end]
    assert generated[1] == [_ACTION_START_TOKEN_ID, action(4), action(4), action(4), action(0), *span_end]
    # Rows are decoded independently.
    for row in range(2):
        assert _decode_constrained(logits[row : row + 1], fast, num_coefficients=7) == [generated[row]]


def test_constrained_fast_decoding_raises_when_the_span_overshoots():
    fast = SimpleNamespace(bpe_tokenizer=SimpleNamespace(decode=lambda bins: "ccc" * len(bins)))
    logits = torch.zeros(1, _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS)
    generated = [[_ACTION_START_TOKEN_ID, _ACTION_TOKEN_START_ID, _ACTION_TOKEN_START_ID]]

    with pytest.raises(RuntimeError, match="exceeded its target coefficient count: 6 > 5"):
        select_constrained_discrete_action_token(
            logits, generated, action_tokenizer=fast, target_coefficients=5
        )


def test_decode_discrete_action_chunk_rejects_a_row_without_action_tokens():
    generated = [[_ACTION_START_TOKEN_ID, _ACTION_END_TOKEN_ID, EOS_TOKEN_ID]]

    with pytest.raises(RuntimeError, match="no decodable action tokens"):
        decode_discrete_action_chunk(generated, SimpleNamespace(), action_horizon=4, action_dim=2)


def test_constrained_fast_decoding_raises_when_no_token_completes_the_chunk():
    fast = SimpleNamespace(bpe_tokenizer=SimpleNamespace(decode=lambda bins: "cc" * len(bins)))
    logits = torch.zeros(1, _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS)

    with pytest.raises(RuntimeError, match="without overshooting 3 coefficients from 2"):
        _decode_constrained(logits, fast, num_coefficients=3)


def _norm_stats_file(sizes, path, **metadata) -> str:
    tag_stats = {
        "action_stats": {
            "q01": [-1.0] * sizes.action_dim,
            "q99": [1.0] * sizes.action_dim,
            "min": [-1.0] * sizes.action_dim,
        },
        "state_stats": {
            "q01": [-4.0] * sizes.state_dim,
            "q99": [4.0] * sizes.state_dim,
            "min": [-4.0] * sizes.state_dim,
        },
    }
    for stats in tag_stats.values():
        dim = len(stats["q01"])
        stats.update(max=[1.0] * dim, mask=[True] * (dim - 1) + [False], names=[f"d{i}" for i in range(dim)])
    tag_stats["state_stats"]["max"] = [4.0] * sizes.state_dim
    (path / "norm_stats.json").write_text(
        json.dumps({"metadata_by_tag": {"libero": {**tag_stats, **metadata}}})
    )
    return str(path / "norm_stats.json")


def _step(pipeline, step_type):
    return next(step for step in pipeline.steps if isinstance(step, step_type))


def test_norm_tag_replaces_the_dataset_stats_and_fills_the_prompt(dataset_meta, sizes, make_config, tmp_path):
    stats_path = _norm_stats_file(
        sizes, tmp_path, setup_type="libero franka", control_mode="delta", camera_keys=[sizes.cameras[1]]
    )
    config = make_config(
        norm_tag="libero", norm_stats_path=stats_path, image_keys=[], action_mode="continuous"
    )

    preprocessor, postprocessor = make_molmoact2_pre_post_processors(
        config, _dataset_stats(sizes), dataset_meta
    )

    state = torch.full((1, sizes.state_dim), 2.0)
    normalized = _step(preprocessor, NormalizerProcessorStep)({TransitionKey.OBSERVATION: {OBS_STATE: state}})
    # The tag's stats (q99 = 4) with its mask: the last dim passes through.
    assert normalized[TransitionKey.OBSERVATION][OBS_STATE].tolist() == [
        [0.5] * (sizes.state_dim - 1) + [2.0]
    ]
    pack = _step(preprocessor, MolmoAct2PackInputsProcessorStep)
    assert (pack.setup_type, pack.control_mode, pack.image_keys) == (
        "libero franka",
        "delta",
        [sizes.cameras[1]],
    )
    restored = _step(postprocessor, UnnormalizerProcessorStep)(
        {TransitionKey.ACTION: torch.full((1, sizes.action_dim), 0.5)}
    )
    assert restored[TransitionKey.ACTION].tolist() == [[0.5] * sizes.action_dim]
    # Values set in the config win over the tag's.
    config = make_config(
        norm_tag="libero", norm_stats_path=stats_path, setup_type="mine", action_mode="continuous"
    )
    pack = _step(
        make_molmoact2_pre_post_processors(config, None, dataset_meta)[0], MolmoAct2PackInputsProcessorStep
    )
    assert (pack.setup_type, pack.image_keys) == ("mine", list(sizes.cameras))
    with pytest.raises(ValueError, match="Unknown norm_tag 'droid'"):
        make_molmoact2_pre_post_processors(make_config(norm_tag="droid", norm_stats_path=stats_path), None)


def test_saved_processors_reload_without_the_norm_tag_or_the_processor_defaults(
    dataset_meta, sizes, make_config, tmp_path, monkeypatch
):
    stats_path = _norm_stats_file(sizes, tmp_path)
    config = make_config(norm_tag="libero", norm_stats_path=stats_path, action_mode="continuous")
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(config, None, dataset_meta)
    preprocessor.save_pretrained(tmp_path / "checkpoint")
    postprocessor.save_pretrained(tmp_path / "checkpoint")
    # The saved pipelines carry their stats and their image processor settings.
    (tmp_path / "norm_stats.json").unlink()
    monkeypatch.setattr(processor_molmoact2, "_PROCESSOR_CONFIG", {"image_processor": {}})

    loaded_pre, loaded_post = make_molmoact2_pre_post_processors_from_pretrained(
        config, tmp_path / "checkpoint"
    )

    assert _step(loaded_pre, MolmoAct2PackInputsProcessorStep).processor_config == sizes.processor_config
    expected, batch = preprocessor(_raw_batch(sizes)), loaded_pre(_raw_batch(sizes))
    assert batch.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(batch[key], value) if isinstance(value, torch.Tensor) else batch[key] == value, key
    action = torch.rand(2, sizes.action_dim) * 2 - 1
    assert torch.equal(loaded_post(action), postprocessor(action))


def test_pretrained_processors_adjust_new_dataset_stats(dataset_meta, sizes, make_config, tmp_path):
    config = make_config(action_mode="continuous")
    for pipeline in make_molmoact2_pre_post_processors(config, _dataset_stats(sizes), dataset_meta):
        pipeline.save_pretrained(tmp_path)
    new_stats = _dataset_stats(sizes)
    for key in (OBS_STATE, ACTION):
        new_stats[key]["q99"][0] = new_stats[key]["max"][0] = 6.0
    # Fine-tuning on a 5-D action with a new prompt: the pack step follows the config.
    config.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(5,))
    config.setup_type, config.random_camera_order = "my robot", "episode"

    preprocessor, postprocessor = make_molmoact2_pre_post_processors_from_pretrained(
        config,
        tmp_path,
        dataset_meta=dataset_meta,
        preprocessor_overrides={"normalizer_processor": {"stats": new_stats}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": new_stats}},
    )

    state = torch.tensor([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3]])
    normalized = _step(preprocessor, NormalizerProcessorStep)({TransitionKey.OBSERVATION: {OBS_STATE: state}})
    # The new range, with the constant joint and the gripper adjusted as when the processors are first built.
    torch.testing.assert_close(
        normalized[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]]),
    )
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]])
    torch.testing.assert_close(postprocessor(action), torch.tensor([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]]))
    pack = _step(preprocessor, MolmoAct2PackInputsProcessorStep)
    assert (pack.action_dim, pack.setup_type, pack.random_camera_order) == (5, "my robot", "episode")
    # Fields the config leaves empty keep their saved values.
    assert pack.image_keys == list(sizes.cameras)


def test_pretrained_processors_map_the_robot_joint_units_to_the_checkpoint(
    dataset_meta, sizes, make_config, tmp_path
):
    # The checkpoint's stats are in radians; the robot and the new dataset use degrees, and flip the second joint.
    config = make_config(action_mode="continuous")
    for pipeline in make_molmoact2_pre_post_processors(config, _dataset_stats(sizes), dataset_meta):
        pipeline.save_pretrained(tmp_path)
    config.joint_scales = [math.pi / 180, -math.pi / 180]
    degree_stats = _dataset_stats(sizes)
    for key in (OBS_STATE, ACTION):
        for name in ("min", "max", "q01", "q99"):
            degree_stats[key][name][:2] *= 180 / math.pi

    preprocessor, postprocessor = make_molmoact2_pre_post_processors_from_pretrained(
        config,
        tmp_path,
        dataset_meta=dataset_meta,
        preprocessor_overrides={"normalizer_processor": {"stats": degree_stats}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": degree_stats}},
    )

    assert _step(preprocessor, JointAffineProcessorStep).scales == config.joint_scales
    assert _step(postprocessor, JointAffineProcessorStep).inverse
    state = torch.tensor([[90.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3]])
    radians = _step(preprocessor, JointAffineProcessorStep)({TransitionKey.OBSERVATION: {OBS_STATE: state}})
    normalized = _step(preprocessor, NormalizerProcessorStep)(radians)
    # Radians normalized with the checkpoint-unit stats ([-2, 2]); the constant joint and the gripper as usual.
    torch.testing.assert_close(
        normalized[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[math.pi / 4, -math.pi / 8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]]),
    )
    action = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3]])
    degrees = 180 / math.pi
    torch.testing.assert_close(
        postprocessor(action), torch.tensor([[degrees, -degrees, 0.0, 0.0, 0.0, 0.0, 0.3]])
    )


def test_pretrained_processors_take_the_norm_tag_prompt_when_the_config_has_none(
    dataset_meta, sizes, make_config, tmp_path
):
    config = make_config(action_mode="continuous")
    for pipeline in make_molmoact2_pre_post_processors(config, _dataset_stats(sizes), dataset_meta):
        pipeline.save_pretrained(tmp_path / "base")
    stats_path = _norm_stats_file(sizes, tmp_path, setup_type="libero franka", control_mode="delta")
    config.norm_tag, config.norm_stats_path = "libero", stats_path

    preprocessor, _ = make_molmoact2_pre_post_processors_from_pretrained(
        config,
        tmp_path / "base",
        dataset_meta=dataset_meta,
        preprocessor_overrides={"normalizer_processor": {"stats": _dataset_stats(sizes)}},
    )

    pack = _step(preprocessor, MolmoAct2PackInputsProcessorStep)
    assert (pack.setup_type, pack.control_mode) == ("libero franka", "delta")


def test_processors_read_the_dimension_names_that_make_policy_attaches(dataset_meta, sizes, make_config):
    config = make_config(action_mode="continuous")
    # lerobot-train passes no dataset_meta to the processors; make_policy attaches it to the config.
    config._runtime_dataset_meta = dataset_meta

    preprocessor, _ = make_molmoact2_pre_post_processors(config, _dataset_stats(sizes))

    state = torch.tensor([[-2.0, 0.0, 2.0, 1.0, -1.0, 0.0, 0.5, 0.3]])
    normalized = _step(preprocessor, NormalizerProcessorStep)({TransitionKey.OBSERVATION: {OBS_STATE: state}})
    # The gripper, named in the metadata, passes through.
    assert normalized[TransitionKey.OBSERVATION][OBS_STATE][0, -1].item() == pytest.approx(0.3)


@pytest.mark.parametrize("action_mode", ["continuous", "both"])
def test_processors_drive_the_policy_end_to_end(dataset_meta, sizes, make_policy, request, action_mode):
    overrides = {"action_mode": action_mode}
    if action_mode == "both":
        overrides["action_tokenizer_name"] = str(request.getfixturevalue("fast_dir"))
    policy = make_policy(**overrides)
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(
        policy.config, _dataset_stats(sizes), dataset_meta
    )

    batch = preprocessor(_raw_batch(sizes))
    assert ("labels" in batch) == (action_mode == "both")
    loss, log = policy.train().forward(batch)
    loss.backward()
    action = postprocessor(policy.select_action(preprocessor(_raw_batch(sizes, with_action=False))))

    assert torch.isfinite(loss) and log.keys() >= {"flow_loss", "loss"}
    assert action.shape == (2, sizes.action_dim)
    # Sampled actions are clipped to [-1, 1]: joints stay in [q01, q99] and the gripper passes through.
    assert action[:, :-1].abs().max() <= 2.0 and action[:, -1].abs().max() <= 1.0


def test_released_processor_files_match_the_model_token_ids():
    """Check the hardcoded token ids against the released vocabulary, when it is in the local Hub cache."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import PreTrainedTokenizerFast

    checkpoint, fast_checkpoint = "allenai/MolmoAct2", "allenai/MolmoAct2-FAST-Tokenizer"
    cached = [try_to_load_from_cache(checkpoint, "processor_config.json")]
    cached += [try_to_load_from_cache(fast_checkpoint, "tokenizer.json")]
    if not all(isinstance(path, str) for path in cached):
        pytest.skip("The allenai/MolmoAct2 processor and tokenizer files are not in the local Hub cache.")

    # The image settings saved in LeRobot checkpoints are those of the released processor_config.json.
    released = json.loads(Path(cached[0]).read_text())
    metadata_keys = {"auto_map", "image_processor_type"}
    released_image = {
        key: value for key, value in released["image_processor"].items() if key not in metadata_keys
    }
    assert _PROCESSOR_CONFIG["image_processor"] == released_image
    assert all(released[key] == value for key, value in _PROCESSOR_CONFIG.items() if key != "image_processor")

    processor = _load_molmoact2_processor(_PROCESSOR_CONFIG, checkpoint)
    tokenizer = processor.tokenizer
    molmo2_config = _get_molmo2_config()
    special = ["<|im_end|>", "<action_start>", "<action_end>", "<action_0>", "<im_start>", "<im_patch>"]
    expected = [EOS_TOKEN_ID, _ACTION_START_TOKEN_ID, _ACTION_END_TOKEN_ID, _ACTION_TOKEN_START_ID]
    expected += [molmo2_config.image_start_token_id, molmo2_config.image_patch_id]
    assert tokenizer.convert_tokens_to_ids(special) == expected
    assert (
        tokenizer.convert_tokens_to_ids(f"<action_{_NUM_ACTION_TOKENS - 1}>")
        == _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS - 1
    )
    # Every FAST token has an `<action_k>` token.
    assert len(PreTrainedTokenizerFast.from_pretrained(fast_checkpoint)) <= _NUM_ACTION_TOKENS
    # A camera becomes a 14x14 grid of patch tokens, without column tokens.
    inputs = processor(
        text=["<|image|>"], images=[np.zeros((256, 320, 3), dtype=np.uint8)], return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    assert inputs["image_grids"].tolist() == [[14, 14, 0, 0]]
    assert tokens == ["<|im_end|>", "<im_start>", *["<im_patch>"] * 196, "<im_end>"]
