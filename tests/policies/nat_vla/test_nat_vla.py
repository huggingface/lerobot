# Copyright 2026 Ahad Jawaid. All rights reserved.
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

"""Tests of the NAT VLA policy with a tiny LFM2-VL and a small NAT tokenizer of the released token layout, saved
in a folder laid out like the pretrained Hub repository."""

import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from transformers import AutoProcessor  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.inference.rtc import supports_rtc_inference, validate_local_chunk_policy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.nat_vla import NATVLAConfig, NATVLAPolicy  # noqa: E402
from lerobot.policies.nat_vla.modeling_nat_vla import ActionStreamHook  # noqa: E402
from lerobot.policies.nat_vla.nat_tokenizer import NATTokenizer  # noqa: E402
from lerobot.policies.nat_vla.processor_nat_vla import (  # noqa: E402
    NATVLATokenizerProcessorStep,
    vlm_config_of,
)

# A small NAT tokenizer with the released tokenizer's token layout.
TOKENIZER = {
    "time_horizon": 32,
    "group_width": 7,
    "encoder": {
        "channels": 1,
        "filters": 8,
        "downsampling_ratios": [8, 7],
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 7,
        "compress": 2,
        "lstm_layers": 1,
        "latent_dim": 32,
    },
    "quantizer": {"levels": [8, 8, 5, 5, 4], "scales": [2, 1, 1, 1]},
    "decoder": {
        "dim": 64,
        "intermediate_dim": 128,
        "num_blocks": 2,
        "condition_embeddings": 4,
        "condition_id": 1,
        "attention_context": 8,
        "group_norm_groups": 8,
    },
}
# Two layers of LFM2-VL with its text vocabulary, for the public LFM2-VL processor.
TINY_VLM = {
    "model_type": "lfm2_vl",
    "projector_hidden_size": 64,
    "text_config": {
        "model_type": "lfm2",
        "vocab_size": 65536,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "eos_token_id": 7,
    },
    "vision_config": {
        "model_type": "siglip2_vision_model",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_patches": 256,
        "patch_size": 16,
    },
}
IDENTITY_STATS = {
    "observation.state": {"q01": -torch.ones(7), "q99": torch.ones(7)},
    "action": {"min": -torch.ones(7), "max": torch.ones(7)},
}


@pytest.fixture(scope="module")
def make_config(tmp_path_factory):
    """Configs of policies whose `vlm` is a folder holding a pretrained NAT VLA as the Hub repository does: the
    policy's config and weights, and vlm/ with the VLM's config and processor."""
    pretrained = tmp_path_factory.mktemp("pretrained")
    AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-450M").save_pretrained(pretrained / "vlm")
    (pretrained / "vlm" / "config.json").write_text(json.dumps(TINY_VLM))
    tokenizer = tokenizer_folder(pretrained / "tokenizer", TOKENIZER)

    def make_config(action_dim=7, **overrides):
        return NATVLAConfig(
            **{
                "device": "cpu",
                "vlm": str(pretrained),
                "action_tokenizer": str(tokenizer),
                "setup": "single arm",
                "control_mode": "absolute joint positions",
                **overrides,
            },
            input_features={
                "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
                "observation.state": PolicyFeature(FeatureType.STATE, (action_dim,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
        )

    torch.manual_seed(0)
    NATVLAPolicy(make_config(vlm_config=vlm_config_of(pretrained / "vlm"))).save_pretrained(pretrained)
    return make_config


def tokenizer_folder(folder, tokenizer_config):
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(tokenizer_config))
    save_file(NATTokenizer(tokenizer_config).state_dict(), folder / "model.safetensors")
    return folder


@pytest.fixture(scope="module")
def policy(make_config):
    return NATVLAPolicy(make_config())


@pytest.fixture(scope="module")
def processors(policy):
    return make_pre_post_processors(policy.config, dataset_stats=IDENTITY_STATS)


@pytest.fixture(scope="module")
def preprocessor(processors):
    return processors[0]


def observations(action_dim=7, size=2, with_actions=False):
    generator = torch.Generator().manual_seed(0)
    batch = {
        "observation.images.front": torch.rand(size, 3, 64, 64, generator=generator),
        "observation.state": torch.rand(size, action_dim, generator=generator) * 2 - 1,
        "task": ["pick up the cube", "open the drawer"][:size],
    }
    if with_actions:
        batch["action"] = torch.rand(size, 32, action_dim, generator=generator).cumsum(1).div(16).sub(1)
    return batch


@pytest.mark.parametrize("action_dim, tokens", [(7, 14), (8, 28), (14, 28)])
def test_action_tokens_cover_every_column_group(policy, action_dim, tokens):
    actions = observations(action_dim, with_actions=True)["action"].clamp(-1, 1)
    action_tokens = policy.action_tokenizer.encode(actions)
    assert action_tokens.shape == (2, tokens) and action_tokens.min() >= 0 and action_tokens.max() < 25600
    assert policy.action_tokenizer.decode(action_tokens, action_dim).shape == actions.shape


@pytest.mark.parametrize("steps, tokens", [(1, 4), (8, 4), (9, 7), (16, 7), (24, 11), (32, 14)])
def test_a_token_prefix_decodes_the_first_steps_of_the_chunk(policy, steps, tokens):
    action_tokens = policy.action_tokenizer.encode(observations(with_actions=True)["action"])
    count, decoded_steps = policy.action_tokenizer.token_count(steps, action_dim=7)
    assert count == tokens and decoded_steps >= steps
    torch.testing.assert_close(
        policy.action_tokenizer.decode(action_tokens[:, :count], 7)[:, :decoded_steps],
        policy.action_tokenizer.decode(action_tokens, 7)[:, :decoded_steps],
    )


def test_the_streaming_decoder_returns_each_latent_as_its_tokens_arrive(policy):
    action_tokens = policy.action_tokenizer.encode(observations(with_actions=True)["action"])
    decoder = policy.action_tokenizer.stream(action_dim=7)
    arrived = [decoder.push(token) for token in action_tokens.unbind(1)]
    assert [index + 1 for index, latent in enumerate(arrived) if latent is not None] == [4, 7, 11, 14]
    torch.testing.assert_close(
        torch.cat([latent for latent in arrived if latent is not None], 1),
        policy.action_tokenizer.decode(action_tokens, 7),
    )


def test_vocabulary_ids_follow_the_text_vocabulary_in_the_pretraining_order(policy):
    vocabulary = policy.vocabulary
    assert (vocabulary.setup_start, vocabulary.state_end, vocabulary.first_state_bin) == (65536, 65541, 65542)
    assert (vocabulary.action_output, vocabulary.action_start, vocabulary.first_action_token) == (
        65798,
        65799,
        65801,
    )
    assert vocabulary.size == policy.vlm.get_input_embeddings().num_embeddings == 65536 + 25865
    assert policy.action_tokenizer.prefix_token_counts == {8: 4, 16: 7, 24: 11, 32: 14}


def test_policy_type_starts_from_the_pretrained_weights(make_config):
    pretrained_weights = load_file(f"{make_config().vlm}/model.safetensors")
    state = NATVLAPolicy(make_config()).state_dict()
    assert all(torch.equal(state[name], tensor) for name, tensor in pretrained_weights.items())


def test_the_preprocessor_writes_the_prompt_tokens_and_image_inputs(policy, preprocessor):
    batch = preprocessor(observations())
    vocabulary, pad_token_id = policy.vocabulary, policy.pad_token_id
    bos_token_id = AutoProcessor.from_pretrained(f"{policy.config.vlm}/vlm").tokenizer.bos_token_id
    assert {"pixel_values", "pixel_attention_mask", "spatial_shapes"} <= set(batch)
    for ids, mask in zip(batch["input_ids"], batch["attention_mask"], strict=True):
        prompt = ids[mask.bool()].tolist()
        assert prompt[0] == bos_token_id
        assert set(range(vocabulary.setup_start, vocabulary.first_state_bin)) <= set(prompt)
        assert prompt[-1] == vocabulary.action_output
        assert sum(vocabulary.first_state_bin <= token < vocabulary.action_output for token in prompt) == 7
        assert (ids[~mask.bool()] == pad_token_id).all()
    assert not batch["attention_mask"].all()


def test_loss_trains_only_the_added_rows_of_the_embedding(policy, preprocessor):
    policy.train()
    policy.zero_grad()
    loss, parts = policy.forward(preprocessor(observations(with_actions=True)))
    loss.backward()
    gradient = policy.vlm.get_input_embeddings().weight.grad
    assert torch.isfinite(loss) and parts["z_loss"] > 0
    first_added_row = policy.vocabulary.setup_start
    assert gradient[:first_added_row].abs().max() == 0 and gradient[first_added_row:].abs().max() > 0
    assert not any(parameter.requires_grad for parameter in policy.action_tokenizer.parameters())


def test_tokens_of_fully_padded_latents_are_not_trained(policy, preprocessor):
    padded_after_step_10 = (torch.arange(32) >= 10).expand(2, -1)
    trained = policy.action_tokenizer.tokens_with_actions(~padded_after_step_10, action_dim=8)
    assert trained.tolist() == [([True] * 7 + [False] * 7) * 2] * 2

    batch = preprocessor(observations(with_actions=True))
    with torch.no_grad():
        assert policy.forward({**batch, "action_is_pad": padded_after_step_10})[0] != policy.forward(batch)[0]


def test_predicted_chunks_and_queued_actions(policy, preprocessor):
    batch = preprocessor(observations())
    assert policy.predict_action_chunk(batch).shape == (2, 32, 7)
    assert policy.predict_action_chunk(batch, steps=8).shape == (2, 8, 7)
    policy.config.n_action_steps = 8
    policy.reset()
    actions = [policy.select_action(batch) for _ in range(9)]
    policy.config.n_action_steps = 32
    policy.reset()
    assert all(action.shape == (2, 7) for action in actions) and not torch.equal(actions[7], actions[8])


def test_predicting_keeps_the_training_mode(policy, preprocessor):
    policy.train()
    policy.predict_action_chunk(preprocessor(observations()), steps=8)
    assert policy.training and policy.vlm.training


def test_streamed_actions_equal_the_predicted_chunk(policy, preprocessor):
    batch = preprocessor(observations())
    policy.config.n_action_steps = 10
    policy.reset()
    try:
        streamed = torch.stack([policy.select_action(batch) for _ in range(11)], 1)
    finally:
        policy.config.n_action_steps = 32
        policy.reset()
    chunk = policy.predict_action_chunk(batch, steps=10)
    assert torch.equal(streamed[:, :10], chunk[:, :10]) and torch.equal(streamed[:, 10], chunk[:, 0])


class StepCountingDecoder:
    """Stands in for the tokenizer's streaming decoder: 4 tokens decode steps 0-7, 7 tokens steps 8-15."""

    decoded_steps = 0

    def __init__(self):
        self.tokens = 0

    def push(self, token):
        self.tokens += 1
        if self.tokens in (4, 7):
            first_step = 0 if self.tokens == 4 else 8
            return torch.arange(first_step, first_step + 8.0).reshape(1, 8, 1).expand(1, 8, 2)
        return None


def test_a_latents_actions_arrive_before_the_rest_of_the_chunk_is_generated():
    rest_of_chunk = threading.Event()

    def generate(streamer):
        streamer.put(torch.zeros(1, 5, dtype=torch.long))
        for token in range(7):
            if token == 4:
                rest_of_chunk.wait(timeout=60)
            streamer.put(torch.tensor([token]))
        streamer.end()

    stream = ActionStreamHook(StepCountingDecoder(), first_action_token=0)
    stream.start(generate, torch.device("cpu"))
    first_latent = torch.stack([stream.action(step) for step in range(8)], 1)
    assert not rest_of_chunk.is_set() and torch.equal(first_latent[0, :, 0], torch.arange(8.0))
    rest_of_chunk.set()
    second_latent = torch.stack([stream.action(step) for step in range(8, 16)], 1)
    assert torch.equal(second_latent[0, :, 1], torch.arange(8.0, 16.0))
    stream.thread.join()


def test_a_failed_generation_raises_in_the_caller():
    def generate(streamer):
        raise RuntimeError("out of memory")

    stream = ActionStreamHook(StepCountingDecoder(), first_action_token=0)
    stream.start(generate, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="out of memory"):
        stream.action(0)


def generation_prompts(policy):
    """Records the input ids of every generate() call."""
    prompts, generate = [], policy.vlm.generate
    policy.vlm.generate = lambda **arguments: (
        prompts.append(arguments["input_ids"].cpu()) or generate(**arguments)
    )
    return prompts


def test_executing_actions_start_the_next_chunk_through_their_tokens(policy, preprocessor):
    batch = preprocessor(observations())
    prompts = generation_prompts(policy)
    policy.config.n_action_steps, policy.config.rtc_steps = 8, 8
    policy.reset()
    try:
        actions = torch.stack([policy.select_action(batch) for _ in range(16)], 1)
    finally:
        del policy.vlm.generate
        policy.config.n_action_steps, policy.config.rtc_steps = 32, 0
        policy.reset()

    assert len(prompts) == 3 and actions.shape == (2, 16, 7)
    for chunk, executing in ((1, actions[:, :8]), (2, actions[:, 8:])):
        prefix = policy.action_tokenizer.tokens_starting_with(executing).cpu()
        assert torch.equal(
            prompts[chunk][:, -prefix.shape[1] :], prefix + policy.vocabulary.first_action_token
        )


def test_rtc_inference_starts_the_chunk_with_the_tokens_of_the_actions_run_meanwhile(policy, preprocessor):
    assert supports_rtc_inference(policy) and validate_local_chunk_policy(policy).prediction_steps == 32
    batch = preprocessor(observations(size=1))
    left_over = torch.linspace(-0.5, 0.5, 20)[:, None].expand(-1, 7)
    prompts = generation_prompts(policy)
    try:
        chunk = policy.predict_action_chunk(batch, inference_delay=3, prev_chunk_left_over=left_over)
    finally:
        del policy.vlm.generate

    prefix = policy.action_tokenizer.tokens_starting_with(left_over[None, :8])
    assert torch.equal(prompts[0][:, -4:], prefix + policy.vocabulary.first_action_token)
    assert chunk.shape == (1, 32, 7)
    torch.testing.assert_close(chunk[:, :8], policy.action_tokenizer.decode(prefix, 7)[:, :8])


def test_rtc_steps_must_fit_whole_latents_in_the_chunk(make_config):
    with pytest.raises(ValueError, match="rtc_steps"):
        make_config(rtc_steps=8)
    with pytest.raises(ValueError, match="whole latents"):
        NATVLAPolicy(make_config(n_action_steps=16, rtc_steps=4))


def test_generation_writes_only_action_tokens(policy, preprocessor):
    captured = []
    decode = policy.action_tokenizer.decode
    policy.action_tokenizer.decode = lambda tokens, action_dim: (
        captured.append(tokens) or decode(tokens, action_dim)
    )
    try:
        policy.predict_action_chunk(preprocessor(observations()))
    finally:
        del policy.action_tokenizer.decode
    assert captured[-1].shape == (2, 14) and min(map(torch.min, captured)) >= 0
    assert max(map(torch.max, captured)) < 25600


def test_the_vlm_follows_the_configured_dtype_and_the_tokenizer_stays_in_float32(
    policy, make_config, preprocessor
):
    float32_policy = NATVLAPolicy(make_config(dtype=torch.float32))
    assert (policy.vlm.dtype, float32_policy.vlm.dtype) == (torch.bfloat16, torch.float32)
    for each_policy in (policy, float32_policy):
        assert all(
            parameter.dtype == torch.float32 for parameter in each_policy.action_tokenizer.parameters()
        )
        chunk = each_policy.predict_action_chunk(preprocessor(observations()))
        assert chunk.dtype == torch.float32 and chunk.isfinite().all()


def test_checkpoints_load_with_their_vlm_config_and_processor_alone(policy, processors, tmp_path):
    batch = processors[0](observations(size=1))
    policy.save_pretrained(tmp_path)
    for processor in processors:
        processor.save_pretrained(tmp_path)
    unreachable = {"vlm": "unreachable/vlm", "action_tokenizer": "unreachable/tokenizer"}
    saved_config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(saved_config | unreachable))

    reloaded = NATVLAPolicy.from_pretrained(tmp_path)
    reloaded_preprocessor = make_pre_post_processors(reloaded.config, pretrained_path=tmp_path)[0]
    for name, tensor in policy.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), name
    torch.testing.assert_close(
        reloaded.predict_action_chunk(reloaded_preprocessor(observations(size=1))),
        policy.predict_action_chunk(batch),
    )


def test_fine_tuning_a_checkpoint_writes_prompts_with_the_new_setup(processors, make_config, tmp_path):
    for processor in processors:
        processor.save_pretrained(tmp_path)
    new_robot = make_config(8, setup="two arms", control_mode="joint velocities")
    loaded = make_pre_post_processors(new_robot, pretrained_path=tmp_path)[0]
    step = next(step for step in loaded.steps if isinstance(step, NATVLATokenizerProcessorStep))
    assert (step.setup, step.control_mode, step.camera_keys) == (
        "two arms",
        "joint velocities",
        ["observation.images.front"],
    )


def test_another_nat_tokenizer_swaps_in_through_the_config_alone(make_config, tmp_path):
    """16 steps in 2 latents, scales [2, 1] of 100 codes: tokens (scale 0, latent 0), (1, 0), (1, 1)."""
    small_tokenizer = {**TOKENIZER, "time_horizon": 16, "quantizer": {"levels": [5, 5, 4], "scales": [2, 1]}}
    tokenizer = tokenizer_folder(tmp_path / "tokenizer", small_tokenizer)
    vlm_config = vlm_config_of(Path(make_config().vlm) / "vlm")
    small = NATVLAPolicy(
        make_config(action_tokenizer=str(tokenizer), chunk_size=16, n_action_steps=8, vlm_config=vlm_config)
    )
    assert small.action_tokenizer.vocab_size == 200 and small.action_tokenizer.prefix_token_counts == {
        8: 2,
        16: 3,
    }
    assert small.vocabulary.size == 65536 + 6 + 256 + 3 + 200

    small_preprocessor = make_pre_post_processors(small.config, dataset_stats=IDENTITY_STATS)[0]
    training_batch = observations(with_actions=True)
    training_batch["action"] = training_batch["action"][:, :16]
    loss, _ = small.forward(small_preprocessor(training_batch))
    loss.backward()
    assert torch.isfinite(loss)

    batch = small_preprocessor(observations())
    assert small.predict_action_chunk(batch).shape == (2, 16, 7)
    streamed = torch.stack([small.select_action(batch) for _ in range(8)], 1)
    torch.testing.assert_close(streamed, small.predict_action_chunk(batch, steps=8))
