# Copyright (c) 2026 Ahad Jawaid
import json
import os
import threading

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers", reason="nat_vla requires the `nat_vla` extra (transformers)")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.nat_vla import NATVLAConfig, NATVLAPolicy
from lerobot.policies.nat_vla.modeling_nat_vla import ActionStreamHook
from lerobot.policies.nat_vla.nat_tokenizer import NATTokenizer

RELEASED_WEIGHTS = "ahad-j/nat-vla-450m"
SMALL_TOKENIZER = {
    "time_horizon": 16,
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
    "quantizer": {"levels": [5, 5, 4], "scales": [2, 1]},
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


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return NATVLAPolicy(config(action_dim=7))


def config(action_dim, **overrides):
    return NATVLAConfig(
        **{"device": "cpu", "setup": "single arm", "control_mode": "absolute joint positions", **overrides},
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (action_dim,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )


def batch(action_dim=7, size=2):
    generator = torch.Generator().manual_seed(0)
    return {
        "observation.images.front": torch.rand(size, 3, 64, 64, generator=generator),
        "observation.state": torch.rand(size, action_dim, generator=generator) * 2 - 1,
        "action": torch.rand(size, 32, action_dim, generator=generator).cumsum(1).div(16).sub(1).clamp(-1, 1),
        "task": ["pick up the cube", "open the drawer"][:size],
    }


@pytest.mark.parametrize("action_dim, tokens", [(7, 14), (8, 28), (14, 28)])
def test_action_tokens_cover_every_column_group(policy, action_dim, tokens):
    actions = batch(action_dim)["action"]
    action_tokens = policy.action_tokenizer.encode(actions)
    assert action_tokens.shape == (2, tokens) and action_tokens.min() >= 0 and action_tokens.max() < 25600
    assert policy.action_tokenizer.decode(action_tokens, action_dim).shape == actions.shape


@pytest.mark.parametrize("steps, tokens", [(1, 4), (8, 4), (9, 7), (16, 7), (24, 11), (32, 14)])
def test_a_token_prefix_decodes_the_first_steps_of_the_chunk(policy, steps, tokens):
    action_tokens = policy.action_tokenizer.encode(batch()["action"])
    count, decoded_steps = policy.action_tokenizer.token_count(steps, action_dim=7)
    assert count == tokens and decoded_steps >= steps
    torch.testing.assert_close(
        policy.action_tokenizer.decode(action_tokens[:, :count], 7)[:, :decoded_steps],
        policy.action_tokenizer.decode(action_tokens, 7)[:, :decoded_steps],
    )


def test_the_streaming_decoder_returns_each_latent_as_its_tokens_arrive(policy):
    action_tokens = policy.action_tokenizer.encode(batch()["action"])
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


def test_the_tokenizer_derives_the_token_counts_its_config_lists(policy):
    assert policy.action_tokenizer.vocab_size == policy.config.action_tokenizer_config["vocab_size"]
    documented = policy.config.action_tokenizer_config["prefix_token_counts"]
    assert {
        str(steps): count for steps, count in policy.action_tokenizer.prefix_token_counts.items()
    } == documented


def test_another_nat_tokenizer_swaps_in_through_the_config_alone(tmp_path):
    """16 steps in 2 latents, scales [2, 1] of 100 codes: tokens (scale 0, latent 0), (1, 0), (1, 1)."""
    tokenizer_folder = tmp_path / "tokenizer"
    tokenizer_folder.mkdir()
    (tokenizer_folder / "config.json").write_text(json.dumps(SMALL_TOKENIZER))
    save_file(NATTokenizer(SMALL_TOKENIZER).state_dict(), tokenizer_folder / "model.safetensors")
    small = NATVLAPolicy(config(7, action_tokenizer=str(tokenizer_folder), chunk_size=16, n_action_steps=8))
    assert small.action_tokenizer.vocab_size == 200 and small.action_tokenizer.prefix_token_counts == {
        8: 2,
        16: 3,
    }
    assert small.vocabulary.size == 65536 + 6 + 256 + 3 + 200
    actions = batch()["action"][:, :16]
    small.train()
    loss, _ = small.forward({**batch(), "action": actions})
    loss.backward()
    assert torch.isfinite(loss) and small.action_tokenizer.encode(actions).max() < 200
    small.save_pretrained(tmp_path / "policy")
    reloaded = NATVLAPolicy.from_pretrained(tmp_path / "policy", config=small.config)
    torch.testing.assert_close(reloaded.action_tokenizer.state_dict(), small.action_tokenizer.state_dict())
    observation = {name: value for name, value in batch().items() if name != "action"}
    assert reloaded.action_tokenizer.token_count(5, action_dim=7) == (2, 8)
    assert reloaded.predict_action_chunk(observation).shape == (2, 16, 7)
    streamed = torch.stack([reloaded.select_action(observation) for _ in range(8)], 1)
    torch.testing.assert_close(streamed, reloaded.predict_action_chunk(observation, steps=8))


def test_prompt_holds_the_setup_state_control_mode_and_action_output(policy):
    prompts, _ = policy.prompts_and_image_inputs(batch())
    vocabulary = policy.vocabulary
    for prompt in prompts:
        assert prompt[0] == policy.processor.tokenizer.convert_tokens_to_ids("<|startoftext|>")
        assert set(range(vocabulary.setup_start, vocabulary.first_state_bin)) <= set(prompt)
        assert prompt[-1] == vocabulary.action_output
        assert sum(vocabulary.first_state_bin <= token < vocabulary.action_output for token in prompt) == 7


def test_loss_trains_only_the_added_rows_of_the_embedding(policy):
    policy.train()
    policy.zero_grad()
    loss, parts = policy.forward(batch())
    loss.backward()
    gradient = policy.vlm.get_input_embeddings().weight.grad
    assert torch.isfinite(loss) and parts["z_loss"] > 0
    first_added_row = policy.vocabulary.setup_start
    assert gradient[:first_added_row].abs().max() == 0 and gradient[first_added_row:].abs().max() > 0
    assert not any(parameter.requires_grad for parameter in policy.action_tokenizer.parameters())


def test_predicted_chunks_and_queued_actions(policy):
    observation = {name: value for name, value in batch().items() if name != "action"}
    assert policy.predict_action_chunk(observation).shape == (2, 32, 7)
    assert policy.predict_action_chunk(observation, steps=8).shape == (2, 8, 7)
    policy.config.n_action_steps = 8
    policy.reset()
    actions = [policy.select_action(observation) for _ in range(9)]
    policy.config.n_action_steps = 32
    policy.reset()
    assert all(action.shape == (2, 7) for action in actions) and not torch.equal(actions[7], actions[8])


def test_streamed_actions_equal_the_predicted_chunk(policy):
    observation = {name: value for name, value in batch().items() if name != "action"}
    policy.config.n_action_steps = 10
    policy.reset()
    try:
        streamed = torch.stack([policy.select_action(observation) for _ in range(11)], 1)
    finally:
        policy.config.n_action_steps = 32
        policy.reset()
    chunk = policy.predict_action_chunk(observation, steps=10)
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


def test_executing_actions_start_the_next_chunk_through_their_tokens(policy):
    observation = {name: value for name, value in batch().items() if name != "action"}
    prompts, generate = [], policy.vlm.generate
    policy.vlm.generate = lambda **arguments: (
        prompts.append(arguments["input_ids"].cpu()) or generate(**arguments)
    )
    policy.config.n_action_steps, policy.config.rtc_steps = 8, 8
    policy.reset()
    try:
        actions = torch.stack([policy.select_action(observation) for _ in range(16)], 1)
    finally:
        del policy.vlm.generate
        policy.config.n_action_steps, policy.config.rtc_steps = 32, 0
        policy.reset()

    assert len(prompts) == 3 and actions.shape == (2, 16, 7)
    for chunk, executing in ((1, actions[:, :8]), (2, actions[:, 8:])):
        padded_actions = torch.cat([executing, executing[:, -1:].expand(-1, 24, -1)], 1)
        prefix = policy.action_tokenizer.encode(padded_actions)[
            :, : policy.action_tokenizer.prefix_token_counts[8]
        ].cpu()
        assert torch.equal(
            prompts[chunk][:, -len(prefix[0]) :], prefix + policy.vocabulary.first_action_token
        )


def test_rtc_steps_must_fit_whole_latents_in_the_chunk():
    with pytest.raises(ValueError, match="rtc_steps"):
        config(7, rtc_steps=8)
    with pytest.raises(ValueError, match="whole latents"):
        NATVLAPolicy(config(7, n_action_steps=16, rtc_steps=4))


def test_generation_writes_only_action_tokens(policy):
    captured = []
    decode = policy.action_tokenizer.decode
    policy.action_tokenizer.decode = lambda tokens, action_dim: (
        captured.append(tokens) or decode(tokens, action_dim)
    )
    try:
        policy.predict_action_chunk({name: value for name, value in batch().items() if name != "action"})
    finally:
        del policy.action_tokenizer.decode
    assert (
        captured[-1].shape == (2, 14)
        and min(map(torch.min, captured)) >= 0
        and max(map(torch.max, captured)) < 25600
    )


def test_save_and_reload_keep_weights_and_predictions(policy, tmp_path):
    observation = {name: value for name, value in batch(size=1).items() if name != "action"}
    policy.save_pretrained(tmp_path)
    reloaded = NATVLAPolicy.from_pretrained(tmp_path, config=policy.config)
    for (name, tensor), reloaded_tensor in zip(
        policy.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        assert torch.equal(tensor, reloaded_tensor), name
    torch.testing.assert_close(
        reloaded.predict_action_chunk(observation), policy.predict_action_chunk(observation)
    )


@pytest.mark.skipif(
    "NAT_VLA_REFERENCE_CASES" not in os.environ, reason="needs reference chunks on LIBERO frames"
)
def test_released_weights_reproduce_reference_chunks():
    """NAT_VLA_REFERENCE_CASES: a torch file of LIBERO cases (images, normalized_state, task, chunk)."""
    cases = torch.load(os.environ["NAT_VLA_REFERENCE_CASES"], weights_only=False)
    libero = config(8, setup="single franka robotic arm in libero", control_mode="delta end-effector pose")
    libero.input_features = {
        "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
    }
    libero.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    released = NATVLAPolicy.from_pretrained(RELEASED_WEIGHTS, config=libero)
    errors = []
    for case in cases:
        observation = {
            name: torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
            for name, image in case["images"].items()
        }
        observation |= {
            "observation.state": torch.as_tensor(case["normalized_state"], dtype=torch.float32)[None],
            "task": [case["task"]],
        }
        errors.append(
            float(
                (released.predict_action_chunk(observation)[0] - torch.as_tensor(case["chunk"])).abs().max()
            )
        )
    assert max(errors) < 0.01, errors
