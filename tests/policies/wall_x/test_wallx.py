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

"""Test script to verify Wall-X policy integration with LeRobot"""

from types import SimpleNamespace

import pytest
import torch
from packaging import version

# Skip if required dependencies are not available
pytest.importorskip("peft")
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.qwen2_5_vision import Qwen2_5VisionGrid  # noqa: E402
from lerobot.policies.common.qwen_vision import prepare_image_grid  # noqa: E402
from lerobot.policies.factory import make_policy_config, make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_x import (
    WallXConfig,  # noqa: E402
    modeling_wall_x,  # noqa: E402
)
from lerobot.policies.wall_x.constant import WALL_X_PROMPT_SEGMENTS  # noqa: E402
from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction, WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import (  # noqa: E402
    WallXPromptProcessorStep,
    make_wall_x_pre_post_processors,
)
from lerobot.policies.wall_x.qwen_model import (  # noqa: E402
    Qwen2_5_VLConfig,
    Qwen2_5_VLMoEModel,
    Qwen2_5_VLTextConfig,
    SparseMoeBlock,
)
from lerobot.policies.wall_x.utils import get_wallx_normal_text, preprocesser_call  # noqa: E402
from lerobot.processor import (  # noqa: E402
    RenderRuntimeMessagesStep,
    RenderTrainingMessagesStep,
    batch_to_transition,
    transition_to_batch,
)
from lerobot.utils.constants import MESSAGES_RENDERED, QUERY_KIND, QUERY_TEXT  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.policies.common.fixed_grid_checks import (  # noqa: E402
    count_grid_builds,
    data_dependent_nodes,
    save_and_load,
)
from tests.utils import require_cuda, require_hf_token  # noqa: E402


def test_from_pretrained_logs_non_strict_state_dict_keys(monkeypatch):
    class FakeConfig:
        text_config = SimpleNamespace(pad_token_id=None)
        pad_token_id = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeTokenizer:
        pad_token_id = 0

        def __len__(self):
            return 1

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeProcessor()

    class FakeModel(Qwen2_5_VLMoEForAction):
        config_class = FakeConfig

        def __init__(self, *args, **kwargs):
            pass

        def resize_token_embeddings(self, *args, **kwargs):
            pass

        def load_state_dict(self, *args, **kwargs):
            assert kwargs["strict"] is False
            return ["missing.weight"], ["unexpected.weight"]

    logged = []
    monkeypatch.setattr(
        modeling_wall_x.Qwen2_5_VLMoEModel,
        "_require_eager_attention",
        staticmethod(lambda *args: None),
    )
    monkeypatch.setattr(modeling_wall_x, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(modeling_wall_x, "cached_file", lambda *args, **kwargs: "weights.safetensors")
    monkeypatch.setattr(modeling_wall_x, "load_file", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        modeling_wall_x,
        "log_model_loading_keys",
        lambda missing, unexpected: logged.append((missing, unexpected)),
    )

    FakeModel.from_pretrained("example/wall-x")

    assert logged == [(["missing.weight"], ["unexpected.weight"])]


def test_moe_model_captures_requested_hidden_states_and_attentions():
    hidden_size = 16
    expert_config = {
        "hidden_size": hidden_size,
        "intermediate_size": 32,
        "hidden_act": "silu",
    }
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [1, 1, 0],
        },
        num_experts=2,
        experts=[expert_config, expert_config],
        dim_inputs=(hidden_size, hidden_size),
        mlp_moe=True,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLMoEModel(config)
    input_ids = torch.tensor([[1, 2, 3]])

    output = model(
        input_ids=input_ids,
        moe_token_types=torch.zeros_like(input_ids),
        output_hidden_states=True,
        output_attentions=True,
    )

    assert len(output.hidden_states) == config.num_hidden_layers + 1
    assert len(output.attentions) == config.num_hidden_layers


def test_moe_block_with_token_counts_routes_like_the_token_mask():
    def expert(size):
        return {"hidden_size": size, "intermediate_size": 32, "hidden_act": "silu"}

    # The third expert gets no tokens and has no width, as with the class defaults of 4 experts and 2 widths.
    config = SimpleNamespace(experts=[expert(16), expert(8), expert(8)], dim_inputs=(16, 8))
    block = SparseMoeBlock(config, num_experts=3)
    hidden_states = torch.randn(2, 7, 16)
    token_types = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])

    with torch.no_grad():
        assert torch.equal(block(hidden_states, token_types, (4, 3)), block(hidden_states, token_types))


def _make_unloaded_policy(**config_values):
    policy = WallXPolicy.__new__(WallXPolicy)
    torch.nn.Module.__init__(policy)
    config_values.setdefault("text_temperature", 0.0)
    config_values.setdefault("text_top_p", 1.0)
    policy.config = SimpleNamespace(chunk_size=3, **config_values)
    return policy


def _tiny_diffusion_model(
    bfloat16: bool = True, vision_attn_implementation: str = "sdpa"
) -> Qwen2_5_VLMoEForAction:
    def expert(size):
        return {"hidden_size": size, "intermediate_size": 2 * size, "hidden_act": "silu"}

    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
            "num_experts": 2,
            "experts": [expert(32), expert(16)],
            "dim_inputs": [32, 16],
            "mlp_moe": True,
            "dof_config": {"arm": 3},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        agent_pos_config={"arm": 4},
        image_token_id=40,
        video_token_id=43,
        vision_start_token_id=41,
        vision_end_token_id=42,
    )
    config._attn_implementation = "eager"
    token_ids = {"<|action|>": 31, "<|propri|>": 32}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=token_ids.__getitem__))
    torch.manual_seed(0)
    model = Qwen2_5_VLMoEForAction(
        config, processor=processor, vision_attn_implementation=vision_attn_implementation
    ).eval()
    if bfloat16:
        model.to_bfloat16_for_selected_params()
    return model


def _diffusion_batch(horizon: int) -> dict[str, torch.Tensor]:
    # The second prompt is left padded, which WALL-X positions differently from text.
    prompt = [41, 40, 40, 42, 7, 32, *[31] * horizon, 2]
    input_ids = torch.tensor([[4, *prompt], [0, *prompt]])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "moe_token_types": input_ids.eq(31),
        "pixel_values": torch.randn(16, 3 * 2 * 2 * 2),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 4]]),
        "dof_mask": torch.ones(2, horizon, 3),
        "proprioception": torch.randn(2, 4),
        "agent_pos_mask": torch.ones(2, 4),
    }


def test_diffusion_prediction_runs_with_float32_weights():
    model = _tiny_diffusion_model(bfloat16=False)
    batch = _diffusion_batch(3)
    text_inputs = []
    model.model.register_forward_pre_hook(
        lambda module, args, kwargs: text_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True
    )
    actions = model.predict(
        **batch, predict_mode="diffusion", pred_horizon=3, action_dim=3, num_inference_timesteps=2
    )["predict_action"]
    assert actions.shape == (2, 3, 3)
    assert actions.isfinite().all()
    # The state embedding goes in at float32, not rounded through bfloat16.
    state = model.action_preprocessor.proprioception_proj(batch["proprioception"], batch["agent_pos_mask"])
    state_tokens = batch["input_ids"] == model.action_token_id_set["propri_token_id"]
    assert torch.equal(text_inputs[0][state_tokens], state.reshape(-1, state.shape[-1]))


def _diffusion_policy(model):
    policy = _make_unloaded_policy(
        prediction_mode="diffusion",
        max_action_dim=3,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    policy._queues = {}
    policy._vision_grid = None
    policy.model = model
    return policy


requires_torch_2_9 = pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.9.0"),
    reason="torch.export before PyTorch 2.9 cannot trace transformers' causal mask with WALL-X's action mask function",
)


def _export_prediction(policy, batch, noise):
    keys = sorted(batch)

    class Predict(torch.nn.Module):
        def forward(self, noise, *tensors):
            return policy.predict_action_chunk(dict(zip(keys, tensors, strict=True)), noise=noise)

    # The public entry point exports under no_grad: with grad enabled, transformers' causal-mask code
    # does not trace.
    with torch.no_grad():
        program = torch.export.export(Predict(), (noise.clone(), *(batch[key] for key in keys)), strict=False)

    def run(inputs, lowered=program):
        return lowered.module()(noise.clone(), *(inputs[key] for key in keys))

    return program, run


@requires_torch_2_9
def test_prepared_diffusion_prediction_exports_and_matches_eager(monkeypatch):
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    input_ids = batch["input_ids"]
    policy = _diffusion_policy(model)
    torch.manual_seed(0)
    noise = torch.randn(2, 3, 3)
    expected = policy.predict_action_chunk(batch, noise=noise.clone())
    with pytest.raises(ValueError, match="moe_token_types must mark exactly the action tokens"):
        policy.predict_action_chunk({**batch, "moe_token_types": input_ids.eq(32)}, noise=noise.clone())

    grid = Qwen2_5VisionGrid.build(model.visual, batch["image_grid_thw"])
    static = policy._static_grid_inputs(batch, grid)
    stock_positions, _ = model.get_rope_index(
        input_ids, batch["image_grid_thw"], attention_mask=batch["attention_mask"]
    )
    assert torch.equal(static["position_ids"], stock_positions)
    with pytest.raises(ValueError, match="image inputs only"):
        policy._static_grid_inputs({**batch, "video_grid_thw": batch["image_grid_thw"]}, grid)

    builds = count_grid_builds(monkeypatch)
    with prepare_image_grid():
        for _ in range(2):  # The second call reuses the prepared grid.
            assert torch.equal(policy.predict_action_chunk(batch, noise=noise.clone()), expected)
    assert len(builds) == 1

    # The switch that turns off the transformers checks must leave the exported checks in place.
    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    program, run = _export_prediction(policy, batch, noise)
    torch.testing.assert_close(run(batch), expected, rtol=0, atol=0)
    assert len(builds) == 1
    # Boolean-mask writes and reads of the action and proprioception tokens would make shapes data-dependent.
    assert data_dependent_nodes(program) == []
    # torch.export cannot save the vmap in transformers' mask code until the program is decomposed.
    lowered = save_and_load(program.run_decompositions())
    # The decompositions round differently in bfloat16.
    torch.testing.assert_close(run(batch, lowered), expected, rtol=0, atol=2e-3)
    # Each malformed input trips one check, since the exported program raises without the check's message: one
    # image token fewer than image features, a state token moved to the other sample, and moe_token_types that
    # mark the state tokens.
    fewer_images = {**batch, "input_ids": input_ids.clone()}
    fewer_images["input_ids"][:, 2] = 7
    moved_state = {**batch, "input_ids": input_ids.clone()}
    moved_state["input_ids"][0, 6], moved_state["input_ids"][1, 5] = 7, 32
    wrong_routing = {**batch, "moe_token_types": input_ids.eq(32)}
    for exported in (program, lowered):
        for wrong in (fewer_images, moved_state, wrong_routing):
            with pytest.raises(RuntimeError, match="Runtime assertion failed"):
                run(wrong, exported)
    # The eager check raises with the switch set, too.
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        policy.predict_action_chunk(fewer_images, noise=noise.clone())


@requires_torch_2_9
def test_preparation_and_export_refuse_hidden_vision_tokens():
    policy = _diffusion_policy(_tiny_diffusion_model())
    batch = {**_diffusion_batch(3), "image_grid_thw": torch.tensor([[1, 2, 4], [1, 4, 2]])}
    noise = torch.randn(2, 3, 3)

    def hide(tokens):
        attention_mask = batch["attention_mask"].clone()
        attention_mask[0, tokens] = 0
        return {**batch, "attention_mask": attention_mask}

    # In the first prompt: the vision start token, an image token, the vision end token, and every token.
    refused = (hide(1), hide(2), hide(4), hide(slice(None)))
    with prepare_image_grid():
        policy.predict_action_chunk(batch, noise=noise.clone())
        for inputs in refused:
            with pytest.raises(ValueError, match="must hide only padding"):
                policy.predict_action_chunk(inputs, noise=noise.clone())

    _, run = _export_prediction(policy, batch, noise)
    for inputs in refused:
        with pytest.raises(RuntimeError, match="Runtime assertion failed"):
            run(inputs)


@require_cuda
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.10.0"),
    reason="WALL-X's packed varlen vision attention needs PyTorch 2.10",
)
def test_export_keeps_the_packed_varlen_vision_attention():
    model = _tiny_diffusion_model(vision_attn_implementation="auto").to("cuda")
    policy = _diffusion_policy(model)
    # Images of 8 and 24 patches, so the packed sequences differ in length, and the second image does not
    # divide into whole attention windows.
    prompts = [[41, *[40] * tokens, 42, 7, 32, 31, 31, 31, 2] for tokens in (2, 6)]
    input_ids = torch.tensor([[0] * 4 + prompts[0], prompts[1]])
    batch = {
        **_diffusion_batch(3),
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "moe_token_types": input_ids.eq(31),
        "pixel_values": torch.randn(32, 3 * 2 * 2 * 2),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 6, 4]]),
    }
    batch = {key: value.cuda() for key, value in batch.items()}
    noise = torch.randn(2, 3, 3, device="cuda")
    with prepare_image_grid():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
    if {block.attn._resolved_backend for block in model.visual.blocks} != {"varlen"}:
        pytest.skip("this GPU runs WALL-X vision attention on SDPA")

    program, run = _export_prediction(policy, batch, noise)
    torch.testing.assert_close(run(batch), expected, rtol=0, atol=0)
    # The exported vision tower runs the same packed varlen kernel as eager, not an SDPA stand-in.
    targets = {
        str(node.target)
        for module in program.graph_module.modules()
        if hasattr(module, "graph")
        for node in module.graph.nodes
    }
    assert any("varlen_attn" in target for target in targets)


def test_eager_diffusion_prediction_builds_no_rope_positions(monkeypatch):
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    calls = []

    class Model:
        def __init__(self):
            self.visual, self.config = model.visual, model.config

        def __call__(self, **inputs):
            calls.append(inputs)
            return {"predict_action": torch.zeros(2, 3, 3)}

    def no_rope_positions(*args, **kwargs):
        raise AssertionError("eager prediction uses the model's own rope positions")

    monkeypatch.setattr("lerobot.policies.wall_x.modeling_wall_x.qwen_image_rope_index", no_rope_positions)
    policy = _diffusion_policy(Model())
    policy.predict_action_chunk(batch)
    assert policy._vision_grid is None  # nothing is cached outside the preparation
    with prepare_image_grid():
        policy.predict_action_chunk(batch)
    assert policy._vision_grid.grid_thw == batch["image_grid_thw"].tolist()
    assert all("vision_grid" not in call and "position_ids" not in call for call in calls)


def test_diffusion_rejects_a_sample_without_its_state_token():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    batch["input_ids"] = batch["input_ids"].clone()
    assert batch["input_ids"][0, 6] == 32
    batch["input_ids"][0, 6] = 7  # sample 0 loses its state token; its state must not go to sample 1
    moved = batch["input_ids"].clone()
    moved[1, 5] = 32  # and sample 1 gets a second one, so only the per-sample count is wrong
    for input_ids in (batch["input_ids"], moved):
        with pytest.raises(ValueError, match="Proprioception features and state tokens do not match"):
            model.predict(
                **{**batch, "input_ids": input_ids},
                predict_mode="diffusion",
                pred_horizon=3,
                action_dim=3,
                num_inference_timesteps=2,
            )


def test_diffusion_reads_the_action_hidden_states_in_token_order():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    hidden_states, read = [], []
    model.model.register_forward_hook(
        lambda module, args, output: hidden_states.append(output.last_hidden_state)
    )
    model.action_preprocessor.action_proj_back.register_forward_pre_hook(
        lambda module, args: read.append(args[0])
    )
    model.predict(**batch, predict_mode="diffusion", pred_horizon=3, action_dim=3, num_inference_timesteps=2)
    action_mask = batch["input_ids"] == 31
    assert read
    for hidden, actions in zip(hidden_states[-len(read) :], read, strict=True):
        assert torch.equal(actions, hidden[action_mask].view(2, 3, -1).float())


def test_diffusion_checks_the_action_token_count_per_sample():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    input_ids = batch["input_ids"].clone()
    input_ids[0, 7], input_ids[1, 5] = 7, 31  # one action token moves from sample 0 to sample 1
    batch = {**batch, "input_ids": input_ids, "moe_token_types": input_ids.eq(31)}
    with pytest.raises(ValueError, match="Expected 3 action tokens per sample"):
        model.predict(
            **batch, predict_mode="diffusion", pred_horizon=3, action_dim=3, num_inference_timesteps=2
        )


def test_diffusion_refuses_inputs_that_do_not_fit_the_prompts():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(1)
    settings = {"predict_mode": "diffusion", "pred_horizon": 1, "action_dim": 3, "num_inference_timesteps": 2}
    # The writes of the noise, the dof mask and the state rows would put one sample's extra rows into the next.
    with pytest.raises(ValueError, match="noise must have shape"):
        model.predict(**batch, **settings, noise=torch.randn(2, 3, 3))
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        model.predict(**{**batch, "dof_mask": torch.ones(2, 3, 3)}, **settings)
    extra_state = {**batch, "proprioception": torch.randn(3, 4), "agent_pos_mask": torch.ones(3, 4)}
    with pytest.raises(ValueError, match="Proprioception has 3 samples but input_ids has 2"):
        model.predict(**extra_state, **settings)
    with pytest.raises(ValueError, match="moe_token_types must be provided"):
        model.predict(**{**batch, "moe_token_types": None}, **settings)


def test_export_before_pytorch_2_9_explains(monkeypatch):
    policy = _diffusion_policy(_tiny_diffusion_model())
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(RuntimeError, match="needs PyTorch 2.9 or later"):
        _export_prediction(policy, _diffusion_batch(3), torch.randn(2, 3, 3))


@requires_torch_2_9
def test_preparation_keeps_its_grid_through_calls_without_images_or_that_fail():
    policy = _diffusion_policy(_tiny_diffusion_model())
    batch = _diffusion_batch(3)
    noise = torch.randn(2, 3, 3)
    expected = policy.predict_action_chunk(batch, noise=noise.clone())
    text_ids = batch["input_ids"].masked_fill(torch.isin(batch["input_ids"], torch.tensor([40, 41, 42])), 7)
    text_only = {**batch, "input_ids": text_ids, "pixel_values": None, "image_grid_thw": None}
    # Another grid with the same image and token counts, in calls that fail after the grid is built: one hides
    # the vision start tokens, one passes noise of the wrong shape.
    changed = {**batch, "image_grid_thw": torch.tensor([[1, 4, 2], [1, 4, 2]])}
    hidden_start = {
        **changed,
        "attention_mask": changed["attention_mask"].index_fill(1, torch.tensor([1]), 0),
    }
    with prepare_image_grid():
        policy.predict_action_chunk(batch, noise=noise.clone())
        policy.predict_action_chunk(text_only, noise=noise.clone())
        with pytest.raises(ValueError, match="must hide only padding"):
            policy.predict_action_chunk(hidden_start, noise=noise.clone())
        with pytest.raises(ValueError, match="noise must have shape"):
            policy.predict_action_chunk(changed, noise=torch.randn(2, 4, 3))
    policy.predict_action_chunk(text_only, noise=noise.clone())
    _, run = _export_prediction(policy, batch, noise)
    torch.testing.assert_close(run(batch), expected)


def _model_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "moe_token_types": torch.zeros(1, 3, dtype=torch.bool),
    }


def test_recipe_prompt_targets_only_selected_assistant_and_keeps_action_supervision():
    step = WallXPromptProcessorStep(image_keys=["observation.images.face_view"], chunk_size=3)
    segments, predicts_action = step._recipe_segments(
        [
            {"role": "user", "content": "What should the robot do?"},
            {"role": "assistant", "content": "Reach for the cup."},
        ],
        ["high_level", "low_level"],
        [1],
        "pick up the cup",
        ["front view"],
    )

    prompt = "".join(segment["text"] for segment in segments)
    spans = []
    offset = 0
    for segment in segments:
        if segment["target"]:
            spans.append((offset, offset + len(segment["text"])))
        offset += len(segment["text"])
    assert predicts_action
    assert len(spans) == 1
    assert prompt[slice(*spans[0])] == "Reach for the cup.<|im_end|>"
    assert prompt.count("<|action|>") == 3
    assert "Proprioception: <|propri|>" in prompt


def test_recipe_and_generation_use_the_same_subtask_user_turn():
    step = WallXPromptProcessorStep(image_keys=["observation.images.face_view"], chunk_size=3)
    messages = [{"role": "user", "content": "clear the table\nPredict the next action in language.\n"}]

    training, _ = step._recipe_segments(messages, ["high_level"], [], "", ["front view"])
    generation = step._generation_segments(messages, ["front view"])

    assert "".join(segment["text"] for segment in generation) == (
        "".join(segment["text"] for segment in training) + "<|im_start|>assistant\n"
    )


def test_low_level_recipe_fallback_uses_the_native_action_prompt():
    step = WallXPromptProcessorStep(image_keys=["observation.images.face_view"], chunk_size=3)
    rendered = step.complementary_data(
        {
            "task": ["clear the table"],
            MESSAGES_RENDERED: [[{"role": "user", "content": "clear the table"}]],
            "message_streams": [["low_level"]],
            "target_message_indices": [[]],
        }
    )

    expected, generated_subtask = get_wallx_normal_text(
        {"instruction": "clear the table"},
        action_chunk_size=3,
        frame_idx=0,
        priority_order=None,
        img_keys=["observation.images.face_view"],
        generate_subtask_ratio=0.0,
    )

    assert not generated_subtask
    assert "".join(segment["text"] for segment in rendered[WALL_X_PROMPT_SEGMENTS][0]) == expected


def test_text_targets_follow_expanded_image_placeholders():
    class Tokenized(dict):
        __getattr__ = dict.__getitem__

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        @staticmethod
        def encode(value):
            return [10_000 if value == "<|action|>" else 10_001]

        @staticmethod
        def __call__(texts, *, return_offsets_mapping, **kwargs):
            del kwargs
            length = len(texts[0])
            tokenized = Tokenized(
                input_ids=torch.arange(1, length + 1).unsqueeze(0),
                attention_mask=torch.ones(1, length, dtype=torch.long),
            )
            if return_offsets_mapping:
                tokenized["offset_mapping"] = torch.tensor([[[index, index + 1] for index in range(length)]])
            return tokenized

    class ImageProcessor:
        merge_size = 1

        @staticmethod
        def __call__(**kwargs):
            del kwargs
            return {"image_grid_thw": torch.tensor([[1, 2, 2]])}

    processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=ImageProcessor())
    text = "prefix <|image_pad|> target"
    target_start = text.index("target")
    inputs = preprocesser_call(
        processor,
        text=text,
        images=[[torch.zeros(3, 2, 2)]],
        target_spans=[[(target_start, target_start + len("target"))]],
    )

    final_target_start = target_start + len("<|image_pad|>") * 3
    assert torch.all(inputs.labels[0, :final_target_start] == -100)
    assert torch.equal(
        inputs.labels[0, final_target_start : final_target_start + len("target")],
        inputs.input_ids[0, final_target_start : final_target_start + len("target")],
    )


def test_policy_combines_text_and_flow_losses_with_configured_weights(monkeypatch):
    policy = _make_unloaded_policy(flow_loss_weight=2.0, text_loss_weight=0.25)
    policy.model = SimpleNamespace(
        __call__=None,
    )
    outputs = SimpleNamespace(
        loss=torch.tensor(99.0),
        flow_loss=torch.tensor(3.0),
        cross_entropy_loss=torch.tensor(4.0),
        channel_loss_dict=None,
    )
    policy.model = lambda **kwargs: outputs
    monkeypatch.setattr(policy, "_pretokenized_inputs", lambda batch, **kwargs: batch)

    loss, metrics = policy.forward({**_model_inputs(), "messages_rendered": [[{"role": "user"}]]})

    assert loss.item() == 7.0
    assert metrics["flow_loss"].item() == 3.0
    assert metrics["cross_entropy_loss"].item() == 4.0


def test_policy_preserves_original_action_only_loss(monkeypatch):
    policy = _make_unloaded_policy(flow_loss_weight=2.0, text_loss_weight=0.25)
    outputs = SimpleNamespace(
        loss=torch.tensor(9.0),
        flow_loss=torch.tensor(3.0),
        cross_entropy_loss=torch.tensor(4.0),
        channel_loss_dict=None,
    )
    policy.model = lambda **kwargs: outputs
    monkeypatch.setattr(policy, "_pretokenized_inputs", lambda batch, **kwargs: batch)

    loss, _ = policy.forward(_model_inputs())

    assert loss.item() == 9.0


def test_generation_preparation_synthesizes_cache_positions():
    input_ids = torch.tensor([[1, 2, 3]])

    prepared = Qwen2_5_VLMoEForAction.prepare_inputs_for_generation(
        SimpleNamespace(),
        input_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    assert torch.equal(prepared["cache_position"], torch.arange(3))


def test_policy_answers_a_rendered_vqa_question():
    assert WallXPolicy.generate_text is not PreTrainedPolicy.generate_text
    assert WallXPolicy.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert not hasattr(WallXPolicy, "generate_texts")

    question = "What is the capital of France?"
    rendered = RenderRuntimeMessagesStep()(batch_to_transition({QUERY_KIND: "vqa", QUERY_TEXT: question}))
    messages = transition_to_batch(rendered)["messages_rendered"]
    assert messages == [{"role": "user", "content": question}]
    assert (
        "".join(
            segment["text"]
            for segment in WallXPromptProcessorStep(
                image_keys=["observation.images.face_view"], chunk_size=3
            )._generation_segments(messages, ["front view"])
        )
        == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nObservation: front view: "
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Instruction: What is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def batch_decode(token_ids, **kwargs):
            del kwargs
            assert torch.equal(token_ids, torch.tensor([[7, 8]]))
            return ["Paris"]

    class Model:
        processor = SimpleNamespace(tokenizer=Tokenizer())

        @staticmethod
        def generate(input_ids, **kwargs):
            del kwargs
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

    policy = _make_unloaded_policy()
    policy.model = Model()
    batch = _model_inputs()

    assert policy.generate_text(batch) == "Paris"
    assert policy.supports_text_generation()


def test_wall_x_runtime_query_is_rendered_by_the_default_input_pipeline():
    config = WallXConfig(device="cpu")
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    stats = {
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }

    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    batch = transition_to_batch(
        preprocessor.steps[0](
            batch_to_transition(
                {
                    "observation.state": torch.zeros(7),
                    "observation.images.face_view": torch.zeros(3, 8, 8),
                    "task": "current subtask",
                    QUERY_KIND: "next_subtask",
                    QUERY_TEXT: "clear the table",
                }
            )
        )
    )

    assert isinstance(preprocessor.steps[0], RenderRuntimeMessagesStep)
    assert isinstance(preprocessor.steps[1], RenderTrainingMessagesStep)
    assert any(isinstance(step, WallXPromptProcessorStep) for step in preprocessor.steps)
    assert batch["messages_rendered"] == [
        {"role": "user", "content": "clear the table\nPredict the next action in language.\n"}
    ]
    assert QUERY_KIND not in batch
    assert QUERY_TEXT not in batch


def test_wall_x_loads_an_explicit_external_recipe(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("messages:\n  - {role: user, content: '${task}', stream: low_level}\n")
    config = WallXConfig(device="cpu", recipe_path=str(path))

    assert config.recipe is not None
    assert config.recipe["messages"] is not None


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = WallXConfig(device="cuda")

    # Set up input_features and output_features in the config
    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(7,),
        ),
        "observation.images.face_view": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "observation.images.face_view": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = WallXPolicy(config)
    preprocessor, postprocessor = make_wall_x_pre_post_processors(config=config, dataset_stats=dataset_stats)
    # Test forward pass with dummy data
    batch_size = 1
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test inference
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="wall_x",
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise


def test_subtask_prompt_is_token_exact_with_the_trained_template():
    """WALL-OSS declares its mode in the user turn, so the template must match upstream."""
    img_keys = ["observation.images.face_view"]
    upstream, generated_subtask = get_wallx_normal_text(
        {"instruction": "clear the table", "subtask_generation": "pick the cup"},
        action_chunk_size=1,
        frame_idx=0,
        priority_order=None,
        img_keys=list(img_keys),
        generate_subtask_ratio=1.0,
    )
    assert generated_subtask

    ours = "".join(
        segment["text"]
        for segment in WallXPromptProcessorStep(
            image_keys=["observation.images.face_view"], chunk_size=1
        )._generation_segments(
            [{"role": "user", "content": "clear the table\nPredict the next action in language.\n"}],
            ["front view"],
        )
    )

    # Compare the user turn only: upstream appends its own assistant target, ours ends
    # at the generation prompt.
    assert ours.split("<|im_start|>assistant")[0] == upstream.split("<|im_start|>assistant")[0]


@pytest.mark.parametrize("recipe_enabled", [False, True])
def test_recipe_config_round_trip_and_optional_rendering(recipe_enabled):
    import draccus

    config = WallXConfig(device="cpu")
    if not recipe_enabled:
        config.recipe = None
    restored = draccus.decode(WallXConfig, draccus.encode(config))
    assert restored.recipe == config.recipe
    runtime = RenderRuntimeMessagesStep(restored.recipe)
    training = RenderTrainingMessagesStep(restored.recipe)
    if recipe_enabled:
        assert runtime.recipe is not None
        assert training.recipe == runtime.recipe
    else:
        from lerobot.processor.converters import create_transition

        transition = create_transition(action=torch.ones(1), complementary_data={"task": "tidy"})
        assert training(transition) is transition
