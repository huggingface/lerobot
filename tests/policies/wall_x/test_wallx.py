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
from lerobot.policies.factory import make_policy_config, make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_x import (
    WallXConfig,  # noqa: E402
    modeling_wall_x,  # noqa: E402
)
from lerobot.policies.wall_x.constant import (  # noqa: E402
    WALL_X_GENERATION_PROMPT_IDS,
    WALL_X_PROMPT_SEGMENTS,
)
from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction, WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import (  # noqa: E402
    WallXPromptProcessorStep,
    make_wall_x_pre_post_processors,
)
from lerobot.policies.wall_x.qwen_model import (  # noqa: E402
    Qwen2_5_VLConfig,
    Qwen2_5_VLMoEModel,
    Qwen2_5_VLTextConfig,
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
from tests.policies.common.fixed_grid_checks import data_dependent_nodes, save_and_load  # noqa: E402
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


def test_moe_model_routes_by_token_counts_like_the_token_mask():
    def expert(size):
        return {"hidden_size": size, "intermediate_size": 32, "hidden_act": "silu"}

    # The third expert gets no tokens and has no width, as with the class defaults of 4 experts and 2 widths.
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0, "mrope_section": [1, 1, 0]},
        num_experts=3,
        experts=[expert(16), expert(8), expert(8)],
        dim_inputs=(16, 8),
        mlp_moe=True,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLMoEModel(config).eval()
    input_ids = torch.randint(32, (2, 7))
    token_types = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])

    with torch.no_grad():
        expected = model(input_ids=input_ids, moe_token_types=token_types).last_hidden_state
        routed = model(input_ids=input_ids, moe_token_types=token_types, moe_tokens_per_expert=(4, 3))
        assert torch.equal(routed.last_hidden_state, expected)
        with pytest.raises(ValueError, match=r"exactly \(5, 2\) tokens of every sample"):
            model(input_ids=input_ids, moe_token_types=token_types, moe_tokens_per_expert=(5, 2))


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
    policy.model = model
    return policy


requires_torch_2_9 = pytest.mark.skipif(
    version.parse(torch.__version__).release < (2, 9),
    reason="torch.export before PyTorch 2.9 cannot trace transformers' causal mask with WALL-X's action mask function",
)


def _export_prediction(policy, batch, noise, grid):
    keys = sorted(batch)

    class Predict(torch.nn.Module):
        def forward(self, noise, *tensors):
            inputs = dict(zip(keys, tensors, strict=True))
            return policy.predict_action_chunk(inputs, noise=noise, vision_grid=grid)

    # The public entry point exports under no_grad: with grad enabled, transformers' causal-mask code
    # does not trace.
    with torch.no_grad():
        program = torch.export.export(Predict(), (noise.clone(), *(batch[key] for key in keys)), strict=False)

    def run(inputs, lowered=program, noise=noise):
        return lowered.module()(noise.clone(), *(inputs[key] for key in keys))

    return program, run


@requires_torch_2_9
def test_diffusion_prediction_on_a_vision_grid_exports_and_matches_eager(monkeypatch):
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    input_ids = batch["input_ids"]
    policy = _diffusion_policy(model)
    torch.manual_seed(0)
    noise = torch.randn(2, 3, 3)
    expected = policy.predict_action_chunk(batch, noise=noise.clone())

    grid = Qwen2_5VisionGrid.build(model.visual, batch["image_grid_thw"])
    stock_positions, _ = model.get_rope_index(
        input_ids, batch["image_grid_thw"], attention_mask=batch["attention_mask"]
    )
    assert torch.equal(model._grid_position_ids(input_ids, batch["attention_mask"], grid), stock_positions)
    assert torch.equal(policy.predict_action_chunk(batch, noise=noise.clone(), vision_grid=grid), expected)

    # The switch that turns off the transformers checks must leave these checks in place.
    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    program, run = _export_prediction(policy, batch, noise, grid)
    torch.testing.assert_close(run(batch), expected, rtol=0, atol=0)
    # Boolean-mask writes and reads of the action and proprioception tokens would make shapes data-dependent.
    assert data_dependent_nodes(program) == []
    # Another batch of the same shape: other pixels, state and noise, and the padding in the other sample.
    ids = input_ids.clone()
    ids[0, 0], ids[1, 0] = 0, 4
    other = {**batch, "input_ids": ids, "attention_mask": ids.ne(0).long()}
    other |= {"pixel_values": torch.randn(16, 24), "proprioception": torch.randn(2, 4)}
    other_noise = torch.randn(2, 3, 3)
    other_expected = policy.predict_action_chunk(other, noise=other_noise.clone())
    torch.testing.assert_close(run(other, noise=other_noise), other_expected, rtol=0, atol=0)
    # torch.export cannot save the vmap in transformers' mask code until the program is decomposed.
    lowered = save_and_load(program.run_decompositions())
    # The decompositions round differently in bfloat16.
    torch.testing.assert_close(run(batch, lowered), expected, rtol=0, atol=2e-3)
    # One image token fewer than image features, and an action token moved to the other sample (with
    # moe_token_types following it).
    fewer_images = {**batch, "input_ids": input_ids.clone()}
    fewer_images["input_ids"][:, 2] = 7
    moved_action = input_ids.clone()
    moved_action[0, 7], moved_action[1, 5] = 7, 31
    moved_action = {**batch, "input_ids": moved_action, "moe_token_types": moved_action.eq(31)}
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        policy.predict_action_chunk(fewer_images, noise=noise.clone())
    with pytest.raises(ValueError, match="Expected 3 action tokens per sample"):
        policy.predict_action_chunk(moved_action, noise=noise.clone())
    # The program, decomposed and saved, refuses moe_token_types shifted one token earlier, onto the
    # state token, a state token moved to the other sample, an image without its vision start token, a hidden
    # image token and another grid with the same counts.
    moved_state = input_ids.clone()
    moved_state[0, 6], moved_state[1, 5] = 7, 32
    lost_start = input_ids.clone()
    lost_start[0, 1] = 7
    hidden_image = batch["attention_mask"].clone()
    hidden_image[0, 2] = 0
    refused = [
        {**batch, "moe_token_types": input_ids.eq(31).roll(-1, dims=1)},
        {**batch, "input_ids": moved_state},
        {**batch, "input_ids": lost_start},
        {**batch, "attention_mask": hidden_image},
        {**batch, "image_grid_thw": torch.tensor([[1, 4, 2], [1, 4, 2]])},
    ]
    for wrong in refused:
        with pytest.raises(RuntimeError, match="Runtime assertion failed"):
            run(wrong, lowered)


@requires_torch_2_9
def test_vision_grid_must_fit_the_diffusion_call():
    policy = _diffusion_policy(_tiny_diffusion_model())
    batch = _diffusion_batch(3)
    noise = torch.randn(2, 3, 3)
    grid = Qwen2_5VisionGrid.build(policy.model.visual, batch["image_grid_thw"])
    text_ids = batch["input_ids"].masked_fill(torch.isin(batch["input_ids"], torch.tensor([40, 41, 42])), 7)
    text_only = {**batch, "input_ids": text_ids, "pixel_values": None, "image_grid_thw": None}
    with pytest.raises(ValueError, match="vision_grid needs pixel_values and image_grid_thw"):
        policy.predict_action_chunk(text_only, noise=noise.clone(), vision_grid=grid)
    with pytest.raises(ValueError, match="image inputs only"):
        video = {**batch, "video_grid_thw": batch["image_grid_thw"]}
        policy.predict_action_chunk(video, noise=noise.clone(), vision_grid=grid)
    # The same image and token counts with the tokens split 1 and 3: the grid is refused before the rope
    # positions read the tokens.
    ids = torch.tensor([[0, 4, 41, 40, 42, 7, 32, 31, 31, 31, 2], [41, 40, 40, 40, 42, 7, 32, 31, 31, 31, 2]])
    split = {**batch, "input_ids": ids, "attention_mask": ids.ne(0).long(), "moe_token_types": ids.eq(31)}
    split["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 6]])
    with pytest.raises(ValueError, match="differs from the one the vision grid was built for"):
        policy.predict_action_chunk(split, noise=noise.clone(), vision_grid=grid)


_DIFFUSION = {"predict_mode": "diffusion", "pred_horizon": 3, "action_dim": 3, "num_inference_timesteps": 2}


@requires_torch_2_9
def test_the_model_checks_a_call_on_a_vision_grid():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    noise = torch.randn(2, 3, 3)
    grid = Qwen2_5VisionGrid.build(model.visual, batch["image_grid_thw"])
    for attention_mask in (None, batch["attention_mask"][:, None, None, :]):
        with pytest.raises(ValueError, match="needs a 2D attention_mask"):
            model.predict(
                **{**batch, "attention_mask": attention_mask},
                **_DIFFUSION,
                noise=noise.clone(),
                vision_grid=grid,
            )


def test_vision_grid_is_refused_outside_the_diffusion_mode():
    policy = _diffusion_policy(_tiny_diffusion_model())
    policy.config.prediction_mode = "fast"
    batch = _diffusion_batch(3)
    # The fast mode predicts one prompt at a time.
    one_prompt = {
        key: batch[key][:1] for key in ("input_ids", "attention_mask", "moe_token_types", "image_grid_thw")
    }
    one_prompt |= {
        "pixel_values": batch["pixel_values"][:8],
        WALL_X_GENERATION_PROMPT_IDS: torch.tensor([7, 32]),
    }
    grid = Qwen2_5VisionGrid.build(policy.model.visual, one_prompt["image_grid_thw"])
    with pytest.raises(ValueError, match="vision_grid is for the diffusion mode only"):
        policy.predict_action_chunk(one_prompt, vision_grid=grid)


def test_text_prediction_takes_a_state_without_a_state_token():
    model = _tiny_diffusion_model()
    model.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
        batch_decode=lambda ids, **kwargs: [row.tolist() for row in ids],
    )
    input_ids = torch.tensor([[1, 7, 5, 6, 9, 2]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "generation_prompt_ids": torch.tensor([5, 6]),
    }
    text = model.predict(predict_mode="text", **inputs)["predict_output_text"]
    # The prompt has no state token, so the state changes nothing.
    state = {"proprioception": torch.randn(1, 4), "agent_pos_mask": torch.ones(1, 4)}
    assert model.predict(predict_mode="text", **inputs, **state)["predict_output_text"] == text


@pytest.mark.skipif(
    version.parse(torch.__version__).release < (2, 10),
    reason="WALL-X installs its own vision attention from PyTorch 2.10",
)
def test_default_vision_attention_exports_on_the_cpu():
    # The default, "auto", installs WALL-X's attention, which runs SDPA per chunk on the CPU.
    model = _tiny_diffusion_model(vision_attn_implementation="auto")
    policy = _diffusion_policy(model)
    batch = _diffusion_batch(3)
    noise = torch.randn(2, 3, 3)
    expected = policy.predict_action_chunk(batch, noise=noise.clone())
    assert {block.attn._resolved_backend for block in model.visual.blocks} == {"sdpa"}
    grid = Qwen2_5VisionGrid.build(model.visual, batch["image_grid_thw"])
    _, run = _export_prediction(policy, batch, noise, grid)
    torch.testing.assert_close(run(batch), expected, rtol=0, atol=0)


@require_cuda
@pytest.mark.skipif(
    version.parse(torch.__version__).release < (2, 10),
    reason="WALL-X's packed varlen vision attention needs PyTorch 2.10",
)
def test_export_keeps_the_packed_varlen_vision_attention():
    if torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("WALL-X's packed varlen vision attention needs an NVIDIA Ampere GPU or newer")
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
    expected = policy.predict_action_chunk(batch, noise=noise.clone())
    assert {block.attn._resolved_backend for block in model.visual.blocks} == {"varlen"}

    grid = Qwen2_5VisionGrid.build(model.visual, batch["image_grid_thw"])
    program, run = _export_prediction(policy, batch, noise, grid)
    torch.testing.assert_close(run(batch), expected, rtol=0, atol=0)
    decomposed = program.run_decompositions()
    # The decompositions round differently in bfloat16.
    torch.testing.assert_close(run(batch, decomposed), expected, rtol=0, atol=2e-3)
    # The exported vision tower runs the same packed varlen kernel as eager, not an SDPA stand-in, also after
    # decomposition.
    for exported in (program, decomposed):
        targets = {
            str(node.target)
            for module in exported.graph_module.modules()
            if hasattr(module, "graph")
            for node in module.graph.nodes
        }
        assert any("varlen_attn" in target for target in targets)


def test_diffusion_reads_the_action_hidden_states_in_token_order():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(3)
    # Right padding puts the second prompt's action tokens in other columns than the first prompt's.
    input_ids = torch.stack([batch["input_ids"][0], batch["input_ids"][1].roll(-1)])
    batch["input_ids"], batch["attention_mask"] = input_ids, input_ids.ne(0).long()
    batch["moe_token_types"] = input_ids.eq(31)
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


def test_diffusion_refuses_inputs_that_do_not_fit_the_prompts():
    model = _tiny_diffusion_model()
    batch = _diffusion_batch(1)
    settings = {"predict_mode": "diffusion", "pred_horizon": 1, "action_dim": 3, "num_inference_timesteps": 2}
    # The writes of the noise and the state rows would put one sample's extra rows into the next.
    with pytest.raises(ValueError, match="noise must have shape"):
        model.predict(**batch, **settings, noise=torch.randn(2, 3, 3))
    with pytest.raises(ValueError, match=r"dof_mask must have shape \(2, 1, 3\)"):
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
        _export_prediction(policy, _diffusion_batch(3), torch.randn(2, 3, 3), None)
    # A pre-release build of 2.9, as NVIDIA's containers ship it, passes the version check and gets as far as
    # asking for the vision grid.
    monkeypatch.setattr(torch, "__version__", "2.9.0a0+nv25.10")
    with pytest.raises(ValueError, match="needs a vision_grid"):
        _export_prediction(policy, _diffusion_batch(3), torch.randn(2, 3, 3), None)


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
