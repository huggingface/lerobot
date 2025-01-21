#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from collections import deque
from typing import List, Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from pytest import Cache
from torch import Tensor, nn
from transformers import AutoTokenizer, GemmaForCausalLM, PaliGemmaForConditionalGeneration

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType, NormalizationMode


def display(x):
    print(x.shape)
    print(f"mean: {x.mean().item()}")
    print(f"std: {x.std().item()}")
    print(f"min: {x.min().item()}")
    print(f"max: {x.max().item()}")


class PI0Policy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "pi0"],
):
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__()
        # config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, dataset_stats)

        self.model = PI0(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[ft.key] for ft in self.config.image_features], dim=-4
            )

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError()


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor * time
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)])
    return pos_emb


import torch
from torch import nn
from transformers import (
    AutoConfig,
    GemmaConfig,
    PaliGemmaConfig,
    PretrainedConfig,
    PreTrainedModel,
)


class PI0PaliGemmaConfig(PretrainedConfig):
    model_type = "PI0"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config=None,
        gemma_config=None,
        state_dim=14,
        action_dim=24,
        width=1024,
        **kwargs,
    ):
        self.paligemma_config = paligemma_config
        self.gemma_expert_config = gemma_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.width = width
        super().__init__(**kwargs)


class PI0PaliGemmaModel(PreTrainedModel):
    def __init__(self, config: PI0PaliGemmaConfig):
        super().__init__(config=config)
        self.config = config
        # self.paligemma = PaliGemmaForConditionalGeneration(config.paligemma_config) #PaliGemmaForConditionalGeneration.from_pretrained("Tinkering/frostpunklab_bf16")
        # self.gemma_expert = AutoModel.from_config(config.gemma_expert_config) #GemmaForCausalLM.from_pretrained('Tinkering/frostpunklab_action_expert_bf16', torch_dtype="bfloat16")

        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            "Tinkering/frostpunklab_bf16", torch_dtype="bfloat16"
        )
        self.gemma_expert = GemmaForCausalLM.from_pretrained(
            "Tinkering/frostpunklab_action_expert_bf16", torch_dtype="bfloat16"
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ):
        models = [self.paligemma.language_model.model, self.gemma_expert.model]

        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            dtype = hidden_states.dtype
            device = hidden_states.device
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

                # TODO: implement kv cache

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            head_dim = self.paligemma.config.text_config.head_dim

            # display(apply_rope(query_states, position_ids)[0,256:256+48])

            key_states = apply_rope(key_states, position_ids)

            if query_states.dtype != dtype:
                raise ValueError(f"{query_states.dtype=}")
            if key_states.dtype != dtype:
                raise ValueError(f"{key_states.dtype=}")
            if value_states.dtype != dtype:
                raise ValueError(f"{value_states.dtype=}")

            # TODO: implement caching

            if use_cache and past_key_values is None:
                # past_key_values = StaticCache(batch_size=batch_size, config=self.config.paligemma_config.text_config)
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    # past_key_values.update(key_states, value_states, layer_idx)
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # key_states = torch.concatenate(past_key_values.key_cache[layer_idx], key_states, dim=1)
                    # value_states = torch.concatenate(past_key_values.value_cache[layer_idx], value_states, dim=1)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            num_att_heads = 8
            num_key_value_heads = 1
            num_key_value_groups = num_att_heads // num_key_value_heads  # TODO from config

            # query_states: batch_size, sequence_length, num_att_head, head_dim
            # key_states: batch_size, sequence_length, num_key_value_head, head_dim
            # value_states: batch_size, sequence_length, num_key_value_head, head_dim

            sequence_length = key_states.shape[1]

            key_states = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states = key_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            value_states = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states = value_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            query_states = query_states.to(dtype=torch.float32)
            key_states = key_states.to(dtype=torch.float32)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            # with autocast(dtype=torch.float32, device_type=device.type):
            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            att_weights *= head_dim**-0.5
            # att_weights: batch_size, num_att_head, sequence_length, sequence_length

            big_neg = -2.3819763e38  # See gemma/modules.py
            masked_att_weights = torch.where(attention_mask[:, None, None, :, :], att_weights, big_neg)

            # with autocast(dtype=torch.bfloat16, device_type=device.type):
            probs = torch.softmax(masked_att_weights, dim=-1, dtype=torch.float32)
            probs = probs.to(dtype=torch.bfloat16)

            # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
            # value_states: batch_size, sequence_length, num_att_heads, head_dim

            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

            att_output = att_output.permute(0, 3, 1, 2, 4)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout

                    # first residual
                    out_emb += hidden_states

                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values


# TODO: for training look at preprocess_observation


class PI0(nn.Module):
    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config
        # TODO Should be derived from config
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        # self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained("Tinkering/frostpunklab_bf16")
        # self.gemma_expert = GemmaForCausalLM.from_pretrained('Tinkering/frostpunklab_action_expert_bf16', torch_dtype="bfloat16")
        self.pi0_paligemma = PI0PaliGemmaModel(
            config=PI0PaliGemmaConfig(
                paligemma_config=PaliGemmaConfig.from_pretrained("Tinkering/frostpunklab_bf16"),
                gemma_config=GemmaConfig.from_pretrained("Tinkering/frostpunklab_action_expert_bf16"),
            )
        )
        # self.pi0_paligemma.from_pretrained("Tinkering/frostpunklab_full_bf16", torch_dtype="bfloat16")

        state_dim = self.config.action_dim
        action_dim = self.config.state_dim
        n_action_steps = self.config.n_action_steps
        width = self.config.action_expert_width

        self.state_proj = nn.Linear(state_dim, width, dtype=torch.float32)
        self.action_in_proj = nn.Linear(action_dim, width, dtype=torch.bfloat16)
        self.action_out_proj = nn.Linear(
            width, action_dim, dtype=torch.bfloat16
        )  # float32 for more precision?

        self.action_time_mlp_in = nn.Linear(width * 2, width, dtype=torch.bfloat16)
        self.action_time_mlp_out = nn.Linear(width, width, dtype=torch.bfloat16)

        self.from_pretrained("/home/remi_cadene/code/openpi/data/aloha_sim/pi0_projs_state_dict.pth")

        # pos_emb = create_sinusoidal_pos_embedding(n_action_steps, width, min_period=4e-3, max_period=4.0)
        # self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

        self._rng = torch.Generator()
        self._rng.manual_seed(42)  # Set an initial seed

    def from_pretrained(self, path):
        state_dict = torch.load(path)

        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        for key in keys:
            module_state_dict = {
                "weight": state_dict[f"{key}.weight"].t(),
                "bias": state_dict[f"{key}.bias"],
            }
            module = getattr(self, key)
            module.load_state_dict(module_state_dict)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        for ft in self.config.image_features:
            batch[ft.key] = resize_with_pad(batch[ft.key], 224, 224)

        akey = self.config.action_feature.key
        skey = self.config.robot_state_feature.key

        if akey in batch:
            batch[akey] = pad_vector(batch[akey], 24)

        batch[skey] = pad_vector(batch[skey], 24)

        # tokenizer works on lists
        # PaliGemma prompt has to end with a new line
        max_length = 48
        tokenized_prompt = self.tokenizer.__call__(
            "Transfer cube\n",
            padding="max_length",
            padding_side="right",
            max_length=max_length,
            return_tensors="pt",
        )

        tokenized_prompt["attention_mask"] = tokenized_prompt["attention_mask"].type(dtype=torch.bool)

        bsize = batch[skey].shape[0]
        device = batch[skey].device

        batch["tokenized_prompt"] = tokenized_prompt["input_ids"].expand(bsize, max_length).to(device=device)
        batch["tokenized_prompt_mask"] = (
            tokenized_prompt["attention_mask"].expand(bsize, max_length).to(device=device)
        )

        actions = self.sample_actions(batch, tokenized_prompt)
        return actions

    def sample_actions(self, batch, tokenized_prompt, noise=None):
        skey = self.config.robot_state_feature.key
        bsize = batch[skey].shape[0]
        dtype = torch.bfloat16
        device = batch[skey].device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.get_prefix_embeddings(batch)

        # create attention mask (shared between prefix and suffix)
        prefix_pad_masks_tensor = torch.cat(prefix_pad_masks, dim=1)
        prefix_att_masks_tensor = torch.tensor(prefix_att_masks, dtype=torch.bool, device=device)
        prefix_att_masks_tensor = prefix_att_masks_tensor.expand(bsize, len(prefix_att_masks_tensor))

        prefix_att_2d_masks = combine_pad_and_att_masks(prefix_pad_masks_tensor, prefix_att_masks_tensor)
        prefix_position_ids = torch.cumsum(prefix_pad_masks_tensor, axis=1) - 1

        use_cache = True  # TODO: from config

        # fill image text cache
        fill_kv_cache = True

        _, past_key_values = self.pi0_paligemma.forward(
            input_ids=None,
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
        )
        fill_kv_cache = False

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=dtype, device=device)

        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(bsize, self.config.n_action_steps, self.config.action_dim),
                dtype=dtype,
                device=device,
            )

        import pickle

        with open("../openpi/data/aloha_sim/noise.pkl", "rb") as f:
            noise = pickle.load(f)
        noise = torch.from_numpy(noise).to(dtype=dtype, device=device)

        x_t = noise
        time = 1.0
        time = torch.tensor(time, dtype=dtype, device=device)
        while time >= -dt / 2:
            # time_batched = x_t[None, ...]
            _, v_t = self.sample_step(
                batch,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                past_key_values,
                fill_kv_cache,
                x_t,
                time,
            )

            x_t_tilde = self.action_out_proj(v_t[:, -self.config.n_action_steps :])

            # Euler step
            x_t += dt * x_t_tilde
            time += dt

        return x_t

    def get_prefix_embeddings(self, batch):
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for ft in self.config.image_features:
            img_key = ft.key
            img_emb = self.pi0_paligemma.paligemma.get_image_features(batch[img_key])
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # TODO: remove normalization?
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * math.sqrt(img_emb_dim)

            bsize, num_img_embs = img_emb.shape[:2]
            device = img_emb.device

            # img_mask = batch[f"{img_key}_mask"].expand(bsize, num_img_embs)
            img_mask = (batch[f"{img_key}_mask"]).expand(bsize, num_img_embs)
            # img_mask = torch.ones(bsize, num_img_embs, dtype=torch.bool, device=device)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # image tokens attend to each other
            att_masks += [0] * num_img_embs

        # TODO: if language
        lang_emb = self.pi0_paligemma.paligemma.language_model.model.embed_tokens(batch["tokenized_prompt"])
        lang_emb = lang_emb.to(dtype=torch.bfloat16)

        # TODO: remove normalization?
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(batch["tokenized_prompt_mask"])

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)

        return embs, pad_masks, att_masks

    def sample_step(
        self,
        batch,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        fill_kv_cache,
        x_t,
        time,
    ):
        # ACTUAL SAMPLE STEP

        embs = []
        pad_masks = []
        att_masks = []

        # add a single state token
        state_emb = self.state_proj(batch["observation.state"])
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])

        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # image/language inputs do not attend to state or actions
        att_masks += [1]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        width = self.config.action_expert_width
        time_emb = create_sinusoidal_pos_embedding(
            time, width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.unsqueeze(0)
        time_emb = time_emb.type(dtype=dtype)
        # time_emb = posemb_sincos(timestep, action_expert_config.width, min_period=4e-3, max_period=4.0)

        # mix timestep + action information using an MLP
        noisy_actions = x_t
        action_emb = self.action_in_proj(noisy_actions)

        bsize, time_dim = time_emb.shape
        time_emb = time_emb.expand(bsize, self.config.n_action_steps, time_dim)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        # action_time_emb = F.swish(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # image/language/state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)

        if self.training:
            prefix_len = embs.shape[1]
            if att_masks[prefix_len] != 1:
                raise ValueError(
                    "Due to prefix-lm decoding, it is very important that the prefix cannot attend to the suffix"
                )

        # create attention mask (shared between prefix and suffix)
        pad_masks = torch.cat(pad_masks, dim=1)
        prefix_pad_masks = torch.cat(prefix_pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks.expand(bsize, len(att_masks))

        suffix_len = pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_masks_expanded = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)

        # att_masks = torch.tensor(prefix_att_masks + att_masks, dtype=torch.bool, device=device)
        # att_masks = att_masks.expand(bsize, len(att_masks))

        # prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        att_2d_masks = combine_pad_and_att_masks(pad_masks, att_masks)

        att_2d_masks = torch.cat([prefix_pad_masks_expanded, att_2d_masks], dim=2)

        # if self.training:
        #     # full forward pass on prefix + suffix at once
        #     positions = torch.cumsum(pad_masks, axis=1) - 1

        #     # TODO: call gemma + gemma expert
        #     _, out = gemma(
        #         tokens=None,
        #         embedded=[prefix_embs, embs],
        #         mask=att_2d_masks,
        #         positions=positions,
        #         decode=False,
        #     )

        #     return self.action_out_proj(out[:, -self.config.n_action_steps :])

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(pad_masks, axis=1) - 1

        use_cache = True  # TODO: from config

        outputs_embeds, _ = self.pi0_paligemma.forward(
            input_ids=None,
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, embs],
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
        )
        return outputs_embeds


def apply_rope(x, positions, max_wavelength=10_000):
    # Copied from Pi0 jax codebase
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        x.shape[-1] // 2, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == torch.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = torch.sin(radians), torch.cos(radians)
    # x1, x2 = jnp.split(x, 2, axis=-1)
    x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    assert res.dtype == torch.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.to(dtype=x.dtype)


def combine_pad_and_att_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    padded_img = F.pad(resized_img, (pad_height, pad_width))
    return padded_img


def pad_vector(vector, new_dim):
    if vector.ndim != 2:
        raise ValueError("Must be batched.")
    if vector.shape[1] == new_dim:
        return vector
    bsize, dim = vector.shape
    new_vector = torch.zeros(bsize, new_dim, dtype=vector.dtype, device=vector.device)
    new_vector[:, :dim] = vector
    return new_vector


def main():
    import json
    import pickle
    from pathlib import Path

    with open("../openpi/data/aloha_sim/obs.pkl", "rb") as f:
        obs = pickle.load(f)

    with open("../openpi/data/aloha_sim/action.pkl", "rb") as f:
        action = pickle.load(f)

    checkpoint_dir = Path("/home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim")

    with open(checkpoint_dir / "assets/norm_stats.json") as f:
        norm_stats = json.load(f)

    len(norm_stats["norm_stats"]["actions"]["mean"])
    len(norm_stats["norm_stats"]["actions"]["std"])
    len(norm_stats["norm_stats"]["state"]["mean"])
    len(norm_stats["norm_stats"]["state"]["std"])

    num_motors = 14

    dataset_stats = {
        "observation.images.top": {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        "observation.state": {
            "mean": torch.tensor(norm_stats["norm_stats"]["state"]["mean"][:num_motors]),
            "std": torch.tensor(norm_stats["norm_stats"]["state"]["std"][:num_motors]),
            "min": torch.zeros(num_motors),
            "max": torch.ones(num_motors),
        },
        "action": {
            "mean": torch.tensor(norm_stats["norm_stats"]["actions"]["mean"][:num_motors]),
            "std": torch.tensor(norm_stats["norm_stats"]["actions"]["std"][:num_motors]),
            "min": torch.zeros(num_motors),
            "max": torch.ones(num_motors),
        },
    }

    cam_top = torch.from_numpy(obs["images"]["cam_high"]).unsqueeze(0) / 255.0 * 2.0 - 1.0
    cam_top = cam_top.to(dtype=torch.float32)
    cam_top_mask = torch.ones(1, dtype=torch.bool)

    state = torch.from_numpy(obs["state"]).unsqueeze(0)
    state = state.to(dtype=torch.float32)

    batch = {
        "observation.images.top": cam_top,
        "observation.images.top_mask": cam_top_mask,
        "observation.images.left_wrist": torch.ones_like(cam_top) * -1,
        "observation.images.left_wrist_mask": torch.zeros_like(cam_top_mask),
        "observation.images.right_wrist": torch.ones_like(cam_top) * -1,
        "observation.images.right_wrist_mask": torch.zeros_like(cam_top_mask),
        "observation.state": state,
    }

    device = "cuda"
    for k in batch:
        batch[k] = batch[k].to(device=device)

    cfg = PI0Config()
    cfg.parse_features_from_dataset(ds_meta=LeRobotDatasetMetadata("lerobot/aloha_sim_transfer_cube_human"))

    cfg_img_left_wrist = PolicyFeature(
        key="observation.images.left_wrist",
        type=FeatureType.VISUAL,
        shape=(3, 480, 640),
        normalization_mode=NormalizationMode.IDENTITY,
    )
    cfg_img_right_wrist = PolicyFeature(
        key="observation.images.right_wrist",
        type=FeatureType.VISUAL,
        shape=(3, 480, 640),
        normalization_mode=NormalizationMode.IDENTITY,
    )
    cfg.image_features.append(cfg_img_left_wrist)
    cfg.image_features.append(cfg_img_right_wrist)

    policy = PI0Policy(cfg, dataset_stats=dataset_stats)
    policy.to(device=device)
    policy.select_action(batch)


if __name__ == "__main__":
    main()
