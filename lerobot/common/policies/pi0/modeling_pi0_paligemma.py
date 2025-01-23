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
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

import torch
from torch import nn
from transformers import (
    AutoConfig,
    GemmaConfig,
    PaliGemmaConfig,
    PretrainedConfig,
    PreTrainedModel,
)

def apply_rope(x, positions, max_wavelength=10_000):
    # Copied from Pi0 jax codebase
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        x.shape[-1] // 2, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].float() / timescale[None, None, :].float()
    radians = radians[..., None, :]
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(dtype=x.dtype)



class PI0PaliGemmaConfig(PretrainedConfig):
    model_type = "PI0"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config=None,
        gemma_expert_config=None,
        state_dim=24,
        action_dim=24,
        width=1024,
        **kwargs,
    ):
        self.paligemma_config = paligemma_config
        self.gemma_expert_config = gemma_expert_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.width = width

        if isinstance(self.paligemma_config, dict):
            paligemma_config["model_type"] = (
                paligemma_config["model_type"] if "model_type" in paligemma_config else "paligemma"
            )
            self.paligemma_config = CONFIG_MAPPING[paligemma_config["model_type"]](**paligemma_config)
        elif paligemma_config is None:
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                intermediate_size=16384,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )
        if isinstance(self.gemma_expert_config, dict):
            gemma_expert_config["model_type"] = gemma_expert_config["model_type"] if "model_type" in gemma_expert_config else "gemma"
            self.gemma_expert_config = CONFIG_MAPPING[gemma_expert_config["model_type"]](**gemma_expert_config)
        elif gemma_expert_config is None:
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                hidden_size=1024,
                num_hidden_layers=18,
                intermediate_size=4096,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=257152,
            )

        super().__init__(**kwargs)


class PI0PaliGemmaModel(PreTrainedModel):
    config_class = PI0PaliGemmaConfig

    def __init__(self, config: PI0PaliGemmaConfig):
        super().__init__(config=config)
        self.config = config
        """
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            "Tinkering/frostpunklab_bf16", torch_dtype="bfloat16"
        )
        self.gemma_expert = GemmaForCausalLM.from_pretrained(
            "Tinkering/frostpunklab_action_expert_bf16_correct", torch_dtype="bfloat16"
        )
        """
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)

        self.state_proj = nn.Linear(self.config.state_dim, self.config.width, dtype=torch.float32)
        self.action_in_proj = nn.Linear(self.config.action_dim, self.config.width, dtype=torch.bfloat16)
        self.action_out_proj = nn.Linear(
            self.config.width, self.config.action_dim, dtype=torch.bfloat16
        )  # float32 for more precision?

        self.action_time_mlp_in = nn.Linear(self.config.width * 2, self.config.width, dtype=torch.bfloat16)
        self.action_time_mlp_out = nn.Linear(self.config.width, self.config.width, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
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
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
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
                    value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

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
            att_weights *= torch.tensor(head_dim**-0.5, dtype=torch.float32)
            # att_weights: batch_size, num_att_head, sequence_length, sequence_length
            #big_neg = torch.finfo(torch.float32).min  # See gemma/modules.py
            big_neg = -2.3819763e38  # See gemma/modules.py
            masked_att_weights = torch.where(attention_mask[:, None, None, :, :], att_weights, big_neg)

            # with autocast(dtype=torch.bfloat16, device_type=device.type):
            probs = torch.softmax(masked_att_weights, dim=-1, dtype=torch.float32)
            # probs = probs.to(dtype=torch.bfloat16)
            value_states = value_states.to(torch.float32)

            # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
            # value_states: batch_size, sequence_length, num_att_heads, head_dim

            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

            att_output = att_output.permute(0, 3, 1, 2, 4)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            ).to(dtype=torch.bfloat16)


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
