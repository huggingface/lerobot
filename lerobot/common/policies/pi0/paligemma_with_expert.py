from typing import List, Optional, Union

import torch
from pytest import Cache
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING


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


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config=None,
        gemma_expert_config=None,
        freeze_vision_encoder=True,
        train_expert_only=True,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        self.gemma_expert = self.gemma_expert.to(dtype=torch.bfloat16)

        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            if not self.config.freeze_vision_encoder:
                raise ValueError()

            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
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

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
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

            head_dim = self.paligemma.config.text_config.head_dim
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

            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            att_weights *= torch.tensor(head_dim**-0.5, dtype=torch.float32)
            # att_weights: batch_size, num_att_head, sequence_length, sequence_length
            big_neg = -2.3819763e38  # See gemma/modules.py
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

            probs = torch.softmax(masked_att_weights, dim=-1, dtype=torch.float32)
            value_states = value_states.to(torch.float32)

            # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
            # value_states: batch_size, sequence_length, num_att_heads, head_dim

            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
            att_output = att_output.to(dtype=torch.bfloat16)
            att_output = att_output.permute(0, 2, 1, 3)
            # we use -1 because sequence length can change
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
