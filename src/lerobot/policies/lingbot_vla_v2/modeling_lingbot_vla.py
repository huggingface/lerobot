from logging import raiseExceptions
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
from functools import partial
import math
from .configuration_lingbot_vla import LingbotVLAConfig
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from transformers import AutoTokenizer
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)


class LossKwargs(TypedDict, total=False):
    labels: Optional[torch.LongTensor]


from transformers.utils.deprecation import deprecate_kwarg
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack

# Qwen2.5-VL tower is unused on the Qwen3-VL v2 path (FlowMatchingV2 builds its own
# QwenvlWithExpertV2Model and never instantiates the v1 QwenvlWithExpertModel).
Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLTextModel = Qwen2_5_VLPreTrainedModel = None

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
)

try:
    from dinov3.hub.backbones import (
        dinov3_vits16,
        dinov3_vits16plus,
        dinov3_vitb16,
    )
except:
    pass
from .utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)
from .utils import apply_rope, our_eager_attention_forward
from .flex_attention import flex_attention_forward
from .flex_attention import build_block_mask, flex_attention_with_block_mask
import time

LingBotVLAWeightLoader = None  # noqa: N816  # lerobot PreTrainedPolicy handles weight loading
TaskTokenDepthHead = None  # lazily wired for the depth-distillation branch (M6)
from .qwen2_action_expert import (
    Qwen2ForCausalLM,
    Qwen2FusedExperts,
    Qwen2TokenMoeBlock,
    FixQwen2RMSNorm,
)

logger = logging.get_logger(__name__)


class QwenvlWithExpertConfig(PretrainedConfig):
    model_type = "QwenvlWithExpertModel"
    sub_configs = {"qwenvl_config": AutoConfig, "qwen_expert_config": AutoConfig}

    def __init__(
        self,
        qwenvl_config: dict | None = None,
        qwen_expert_config: dict | None = None,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 257152,
        use_lm_head: bool = False,
        attention_implementation: str = "eager",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        if qwenvl_config is None:
            self.qwenvl_config = CONFIG_MAPPING["qwen2_5_vl"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=151656,
                hidden_act="silu",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=11008,
                max_position_embeddings=128000,
                max_window_layers=70,
                model_type="qwen2_5_vl",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.41.2",
                use_cache=True,
                use_sliding_window=False,
                vision_config={
                    "depth": 32,
                    "hidden_act": "silu",
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_heads": 16,
                    "in_chans": 3,
                    "out_hidden_size": 2048,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "window_size": 112,
                    "fullatt_block_indexes": [7, 15, 23, 31],
                    "tokens_per_second": 2,
                    "temporal_patch_size": 2,
                },
                rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
                vocab_size=151936,
            )
        elif isinstance(self.qwenvl_config, dict):
            if "model_type" not in qwen_expert_config:
                qwenvl_config["model_type"] = "qwen2_5_vl"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwenvl_config = cfg_cls(**qwenvl_config)

        if qwen_expert_config is None:
            self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                hidden_act="silu",
                hidden_size=expert_hidden_size,
                head_dim=128,
                initializer_range=0.02,
                intermediate_size=expert_intermediate_size,
                max_position_embeddings=32768,
                max_window_layers=21,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.43.1",
                use_cache=use_cache,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif isinstance(self.qwen_expert_config, dict):
            if "model_type" not in qwen_expert_config:
                qwen_expert_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwen_expert_config = cfg_cls(**qwen_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        # cond = cond.to(torch.float32)
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


class FixAdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        cond = cond.to(torch.float32)
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


# HACK: show directly use this norm during initialization
# TODO: clear the logics
def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, final_norm_adanorm):
    for name, child in module.named_children():
        if final_norm_adanorm:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            elif isinstance(child, FixQwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, FixAdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)
        else:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)


class QwenvlWithExpertModel(PreTrainedModel):
    config_class = QwenvlWithExpertConfig

    def __init__(self, config: QwenvlWithExpertConfig, eval=False):
        super().__init__(config=config)
        self.config = config
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
        vlm_config.vision_config.initializer_range = 0.02
        print(f"=====Vocab_size in Config is {self.config.vocab_size}=====")
        if (
            self.config.vocab_size != 0
            and self.config.vocab_size != 257152
            and vlm_config.vocab_size != self.config.vocab_size
        ):
            vlm_config.vocab_size = self.config.vocab_size
        print(f"====Vocabulary Size is {vlm_config.vocab_size}====")
        vlm_config._attn_implementation = "flash_attention_2"
        vlm_config.vision_config._attn_implementation = self.config.vit_attn_implementation
        self.qwenvl = Qwen2_5_VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()
        self.config.qwen_expert_config._attn_implementation = "flash_attention_2"
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )
        if getattr(self.config, "use_moe", False):
            bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
            hidden_size = self.config.qwen_expert_config.hidden_size  # 768

            token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

            if token_moe_layers:
                token_config = CONFIG_MAPPING["qwen2_moe"](
                    num_experts=getattr(self.config, "token_num_experts", 32),
                    num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                    norm_topk_prob=True,
                    hidden_size=hidden_size,
                    moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                    shared_expert_intermediate_size=getattr(
                        self.config, "token_shared_intermediate_size", 256
                    ),
                    output_router_logits=False,
                )
                token_config.bias_update_speed = bias_update_speed
                token_config._moe_implementation = getattr(self.config, "_moe_implementation", None)
                token_config.router_activation = getattr(self.config, "router_activation", "softmax")
                token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
                token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
                for idx in token_moe_layers:
                    self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)
        # Precomputed grid_thw cache (populated on first call when precompute_grid_thw=True)
        self.rotary_pos_emb = None
        self.window_index = None
        self.cu_window_seqlens = None
        self.cu_seqlens = None

        # Remove unused embed_tokens
        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )
        self.attention_interface = self.get_attention_interface()

        # self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        """sets the requires_grad attribute of the model parameters based on the configuration.
        If `freeze_vision_encoder` is True, the vision tower parameters are frozen.
        If `train_expert_only` is True, the entire Qwenvl model is frozen.
        """
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
            for params in self.qwenvl.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()

    def to_bfloat16_like_physical_intelligence(self):
        """casts the model to bfloat16.

        Modules not casted to bfloat16:
        - .qwenvl.model.embed_tokens.weight
        - .qwenvl.model.norm.weight
        - qwen_expert.model.norm.weight
        - qwen_expert.lm_head.weight
        """
        self.qwenvl = self.qwenvl.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            ".qwenvl.model.layers",
            "qwen_expert.model.layers",
            "visual",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        precompute_grid_thw: bool = False,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            precompute_grid_thw (`bool`): If True, compute and cache rotary_pos_emb/window_index/cu_seqlens on first call.
        """
        if precompute_grid_thw and self.rotary_pos_emb is None:
            (self.rotary_pos_emb, self.window_index, self.cu_window_seqlens, self.cu_seqlens) = (
                self.qwenvl.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
            )
        image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            rotary_pos_emb=self.rotary_pos_emb,
            window_index=self.window_index,
            cu_window_seqlens=self.cu_window_seqlens,
            cu_seqlens=self.cu_seqlens,
        )
        split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = torch.stack(image_embeds, dim=0)
        return image_embeds

    def embed_image(
        self, image: torch.Tensor, patch_size=14, temporal_patch_size=2, precompute_grid_thw=False
    ):
        h = w = int(image.shape[1] ** 0.5)
        image_grid_thw = torch.tensor([[1, h, w]] * image.shape[0], device=image.device)
        image_embeds = self.get_image_features(
            image, image_grid_thw=image_grid_thw, precompute_grid_thw=precompute_grid_thw
        )
        return image_embeds
        # return torch.randn(72, 64, 2048).to(device=image.device, dtype=torch.bfloat16)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.embed_tokens(tokens)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        ada_cond: List[torch.FloatTensor] = None,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [
            self.qwenvl.model,
            self.qwen_expert.model,
        ]  # Qwen2_5_VLTextModel, Qwen2Model (We have re-writeen their forward as follows:)

        # RMSNorm
        num_layers = self.qwenvl.config.num_hidden_layers  # 36
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers  # 36
        assert action_num_layers == num_layers, (
            "Action expert and VLM must have the same number of layers "
            f"(got action={action_num_layers}, vlm={num_layers})."
        )

        router_logits_list = []
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1:  # For action expert
                    query_state, key_state, value_state = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True, ada_cond=ada_cond
                    )
                else:  # For VLM
                    query_state, key_state, value_state = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True
                    )

                if query_state.dtype != torch.float32:
                    query_state, key_state, value_state = (
                        query_state.to(torch.float32),
                        key_state.to(torch.float32),
                        value_state.to(torch.float32),
                    )
                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length (img, lang, state, action), H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
            if self.config.attention_implementation == "flex_cached":
                if layer_idx == 0:
                    _full_len = query_states.shape[1]
                    _full_block_mask = build_block_mask(
                        attention_mask, self.qwenvl.config.num_attention_heads, _full_len, _full_len
                    )
                att_output = flex_attention_with_block_mask(
                    query_states, key_states, value_states, _full_block_mask, query_states.shape[1]
                )
            else:
                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    if i == 1:
                        out_emb, _router_logits = models[i].layers[layer_idx](
                            hidden_states, att_output, start, end, output_atten=True, ada_cond=ada_cond
                        )
                        if _router_logits is not None:
                            router_logits_list.append(_router_logits)
                    else:
                        out_emb = models[i].layers[layer_idx](
                            hidden_states, att_output, start, end, output_atten=True
                        )
                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.config.final_norm_adanorm:
                    if i == 1:
                        out_emb, _ = models[i].norm(hidden_states, ada_cond)
                    else:
                        out_emb = models[i].norm(hidden_states)
                else:
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            print("=====Using Flex Attn=====")
            attention_interface = flex_attention_forward
        elif self.config.attention_implementation == "eager":
            print("=====Using Eager Attn=====")
            attention_interface = our_eager_attention_forward
        elif self.config.attention_implementation == "flex_cached":
            print("=====Using Flex Cached (prebuilt BlockMask) Attn=====")
            attention_interface = flex_attention_forward  # fallback
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'flex_cached', 'eager', 'xformer']."
            )
        return attention_interface


class LingbotVlaPolicy(PreTrainedModel):
    config_class = LingbotVLAConfig
    name = "torch_lingbot_vla"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Qwen2DecoderLayer",
        "FixQwen2RMSNorm",
        "FixAdaRMSNorm",
    ]  # NOTE: if moudule in Qwen2DecoderLayer, it doesn't need to specify in _no_split_modules

    def get_parallel_plan(self):
        from lingbotvla.distributed.parallel_plan import ParallelPlan
        from torch.distributed._tensor import Shard

        ep_plan = {
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.gate_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.up_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(ep_plan=ep_plan)

    @classmethod
    def get_weight_loader(cls):
        return LingBotVLAWeightLoader()

    def __init__(
        self,
        config: LingbotVLAConfig,
        eval: bool = False,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatching(config, eval)

        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        self.reset()
        torch.set_float32_matmul_precision("high")

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(
        self,
        images,
        img_masks,
        state,
        lang_tokens,
        lang_masks,
        actions,
        joint_mask=None,
        action_is_pad=None,
        noise=None,
        time=None,
        vlm_causal=False,
        depth_targets=None,
        precompute_grid_thw=False,
        future_depth_targets=None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}
        # Keep state and actions in fp32 for action learning precision
        if getattr(self.config, "action_fp32", False):
            state = state.float()
            actions = actions.float()
        losses, loss_depth, depth_preds, seq_wise_loss, moe_metrics = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            vlm_causal,
            self.config.loss_type,
            depth_targets,
            precompute_grid_thw=precompute_grid_thw,
            future_depth_targets=future_depth_targets,
        )

        if joint_mask is not None:
            if "repeat" in self.config.loss_type:
                joint_mask = joint_mask.repeat(2, 1)
            mask_expanded = joint_mask.unsqueeze(1).expand(-1, losses.size(1), -1)  # (B, T, D)
            masked_losses = losses * mask_expanded

            valid_counts = mask_expanded.sum(dim=(1, 2)).clamp(min=1)
            batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
            loss_vla = masked_losses.sum() / mask_expanded.sum().clamp(min=1)
        else:
            losses = losses[:, :, : self.config.action_dim]
            batch_mean_losses = losses.mean(dim=(1, 2))
            loss_vla = losses.mean()

        loss_dict["batch_mean_losses"] = batch_mean_losses.detach()

        total_loss = loss_vla + loss_depth + seq_wise_loss

        # Attach MoE monitoring metrics to loss_dict
        if moe_metrics:
            loss_dict.update(moe_metrics)

        return total_loss, loss_vla, loss_depth, seq_wise_loss, loss_dict, depth_preds


class FlowMatching(nn.Module):
    def __init__(self, config, eval):
        super().__init__()
        self.config = config

        # qwenvl with action expert
        qwenvl_with_export_config = QwenvlWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
        )
        qwenvl_with_export_config.adanorm_time = getattr(config, "adanorm_time", False)
        qwenvl_with_export_config.final_norm_adanorm = getattr(config, "final_norm_adanorm", False)
        qwenvl_with_export_config.vit_attn_implementation = getattr(
            config, "vit_attn_implementation", "flash_attention_2"
        )
        if getattr(config, "use_moe", False):
            qwenvl_with_export_config.use_moe = config.use_moe
            qwenvl_with_export_config.bias_update_speed = getattr(config, "bias_update_speed", 0.001)
            qwenvl_with_export_config.token_moe_layers = getattr(config, "token_moe_layers", None)
            qwenvl_with_export_config.token_num_experts = getattr(config, "token_num_experts", 32)
            qwenvl_with_export_config.token_top_k = getattr(config, "token_top_k", 1)
            qwenvl_with_export_config.token_moe_intermediate_size = getattr(
                config, "token_moe_intermediate_size", 256
            )
            qwenvl_with_export_config.token_shared_intermediate_size = getattr(
                config, "token_shared_intermediate_size", 256
            )
        # Pass _moe_implementation through for EP/fused support
        qwenvl_with_export_config._moe_implementation = getattr(config, "_moe_implementation", None)
        self.qwenvl_with_expert = QwenvlWithExpertModel(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(
            qwenvl_with_export_config.qwen_expert_config, "initializer_range", None
        )
        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.config.align_params = getattr(self.config, "align_params", {})
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False

        self.set_requires_grad()

    def init_depth_heads(self, config):
        self.llm_image_token_size = config["llm"]["image_token_size"]
        self.llm_image_input_size = config["llm"]["image_input_size"]
        self.depth_token_size = config["depth"]["token_size"]
        self.depth_input_size = config["depth"]["input_size"]
        self.align_type = config.get("mode", None)
        self.model_type = config["depth"]["model_type"]
        if self.align_type != "query":
            raise ValueError(f"Only query depth alignment is supported, got {self.align_type!r}.")
        if self.model_type != "MoRGBD":
            raise ValueError(f"Only MoRGBD depth distillation is supported, got {self.model_type!r}.")
        self.use_future_depth = (config.get("depth") or {}).get("use_future_depth", False)
        self.block_future_depth_to_action = (config.get("depth") or {}).get(
            "block_future_depth_to_action", False
        )
        self.detach_future_depth_image_feats = bool(
            (config.get("depth") or {}).get("detach_future_image_feats", False)
        )
        self.use_future_video = bool(config.get("use_future_video", False))
        self.use_future_video_patch = False
        self.use_current_video_patch = False
        self.use_current_shared_task_proj = False
        self.use_future_video_cls = False
        self.use_shared_future_task_proj = False
        self.future_video_share_future_depth_query = False
        self.num_task_tokens = config["num_task_tokens"]
        assert config["depth"]["num_backbone_tokens"] % self.num_task_tokens == 0
        self.depth_align_embs = nn.Parameter(
            torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
        )
        self.depth_align_embs.requires_grad = True

        self.depth_align_head = TaskTokenDepthHead(
            config["depth"], llm_hidden_size=config["llm"]["dim_out"]
        ).to(dtype=torch.bfloat16)

        for p in self.depth_align_head.parameters():
            p.requires_grad = True

        if self.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(
                torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
            )
            self.future_depth_align_embs.requires_grad = True

            self.future_depth_align_head = TaskTokenDepthHead(
                config["depth"], llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)

            for p in self.future_depth_align_head.parameters():
                p.requires_grad = True

    def init_video_heads(self, config):
        if self.align_type != "query":
            raise ValueError("future-video alignment is only supported for query align mode.")

        video_config = dict(config.get("depth", {}))
        video_config.update(config.get("video", {}))
        required_keys = ("num_backbone_tokens", "dim_out", "num_layers", "num_heads", "dim_head", "ff_mult")
        missing = [key for key in required_keys if key not in video_config]
        if missing:
            raise ValueError(f"video align config missing required keys: {missing}")
        self.use_future_video_patch = bool(video_config.get("use_patch_loss", True))
        self.use_current_video_patch = bool(video_config.get("use_current_patch_loss", False))
        if self.use_current_video_patch and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_current_patch_loss=True requires "
                "align_params.video.use_patch_loss=True."
            )
        self.use_current_shared_task_proj = bool(
            video_config.get("use_current_shared_task_proj", self.use_current_video_patch)
        )
        if self.use_current_shared_task_proj and not self.use_current_video_patch:
            raise ValueError(
                "align_params.video.use_current_shared_task_proj=True requires "
                "align_params.video.use_current_patch_loss=True."
            )
        self.use_future_video_cls = bool(video_config.get("use_cls_loss", False))
        self.future_video_share_future_depth_query = bool(video_config.get("share_future_depth_query", False))
        self.use_shared_future_task_proj = bool(video_config.get("use_shared_future_task_proj", False))
        if self.use_shared_future_task_proj and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires "
                "align_params.video.use_patch_loss=True."
            )
        if self.use_shared_future_task_proj and not self.future_video_share_future_depth_query:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires "
                "align_params.video.share_future_depth_query=True."
            )
        if self.future_video_share_future_depth_query:
            if not self.use_future_depth:
                raise ValueError(
                    "align_params.video.share_future_depth_query=True requires "
                    "align_params.depth.use_future_depth=True."
                )
            if int(video_config["num_backbone_tokens"]) != int(config["depth"]["num_backbone_tokens"]):
                raise ValueError(
                    "future-video shared query requires video.num_backbone_tokens "
                    "to match depth.num_backbone_tokens."
                )

        self.block_suffix_to_future_video = bool(video_config.get("block_suffix_to_future_video", False))
        self.future_video_context_mode = str(video_config.get("context_mode", "img_query")).lower()
        if self.future_video_context_mode not in ("img_query", "query_only"):
            raise ValueError(
                "future-video context_mode must be 'img_query' or 'query_only', "
                f"got {self.future_video_context_mode!r}."
            )
        if self.use_future_video_patch:
            if self.use_current_video_patch:
                self.current_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
                self.current_video_align_embs.requires_grad = True
                if self.use_current_shared_task_proj:
                    self.current_shared_task_proj = nn.Linear(
                        config["llm"]["dim_out"] * 2,
                        config["llm"]["dim_out"],
                    )
                    for p in self.current_shared_task_proj.parameters():
                        p.requires_grad = True
                self.current_video_align_head = TaskTokenDepthHead(
                    video_config, llm_hidden_size=config["llm"]["dim_out"]
                ).to(dtype=torch.bfloat16)
                for p in self.current_video_align_head.parameters():
                    p.requires_grad = True

            if not self.future_video_share_future_depth_query or self.use_shared_future_task_proj:
                self.future_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
                self.future_video_align_embs.requires_grad = True
            if self.use_shared_future_task_proj:
                self.future_shared_task_proj = nn.Linear(
                    config["llm"]["dim_out"] * 2,
                    config["llm"]["dim_out"],
                )
                for p in self.future_shared_task_proj.parameters():
                    p.requires_grad = True
            self.future_video_align_head = TaskTokenDepthHead(
                video_config, llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)
            for p in self.future_video_align_head.parameters():
                p.requires_grad = True

        if self.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, config["llm"]["dim_out"])
            self.future_video_cls_head = nn.Sequential(
                nn.LayerNorm(config["llm"]["dim_out"]),
                nn.Linear(config["llm"]["dim_out"], video_config["dim_out"]),
            ).to(dtype=torch.bfloat16)
            for p in self.future_video_cls_head.parameters():
                p.requires_grad = True

    def _future_depth_token_count(self):
        return self.num_task_tokens if getattr(self, "use_future_depth", False) else 0

    def _future_video_own_token_count(self):
        if not getattr(self, "use_future_video", False):
            return 0
        count = 1 if getattr(self, "use_future_video_cls", False) else 0
        if getattr(self, "use_future_video_patch", True) and not getattr(
            self, "future_video_share_future_depth_query", False
        ):
            count += self.num_task_tokens
        return count

    def _future_video_own_span(self, hidden_states):
        own_count = self._future_video_own_token_count()
        future_depth_count = self._future_depth_token_count()
        end = hidden_states.shape[1] - future_depth_count
        start = end - own_count
        return start, end

    def _future_depth_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_depth", False):
            raise ValueError("future-depth query tokens are not enabled.")
        return hidden_states[:, -self.num_task_tokens :, :]

    def _future_video_cls_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_video_cls", False):
            return None
        start, _ = self._future_video_own_span(hidden_states)
        return hidden_states[:, start : start + 1, :]

    def _future_video_patch_task_tokens(self, hidden_states):
        if getattr(self, "future_video_share_future_depth_query", False):
            return self._future_depth_task_tokens(hidden_states)
        start, end = self._future_video_own_span(hidden_states)
        if getattr(self, "use_future_video_cls", False):
            start += 1
        return hidden_states[:, start:end, :]

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        image_token_len = chunk_size + (
            2 if getattr(self.config, "qwen3vl_use_vision_boundaries", False) else 0
        )
        if getattr(self, "use_future_depth", False):
            start = num_images * image_token_len
            return hidden_states[:, start : start + self.num_task_tokens, :]
        end = hidden_states.shape[1] - self._future_video_own_token_count()
        start = end - self.num_task_tokens
        return hidden_states[:, start:end, :]

    def _future_video_query_span(self, prefix_len):
        if not getattr(self, "use_future_video", False):
            return prefix_len, prefix_len
        future_depth_count = self._future_depth_token_count()
        own_count = self._future_video_own_token_count()
        end = prefix_len - future_depth_count
        return end - own_count, end

    def _block_suffix_to_future_video_(self, att_2d_masks, suffix_row_start, prefix_len):
        start, end = self._future_video_query_span(prefix_len)
        if end <= start:
            return att_2d_masks
        att_2d_masks[:, suffix_row_start:, start:end] = False
        return att_2d_masks

    def _block_suffix_to_future_video_if_enabled_(
        self,
        att_2d_masks,
        suffix_row_start,
        prefix_len,
    ):
        if not getattr(self, "block_suffix_to_future_video", False):
            return att_2d_masks
        return self._block_suffix_to_future_video_(
            att_2d_masks,
            suffix_row_start=suffix_row_start,
            prefix_len=prefix_len,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2FusedExperts):
            module.initializer_range = std
            module.reset_parameters()
        reset_post_init = getattr(module, "_reset_post_init_parameters", None)
        if reset_post_init is not None:
            reset_post_init()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    @staticmethod
    def _fp32_linear(module, x):
        """Compute linear layer in fp32 regardless of module's current parameter dtype."""
        return F.linear(
            x.float(), module.weight.float(), module.bias.float() if module.bias is not None else None
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype

        # embed image
        if images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        elif images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 3:  # For inference bs=1
            bsize = 1
        img_emb = self.qwenvl_with_expert.embed_image(images, precompute_grid_thw=precompute_grid_thw)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)  # bsize = 24
        num_img_embs = img_emb.shape[1]
        if img_masks.ndim == 1:  # For inference bs=1
            img_masks = img_masks.unsqueeze(0)
        if self.use_depth_align and self.align_type == "query":
            align_masks = einops.repeat(img_masks, "b n -> b (n l)", l=self.num_task_tokens)
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(
                    self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1]
                )
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_embs = (
                _get_align_tokens(self.depth_align_embs)
                .repeat(img_emb.size(0), 1, 1)
                .to(img_emb.device, img_emb.dtype)
            )
            # align_masks = einops.rearrange(img_masks, "b (n l) -> b n l", n=3)
            # align_masks = align_masks[:, :, 0]
            # align_masks = einops.repeat(align_masks, "b n -> b (n l)", l=self.num_task_tokens)
            embs = torch.cat([img_emb, align_embs, align_embs, align_embs, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, align_masks, lang_masks], dim=1)
        else:
            # assemble embeddings
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        if not vlm_causal:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )  # 1, bs_img*(768+48)
            else:
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )  # 1, bs_img*(768+48)
        else:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )  # 1, bs_img*(768+48)
            else:
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )  # 1, bs_img*(768+48)
        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state, noisy_actions, timestep
    ):  # (torch.Size([state_bs, 32]), torch.Size([1, state_bs*50, 32]), torch.Size([1]))
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        _fp32 = getattr(self.config, "action_fp32", False)
        # embed state
        state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.config.proj_width,  # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = (
            self._fp32_linear(self.action_in_proj, noisy_actions)
            if _fp32
            else self.action_in_proj(noisy_actions)
        )  # torch.Size([1, state_bs*50, 1024])
        time_emb = einops.repeat(
            time_emb, "b d -> b n d", n=action_emb.shape[1]
        )  # [1, 1024] -> [1, state_bs*50, 1024]
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)  # [1, state_bs*50, 2048]

        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_in, action_time_emb)
            if _fp32
            else self.action_time_mlp_in(action_time_emb)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_out, action_time_emb)
            if _fp32
            else self.action_time_mlp_out(action_time_emb)
        )  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        vlm_causal=False,
        loss_type="fm",
        depth_targets=None,
        precompute_grid_thw=False,
        future_depth_targets=None,
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)

        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw=precompute_grid_thw
        )  # 1,bs_img*(768+48),2048  1,bs_img*(768+48)  1,bs_img*(768+48)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )  # [1, state_bs*(50+1), 1024], [1, state_bs*(50+1)], [1, state_bs*(50+1)]   state_bs=bs_img

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)  # 1,state_bs*(768+48+50+1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)  # 1,state_bs*(768+48+50+1)

        # pad_masks = pad_masks.reshape(state.size(0), -1)
        # att_masks = att_masks.reshape(state.size(0), -1)
        att_2d_masks = make_att_2d_masks(
            pad_masks, att_masks
        )  # torch.Size([state_bs, 768+48+50+1, 768+48+50+1])
        position_ids = torch.cumsum(pad_masks, dim=1) - 1  # torch.Size([state_bs, 768+48+50+1])
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # prefix_embs = prefix_embs.reshape(state.size(0), -1, prefix_embs.size(-1))
        # suffix_embs = suffix_embs.reshape(state.size(0), -1, suffix_embs.size(-1))
        (outputs_embeds, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],  # bs_img,(768+48),2048  [state_bs, (50+1), 1024]
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        if self.config.align_params != {}:
            loss_depth, depth_preds = self.depth_emb_forward(outputs_embeds, depth_targets, img_masks)
            loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
            self.steps += 1
        else:
            loss_depth = 0
            depth_preds = None
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        # u_t = u_t.reshape(images.size(0), -1, u_t.size(-1))
        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
            # losses = torch.mean((v_t - u_t)**2, dim=-1)
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")

        # Sequence-wise balance loss (DeepSeek-V3 style, for token-MoE only)
        seq_wise_loss_coeff = getattr(self.config, "sequence_wise_loss_coeff", 0)
        seq_wise_loss = 0

        if seq_wise_loss_coeff > 0 and router_logits_list:
            from .moe_loss import sequence_wise_balance_loss as triton_sequence_wise_balance_loss

            token_moe_layers_set = set(getattr(self.config, "token_moe_layers", None) or [])
            token_moe_layers_list = sorted(token_moe_layers_set)
            token_router_logits = tuple(
                logits
                for i, logits in enumerate(router_logits_list)
                if not token_moe_layers_list
                or (token_moe_layers_list[i] if i < len(token_moe_layers_list) else i) in token_moe_layers_set
            )

            if token_router_logits:
                token_top_k = getattr(self.config, "token_top_k", 4)

                # Batch-wise balance loss: treat all B×T tokens as one group.
                # seq_lengths=None makes the function use all tokens at once,
                # giving stable f_i statistics (B×T×K assignments / E experts).
                layer_losses = triton_sequence_wise_balance_loss(
                    router_logits_list=token_router_logits,
                    top_k=token_top_k,
                    seq_lengths=None,
                    padding_len=0,
                )
                if layer_losses:
                    seq_wise_loss = seq_wise_loss_coeff * torch.stack(layer_losses).mean()

        # MoE monitoring metrics for token-MoE.
        moe_metrics = {}
        if router_logits_list:
            all_moe_indices = sorted(getattr(self.config, "token_moe_layers", None) or [])
            token_expert_counts = []

            with torch.no_grad():
                for i, logits in enumerate(router_logits_list):
                    layer_id = all_moe_indices[i] if i < len(all_moe_indices) else i
                    num_experts = logits.shape[-1]
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)

                    moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[layer_id].mlp
                    if hasattr(moe_block, "last_tokens_per_expert"):
                        counts = moe_block.last_tokens_per_expert.clone()
                    else:
                        _, selected = torch.topk(routing_probs, 1, dim=-1)
                        expert_indices = selected.squeeze(-1)
                        counts = F.one_hot(expert_indices, num_classes=num_experts).float().sum(dim=0)

                    token_expert_counts.append((layer_id, counts))

                    # MaxVio: (max_load - avg_load) / avg_load (paper 2408.15664)
                    avg_load = counts.mean()
                    maxvio = (counts.max() - avg_load) / avg_load.clamp(min=1e-9)
                    moe_metrics[f"token_moe/layer{layer_id}_maxvio"] = maxvio

                    per_sample_entropy = -(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1)
                    moe_metrics[f"token_moe/layer{layer_id}_entropy"] = per_sample_entropy.mean()

                # Compute average MaxVio across token-MoE layers
                token_maxvio_values = [
                    moe_metrics[k]
                    for k in moe_metrics
                    if k.startswith("token_moe/") and k.endswith("_maxvio")
                ]
                if token_maxvio_values:
                    moe_metrics["token_moe/avg_maxvio"] = torch.stack(token_maxvio_values).mean()

                # Avg top-K sigmoid score (before norm) across token-MoE layers
                token_moe_layers_list = sorted(getattr(self.config, "token_moe_layers", None) or [])
                if token_moe_layers_list:
                    sigmoid_scores = []
                    for lid in token_moe_layers_list:
                        moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[lid].mlp
                        if hasattr(moe_block, "avg_topk_sigmoid_score"):
                            sigmoid_scores.append(moe_block.avg_topk_sigmoid_score.detach().to(losses.device))
                    if sigmoid_scores:
                        moe_metrics["token_moe/avg_topk_sigmoid"] = torch.stack(sigmoid_scores).mean()

                if token_expert_counts:
                    moe_metrics["_token_moe_expert_counts"] = token_expert_counts

        return losses, loss_depth, depth_preds, seq_wise_loss, moe_metrics

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, vlm_causal=False, noise=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )  # bs, prefix_len, prefix_len
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Euler step
            x_t += dt * v_t
            time += dt
        print(f"Denoise {count} steps")
        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )  # bs, suffix_len, prefix_len+suffix_len

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            v_t = self.action_out_proj(suffix_out)
        return v_t

    def depth_emb_forward(self, hidden_states, depth_targets=None, img_masks=None, future_depth_targets=None):
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        num_images = img_masks.shape[1] if img_masks is not None and img_masks.ndim == 2 else 3
        if img_masks is not None:
            img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        align_embs = self._current_depth_task_tokens(hidden_states, num_images=num_images)
        align_embs = torch.cat([image_embs, align_embs], dim=1)
        depth_preds = self.depth_align_embs.repeat(align_embs.shape[0], 1, 1).to(
            dtype=align_embs.dtype, device=align_embs.device
        )
        depth_preds = self.depth_align_head(align_embs, depth_preds).contiguous().float()
        current_loss = self._emb_loss(depth_preds, depth_targets)

        if self.use_future_depth:
            future_align_embs = self._future_depth_task_tokens(hidden_states)
            future_image_embs = (
                image_embs.detach() if getattr(self, "detach_future_depth_image_feats", False) else image_embs
            )
            future_align_embs = torch.cat([future_image_embs, future_align_embs], dim=1)
            future_depth_preds = self.future_depth_align_embs.repeat(future_align_embs.shape[0], 1, 1).to(
                dtype=future_align_embs.dtype, device=future_align_embs.device
            )
            future_depth_preds = (
                self.future_depth_align_head(future_align_embs, future_depth_preds).contiguous().float()
            )
            future_loss = self._emb_loss(future_depth_preds, future_depth_targets)
            return current_loss, future_loss, depth_preds, future_depth_preds

        return current_loss, 0, depth_preds, None

    def video_emb_forward(
        self,
        hidden_states,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
    ):
        if self.align_type != "query":
            raise ValueError("future-video alignment is only supported for query align mode.")

        use_patch = getattr(self, "use_future_video_patch", True)
        use_cls = getattr(self, "use_future_video_cls", False)
        if not use_patch and not use_cls:
            raise ValueError("future-video alignment requires use_patch_loss or use_cls_loss to be enabled.")
        if use_patch and future_video_targets is None:
            raise ValueError("future_video_targets is required when use_patch_loss=True.")

        align_params = getattr(getattr(self, "config", None), "align_params", {}) or {}
        video_cfg = align_params.get("video", {}) if hasattr(align_params, "get") else {}
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        image_embs_for_video = (
            image_embs.detach() if bool(video_cfg.get("detach_image_feats", False)) else image_embs
        )

        cls_preds = None
        if use_cls:
            if future_video_cls_targets is None:
                raise ValueError("future_video_cls_targets is required when use_cls_loss=True.")
            cls_task_embs = self._future_video_cls_task_tokens(hidden_states)
            cls_delta = self.future_video_cls_head(cls_task_embs.squeeze(1))
            cls_preds = cls_delta.contiguous().float()

        loss = None
        metrics = {}
        video_preds = None
        if use_patch:
            video_task_embs = self._future_video_patch_task_tokens(hidden_states)
            context_mode = str(
                video_cfg.get(
                    "context_mode",
                    getattr(self, "future_video_context_mode", "img_query"),
                )
            ).lower()
            if context_mode == "query_only":
                video_align_embs = video_task_embs
            else:
                video_align_embs = torch.cat([image_embs_for_video, video_task_embs], dim=1)
            if getattr(self, "future_video_share_future_depth_query", False) and not getattr(
                self, "use_shared_future_task_proj", False
            ):
                query_embs = self.future_depth_align_embs
            else:
                query_embs = self.future_video_align_embs
            video_preds = query_embs.repeat(video_align_embs.shape[0], 1, 1).to(
                dtype=video_align_embs.dtype, device=video_align_embs.device
            )
            video_preds = self.future_video_align_head(video_align_embs, video_preds).contiguous().float()
            loss, metrics = self._video_emb_loss(video_preds, future_video_targets)
        if use_cls:
            cls_loss, cls_metrics = self._video_cls_loss(cls_preds, future_video_cls_targets)
            loss = cls_loss if loss is None else loss + cls_loss
            metrics.update(cls_metrics)
        return loss, video_preds, metrics

    def current_video_emb_forward(
        self,
        hidden_states,
        current_video_targets=None,
    ):
        if self.align_type != "query":
            raise ValueError("current-video alignment is only supported for query align mode.")
        if not getattr(self, "use_current_video_patch", False):
            raise ValueError("current-video alignment requires use_current_patch_loss=True.")
        if current_video_targets is None:
            raise ValueError("current_video_targets is required for current-video alignment.")

        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        current_task_embs = self._current_depth_task_tokens(hidden_states)
        align_embs = torch.cat([image_embs, current_task_embs], dim=1)
        queries = self.current_video_align_embs.repeat(align_embs.shape[0], 1, 1).to(
            dtype=align_embs.dtype,
            device=align_embs.device,
        )
        preds = self.current_video_align_head(align_embs, queries).contiguous().float()
        loss, metrics = self._video_emb_loss(
            preds,
            current_video_targets,
            metric_prefix="current_video",
        )
        return loss, preds, metrics

    def _video_emb_loss(self, video_preds, future_video_targets, metric_prefix="future_video"):
        align_params = getattr(getattr(self, "config", None), "align_params", {}) or {}
        video_cfg = align_params.get("video", {}) if hasattr(align_params, "get") else {}
        use_smooth_l1 = bool(video_cfg.get("use_smooth_l1_loss", True))
        use_mse = bool(video_cfg.get("use_mse_loss", False))
        use_cosine = bool(video_cfg.get("use_cosine_loss", False))
        if not use_smooth_l1 and not use_mse and not use_cosine:
            raise ValueError(f"{metric_prefix} loss requires smooth-L1, MSE, and/or cosine loss.")

        metrics = {}
        loss = None
        if use_smooth_l1:
            smooth_l1_loss = self._emb_loss(video_preds, future_video_targets)
            metrics[f"align/{metric_prefix}_smooth_l1_loss"] = smooth_l1_loss.detach()
            loss = smooth_l1_loss
        if use_mse:
            target = future_video_targets.to(dtype=video_preds.dtype, device=video_preds.device)
            mse_loss = F.mse_loss(video_preds.float(), target.float().detach())
            mse_weight = float(video_cfg.get("mse_loss_weight", 1.0))
            metrics[f"align/{metric_prefix}_mse_loss"] = mse_loss.detach()
            metrics[f"align/{metric_prefix}_mse_loss_weighted"] = (mse_loss * mse_weight).detach()
            weighted_mse_loss = mse_loss * mse_weight
            loss = weighted_mse_loss if loss is None else loss + weighted_mse_loss
        if use_cosine:
            target = future_video_targets.to(dtype=video_preds.dtype, device=video_preds.device)
            pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
            target_norm = F.normalize(target.float().detach(), dim=-1, eps=1e-6)
            cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
            cosine_weight = float(video_cfg.get("cosine_loss_weight", 1.0))
            metrics[f"align/{metric_prefix}_cosine_loss"] = cosine_loss.detach()
            metrics[f"align/{metric_prefix}_cosine_loss_weighted"] = (cosine_loss * cosine_weight).detach()
            weighted_cosine_loss = cosine_loss * cosine_weight
            loss = weighted_cosine_loss if loss is None else loss + weighted_cosine_loss
        return loss, metrics

    def _video_cls_loss(self, cls_preds, future_video_cls_targets):
        align_params = getattr(getattr(self, "config", None), "align_params", {}) or {}
        video_cfg = align_params.get("video", {}) if hasattr(align_params, "get") else {}
        cls_loss_type = str(video_cfg.get("cls_loss_type", "cosine")).lower()
        cls_weight = float(video_cfg.get("cls_loss_weight", 1.0))
        target = future_video_cls_targets.to(dtype=cls_preds.dtype, device=cls_preds.device)
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        metrics = {}
        loss = None
        if cls_loss_type in ("smooth_l1", "smoothl1", "huber"):
            smooth_l1_loss = F.smooth_l1_loss(cls_preds.float(), target.float().detach())
            metrics["align/future_video_cls_smooth_l1_loss"] = smooth_l1_loss.detach()
            loss = smooth_l1_loss
        if cls_loss_type in ("mse", "mse_cosine", "cosine_mse"):
            mse_loss = F.mse_loss(cls_preds.float(), target.float().detach())
            metrics["align/future_video_cls_mse_loss"] = mse_loss.detach()
            loss = mse_loss
        if cls_loss_type in ("cosine", "mse_cosine", "cosine_mse"):
            pred_norm = F.normalize(cls_preds.float(), dim=-1, eps=1e-6)
            target_norm = F.normalize(target.float().detach(), dim=-1, eps=1e-6)
            cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
            metrics["align/future_video_cls_cosine_loss"] = cosine_loss.detach()
            loss = cosine_loss if loss is None else loss + cosine_loss
        if loss is None:
            raise ValueError(f"Unsupported future-video CLS loss type: {cls_loss_type}")
        weighted_loss = loss * cls_weight
        metrics["align/future_video_cls_loss"] = loss.detach()
        metrics["align/future_video_cls_loss_weighted"] = weighted_loss.detach()
        return weighted_loss, metrics

    def _emb_loss(self, emb_preds, emb_targets):
        l1_loss = F.smooth_l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")
        return l1_loss.mean()


ModelClass = LingbotVlaPolicy

__all__ = [
    "LingbotVlaPolicy",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLTextModel",
    "Qwen2ForCausalLM",
    "Qwen2_5_VLPreTrainedModel",
]
