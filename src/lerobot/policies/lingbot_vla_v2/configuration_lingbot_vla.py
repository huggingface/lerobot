# Copyright 2026 Robbyant Team and/or its affiliates
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


from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from transformers import AutoConfig, PretrainedConfig


class LingbotVLAConfig(PretrainedConfig):
    """Configuration class for Lingbot-VLA.
    This is the configuration class to store the configuration of a [`Lingbot-VLA`].
    """

    model_type = "lingbotvla"
    is_composition = True

    def __init__(
        self,
        vlm_repo_id: Optional[str] = None,
        expert_vision_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        post_training: bool = False,
        adanorm_time: bool = False,
        split_gate_liner: bool = False,
        nosplit_gate_liner: bool = False,
        separate_time_proj: bool = False,
        final_norm_adanorm: bool = False,
        enable_expert_vision: bool = False,
        expert_vision_type: Optional[str] = None,
        freeze_vision_encoder: bool = False,
        incremental_training: bool = False,
        depth_incremental_training: bool = False,
        reinit_mismatched_weights: bool = False,
        action_dim: int = 14,
        max_action_dim: int = 14,
        max_state_dim: int = 14,
        chunk_size: int = 50,
        vlm_causal: bool = False,
        tokenizer_max_length: int = 48,
        loss_type: str = "fm",
        norm_qkv: bool = False,
        align_params: Optional[Dict[str, Any]] = None,
        use_compile: bool = False,
        use_moe: bool = False,
        token_moe_layers: Optional[list] = None,
        token_num_experts: int = 32,
        token_top_k: int = 1,
        token_moe_intermediate_size: int = 256,
        token_shared_intermediate_size: int = 256,
        bias_update_speed: float = 0.001,
        sequence_wise_loss_coeff: float = 0.001,
        sequence_wise_mode: str = "per_sequence",
        router_z_loss_coeff: float = 0.0,
        router_activation: str = "softmax",
        routed_scaling_factor: float = 1.0,
        use_shared_expert_gate: bool = True,
        moe_implementation: Optional[Literal[None, "eager", "fused"]] = None,
        split_fused_experts_from_decoder_fsdp: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 16,
        action_num_key_value_heads: int = 2,
        action_head_dim: int = 128,
        action_fp32: bool = False,
        use_qwen3_chat_template: bool = False,
        return_image_grid_thw: bool = False,
        qwen3vl_use_vision_boundaries: bool = False,
        precompute_grid_thw: bool = False,
        use_qwen3_fixed_grid_cache: bool = False,
        use_lm_head: bool = False,
        vocab_size: int = 0,
        vit_attn_implementation: str = "flash_attention_2",
        attention_implementation: str = "flex",
        train_expert_only: bool = False,
        train_state_proj: bool = True,
        **kwargs,
    ):
        super().__init__()
        if moe_implementation is None:
            moe_implementation = kwargs.pop("_moe_implementation", None)
        self.architectures = ["LingbotVlaPolicy"]
        self.train_state_proj = train_state_proj
        self.train_expert_only = train_expert_only
        self.use_cache = False
        self.attention_implementation = attention_implementation
        self.num_steps = 10
        self.n_obs_steps = 1

        assert not (split_gate_liner and nosplit_gate_liner), (
            "split_gate_liner and nosplit_gate_liner can not be both True."
        )

        self.vlm_repo_id = vlm_repo_id
        self.expert_vision_path = expert_vision_path
        self.tokenizer_path = tokenizer_path
        self.post_training = post_training
        self.adanorm_time = adanorm_time
        self.split_gate_liner = split_gate_liner
        self.nosplit_gate_liner = nosplit_gate_liner
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.incremental_training = incremental_training
        self.depth_incremental_training = depth_incremental_training
        self.reinit_mismatched_weights = reinit_mismatched_weights
        self.norm_qkv = norm_qkv
        self.use_compile = use_compile
        self.loss_type = loss_type
        self.separate_time_proj = separate_time_proj
        self.final_norm_adanorm = final_norm_adanorm
        self.freeze_vision_encoder = freeze_vision_encoder
        self.tokenizer_max_length = tokenizer_max_length
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.vlm_causal = vlm_causal
        self.align_params = align_params
        self.use_moe = use_moe
        if self.use_moe:
            self.token_moe_layers = token_moe_layers
            self.token_num_experts = token_num_experts
            self.token_top_k = token_top_k
            self.token_moe_intermediate_size = token_moe_intermediate_size
            self.token_shared_intermediate_size = token_shared_intermediate_size
        self.bias_update_speed = bias_update_speed
        self.sequence_wise_loss_coeff = sequence_wise_loss_coeff
        self.sequence_wise_mode = sequence_wise_mode
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_activation = router_activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert_gate = use_shared_expert_gate
        self.moe_implementation = moe_implementation
        if moe_implementation is not None:
            if moe_implementation not in ("eager", "fused"):
                raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
            self._moe_implementation = moe_implementation
        self.split_fused_experts_from_decoder_fsdp = split_fused_experts_from_decoder_fsdp
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        self.action_fp32 = action_fp32
        self.use_qwen3_chat_template = use_qwen3_chat_template
        self.return_image_grid_thw = return_image_grid_thw
        self.qwen3vl_use_vision_boundaries = qwen3vl_use_vision_boundaries
        self.precompute_grid_thw = precompute_grid_thw
        self.use_qwen3_fixed_grid_cache = use_qwen3_fixed_grid_cache
        self.use_lm_head = use_lm_head
        if vocab_size == 0:
            if vlm_repo_id and "paligemma" in vlm_repo_id.lower():
                self.vocab_size = 257216
            elif vlm_repo_id and "qwen" in vlm_repo_id.lower():
                self.vocab_size = 151936
            else:
                self.vocab_size = 257152
        else:
            self.vocab_size = vocab_size
        self.vit_attn_implementation = vit_attn_implementation


class LingbotVLAV2Config(LingbotVLAConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("attention_implementation", "flex_cached")
        kwargs.setdefault("vit_attn_implementation", "flash_attention_2")
        kwargs.setdefault("action_num_attention_heads", 32)
        kwargs.setdefault("action_num_key_value_heads", 8)
        kwargs.setdefault("action_head_dim", 128)
        kwargs.setdefault("expert_hidden_size", 768)
        kwargs.setdefault("use_qwen3_chat_template", True)
        kwargs.setdefault("return_image_grid_thw", True)
        kwargs.setdefault("qwen3vl_use_vision_boundaries", True)
        kwargs.setdefault("use_qwen3_fixed_grid_cache", True)
        super().__init__(**kwargs)
        self.architectures = ["LingbotVlaV2Policy"]
        self.vlm_family = "qwen3_vl"


ConfigClass = [LingbotVLAConfig, LingbotVLAV2Config]
__all__ = ["LingbotVLAConfig", "LingbotVLAV2Config"]
