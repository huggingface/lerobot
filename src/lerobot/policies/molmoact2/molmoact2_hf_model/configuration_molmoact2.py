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


"""
MolmoAct2 configuration
"""

from typing import Any

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MolmoAct2VitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoAct2VisionTransformer`].
    It is used to instantiate a `MolmoAct2VisionTransformer` according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import MolmoAct2VitConfig, MolmoAct2VisionTransformer

    >>> # Initializing a MolmoAct2VitConfig
    >>> configuration = MolmoAct2VitConfig()

    >>> # Initializing a MolmoAct2VisionTransformer (with random weights)
    >>> model = MolmoAct2VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact2"
    base_config_key = "vit_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        image_default_input_size: tuple[int, int] = (378, 378),
        image_patch_size: int = 14,
        image_num_pos: int = 577,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        initializer_range: float = 0.02,
        float32_attention: bool = True,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.image_default_input_size = image_default_input_size
        self.image_patch_size = image_patch_size
        self.image_num_pos = image_num_pos
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.initializer_range = initializer_range
        self.float32_attention = float32_attention

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class MolmoAct2AdapterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of MolmoAct2Adapter. With MolmoAct2VitConfig,
    It is used to instantiate an MolmoAct2VisionBackbone according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MolmoAct2VitConfig, MolmoAct2AdapterConfig, MolmoAct2VisionBackbone

    >>> # Initializing a MolmoAct2VitConfig and a MolmoAct2AdapterConfig
    >>> vit_config = MolmoAct2VitConfig()
    >>> adapter_config = MolmoPoolingConfig()

    >>> # Initializing a MolmoAct2VisionBackbone (with random weights)
    >>> model = MolmoAct2VisionBackbone(vit_config, adapter_config)

    >>> # Accessing the model configuration
    >>> vit_configuration = model.vit_config
    >>> adapter_configuration = model.adapter_config
    ```"""

    model_type = "molmoact2"
    base_config_key = "adapter_config"

    def __init__(
        self,
        vit_layers: tuple = (-3, -9),
        pooling_attention_mask: bool = False,
        hidden_size: int = 1152,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        text_hidden_size: int = 3584,
        image_feature_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        self.vit_layers = vit_layers
        self.pooling_attention_mask = pooling_attention_mask
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_feature_dropout = image_feature_dropout
        self.initializer_range = initializer_range


class MolmoAct2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoAct2TextModel`]. It is used to instantiate a
    `MolmoAct2TextModel` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import MolmoAct2TextConfig, MolmoAct2TextModel

    >>> # Initializing a MolmoAct2TextConfig
    >>> configuration = MolmoAct2TextConfig()

    >>> # Initializing a MolmoAct2TextModel (with random weights)
    >>> model = MolmoAct2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.self_attn.att_proj": "colwise",
        "blocks.*.self_attn.attn_out": "rowwise",
        "blocks.*.mlp.ff_proj": "colwise",
        "blocks.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "wte": (["input_ids"], ["inputs_embeds"]),
        "blocks": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "ln_f": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int | None = 4,
        head_dim: int = 128,
        vocab_size: int = 152064,
        additional_vocab_size: int = 128,
        qkv_bias: bool = True,
        num_hidden_layers: int = 48,
        intermediate_size: int = 18944,
        hidden_act: str = "silu",
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] = None,
        rope_scaling_layers: list[int] | None = None,
        use_qk_norm: bool = False,
        qk_norm_type: str = "olmo",
        layer_norm_eps: int = 1e-6,
        norm_after: bool = False,
        initializer_range: float = 0.02,
        use_cache=True,
        tie_word_embeddings=False,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(
            tie_word_embeddings=tie_word_embeddings, attn_implementation=attn_implementation, **kwargs
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.qkv_bias = qkv_bias
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_scaling_layers = rope_scaling_layers
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.norm_after = norm_after
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)


class MolmoAct2ActionExpertConfig(PretrainedConfig):
    r"""Configuration for the MolmoAct2 modern action expert."""

    model_type = "molmoact2_action_expert"
    base_config_key = "action_expert_config"

    def __init__(
        self,
        max_action_horizon: int = 32,
        max_action_dim: int = 32,
        hidden_size: int = 1024,
        num_layers: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 8.0 / 3.0,
        ffn_multiple_of: int = 256,
        timestep_embed_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        context_layer_norm: bool = True,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
        causal_attn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_action_horizon = max_action_horizon
        self.max_action_dim = max_action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.ffn_multiple_of = ffn_multiple_of
        self.timestep_embed_dim = timestep_embed_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.context_layer_norm = context_layer_norm
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.rope = rope
        self.causal_attn = causal_attn

    def to_dict(self):
        output = super().to_dict()
        # These are derived from the parent MolmoAct2Config for HF exports. Keeping
        # them out of the public nested config avoids duplicated sources of truth.
        output.pop("max_action_horizon", None)
        output.pop("max_action_dim", None)
        return output


class MolmoAct2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoAct2ForConditionalGeneration`].
    It is used to instantiate an MolmoAct2 model according to the specified arguments, defining the model architecture.

    Example:

    ```python
    >>> from transformers import MolmoAct2Config, MolmoAct2VitConfig, MolmoAct2AdapterConfig, MolmoAct2TextConfig

    >>> # Initializing a MolmoAct2VitConfig
    >>> vit_config = MolmoAct2VitConfig()

    >>> # Initializing a MolmoAct2AdapterConfig
    >>> adapter_config = MolmoAct2AdapterConfig()

    >>> # Initializing a MolmoAct2TextConfig
    >>> text_config = MolmoAct2TextConfig()

    >>> # Initializing a MolmoAct2Config
    >>> configuration = MolmoAct2Config(
    >>>     vit_config=vit_config,
    >>>     adapter_config=adapter_config,
    >>>     text_config=text_config,
    >>>     image_start_token_id=151936,
    >>>     image_end_token_id=151937,
    >>>     image_patch_id=151938,
    >>>     image_col_id=151939,
    >>>     low_res_image_start_token_id=151940,
    >>>     image_low_res_id=151942,
    >>>     frame_start_token_id=151943,
    >>>     frame_end_token_id=151944,
    >>> )

    >>> # Initializing a model
    >>> model = MolmoAct2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact2"
    sub_configs = {
        "text_config": MolmoAct2TextConfig,
        "vit_config": MolmoAct2VitConfig,
        "adapter_config": MolmoAct2AdapterConfig,
        "action_expert_config": MolmoAct2ActionExpertConfig,
    }

    def __init__(
        self,
        vit_config: MolmoAct2VitConfig = None,
        adapter_config: MolmoAct2AdapterConfig = None,
        text_config: MolmoAct2TextConfig = None,
        action_expert_config: MolmoAct2ActionExpertConfig = None,
        image_start_token_id: int = None,
        low_res_image_start_token_id: int = None,
        image_end_token_id: int = None,
        image_low_res_id: int = None,
        image_patch_id: int = None,
        image_col_id: int = None,
        frame_start_token_id: int = None,
        frame_end_token_id: int = None,
        use_frame_special_tokens: bool = True,
        initializer_range: float = 0.02,
        add_action_expert: bool = True,
        max_action_dim: int = 32,
        max_action_horizon: int = 30,
        n_obs_steps: int = 30,
        action_mode: str = "both",
        state_format: str = "discrete",
        flow_matching_num_steps: int = 10,
        flow_matching_cutoff: float = 1.0,
        flow_matching_time_offset: float = 0.001,
        flow_matching_time_scale: float = 0.999,
        flow_matching_beta_alpha: float = 1.0,
        flow_matching_beta_beta: float = 1.5,
        mask_action_dim_padding: bool = True,
        enable_depth_reasoning: bool = False,
        depth_mode: int = 2,
        num_depth_codes: int = 100,
        action_expert_depth_gate: bool = False,
        action_expert_depth_gate_per_layer: bool = False,
        action_expert_depth_gate_init_bias: float = -4.0,
        action_output_token_id: int = None,
        action_start_token_id: int = None,
        action_end_token_id: int = None,
        action_token_start_id: int = None,
        num_action_tokens: int = 0,
        depth_output_token_id: int = None,
        depth_start_token_id: int = None,
        depth_end_token_id: int = None,
        depth_token_start_id: int = None,
        num_depth_tokens: int = 0,
        state_start_token_id: int = None,
        state_end_token_id: int = None,
        state_token_start_id: int = None,
        num_state_tokens: int = 0,
        add_setup_tokens: bool = True,
        add_control_tokens: bool = True,
        norm_stats_filename: str = "norm_stats.json",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vit_config is None:
            self.vit_config = MolmoAct2VitConfig()
        elif isinstance(vit_config, dict):
            self.vit_config = MolmoAct2VitConfig(**vit_config)
        else:
            self.vit_config = vit_config
        if adapter_config is None:
            self.adapter_config = MolmoAct2AdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = MolmoAct2AdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        if text_config is None:
            self.text_config = MolmoAct2TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = MolmoAct2TextConfig(**text_config)
        else:
            self.text_config = text_config
        self.add_action_expert = bool(add_action_expert)
        if not self.add_action_expert:
            self.action_expert_config = None
        elif action_expert_config is None:
            self.action_expert_config = MolmoAct2ActionExpertConfig(
                max_action_horizon=max_action_horizon,
                max_action_dim=max_action_dim,
                num_layers=self.text_config.num_hidden_layers,
            )
        elif isinstance(action_expert_config, dict):
            self.action_expert_config = MolmoAct2ActionExpertConfig(**action_expert_config)
        else:
            self.action_expert_config = action_expert_config
        if self.add_action_expert:
            self.action_expert_config.max_action_dim = int(max_action_dim)
            self.action_expert_config.max_action_horizon = int(max_action_horizon)
            self._validate_release_action_config(
                state_format=state_format,
            )
        self.image_start_token_id = image_start_token_id
        self.low_res_image_start_token_id = low_res_image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_low_res_id = image_low_res_id
        self.image_high_res_id = image_patch_id
        self.image_patch_id = image_patch_id
        self.image_col_id = image_col_id
        self.frame_start_token_id = frame_start_token_id
        self.frame_end_token_id = frame_end_token_id
        self.use_frame_special_tokens = use_frame_special_tokens
        self.initializer_range = initializer_range
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        self.n_obs_steps = n_obs_steps
        self.action_mode = action_mode
        self.state_format = state_format
        self.flow_matching_num_steps = flow_matching_num_steps
        self.flow_matching_cutoff = flow_matching_cutoff
        self.flow_matching_time_offset = flow_matching_time_offset
        self.flow_matching_time_scale = flow_matching_time_scale
        self.flow_matching_beta_alpha = flow_matching_beta_alpha
        self.flow_matching_beta_beta = flow_matching_beta_beta
        self.mask_action_dim_padding = mask_action_dim_padding
        self.enable_depth_reasoning = enable_depth_reasoning
        self.depth_mode = depth_mode
        self.num_depth_codes = num_depth_codes
        self.action_expert_depth_gate = action_expert_depth_gate
        self.action_expert_depth_gate_per_layer = action_expert_depth_gate_per_layer
        self.action_expert_depth_gate_init_bias = action_expert_depth_gate_init_bias
        self.action_output_token_id = action_output_token_id
        self.action_start_token_id = action_start_token_id
        self.action_end_token_id = action_end_token_id
        self.action_token_start_id = action_token_start_id
        self.num_action_tokens = num_action_tokens
        self.depth_output_token_id = depth_output_token_id
        self.depth_start_token_id = depth_start_token_id
        self.depth_end_token_id = depth_end_token_id
        self.depth_token_start_id = depth_token_start_id
        self.num_depth_tokens = num_depth_tokens
        self.state_start_token_id = state_start_token_id
        self.state_end_token_id = state_end_token_id
        self.state_token_start_id = state_token_start_id
        self.num_state_tokens = num_state_tokens
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens
        self.norm_stats_filename = norm_stats_filename

    @staticmethod
    def _validate_release_action_config(
        *,
        state_format: str,
    ) -> None:
        if state_format != "discrete":
            raise ValueError("MolmoAct2 HF export supports only state_format='discrete'.")

    @property
    def image_num_patch(self):
        assert self.vit_config is not None
        return self.vit_config.image_num_patch

    @property
    def num_attention_heads(self):
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self):
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.text_config.head_dim

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def max_position_embeddings(self):
        return self.text_config.max_position_embeddings


MolmoAct2VitConfig.register_for_auto_class()
MolmoAct2AdapterConfig.register_for_auto_class()
MolmoAct2TextConfig.register_for_auto_class()
MolmoAct2ActionExpertConfig.register_for_auto_class()
MolmoAct2Config.register_for_auto_class()
