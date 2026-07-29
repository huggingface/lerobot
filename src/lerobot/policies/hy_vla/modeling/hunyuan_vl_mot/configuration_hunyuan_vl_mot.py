# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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

"""Configuration class for HunYuanVL-MoT (Mixture of Transformers) model."""

from transformers.configuration_utils import PretrainedConfig


class HunYuanVLMoTVisionConfig(PretrainedConfig):
    """
    Configuration class for HunYuanVL-MoT Vision Transformer.
    """

    model_type = "hunyuan_vl_mot"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode="bilinear",
        rms_norm_eps=1e-05,
        attention_dropout=0.0,
        learnable_mlp_pooling_size=0,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        remove_prenorm=True,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        text_hidden_size=4096,
        anyres_vit_max_image_size=2048,
        cat_extra_token=1,
        img_max_token_num=4096,
        max_image_size=2048,
        max_vit_seq_len=16384,
        min_image_size=512,
        resize_resolution=2048,
        vision_full_attention=False,
        video_max_image_size=768,
        video_min_image_size=256,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.learnable_mlp_pooling_size = learnable_mlp_pooling_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.remove_prenorm = remove_prenorm
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.text_hidden_size = text_hidden_size
        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.cat_extra_token = cat_extra_token
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.min_image_size = min_image_size
        self.resize_resolution = resize_resolution
        self.vision_full_attention = vision_full_attention
        self.video_max_image_size = video_max_image_size
        self.video_min_image_size = video_min_image_size
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm


class HunYuanVLMoTTextConfig(PretrainedConfig):
    r"""
    Configuration class for HunYuanVL-MoT language model with Mixture of Transformers attention.

    The Mixture of Transformers (MoT) mechanism enables:
    - Separate attention paths for text and vision tokens
    - Packed variable-length sequences with eager attention
    - Different causal masks per modality (text=causal, vision=non-causal)

    Args:
        vocab_size (`int`, *optional*, defaults to 120818):
            Vocabulary size of the HunYuan MoT model (Chinese-focused).
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the feed-forward (MLP) representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers (decoder layers with MoT attention).
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of query attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads (Grouped Query Attention). Defaults to `num_attention_heads` if None.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (SiLU for MoT).
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length (supports very long sequences with RoPE).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS norm layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache for efficient generation.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE (Rotary Position Embeddings).
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for RoPE (e.g., dynamic scaling).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP projections.
        attention_dropout (`float`, *optional*, defaults to 0.0`):
            Dropout rate for attention (typically 0 for MoT).
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use layer normalization on Q and K in attention (important for MoT stability).
        use_rotary_pos_emb (`bool`, *optional*, defaults to `True`):
            Whether to use RoPE (Rotary Position Embeddings).
    """

    model_type = "hunyuan_vl_mot"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=120818,
        org_vocab_size=120818,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        attention_head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=120002,
        bos_token_id=120000,
        eos_token_id=120020,
        tie_word_embeddings=True,
        pretraining_tp=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_qk_norm=True,
        use_rotary_pos_emb=True,
        # MoT-specific parameters
        vision_full_attention=False,
        head_dim=128,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.vision_full_attention = vision_full_attention

        # Attention head dimension
        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        self.head_dim = head_dim if head_dim is not None else self.attention_head_dim


class HunYuanVLMoTConfig(PretrainedConfig):
    r"""
    This is the config class to store the configuration of a [`HunYuanVLMoTForConditionalGeneration`] model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`HunYuanVLMoTTextConfig`, *optional*):
            Model configuration class with all the parameters of the text model.
        vision_config (`HunYuanVLMoTVisionConfig`, *optional*):
            Model configuration class with all the parameters of the vision model.
        image_start_token_id (`int`, *optional*, defaults to 120119):
            The id of the token used for image start marker.
        image_end_token_id (`int`, *optional*, defaults to 120120):
            The id of the token used for image end marker.
        image_token_id (`int`, *optional*, defaults to 120687):
            The id of the token used for image embeddings (MoT uses different ID than hunyuan_vl).
        video_start_token_id (`int`, *optional*, defaults to 120122):
            The id of the token used for video start marker.
        video_end_token_id (`int`, *optional*, defaults to 120123):
            The id of the token used for video end marker.
    """

    model_type = "hunyuan_vl_mot"
    sub_configs = {
        "vision_config": HunYuanVLMoTVisionConfig,
        "text_config": HunYuanVLMoTTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=120119,
        image_end_token_id=120120,
        image_token_id=120687,  # Different from hunyuan_vl (120120)
        video_start_token_id=120122,
        video_end_token_id=120123,
        latent_token_id=120690,  # MoT-specific
        **kwargs,
    ):
        kwargs.pop("attn_implementation", None)
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        # Image/video token IDs (MoT-specific)
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id  # 120687
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.latent_token_id = latent_token_id  # 120690

        # Align vision config text hidden size with actual text model size
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        self._attn_implementation = "eager"

    @property
    def is_moe(self) -> bool:
        """For future MoE support in MoT variant."""
        return False  # MoT variant does not use MoE, uses Mixture of Transformers instead

    def __setattr__(self, key, value):
        """
        Proxy attributes to text_config for convenience.
        Allows `config.hidden_size` instead of `config.text_config.hidden_size`.
        """
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        """
        Proxy attributes from text_config for convenience.
        Allows `config.hidden_size` instead of `config.text_config.hidden_size`.
        """
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)
