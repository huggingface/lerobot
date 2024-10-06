# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2VL model configuration"""
import os
from typing import Union
from transformers.utils import logging # Using standard Python logging module instead of `transformers.utils.logging`
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

logger = logging.get_logger(__name__)

class Qwen2VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_vl"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
   
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "qwen2_vl":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

      

class Qwen2VLConfig(PretrainedConfig):
    r"""
    A simplified version of the Qwen2VL model configuration class without the `transformers` dependencies.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064): Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 8192): Dimension of hidden representations.
        input_shapes (`dict`, *optional*): Dictionary specifying the input shapes, including observation state.
        output_shapes (`dict`, *optional*): Dictionary specifying the output shapes, including action.
    """
    model_type = "qwen2_vl"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_decoder_layers = 1,
        _attn_implementation="eager",
        num_attention_heads=64,
        num_key_value_heads=8,
        dim_feedforward = 3200,
        hidden_act="silu",
        pad_token_id=0,
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling={"type": "mrope", "mrope_section": [2, 2, 2]},
        input_shapes={
        "observation.state": [128]},
        output_shapes={
        "action": [64]},
        pruned_heads=None,
        **kwargs,
    ):
        # Initialize vision config
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()

        # Input and output shapes
        self.input_shapes = input_shapes if input_shapes is not None else {}
        self.output_shapes = output_shapes if output_shapes is not None else {}

        # Model hyperparameters
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.pad_token_id = pad_token_id 
        self._attn_implementation=_attn_implementation
        self.pruned_heads = pruned_heads or {}
        self.rope_scaling = rope_scaling
        self.num_decoder_layers = num_decoder_layers

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Custom loading logic from a pre-trained model or path
        logger.info(f"Loading pretrained config from {pretrained_model_name_or_path}...")
        # Add custom logic here to load a pretrained configuration
        return cls(**kwargs)
