#!/usr/bin/env python

# Copyright 2026 S-Lab and The HuggingFace Inc. team. All rights reserved.
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

import functools
import logging
import typing

import torch
import torch.nn.functional as F
from timm.layers import DropPath, SqueezeExcite
from transformers import (
    AutoModel,
    GenerationMixin,
    LlamaConfig,
    LlamaModel,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class KwargsForCausalLM(FlashAttentionKwargs, TransformersKwargs): ...


class FastVLMBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: typing.Optional[torch.FloatTensor] = None
    past_key_values: typing.Optional[Cache] = None
    hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None
    attentions: typing.Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None


class FastVLMCausalLMOutputWithPast(ModelOutput):
    loss: typing.Optional[torch.FloatTensor] = None
    logits: typing.Optional[torch.FloatTensor] = None
    past_key_values: typing.Optional[Cache] = None
    hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None
    attentions: typing.Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None


class FastViTConfig(PretrainedConfig):
    model_type = "fastvit"
    base_config_key = "vision_config"

    def __init__(
        self,
        in_channels=3,
        out_channels=768,
        image_size=1024,
        patch_size=64,
        n_blocks=[2, 12, 24, 4, 2],
        embed_dims=[96, 192, 384, 768, 1536],
        mlp_ratios=[4, 4, 4, 4, 4],
        downsample=[True, True, True, True, True],
        downsample_patch_size=7,
        downsample_stride=2,
        downsample_use_se=None,
        position_embeddings=None,
        token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
        repmixer_kernel_size=3,
        use_scale_branch=True,
        use_layer_scale=True,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        clsss_ratio=2,
        inference_mode=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_blocks = n_blocks
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.downsample = downsample
        self.downsample_patch_size = downsample_patch_size
        self.downsample_stride = downsample_stride
        self.downsample_use_se = downsample_use_se
        self.position_embeddings = position_embeddings
        self.token_mixers = token_mixers
        self.repmixer_kernel_size = repmixer_kernel_size
        self.use_scale_branch = use_scale_branch
        self.use_layer_scale = use_layer_scale
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.clsss_ratio = clsss_ratio
        self.inference_mode = inference_mode


class FastVLMConfig(PretrainedConfig):

    model_type = "fastvlm"
    sub_configs = {"text_config": LlamaConfig | Qwen2Config, "vision_config": FastViTConfig}

    def __init__(
        self,
        use_cache=True,
        tie_word_embeddings=False,
        max_position_embeddings=None,
        text_config=None,
        vision_config=None,
        image_token_id=128_257,
        pad_token_id=128_002,
        **kwargs,
    ) -> None:
        # ViTConfig
        if vision_config is None:
            self.vision_config = FastViTConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = FastViTConfig(**vision_config)
        elif isinstance(vision_config, FastViTConfig):
            self.vision_config = vision_config
        else:
            raise ValueError("No valid vision_config is provided.")

        # LlamaConfig
        if text_config is None:
            self.text_config = LlamaConfig()
        elif isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif isinstance(text_config, LlamaConfig) or isinstance(text_config, Qwen2Config):
            self.text_config = text_config
        else:
            raise ValueError("No valid text_config is provided.")

        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        if max_position_embeddings is not None:
            self.text_config.max_position_embeddings = max_position_embeddings

        super().__init__(
            **kwargs, pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings
        )


class FastVLMPreTrainedModel(PreTrainedModel):
    config_class = FastVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(
            self.config,
            "initializer_range",
            self.config.get_text_config().initializer_range,
        )

        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class FastVLMForConditionalGeneration(FastVLMPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = FastVLMModel(config)
        self.lm_head = torch.nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.vocab_size = config.text_config.vocab_size
        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning
        adapter weights while keeping the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = (
            self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        )
        self._vision_require_grads_hook = (
            self.model.vision_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grads
            )
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: typing.Optional[torch.LongTensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        pixel_values: typing.Optional[torch.FloatTensor] = None,
        pixel_attention_mask: typing.Optional[torch.BoolTensor] = None,
        image_hidden_states: typing.Optional[torch.FloatTensor] = None,
        labels: typing.Optional[torch.LongTensor] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        use_cache: typing.Optional[bool] = None,
        cache_position: typing.Optional[torch.LongTensor] = None,
        return_dict: typing.Optional[bool] = None,
        logits_to_keep: typing.Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> typing.Union[typing.Tuple, FastVLMCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not
        # computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return FastVLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


class FastVLMModel(FastVLMPreTrainedModel):
    def __init__(self, config: FastVLMConfig) -> None:
        super().__init__(config)
        self.vision_model = FastViT(config.vision_config)
        self.connector = FastVLMConnector(config)
        self.text_model: LlamaModel = AutoModel.from_config(config.text_config)
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's
        embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = (
            self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        )
        self._vision_require_grads_hook = get_lowest_module(
            self.vision_model
        ).register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self) -> None:
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor):
        batch_size, num_images, _, _, _ = pixel_values.shape
        pixel_values = pixel_values.view(
            batch_size * num_images, *pixel_values.shape[2:]
        )
        # Modality projection & resampling
        image_features = self.vision_model(pixel_values).flatten(2).transpose(1, 2)
        image_features = self.connector(image_features)
        return image_features

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ):
        _, patch_size, _ = image_hidden_states.shape
        image_mask = input_ids == self.config.image_token_id
        num_image_tokens = image_mask.sum(dim=1)

        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError(
                "At least one sample has <image> tokens not divisible by patch_size."
            )

        blocks_per_sample = num_image_tokens // patch_size
        offsets = torch.nn.functional.pad(
            blocks_per_sample.cumsum(dim=0), (1, 0), value=0
        )
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[
            block_idx[image_mask], local_idx[image_mask], :
        ]

        merged_embeds = torch.where(
            image_mask.unsqueeze(-1), image_embeds, inputs_embeds
        )
        return merged_embeds

    def forward(
        self,
        input_ids: typing.Optional[torch.LongTensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        pixel_values: typing.Optional[torch.FloatTensor] = None,
        image_hidden_states: typing.Optional[torch.FloatTensor] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        use_cache: typing.Optional[bool] = None,
        cache_position: typing.Optional[torch.LongTensor] = None,
        return_dict: typing.Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> FastVLMBaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            _, _ = input_ids.shape
        elif inputs_embeds is not None:
            _, _, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()

            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError(
                "When first calling the model, if input_embeds are passed, input_ids "
                "should not be None."
            )

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(
                input_ids.device
            )

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "You cannot specify both pixel_values and image_hidden_states at the "
                "same time"
            )
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values).to(
                dtype=self.dtype, device=input_ids.device
            )
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.dtype, device=input_ids.device
            )

        if inputs_embeds is not None and image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        return FastVLMBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class FastViT(torch.nn.Module):
    """The FastViT model. <https://arxiv.org/pdf/2303.14189.pdf>"""

    def __init__(self, config: FastViTConfig) -> None:
        super().__init__()
        n_stages = len(config.n_blocks)
        if config.position_embeddings is None:
            config.position_embeddings = [None] * n_stages
        if config.downsample_use_se is None:
            config.downsample_use_se = [False] * n_stages

        # Stem
        self.patch_embedding = torch.nn.Sequential(
            MobileOneBlock(
                in_channels=config.in_channels,
                out_channels=config.embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                inference_mode=config.inference_mode,
                use_se=False,
                num_conv_branches=1,
                use_scale_branch=config.use_scale_branch,
            ),
            MobileOneBlock(
                in_channels=config.embed_dims[0],
                out_channels=config.embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=config.embed_dims[0],
                inference_mode=config.inference_mode,
                use_se=False,
                num_conv_branches=1,
                use_scale_branch=config.use_scale_branch,
            ),
            MobileOneBlock(
                in_channels=config.embed_dims[0],
                out_channels=config.embed_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=config.inference_mode,
                use_se=False,
                num_conv_branches=1,
                use_scale_branch=config.use_scale_branch,
            ),
        )
        # Stage Blocks
        stages = []
        for i in range(n_stages):
            # Add position embeddings as requested
            pe = self._get_position_embedding(config.position_embeddings[i])
            if pe is not None:
                stages.append(
                    pe(
                        config.embed_dims[i],
                        config.embed_dims[i],
                        inference_mode=config.inference_mode,
                    )
                )

            stage = self._get_stage_blocks(
                config.embed_dims[i],
                i,
                config.n_blocks,
                token_mixer_type=config.token_mixers[i],
                kernel_size=config.repmixer_kernel_size,
                mlp_ratio=config.mlp_ratios[i],
                norm_layer=LayerNormChannel,
                drop_path_rate=config.drop_path_rate,
                use_layer_scale=config.use_layer_scale,
                layer_scale_init_value=config.layer_scale_init_value,
                inference_mode=config.inference_mode,
            )
            stages.append(stage)
            # Patch merging/downsampling between stages
            if i == n_stages - 1:
                break
            if config.downsample[i] or config.embed_dims[i] != config.embed_dims[i + 1]:
                stages.append(
                    PatchEmbedding(
                        patch_size=config.downsample_patch_size,
                        stride=config.downsample_stride,
                        in_channels=config.embed_dims[i],
                        embed_dim=config.embed_dims[i + 1],
                        inference_mode=config.inference_mode,
                        use_se=config.downsample_use_se[i + 1],
                    )
                )

        self.layers = torch.nn.ModuleList(stages)
        self.conv_exp = MobileOneBlock(
            in_channels=config.embed_dims[-1],
            out_channels=config.embed_dims[-1] * config.clsss_ratio,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.embed_dims[-1],
            inference_mode=config.inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def _get_position_embedding(self, config):
        if config is None:
            return None
        elif config["name"] == "RepCPE":
            return functools.partial(RepCPE, spatial_shape=config["spatial_shape"])
        else:
            raise ValueError(f"Position embedding {config['name']} not supported.")

    def _get_stage_blocks(
        self,
        dim: int,
        block_index: int,
        num_blocks: int,
        token_mixer_type: str,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.BatchNorm2d,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode=False,
    ) -> torch.nn.Sequential:
        blocks = []
        for block_idx in range(num_blocks[block_index]):
            block_dpr = (
                drop_path_rate
                * (block_idx + sum(num_blocks[:block_index]))
                / (sum(num_blocks) - 1)
            )
            if token_mixer_type == "repmixer":
                blocks.append(
                    RepMixerBlock(
                        dim,
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        drop=drop_rate,
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                        inference_mode=inference_mode,
                    )
                )
            elif token_mixer_type == "attention":
                blocks.append(
                    AttentionBlock(
                        dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        drop=drop_rate,
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                raise ValueError(
                    "Token mixer type: {} not supported".format(token_mixer_type)
                )

        return torch.nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x)

        return self.conv_exp(x)


class FastVLMConnector(torch.nn.Module):
    def __init__(self, config: FastVLMConfig) -> None:
        super().__init__()

        vision_out_dim = (
            config.vision_config.embed_dims[-1] * config.vision_config.clsss_ratio
        )
        llm_hidden_size = config.text_config.hidden_size
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(vision_out_dim, llm_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.connector(x)


# NOTE: The following components are copied from
# https://huggingface.co/apple/FastVLM-0.5B/blob/main/llava_qwen.py
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


class RepCPE(torch.nn.Module):
    """Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: typing.Union[int, typing.Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, typing.Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = torch.nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        self.__delattr__("pe")


class LayerNormChannel(torch.nn.Module):
    """LayerNorm only for Channel Dimension"""

    def __init__(self, num_features, eps=1e-05) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
            -1
        ).unsqueeze(-1)

        return x


class MobileOneBlock(torch.nn.Module):
    """The MobileOne building block. <https://arxiv.org/pdf/2206.04040.pdf>"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: torch.nn.Module = torch.nn.GELU(),
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = torch.nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = torch.nn.Identity()

        if inference_mode:
            self.reparam_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            # Fallback, sometimes batchnorm tensors
            # do not get instantiated correctly on some processes
            # when using deepspeed + accelerate
            norm_layer = torch.nn.BatchNorm2d(num_features=in_channels)
            if norm_layer.weight.shape[0] == 0:
                norm_layer.weight = torch.nn.Parameter(torch.zeros(in_channels))
            if norm_layer.bias.shape[0] == 0:
                norm_layer.bias = torch.nn.Parameter(torch.zeros(in_channels))

            self.rbr_skip = (
                norm_layer if out_channels == in_channels and stride == 1 else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = torch.nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: typing.Union[torch.nn.Sequential, torch.nn.BatchNorm2d]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        """
        if isinstance(branch, torch.nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, torch.nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups

                kernel_size = self.kernel_size
                if isinstance(self.kernel_size, int):
                    kernel_size = (self.kernel_size, self.kernel_size)

                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> torch.nn.Sequential:
        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = torch.nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = torch.nn.Parameter(torch.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = torch.nn.Parameter(torch.zeros(self.out_channels))

        mod_list = torch.nn.Sequential()
        mod_list.add_module(
            "conv",
            torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list


class SEBlock(torch.nn.Module):
    """The Squeeze and Excite module. <https://arxiv.org/pdf/1709.01507.pdf>"""

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = torch.nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        # x = F.avg_pool2d(inputs, kernel_size=[16, 16])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class RepMixerBlock(torch.nn.Module):
    """The Metaformer block with RepMixer as token mixer <https://arxiv.org/pdf/2111.11418.pdf>"""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: torch.nn.Module = torch.nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = torch.nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class RepMixer(torch.nn.Module):
    """Reparameterizable token mixer. <https://arxiv.org/pdf/2303.14189.pdf>"""

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = torch.nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = torch.nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = torch.nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(torch.nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: typing.Optional[int] = None,
        out_channels: typing.Optional[int] = None,
        act_layer: torch.nn.Module = torch.nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = torch.nn.Sequential()
        self.conv.add_module(
            "conv",
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )

        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = torch.nn.BatchNorm2d(num_features=out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = torch.nn.Parameter(torch.zeros(out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = torch.nn.Parameter(torch.zeros(out_channels))

        self.conv.add_module("bn", norm_layer)
        self.fc1 = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = torch.nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(torch.nn.Module):
    """The metaformer block with MHSA as token mixer. <https://arxiv.org/pdf/2111.11418.pdf>"""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()

        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer_ = norm_layer(num_features=dim)
        if norm_layer_.weight.shape[0] == 0:
            norm_layer_.weight = torch.nn.Parameter(torch.zeros(dim))
        if norm_layer_.bias.shape[0] == 0:
            norm_layer_.bias = torch.nn.Parameter(torch.zeros(dim))

        self.norm = norm_layer_
        self.token_mixer = MHSA(dim=dim)
        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = torch.nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = torch.nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


class MHSA(torch.nn.Module):
    """Multi-headed Self Attention module. Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbedding(torch.nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = torch.nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class ReparamLargeKernelConv(torch.nn.Module):
    """The Building Block of RepLKNet <https://arxiv.org/abs/2203.06717>"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: torch.nn.Module = torch.nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        # Check if SE is requested
        if use_se:
            self.se = SqueezeExcite(out_channels, rd_ratio=0.25)
        else:
            self.se = torch.nn.Identity()

        if inference_mode:
            self.lkb_reparam = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        return self.activation(self.se(out))

    def get_kernel_bias(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += torch.nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: torch.nn.BatchNorm2d
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> torch.nn.Sequential:
        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = torch.nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = torch.nn.Parameter(torch.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = torch.nn.Parameter(torch.zeros(self.out_channels))

        mod_list = torch.nn.Sequential()
        mod_list.add_module(
            "conv",
            torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list
