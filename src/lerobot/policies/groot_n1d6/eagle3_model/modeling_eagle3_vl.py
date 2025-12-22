# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import inspect
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.utils.checkpoint as cp
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from .modeling_siglip2 import Siglip2VisionModel
from peft import LoraConfig, get_peft_model
from transformers.generation import GenerationMixin
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from .configuration_eagle3_vl import Eagle3_VLConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from collections import defaultdict
logger = logging.get_logger(__name__)
    
# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1
EAGLE3_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Eagle3_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Eagle3_VL Model outputting raw hidden-states without any specific head on top.",
    EAGLE3_VL_START_DOCSTRING,
)
class Eagle3_VLPreTrainedModel(PreTrainedModel):
    config_class = Eagle3_VLConfig
    base_model_prefix = "model"
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "LlamaDecoderLayer" ,"Siglip2EncoderLayer", "SiglipEncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Eagle3_VLForConditionalGeneration(Eagle3_VLPreTrainedModel, GenerationMixin):
    config_class = Eagle3_VLConfig
    def __init__(self, config: Eagle3_VLConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.loss_version = config.loss_version
        self.mlp_checkpoint = config.mlp_checkpoint

        logger.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'intern_vit_6b':
                self.vision_model = InternVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip2_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = Siglip2VisionModel(config.vision_config)
            elif config.vision_config.model_type == 'radio':
                self.vision_model = RADIOModel(config.vision_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen2 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen3 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise NotImplementedError(f'{config.text_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        self.image_token_index = config.image_token_index
        self.neftune_alpha = None


        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora 
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
            
        self.check_forward_kwargs()
        
    def check_forward_kwargs(self):
        # We intentionally avoid using **kwargs in forward because Hugging Face Transformers
        # has special handling for functions with **kwargs parameters that would affect
        # how our model is processed during training and inference.
        forward_params = inspect.signature(self.forward).parameters
        assert not any(k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values())

        
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj',
                            'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        self.use_llm_lora = True
        
    def forward(
            self,
            pixel_values: List[torch.FloatTensor],
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        num_images = len(pixel_values)
        
        if image_flags is not None:
            image_flags = image_flags.view(-1)

        vit_embeds = self.extract_feature(pixel_values, image_flags)


        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.image_token_index)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        except Exception as e:
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle_back(self, vit_embeds, spatial_shapes):
        # Assume vit_embeds: [1, 15020, 1152], spatial_shapes: [(h1,w1), (h2,w2), ...] length 64
        B, N, C = vit_embeds.shape
        shapes = spatial_shapes.tolist()  # List of (h, w)

        # 1) Split at once
        lengths = [h * w for (h, w) in shapes]               # Number of patches per image
        slices = torch.split(vit_embeds.view(-1, C), lengths, dim=0)
        # slices[i]: [hi*wi, C]

        # 2) Convert to [C, H, W]
        features = [
            sl.transpose(0, 1).reshape(C, h, w)
            for sl, (h, w) in zip(slices, shapes)
        ]  # Each item [C, hi, wi]
        # visualize_tensor_list(features, 'features.jpg')

        # 3) Group by scale and batch unshuffle
        down_feats = [None] * len(features)
        grouped: dict = defaultdict(list)
        for idx, (h, w) in enumerate(shapes):
            grouped[(h, w)].append(idx)

        for (h, w), idxs in grouped.items():
            # Stack features of the same scale -> [n, C, H, W]
            grp = torch.stack([features[i] for i in idxs], dim=0)
            # Pixel Unshuffle at once
            out = F.pixel_unshuffle(grp, downscale_factor=int(1/self.downsample_ratio))  # [n, C*4, H//2, W//2]
            out = out.flatten(start_dim=2).transpose(1, 2)  # [n, H//2 * W//2, C*4]
            # Split back to respective positions
            for i, feat in zip(idxs, out):
                down_feats[i] = feat
        
        down_feats = torch.cat(down_feats, dim=0).unsqueeze(0)
        return down_feats, (spatial_shapes*self.downsample_ratio).to(torch.int32)

    def mask_valid_tokens(self, vit_embeds, spatial_shapes, image_flags):
        """
        vit_embeds: Tensor, shape [1, N, C] or [N, C]
        spatial_shapes: Tensor of shape [num_images, 2], each row is (H, W)
        image_flags: list[int], e.g. [1, 0, 1, ...]
        Returns:
        valid_tokens: Tensor [num_valid_tokens, C]
        """

        lengths = spatial_shapes[:, 0] * spatial_shapes[:, 1]  # [num_images]
        valid_mask = []
        for flag, length in zip(image_flags, lengths):
            valid_mask.extend([flag] * length)

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=vit_embeds.device)
        valid_tokens = vit_embeds[valid_mask]  # [num_valid_tokens, C]

        return valid_tokens
    
    def extract_feature(self, pixel_values, image_flags=None):

        if self.select_layer == -1:
            vision_model_output = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True)
            if hasattr(vision_model_output, 'last_hidden_state'):
                vit_embeds = vision_model_output.last_hidden_state
            if hasattr(vision_model_output, 'spatial_shapes'):
                spatial_shapes = vision_model_output.spatial_shapes
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds, spatial_shapes = self.pixel_shuffle_back(vit_embeds, spatial_shapes)


        if self.mlp_checkpoint and vit_embeds.requires_grad:
            vit_embeds = cp.checkpoint(self.mlp1, vit_embeds)
        else:
            vit_embeds = self.mlp1(vit_embeds)
        
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)
        
        if image_flags is not None and any(image_flags==0):
            vit_embeds = self.mask_valid_tokens(vit_embeds, spatial_shapes, image_flags)
            
        return vit_embeds

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            image_sizes: Optional[List[Tuple[int, int]]] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                pixel_values = [each.to(self.device) for each in pixel_values]
                import time
                torch.cuda.synchronize()
                begin_time = time.time()
                for _ in range(10):
                    vit_embeds = self.extract_feature(pixel_values)
                torch.cuda.synchronize()
                end_time = time.time()

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.config.image_token_index)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            
        if 'use_cache' not in generate_kwargs:
            generate_kwargs['use_cache'] = True
            
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        return outputs

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

