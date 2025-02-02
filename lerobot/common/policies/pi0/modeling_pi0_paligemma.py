from typing import List, Optional, Union, Tuple

import torch
from pytest import Cache
from torch import nn
import torch.version
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)

from packaging.version import Version


from transformers.models.auto import CONFIG_MAPPING, AutoConfig


if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask


import torch.nn.functional as F
                
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_mask,
    create_block_mask,
    _round_up_to_multiple,
)
####
# @torch.compile(dynamic=False)
def flex_attention_forward(
    attention_mask: torch.Tensor,
    batch_size: int,             
    head_dim: int,               
    query_states: torch.Tensor,  
    key_states: torch.Tensor,    
    value_states: torch.Tensor,  
    scaling=None
):
    '''
    This is defined out of classes to make compile happy.
    '''

    original_dtype = query_states.dtype
    num_att_heads = 8
    num_key_value_heads = 1
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = key_states[:, :, :, None, :]
    key_states = key_states.expand(
        batch_size, key_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, key_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )

    value_states = value_states[:, :, :, None, :]
    value_states = value_states.expand(
        batch_size, value_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, value_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.transpose(1, 2)
    key_states   = key_states.transpose(1, 2)  
    value_states = value_states.transpose(1, 2)

    query_states = query_states.to(torch.float32)
    key_states   = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)


    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, :key_states.shape[2]]

        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]
        return mask_mod
    B_mask, H_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask

    BLOCK_SIZE = 128
    q_len_rounded = _round_up_to_multiple(q_len, BLOCK_SIZE)
    kv_len_rounded = _round_up_to_multiple(kv_len, BLOCK_SIZE)

    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len

    padded_causal_mask = F.pad(
        causal_mask,
        (0, pad_k, 0, pad_q), 
        value=0.0 
    )
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=B_mask,  
        H=H_mask,  
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
        _compile=False,
    )

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=B_mask,
        H=H_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=BLOCK_SIZE,
        device=causal_mask.device,
        _compile=False, 
    )

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output, attention_weights = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,             # because we shaped query/key states for GQA
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
    )

    attn_output = attn_output.to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    return attn_output


####

def causal_mask_condition(b, h, q_idx, kv_idx):
    """
    Returns True if q_idx can attend to kv_idx based on the desired constraints:
      - Image tokens attend only to image (non-padding) + text (non-padding).
      - Image padding attends to nothing, and no one attends to image padding.
      - Text tokens attend only to image (non-padding) + text (non-padding).
      - Text padding attends to nothing, and no one attends to text padding.
      - State tokens attend to image, text, and previously defined state tokens (triangular among states, even though there's just one).
      - Action tokens attend to image, text, all states, and all actions (but never padding).
    """

    N_IMG_TOKENS=256
    N_EMPTY_IMG_TOKENS=512
    N_TXT_TOKENS=4
    N_TXT_PADDING_TOKENS=44
    N_STATE_TOKENS=1
    N_ACTION_TOKENS=50 

    img_pad_start = N_IMG_TOKENS
    txt_start = img_pad_start + N_EMPTY_IMG_TOKENS
    txt_pad_start = txt_start + N_TXT_TOKENS
    state_start = txt_pad_start + N_TXT_PADDING_TOKENS
    action_start = state_start + N_STATE_TOKENS
    N = action_start + N_ACTION_TOKENS  # total

    # If kv is in image or text padding => False
    if (img_pad_start <= kv_idx and  kv_idx < txt_start) or (txt_pad_start <= kv_idx and kv_idx < state_start):
        return False

    # If q is in image padding or text padding => attends to nothing
    if img_pad_start <= q_idx < txt_start or txt_pad_start <= q_idx < state_start:
        return False

    is_image = (q_idx < N_IMG_TOKENS)
    is_text = (txt_start <= q_idx < txt_start + N_TXT_TOKENS)
    is_state = (state_start <= q_idx < action_start)
    is_action = (action_start <= q_idx < N)

    if is_image:
        # Image tokens: attend to (non-padding) image tokens + (non-padding) text tokens only
        if kv_idx < N_IMG_TOKENS:  # other image tokens
            return True
        if txt_start <= kv_idx < txt_start + N_TXT_TOKENS:  # text that is not padded
            return True
        return False

    elif is_text:
        # Text tokens: attend to (non-padding) image + text
        if kv_idx < N_IMG_TOKENS:  # image
            return True
        if txt_start <= kv_idx < txt_start + N_TXT_TOKENS:  # text
            return True
        return False

    elif is_state:
        # state tokens: attend to non-padding image, non-padding text, and states up to q_idx
        if kv_idx < N_IMG_TOKENS:  # image
            return True
        if txt_start <= kv_idx < txt_start + N_TXT_TOKENS:  # text
            return True
        if state_start <= kv_idx <= q_idx:  # triangular among states
            return True
        return False

    elif is_action:
        # action tokens: attend to non-padding image, non-padding text, all states, and all actions
        if kv_idx < N_IMG_TOKENS:  # image
            return True
        if txt_start <= kv_idx < txt_start + N_TXT_TOKENS:  # text
            return True
        if state_start <= kv_idx < action_start:  # any state
            return True
        if action_start <= kv_idx < N:  # any action (including self I think) (yes)
            return True
        return False

    return False



def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    
    radians = radians[..., None, :]

    sin = torch.sin(radians)#.to(dtype=dtype)
    cos = torch.cos(radians)#.to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


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
            gemma_expert_config["model_type"] = (
                gemma_expert_config["model_type"] if "model_type" in gemma_expert_config else "gemma"
            )
            self.gemma_expert_config = CONFIG_MAPPING[gemma_expert_config["model_type"]](
                **gemma_expert_config
            )
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
        # for pname, params in self.paligemma.named_parameters():
        #     if "language_model.model.embed_tokens" in pname:
        #         continue
        #     params.data = params.data.to(dtype=torch.bfloat16)
        #self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        # TODO: finetune expert only as an option
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)

        #self.gemma_expert = self.gemma_expert.to(dtype=torch.bfloat16)

        # In the original impl, all projections are done in float32.
        self.state_proj = nn.Linear(self.config.state_dim, self.config.width, dtype=torch.float32)
        self.action_in_proj = nn.Linear(self.config.action_dim, self.config.width, dtype=torch.float32)
        self.action_out_proj = nn.Linear(self.config.width, self.config.action_dim, dtype=torch.float32)

        self.action_time_mlp_in = nn.Linear(self.config.width * 2, self.config.width, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(self.config.width, self.config.width, dtype=torch.float32)


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
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            dtype = hidden_states.dtype
            device = hidden_states.device
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim
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

                # TODO: implement kv cache
            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            # if query_states.dtype != dtype:
            #     raise ValueError(f"{query_states.dtype=}")
            # if key_states.dtype != dtype:
            #     raise ValueError(f"{key_states.dtype=}")
            # if value_states.dtype != dtype:
            #     raise ValueError(f"{value_states.dtype=}")

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
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )
            
            # TODO use a from config instantiation with supported attention classes
            use_fa2 = False
            use_flex = True
            if use_fa2:
                attention_interface = self.flash_attention_forward
            elif use_flex:
                attention_interface = flex_attention_forward
            else:
                attention_interface = self.eager_attention_forward
            att_output = attention_interface(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
            att_output = att_output.to(dtype)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
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

    def flash_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        raise NotImplementedError("FA2 is not implemented (yet)")
    
    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):

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

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        # with autocast(dtype=torch.float32, device_type=device.type):
        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        #att_weights *= torch.tensor(head_dim**-0.5, dtype=torch.float32)
        # att_weights: batch_size, num_att_head, sequence_length, sequence_length
        # big_neg = torch.finfo(torch.float32).min  # See gemma/modules.py
        big_neg = -2.3819763e38  # See gemma/modules.py
        
        # masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        # with autocast(dtype=torch.bfloat16, device_type=device.type):
        # probs = nn.functional.softmax(masked_att_weights, dim=-1, dtype=torch.float32)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)
        # value_states = value_states.to(torch.float32)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        # att_output = torch.matmul(probs, value_states.permute(0, 3, 2, 1,4))

        # Now, we recast to the original dtype.

        # att_output = att_output.to(dtype=dtype)
        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )
        
        return att_output
