import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from configuration_vla import Qwen2VLConfig, Qwen2VLVisionConfig
from modeling_vision import Qwen2VisionTransformerPretrainedModel
from modeling_language import Qwen2VLDecoderLayer, Qwen2RMSNorm, Qwen2VLRotaryEmbedding
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
from transformers.cache_utils import Cache, StaticCache  
from transformers.modeling_utils import PreTrainedModel

class VLAPolicy(nn.Module):
    """
    Vision-Language Action Policy (VLAPolicy).
    This policy uses a Vision-Language Model (VLA) for action prediction based on vision and language inputs.
    """

    def __init__(self, config: Qwen2VLConfig):
        """
        Initialize the VLAPolicy class with model configuration.
        
        Args:
        config (Qwen2VLConfig): Configuration for the Qwen2VL model.
        """
        super().__init__()
        self.model = VLA(config)  # Use the VLA model instead of directly using Qwen2VLModel

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch (dict): Dictionary containing the following keys:
                - "input_ids": Tensor of tokenized inputs (input to language model).
                - "attention_mask": Tensor mask for the input tokens.
                - "observation.state": Tensor containing the robot's state.
                - "action": Tensor containing the ground-truth actions (optional for training).

        Returns:
            dict: A dictionary containing the loss and predicted actions.
        """
        # Extract inputs for the model
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        # Forward pass through the VLA model
        predicted_actions = self.model(batch, input_ids=input_ids, attention_mask=attention_mask)

        loss_dict = {}

        # If ground-truth actions are available, compute L2 loss for training
        if "action" in batch:
            true_actions = batch["action"]  # Ground-truth actions
            l2_loss = F.mse_loss(predicted_actions, true_actions, reduction="mean")  # L2 loss
            loss_dict["l2_loss"] = l2_loss.item()
            loss_dict["loss"] = l2_loss

        return loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select an action based on the model's prediction given an input batch.
        
        Args:
            batch (dict): A batch containing "input_ids", "attention_mask", and "observation.state".

        Returns:
            torch.Tensor: The predicted actions.
        """
        # Forward pass without computing gradients
        predicted_actions = self.model(batch, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return predicted_actions

class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Initialize flags for using robot state and environment state
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes

        # Embedding layers for robot observation state and action
        if self.use_robot_state:
            self.robot_state_embed = nn.Linear(
                config.input_shapes["observation.state"][0], config.hidden_size
            )
        # Embedding layer for robot action
        self.action_embed = nn.Linear(
            config.output_shapes["action"][0], config.hidden_size
        )

        # Token embedding for text
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Layers for processing
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)
       
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        batch: dict,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Step 1: Embed the text input if provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Step 2: Get robot state and action embeddings if applicable
        robot_state_embedding = None
        if self.use_robot_state:
            robot_state_embedding = self.robot_state_embed(batch["observation.state"]).unsqueeze(1)
        
        action_embedding = self.action_embed(batch["action"]).unsqueeze(1)

        # Step 3: Combine text, robot state, and action embeddings
        if robot_state_embedding is not None:
            combined_embedding = torch.cat([inputs_embeds, robot_state_embedding, action_embedding], dim=1)
        else:
            combined_embedding = torch.cat([inputs_embeds, action_embedding], dim=1)

        # Step 4: Proceed with the rest of the forward pass
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + combined_embedding.shape[1], device=combined_embedding.device
            )

        # The hard coded `3` is for temporal, height, and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, combined_embedding.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
   
        causal_mask = self._update_causal_mask(
            attention_mask, combined_embedding, cache_position, past_key_values, output_attentions
        )

        hidden_states = combined_embedding

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # Causal mask logic
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # Causal mask generation logic
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
                            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class ActionDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout)
        self.cross_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.attention_dropout)
        self.dropout2 = nn.Dropout(config.attention_dropout)
        self.dropout3 = nn.Dropout(config.attention_dropout)

        self.activation = nn.GELU()
        self.pre_norm = True  # Assumed pre-norm architecture; can adjust based on config

    def maybe_add_pos_embed(self, tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_pos_embed: torch.Tensor | None = None,
        encoder_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Hidden Size) tensor of input tokens.
            encoder_out: (Encoder Sequence, Batch, Hidden Size) output features from the last layer of the encoder we are cross-attending with.
            decoder_pos_embed: (Sequence, 1, Hidden Size) positional embedding for decoder queries.
            encoder_pos_embed: (Sequence, 1, Hidden Size) positional embedding for encoder keys.
        Returns:
            (Sequence, Batch, Hidden Size) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        
        # Self-attention
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not attention weights
        x = skip + self.dropout1(x)
        
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # Cross-attention with encoder outputs
        x = self.cross_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not attention weights
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # Feed-forward network
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        
        if not self.pre_norm:
            x = self.norm3(x)

        return x

class ActionDecoder(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        """Runs multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ActionDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_pos_embed: torch.Tensor | None = None,
        encoder_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        x = self.norm(x)
        return x

class VLA(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super(VLA, self).__init__()
        
        # Initialize the Qwen2VLModel and ActionDecoder
        self.model = Qwen2VLModel(config)
        self.action_decoder = ActionDecoder(config)  # Use the updated ActionDecoder
        self.action_head = nn.Linear(config.hidden_size, config.output_shapes["action"][0])

    def forward(self, batch, input_ids=None, attention_mask=None):
        # Get the hidden states from the Qwen2VLModel
        model_output = self.model(
            batch=batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = model_output.last_hidden_state
        action_embedding = hidden_states[:, -1:, :]
        # Decoder input preparation (you may want to adjust this depending on the input requirements)
        encoder_out = action_embedding
        
        # Decode the action
        action_logits = self.action_decoder(
            x=action_embedding,
            encoder_out=encoder_out
        )
        breakpoint()
        # Final action logits through the action head
        action_logits = self.action_head(action_logits.squeeze())
        
        return action_logits

