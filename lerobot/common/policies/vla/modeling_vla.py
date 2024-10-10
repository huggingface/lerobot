import inspect
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.vla.configuration_vla import VLAConfig
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class VLAPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "act"],
):
    """
    Vision-Language Action Policy (VLAPolicy).
    This policy uses a Vision-Language Model (VLA) for action prediction based on vision and language inputs.
    """

    name = "vla"

    def __init__(
        self,
        config: VLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Initialize the VLAPolicy class with model configuration.

        Args:
        config (VLAConfig): Configuration for the Qwen2VL model.
        """
        super().__init__()
        if config is None:
            config = VLAConfig()
        self.config: VLAConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        
        self.language_model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map = 'cuda')
        self.device= self.language_model.device
        self.model = VLA(config).to(self.device)
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")# Updated Qwen2VL without loss and lm_head
        

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        #batch = self.normalize_inputs(batch)
        
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4).to(self.device)
        
        batch["prompt"] = self.config.prompt.to(self.device)
        
        # Process inputs (text and images) using the processor
        processed_inputs = self.processor(
            text=batch["prompt"], videos=list(batch["observation.images"]), return_tensors="pt", padding=True, do_rescale=False
        ).to(self.device)

        processed_inputs["pixel_values_videos"] = processed_inputs["pixel_values_videos"].to(self.device).to(torch.float16)
        breakpoint()
        # Forward pass through Llava (to get hidden states)
        llava_output = self.language_model(  # Calling the Llava model inside VLA
            **processed_inputs,
            return_dict=True,
            output_hidden_states=True
        )
        
        hidden_states = llava_output.hidden_states[-1]  # Use last layer's hidden state
        hidden_states = hidden_states[:, -4:, :] #make 4 a config parameter 

        # Pass the hidden states to the VLA model for action decoding
        predicted_actions = self.model(hidden_states)

        if len(self._action_queue) == 0:
            actions = self.unnormalize_outputs({"action": predicted_actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
       
        #batch = self.normalize_inputs(batch)
        
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4).to(self.device)
        
        batch["prompt"] = self.config.prompt
 
        processed_inputs = self.processor(
            text=batch["prompt"], videos=list(batch["observation.images"]), return_tensors="pt", padding=True, do_rescale=False
        ).to(self.device)
        processed_inputs["pixel_values_videos"] = processed_inputs["pixel_values_videos"].to(self.device).to(torch.float16)
     
        # Pass inputs through Llava and VLA
        llava_output = self.language_model(
            **processed_inputs,
            return_dict=True,
            output_hidden_states=True
        )
        hidden_states = llava_output.hidden_states[-1]
        hidden_states = hidden_states[:, -4:, :]
        #hidden_states.to(dtype=torch.float16).to(self.device)
        breakpoint()
        # Forward pass through VLA
        predicted_actions = self.model(hidden_states)

        loss_dict = {}
        if "action" in batch:
            true_actions = batch["action"]
            breakpoint()
            l2_loss = F.mse_loss(predicted_actions, true_actions, reduction="mean")
            loss_dict["l2_loss"] = l2_loss.item()
            loss_dict["loss"] = l2_loss

        return loss_dict

class ActionDecoderLayer(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout
        )
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout
        )

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
        breakpoint()

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
    def __init__(self, config: VLAConfig):
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
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        x = self.norm(x)
        return x



class VLA(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()

        # Initialize the Qwen2VLForConditionalGeneration and ActionDecoder
        #qwen2_vl_config = make_qwen2_vl_config(config)
        self.chunk_size = config.chunk_size
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.hidden_size)
        self.action_decoder = ActionDecoder(config)  # Use the updated ActionDecoder
        self.action_head = nn.Linear(config.hidden_size, config.output_shapes["action"][0])
        
       
        self.half()

    def forward(self, hidden_states):
            """
            Forward pass to compute action logits using hidden states from Qwen2VL (Llava).
            
            Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size] from Llava model.
            
            Returns:
            action_logits: Tensor of predicted actions.
            """
            batch_size = hidden_states.size(0)  # Ensure batch size is extracted
            seq_len = hidden_states.size(1)  # Sequence length of hidden states
            hidden_size = hidden_states.size(2)  # Hidden size

            # Ensure encoder_out has the correct shape [chunk_size, batch_size, seq_len, hidden_size]
            # Repeat the encoder output for chunk size across the batch dimension
            #encoder_out = hidden_states.unsqueeze(0).repeat(self.chunk_size, 1, 1, 1)  # [chunk_size, batch_size, seq_len, hidden_size]
            #encoder_out = encoder_out.view(self.chunk_size * seq_len, batch_size, hidden_size)

            # Repeat the decoder input (hidden states) as well, maintaining batch and hidden size
            repeated_hidden_states = hidden_states.unsqueeze(0).repeat(self.chunk_size//seq_len, 1, 1, 1)  # [chunk_size, batch_size, seq_len, hidden_size]
            breakpoint()
            repeated_hidden_states = repeated_hidden_states.view(self.chunk_size, batch_size, hidden_size)
        
            # Generate positional embeddings for the decoder
            decoder_pos_embeddings = self.decoder_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

            # Decode the action with positional embeddings and encoder output
            action_logits = self.action_decoder(
                x=repeated_hidden_states, 
                encoder_out=repeated_hidden_states ,
                decoder_pos_embed=decoder_pos_embeddings
            )

            # Final action logits through the action head
            action_logits = self.action_head(action_logits)

            action_logits = action_logits.transpose(0, 1)
            return action_logits
