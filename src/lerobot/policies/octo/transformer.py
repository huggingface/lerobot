# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.octo.base import (
    RULE_MAP,
    AttentionRule,
    PrefixGroup,
    TimestepGroup,
    find_match,
)
from lerobot.policies.octo.tokenizers import ImageTokenizer, LanguageTokenizer, SmallStem16


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        mlp_dim: int,
        dtype: torch.dtype = torch.float32,
        out_dim: int | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        # First linear layer
        self.dense1 = nn.Linear(mlp_dim if out_dim is None else out_dim, mlp_dim, dtype=self.dtype)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second linear layer
        self.dense2 = nn.Linear(mlp_dim, mlp_dim if out_dim is None else out_dim, dtype=self.dtype)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense1(inputs)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout1(x)

        output = self.dense2(x)
        output = self.dropout2(output)

        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        mlp_dim: int,
        num_heads: int,
        d_model: int = 768,
        dtype: torch.dtype = torch.float32,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model, eps=1e-6, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(self.d_model, eps=1e-6, dtype=self.dtype)

        # MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            bias=True,
            batch_first=True,
        )

        # MLP block
        self.mlp = MLPBlock(mlp_dim=mlp_dim, dtype=dtype, out_dim=self.d_model, dropout_rate=dropout_rate)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert inputs.dim() == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

        # Attention block
        x = self.norm1(inputs)

        # # TODO (lilkm): I need to check this
        # # Process attention mask
        # if attention_mask is not None:
        #     # Attention_mask comes in as (batch, 1, seq, seq)
        #     # PyTorch MultiheadAttention expects (batch * num_heads, seq, seq) or (seq, seq)
        #     # We'll use the simpler (seq, seq) format by taking the first batch
        #     # if attention_mask.dim() == 4:
        #     #     # Take the first batch and squeeze out the head dimension
        #     #     attention_mask = attention_mask[0, 0]  # (seq, seq)

        # Convert boolean mask to additive mask (True -> 0, False -> -inf)
        if attention_mask.dtype == torch.bool:
            attention_mask = (
                attention_mask.float()
                .masked_fill(~attention_mask, float("-inf"))
                .masked_fill(attention_mask, 0.0)
            )

            # batch_size, seq_len = attention_mask.shape[0], attention_mask.shape[2]

            # # attention_mask = attention_mask.unsqueeze(1)
            # attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            # attention_mask = attention_mask.view(batch_size * self.num_heads, seq_len, seq_len)

        # Apply attention
        x, _ = self.attention(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = self.dropout(x)
        x = x + inputs

        # MLP block
        y = self.norm2(x)
        y = self.mlp(y)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        d_model: int = 768,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.add_position_embedding = add_position_embedding

        # Encoder blocks - initialize with d_model
        self.encoder_blocks = nn.ModuleList(
            [
                Encoder1DBlock(
                    mlp_dim=mlp_dim,
                    num_heads=num_attention_heads,
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 3, f"Expected (batch, len, emb) got {x.shape}"

        # Apply encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)

        # Final layer norm
        encoded = self.encoder_norm(x)

        return encoded


class BlockTransformer(nn.Module):
    """A transformer that acts on multiple groups of tokens, which may attend to each other
    (in complex patterns)."""

    def __init__(self, transformer_kwargs: dict[str, Any], enforce_causal: bool = True):
        super().__init__()
        self.transformer_kwargs = transformer_kwargs
        self.enforce_causal = enforce_causal

        # Create transformer
        self.transformer = Transformer(**transformer_kwargs)

    def forward(
        self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]
    ) -> tuple[list[PrefixGroup], list[TimestepGroup]]:
        horizon = timestep_groups[0].tokens.shape[1]
        assert all(group.tokens.shape[1] == horizon for group in timestep_groups)

        token_dim = timestep_groups[0].tokens.shape[-1]
        assert all(group.tokens.shape[-1] == token_dim for group in prefix_groups)
        assert all(group.tokens.shape[-1] == token_dim for group in timestep_groups)

        # Assemble input tokens
        input_tokens = self._assemble_input_tokens(prefix_groups, timestep_groups)

        # Generate attention mask
        attention_mask = self._generate_attention_mask(prefix_groups, timestep_groups)

        # Run transformer
        output = self.transformer(input_tokens, attention_mask)

        # Split output into prefix and timestep groups
        prefix_outputs, timestep_outputs = self._split_output_tokens(output, prefix_groups, timestep_groups)

        return prefix_outputs, timestep_outputs

    def _assemble_input_tokens(
        self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]
    ) -> torch.Tensor:
        """Assemble input tokens from prefix and timestep groups."""
        batch_size = timestep_groups[0].tokens.shape[0]
        token_dim = timestep_groups[0].tokens.shape[-1]

        # Concatenate prefix tokens
        if len(prefix_groups) > 0:
            all_prefix_tokens = torch.cat([group.tokens for group in prefix_groups], dim=1)
        else:
            all_prefix_tokens = torch.zeros(
                (batch_size, 0, token_dim),
                dtype=timestep_groups[0].tokens.dtype,
                device=timestep_groups[0].tokens.device,
            )

        # Concatenate timestep tokens and fold horizon into sequence dimension
        all_timestep_tokens = torch.cat([group.tokens for group in timestep_groups], dim=2)
        # Reshape from (batch, horizon, n_tokens, d) to (batch, horizon * n_tokens, d)
        batch_size, horizon, n_tokens, d = all_timestep_tokens.shape
        all_timestep_tokens = all_timestep_tokens.view(batch_size, horizon * n_tokens, d)

        # Concatenate prefix and timestep tokens
        tokens = torch.cat([all_prefix_tokens, all_timestep_tokens], dim=1)

        return tokens

    def _split_output_tokens(
        self,
        output_tokens: torch.Tensor,
        prefix_groups: list[PrefixGroup],
        timestep_groups: list[TimestepGroup],
    ) -> tuple[list[PrefixGroup], list[TimestepGroup]]:
        """Split output tokens back into prefix and timestep groups."""
        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        # Split prefix and timestep tokens
        prefix_embeddings, timestep_embeddings = torch.split(
            output_tokens, [n_prefix_tokens, output_tokens.shape[1] - n_prefix_tokens], dim=1
        )

        # Process prefix group outputs
        all_prefix_outputs = []
        if len(prefix_groups) > 0:
            prefix_splits = torch.split(prefix_embeddings, tokens_per_prefix_group, dim=1)
            for group, embeddings in zip(prefix_groups, prefix_splits, strict=True):
                all_prefix_outputs.append(
                    PrefixGroup(
                        tokens=embeddings,
                        mask=group.mask,
                        name=group.name,
                        attention_rules=group.attention_rules,
                    )
                )

        # Process timestep group outputs
        # Reshape from (batch, horizon * n_tokens, d) to (batch, horizon, n_tokens, d)
        batch_size, total_timestep_tokens, d = timestep_embeddings.shape
        n_tokens_per_timestep = total_timestep_tokens // horizon
        timestep_embeddings = timestep_embeddings.view(batch_size, horizon, n_tokens_per_timestep, d)

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        timestep_splits = torch.split(timestep_embeddings, tokens_per_timestep_group, dim=2)

        all_timestep_outputs = []
        for group, embeddings in zip(timestep_groups, timestep_splits, strict=True):
            all_timestep_outputs.append(
                TimestepGroup(
                    tokens=embeddings, mask=group.mask, name=group.name, attention_rules=group.attention_rules
                )
            )

        return all_prefix_outputs, all_timestep_outputs

    def _generate_attention_mask(
        self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]
    ) -> torch.Tensor:
        """Generate attention mask based on group attention rules."""
        if self.enforce_causal:
            self._verify_causality(prefix_groups, timestep_groups)

        device = timestep_groups[0].tokens.device
        horizon = timestep_groups[0].tokens.shape[1]

        all_groups = prefix_groups + timestep_groups
        num_groups = len(all_groups)

        prefix_token_counts = [g.tokens.shape[1] for g in prefix_groups]
        timestep_token_counts = [g.tokens.shape[2] for g in timestep_groups]

        tokens_for_prefix = sum(prefix_token_counts)
        tokens_per_time_step = sum(timestep_token_counts)
        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon

        group_ids = torch.zeros(total_tokens, dtype=torch.long, device=device)
        timesteps = torch.full((total_tokens,), -1, dtype=torch.long, device=device)

        current_pos = 0
        for i, count in enumerate(prefix_token_counts):
            group_ids[current_pos : current_pos + count] = i
            current_pos += count

        timestep_group_ids_per_step = []
        for i, count in enumerate(timestep_token_counts):
            timestep_group_ids_per_step.append(
                torch.full(
                    (count,),
                    len(prefix_groups) + i,
                    dtype=torch.long,
                    device=device,
                )
            )
        timestep_group_ids_per_step = torch.cat(timestep_group_ids_per_step)

        start_pos = tokens_for_prefix
        for t in range(horizon):
            end_pos = start_pos + tokens_per_time_step
            group_ids[start_pos:end_pos] = timestep_group_ids_per_step
            timesteps[start_pos:end_pos] = t
            start_pos = end_pos

        rules_table = torch.zeros(num_groups, num_groups, dtype=torch.long, device=device)
        for i, group_i in enumerate(all_groups):
            for j, group_j in enumerate(all_groups):
                rule = find_match(group_i.attention_rules, group_j.name, AttentionRule.NEVER)
                rules_table[i, j] = RULE_MAP[rule]

        attending_rules = rules_table[group_ids[:, None], group_ids[None, :]]

        timesteps_i = timesteps[:, None]
        timesteps_j = timesteps[None, :]

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
        mask |= (attending_rules == RULE_MAP[AttentionRule.CAUSAL]) & (timesteps_j <= timesteps_i)
        mask |= (attending_rules == RULE_MAP[AttentionRule.CURRENT]) & (timesteps_j == timesteps_i)
        mask |= (attending_rules == RULE_MAP[AttentionRule.STRICT_PAST]) & (timesteps_j < timesteps_i)
        mask |= attending_rules == RULE_MAP[AttentionRule.ALL]

        # Combine with padding mask
        pad_attention_mask = self._generate_pad_attention_mask(prefix_groups, timestep_groups)
        batch_size = pad_attention_mask.shape[0]

        attention_mask = mask.unsqueeze(0) & pad_attention_mask

        num_attention_heads = self.transformer_kwargs["num_attention_heads"]
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, num_attention_heads, total_tokens, total_tokens
        )
        attention_mask = attention_mask.reshape(batch_size * num_attention_heads, total_tokens, total_tokens)

        return attention_mask

    def _generate_pad_attention_mask(
        self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]
    ) -> torch.Tensor:
        """Generate padding attention mask."""
        batch_size = timestep_groups[0].tokens.shape[0]
        horizon = timestep_groups[0].tokens.shape[1]

        # Concatenate prefix masks
        if len(prefix_groups) > 0:
            prefix_pad_mask = torch.cat([group.mask for group in prefix_groups], dim=1)
        else:
            prefix_pad_mask = torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=timestep_groups[0].tokens.device
            )

        # Concatenate timestep masks and flatten
        timestep_pad_mask = torch.cat([group.mask for group in timestep_groups], dim=2)
        # Reshape from (batch, horizon, n_tokens) to (batch, horizon * n_tokens)
        batch_size, horizon, n_tokens = timestep_pad_mask.shape[:3]
        timestep_pad_mask = timestep_pad_mask.view(batch_size, -1)

        # Combine masks
        pad_mask = torch.cat([prefix_pad_mask, timestep_pad_mask], dim=1)

        # Broadcast to attention mask shape (batch, 1, total_tokens, total_tokens)
        # This matches the JAX implementation's broadcasting
        total_tokens = pad_mask.shape[1]
        pad_mask = pad_mask.unsqueeze(1)  # (batch, 1, total_tokens)
        pad_mask = pad_mask.expand(batch_size, total_tokens, total_tokens)

        return pad_mask

    def _verify_causality(self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]):
        """Verify that attention rules don't break causality."""
        # Simplified verification - in full implementation would check all attention rules
        pass


class OctoTransformer(nn.Module):
    """Implementation of the Octo transformer with block-wise attention using BlockTransformer"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        repeat_task_tokens: bool = False,
    ):
        super().__init__()
        self.repeat_task_tokens = repeat_task_tokens

        # Create BlockTransformer with appropriate transformer kwargs
        transformer_kwargs = {
            "num_layers": num_layers,
            "mlp_dim": dim_feedforward,
            "num_attention_heads": nhead,
            "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "add_position_embedding": False,
            "d_model": d_model,
        }

        self.transformer = BlockTransformer(
            transformer_kwargs=transformer_kwargs,
            enforce_causal=True,
        )

    def forward(
        self, prefix_groups: list[PrefixGroup], timestep_groups: list[TimestepGroup]
    ) -> tuple[list[PrefixGroup], list[TimestepGroup]]:
        """
        A simple wrapper around the BlockTransformer.

        Args:
            prefix_groups: List of PrefixGroup objects for the transformer.
            timestep_groups: List of TimestepGroup objects for the transformer.

        Returns:
            A tuple of (prefix_outputs, timestep_outputs).
        """
        prefix_outputs, timestep_outputs = self.transformer(
            prefix_groups=prefix_groups, timestep_groups=timestep_groups
        )

        return prefix_outputs, timestep_outputs


class OctoWithoutHead(nn.Module):
    def __init__(
        self,
        model_name: str = "octo-base",
        repeat_task_tokens: bool = False,
        freeze_language_encoder: bool = True,
        token_embedding_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        chunk_size: int = 10,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.model_name = model_name
        self.token_embedding_size = token_embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_horizon = chunk_size
        self.repeat_task_tokens = repeat_task_tokens

        # Create positional embeddings
        # Primary observation tokens: 256 tokens with d_model dimensions
        self.obs_primary_pos_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 256, self.token_embedding_size) * 0.02
        )

        # Wrist observation tokens: 64 tokens with d_model dimensions
        self.obs_wrist_pos_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 64, self.token_embedding_size) * 0.02
        )

        # Language task tokens: 16 tokens with d_model dimensions
        self.task_language_pos_embedding = nn.Parameter(torch.randn(1, 16, self.token_embedding_size) * 0.02)

        # Readout token embeddings
        self.readout_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 1, self.token_embedding_size) * 0.02
        )

        # Initialize components
        self.observation_tokenizers = nn.ModuleDict()
        self.task_tokenizers = nn.ModuleDict()

        # Initialize transformer
        self.transformer = OctoTransformer(
            d_model=self.token_embedding_size,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.mlp_dim,
            repeat_task_tokens=self.repeat_task_tokens,
        )

        # Attributes to store intermediate values for debugging
        self.prefix_groups = None
        self.timestep_groups = None

        # Caching
        self.cached_prefix_groups = None

        # Projections
        self.obs_primary_projection = nn.Linear(512, self.token_embedding_size)
        self.obs_wrist_projection = nn.Linear(512, self.token_embedding_size)
        self.task_language_projection = nn.Linear(768, self.token_embedding_size)

        # Initialize tokenizers
        self._init_tokenizers(freeze_language_encoder)

    def _init_tokenizers(self, freeze_language_encoder: bool):
        """Initialize observation and task tokenizers"""
        # Primary image tokenizer (256x256)
        self.observation_tokenizers["image_primary"] = ImageTokenizer(
            encoder=SmallStem16(),
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            task_film_keys=[],
        )

        # Wrist image tokenizer (128x128)
        self.observation_tokenizers["image_wrist"] = ImageTokenizer(
            encoder=SmallStem16(),
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            task_film_keys=[],
        )

        # Language tokenizer
        self.task_tokenizers["language_instruction"] = LanguageTokenizer(
            finetune_encoder=not freeze_language_encoder
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        tasks: dict[str, torch.Tensor],
        timestep_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        batch_size, horizon = timestep_pad_mask.shape

        # Define attention rules
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}
        obs_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }

        # Create prefix groups for task tokens
        prefix_groups = []
        for name, tokenizer in self.task_tokenizers.items():
            if name in tasks:
                token_group = tokenizer(tasks[name], tasks)
                projected_tokens = self.task_language_projection(token_group.tokens)

                # Add positional embedding
                group_name = f"task_{name}"
                pos_embedding = self.task_language_pos_embedding[:, : projected_tokens.shape[1]]
                processed_tokens = projected_tokens + pos_embedding

                prefix_groups.append(
                    PrefixGroup(
                        tokens=processed_tokens,
                        mask=token_group.mask,
                        name=group_name,
                        attention_rules=task_attention_rules,
                    )
                )

        # with torch.profiler.record_function("Octo/TaskTokenization"):
        #     if self.cached_prefix_groups is not None:
        #         # Expand cached groups to current batch size and device
        #         prefix_groups = [
        #             PrefixGroup(
        #                 tokens=g.tokens.to(timestep_pad_mask.device).expand(batch_size, -1, -1),
        #                 mask=g.mask.to(timestep_pad_mask.device).expand(batch_size, -1),
        #                 name=g.name,
        #                 attention_rules=g.attention_rules,
        #             )
        #             for g in self.cached_prefix_groups
        #         ]
        #     else:
        #         prefix_groups = []
        #         for name, tokenizer in self.task_tokenizers.items():
        #             if name in tasks:
        #                 token_group = tokenizer(tasks[name], tasks)
        #                 projected_tokens = self.task_language_projection(token_group.tokens)

        #                 # Add positional embedding
        #                 pos_embedding = self.task_language_pos_embedding[:, : projected_tokens.shape[1]]
        #                 processed_tokens = projected_tokens + pos_embedding

        #                 # ✅ store only one exemplar in cache (batch = 1)
        #                 processed_tokens = processed_tokens[:1].detach().cpu()
        #                 token_mask = token_group.mask[:1].detach().cpu()

        #                 prefix_groups.append(
        #                     PrefixGroup(
        #                         tokens=processed_tokens.to(timestep_pad_mask.device).expand(batch_size, -1, -1),
        #                         mask=token_mask.to(timestep_pad_mask.device).expand(batch_size, -1),
        #                         name=f"task_{name}",
        #                         attention_rules=task_attention_rules,
        #                     )
        #                 )

        #         # ✅ cache the single exemplar (not the expanded version)
        #         self.cached_prefix_groups = [
        #             PrefixGroup(
        #                 tokens=g.tokens[:1].detach().cpu(),
        #                 mask=g.mask[:1].detach().cpu(),
        #                 name=g.name,
        #                 attention_rules=g.attention_rules,
        #             )
        #             for g in prefix_groups
        #         ]

        # Create timestep groups for observation tokens
        timestep_groups = []
        for name, tokenizer in self.observation_tokenizers.items():
            if name in observations:
                token_group = tokenizer(observations, tasks)

                # Project tokens
                if name == "image_primary":
                    projected_tokens = self.obs_primary_projection(token_group.tokens)
                    pos_embedding = self.obs_primary_pos_embedding
                elif name == "image_wrist":
                    projected_tokens = self.obs_wrist_projection(token_group.tokens)
                    pos_embedding = self.obs_wrist_pos_embedding
                else:
                    projected_tokens = token_group.tokens
                    pos_embedding = None

                # Add positional embedding
                if pos_embedding is not None:
                    processed_tokens = projected_tokens + pos_embedding[:, : projected_tokens.shape[1]]
                else:
                    processed_tokens = projected_tokens

                # Create mask
                mask = torch.logical_and(timestep_pad_mask.unsqueeze(-1), token_group.mask)

                timestep_groups.append(
                    TimestepGroup(
                        tokens=processed_tokens,
                        mask=mask,
                        name=f"obs_{name}",
                        attention_rules=obs_attention_rules,
                    )
                )

        # Apply repeat_task_tokens logic if enabled
        if self.repeat_task_tokens:
            for task_group in prefix_groups:
                task_tokens = task_group.tokens.unsqueeze(1)
                ws = timestep_groups[0].tokens.shape[1]
                task_tokens = task_tokens.repeat(1, ws, 1, 1)

                task_pad_mask = task_group.mask.unsqueeze(1)
                task_pad_mask = task_pad_mask.repeat(1, ws, 1)

                group_name = f"obs_{task_group.name}"
                timestep_groups.append(
                    TimestepGroup(
                        tokens=task_tokens,
                        mask=task_pad_mask,
                        name=group_name,
                        attention_rules=obs_attention_rules,
                    )
                )

        # Add readout tokens
        readout_tokens = torch.zeros(
            (batch_size, horizon, 1, self.token_embedding_size), device=timestep_pad_mask.device
        )
        readout_tokens += self.readout_embedding[:, :horizon]
        readout_mask = torch.ones((batch_size, horizon, 1), dtype=torch.bool, device=timestep_pad_mask.device)
        readout_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
            "readout_action": AttentionRule.CAUSAL,
        }
        timestep_groups.append(
            TimestepGroup(
                tokens=readout_tokens,
                mask=readout_mask,
                name="readout_action",
                attention_rules=readout_attention_rules,
            )
        )

        # Store groups for debugging
        self.prefix_groups = prefix_groups
        self.timestep_groups = timestep_groups

        # Run transformer
        _, timestep_outputs = self.transformer(prefix_groups, timestep_groups)

        # Create transformer outputs dict for action head
        transformer_outputs = {}
        for group in timestep_outputs:
            transformer_outputs[group.name] = group

        return transformer_outputs
