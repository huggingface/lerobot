"""
TODO(now)
"""

from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.d_model = d_model
        self.nhead = nhead
        self._init_params()  # TODO(now): move to somewhere common

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, encoder_pos, decoder_pos):
        """
        Args:
            x: ((E)ncoder (S)equence, (B)atch, (C)hannels)
            decoder_pos: (Decoder Sequence, C) tensor for the decoder's positional embedding.
            encoder_pos: (ES, C) tenso
        """
        # TODO flatten only when input has H and W
        bs = x.shape[1]

        encoder_out = self.encoder(x, pos=encoder_pos)
        decoder_in = torch.zeros(
            (decoder_pos.shape[0], bs, decoder_pos.shape[2]),
            dtype=decoder_pos.dtype,
            device=decoder_pos.device,
        )
        decoder_out = self.decoder(decoder_in, encoder_out, encoder_pos=encoder_pos, decoder_pos=decoder_pos)
        return decoder_out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if normalize_before else nn.Identity()

    def forward(self, x, pos: Optional[Tensor] = None):
        for layer in self.layers:
            x = layer(x, pos=pos)
        x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, x, pos: Optional[Tensor] = None):
        skip = x
        if self.normalize_before:
            x = self.norm1(x)
        q = k = x if pos is None else x + pos
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.normalize_before:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, decoder_pos: Tensor | None = None, encoder_pos: Tensor | None = None):
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos=decoder_pos, encoder_pos=encoder_pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def maybe_add_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos: Tensor | None = None,
        encoder_pos: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.normalize_before:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.normalize_before:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.normalize_before:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.normalize_before:
            x = self.norm3(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
