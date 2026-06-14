import torch.nn as nn
import torch
from typing import Optional, Tuple
from .modules import CategorySpecificMLP
from .pos_embs import Fixed2DPositionalEncoding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """
    Pre-norm Transformer block with Adaptive Layer Normalization (AdaLN) conditioning.

    Each block receives a conditioning vector c ~ [B, D] (action embedding) and
    uses a per-block linear to produce (shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp), following the DiT design.

    The gate weights are zero-initialized so the block starts as an identity,
    giving stable training from scratch.
    """

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        ffn_dim = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        # SiLU → Linear produces 6 modulation params: (shift/scale/gate) × (attn, ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-init: gates start at 0 → blocks are identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, D] visual tokens
            c: [B, D]    action conditioning vector
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        # unsqueeze token dim for broadcasting: [B, D] → [B, 1, D]
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa  = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp  = gate_mlp.unsqueeze(1)

        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + gate_msa * attn_out

        normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(normed)
        return x


class LAMDecoder_v2(nn.Module):
    def __init__(
        self,
        context_dim,
        input_dim: int = 1024,
        num_queries: int = 1,
        num_layers = 6,
        num_heads = 16,
        dropout = 0.1,
        grid_hw: Tuple[int, int] = (16, 16),
        train_in_latent: bool = True,
        ffn_expansion_factor = 2,
        num_embodiments: int = 32,
        code_dim: Optional[int] = None,
        last_ln: bool = True,
    ):
        super().__init__()
        self.feature_dim = context_dim
        self.input_dim = input_dim
        self.train_in_latent = train_in_latent
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
        self.num_queries = num_queries
        self.use_last_ln = bool(last_ln)

        self.dec_layers = nn.ModuleList([
            AdaLNBlock(context_dim, num_heads, ffn_expansion_factor=ffn_expansion_factor, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pos_embed = Fixed2DPositionalEncoding(context_dim, self.grid_height, self.grid_width)
        self.last_ln = nn.LayerNorm(input_dim) if self.use_last_ln else nn.Identity()

        if input_dim == context_dim:
            self.project_input = nn.Identity()
            self.project_output = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, context_dim)
            self.project_output = nn.Linear(context_dim, input_dim)

        if code_dim is not None and code_dim != context_dim:
            self.action_in_proj = nn.Linear(code_dim, context_dim)
        else:
            self.action_in_proj = nn.Identity()

        if not train_in_latent:
            self.to_pixel = nn.ConvTranspose2d(input_dim, 3, kernel_size=16, stride=16)

    def forward(self, features, actions):
        c = self.action_in_proj(actions)           # [B, 1, context_dim]
        if c.dim() == 3:
            c = c.squeeze(1)                       # [B, context_dim]

        features_tokens = self.project_input(features)
        if features_tokens.dim() == 4:
            features_tokens = features_tokens.squeeze(1)   # [B, K, context_dim]

        expected_tokens = int(self.grid_height * self.grid_width)
        if features_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Decoder token mismatch: got K={features_tokens.shape[1]}, expected K={expected_tokens} "
                f"for grid_hw=({self.grid_height},{self.grid_width})."
            )

        x = self.pos_embed(features_tokens)        # [B, K, context_dim]

        for layer in self.dec_layers:
            x = layer(x, c)

        reconstructed_features = self.project_output(x)   # [B, K, input_dim]
        reconstructed_features = self.last_ln(reconstructed_features)

        if not self.train_in_latent:
            B, K, D = reconstructed_features.shape
            rec_img = self.to_pixel(
                reconstructed_features.transpose(1, 2).reshape(B, D, self.grid_height, self.grid_width)
            )
            return rec_img.unsqueeze(1)            # [B, 1, 3, H, W]
        else:
            return reconstructed_features.unsqueeze(1)     # [B, 1, K, input_dim]


class StatePredictor(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        dropout: float = 0.1,
        num_embodiments: int = 32,
        num_queries: int = 1,
        max_state_dim: int = 32,
        code_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_embodiments = int(num_embodiments)
        self.max_state_dim = int(max_state_dim)
        z_input_dim = int(code_dim) if code_dim is not None else int(latent_dim)
        self.z_encoder = nn.Linear(z_input_dim, latent_dim)
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.num_embodiments,
            input_dim=self.max_state_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
        )
        self.state_decoder = CategorySpecificMLP(
            num_categories=self.num_embodiments,
            input_dim=2 * latent_dim,
            hidden_dim=latent_dim,
            output_dim=self.max_state_dim,
        )

    def forward(self, z_t: torch.Tensor, state_0: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        B = state_0.size(0)
        if not isinstance(embodiment_id, torch.Tensor):
            raise TypeError(
                f"StatePredictor expects `embodiment_id` as torch.Tensor, got {type(embodiment_id).__name__}."
            )
        emb = embodiment_id
        if emb.ndim == 2 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        elif emb.ndim != 1:
            raise ValueError(
                f"`embodiment_id` must be [B] or [B,1], got {tuple(emb.shape)}"
            )
        if emb.shape[0] != B:
            raise ValueError(
                f"embodiment_id batch mismatch: got {emb.shape[0]}, expected {B}"
            )
        emb = emb.to(device=state_0.device, dtype=torch.long)
        single_timestep = state_0.dim() == 3 and state_0.size(1) == 1
        if single_timestep:
            state_0 = state_0.squeeze(1)
        z_mean = z_t.mean(dim=1)
        expected_z_dim = int(self.z_encoder.in_features)
        if z_mean.size(-1) != expected_z_dim:
            raise ValueError(
                f"Unexpected z_t feature dim {z_mean.size(-1)}. Expected {expected_z_dim}."
            )
        z_embed = self.z_encoder(z_mean)
        state_embed = self.state_encoder(state_0.contiguous(), emb)
        if state_embed.dim() == 3 and z_embed.dim() == 2:
            z_embed = z_embed.unsqueeze(1).expand(-1, state_embed.size(1), -1)
        fused = torch.cat([z_embed, state_embed], dim=-1)
        s_pred = torch.tanh(self.state_decoder(fused, emb))
        if single_timestep and s_pred.dim() == 3 and s_pred.size(1) == 1:
            s_pred = s_pred.squeeze(1)
        return s_pred
