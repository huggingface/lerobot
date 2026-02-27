import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm

###############################################################################
# Utility Functions
###############################################################################

def find_multiple(n: int, k: int) -> int:
    """
    Returns the smallest number greater than or equal to n that is a multiple of k.
    """
    return n if n % k == 0 else n + k - (n % k)

def stateless_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes x without maintaining running statistics.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-6)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies a modulation to x given shift and scale signals.
    The modulation formula: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


###############################################################################
# SwiGlu MLP
###############################################################################

class SwiGlu(nn.Module):
    """
    An implementation of the SwiGlu MLP activation as used in transformer feedforward layers.
    
    Args:
        dim: Input dimension.
        hidden_dim: Dimension of the hidden layer. If None, defaults to 4 * dim.
        dropout: Dropout probability.
        output_dim: Output dimension. Defaults to dim.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, output_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        # Following the original design: use 2/3 of hidden_dim (rounded to a multiple of 256)
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        if output_dim is None:
            output_dim = dim
        self.fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGlu MLP.
        """
        x1 = F.silu(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.dropout(x)
        x = self.proj(x)
        return x

###############################################################################
# Rotary Positional Embedding Helpers
###############################################################################

def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency matrices for 1D rotary embeddings.
    Returns:
        (cosine, sine): Tensors of shape [max_seq_len, dim/2].
    """
    freqs = torch.arange(0, dim, 2).float()  # [dim/2]
    freqs = theta ** (-freqs / dim)
    positions = torch.arange(max_seq_len).float()  # [max_seq_len]
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_seq_len, dim/2]
    return angles.cos(), angles.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension in half and rotates the halves.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to queries and keys.
    
    Args:
        q: Query tensor of shape [B, heads, seq_len, head_dim].
        k: Key tensor with the same shape as q.
        cos, sin: Cosine and sine frequency tensors of shape [max_seq_len, head_dim/2].
        position_ids: Optional tensor with position indices; if None, uses sequential positions.
    
    Returns:
        A tuple (q_rot, k_rot) with rotary embeddings applied.
    """
    seq_len = q.size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    cos = cos[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    sin = sin[position_ids].unsqueeze(0).unsqueeze(0)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

###############################################################################
# Attention Modules
###############################################################################

class FlowerAttention(nn.Module):
    """
    Multi-head self-attention module with optional rotary positional embeddings.
    
    Args:
        dim: Input dimension.
        n_heads: Number of attention heads.
        attn_pdrop: Dropout rate on the attention probabilities.
        resid_pdrop: Dropout rate on the output projection.
        use_rope: Whether to apply rotary embeddings.
        max_seq_len: Maximum sequence length for precomputed rotary frequencies.
        rope_theta: Theta value for rotary embeddings.
    """
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 use_rope: bool = False,
                 max_seq_len: int = 120,
                 rope_theta: float = 32) -> None:
        super().__init__()
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            self.rope_theta = rope_theta
            cos, sin = precompute_freqs_1d(self.head_dim, max_seq_len, theta=rope_theta)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor,
                custom_attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape [B, seq_len, dim].
            custom_attn_mask: Optional attention mask.
            is_causal: If True, applies causal masking.
        
        Returns:
            Tensor of shape [B, seq_len, dim] after attention and projection.
        """
        B, T, C = x.size()
        # Compute query, key, value and reshape for multi-head attention.
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        # Build causal mask if needed.
        if is_causal and custom_attn_mask is None:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif custom_attn_mask is not None:
            mask = custom_attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        else:
            mask = None
        # Use PyTorch's built-in scaled dot-product attention.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else ~mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
            # is_causal=False,
            is_causal=is_causal if custom_attn_mask is None else False
        )
        out = attn_output.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out
    

class FlowerCrossAttention(nn.Module):
    """
    Cross-attention module with optional rotary embeddings.
    
    Args:
        dim: Input and output dimension.
        n_heads: Number of attention heads.
        attn_pdrop: Dropout rate on the attention weights.
        resid_pdrop: Dropout rate on the output.
        use_rope: Whether to apply rotary embeddings.
        query_seq_len: Maximum length for queries.
        context_seq_len: Maximum length for context.
        rope_theta: Theta for query rotary embeddings.
        context_rope_theta: Theta for context rotary embeddings.
    """
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 use_rope: bool = False,
                 query_seq_len: int = 64,
                 context_seq_len: int = 384,
                 rope_theta: float = 32,
                 context_rope_theta: float = 1000.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads."
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            q_cos, q_sin = precompute_freqs_1d(self.head_dim, query_seq_len, theta=rope_theta)
            k_cos, k_sin = precompute_freqs_1d(self.head_dim, context_seq_len, theta=context_rope_theta)
            self.register_buffer("q_cos", q_cos)
            self.register_buffer("q_sin", q_sin)
            self.register_buffer("k_cos", k_cos)
            self.register_buffer("k_sin", k_sin)
            self.query_seq_len = query_seq_len
            self.context_seq_len = context_seq_len
            self.rope_theta = rope_theta
            self.context_rope_theta = context_rope_theta

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                custom_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies cross-attention between x (queries) and context (keys and values).
        
        Args:
            x: Query tensor of shape [B, seq_len, dim].
            context: Context tensor of shape [B, context_len, dim].
            custom_attn_mask: Optional attention mask.
        
        Returns:
            Tensor of shape [B, seq_len, dim].
        """
        B, T, C = x.size()
        _, S, _ = context.size()
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, _ = apply_rotary_pos_emb(q, q, self.q_cos, self.q_sin)
            k, _ = apply_rotary_pos_emb(k, k, self.k_cos, self.k_sin)
        if custom_attn_mask is not None:
            # First resh ape the mask to match q's sequence length
            mask = custom_attn_mask.unsqueeze(1).unsqueeze(2)  # [32, 1, 1, 101]
            mask = mask.expand(-1, self.n_heads, q.size(2), -1)  # [32, 16, 10, 101]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
                is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
                is_causal=False
            )
        out = attn_output.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out

###############################################################################
# Main FlowBlock
###############################################################################

class FlowBlock(nn.Module):
    """
    A transformer block for flow-based diffusion. Combines self-attention,
    (optional) cross-attention, and a SwiGlu MLP with adaptive layer normalization modulation.
    
    Args:
        dim: Input dimension.
        heads: Number of attention heads.
        attn_pdrop: Attention dropout rate.
        resid_pdrop: Residual dropout rate.
        mlp_pdrop: MLP dropout rate.
        use_cross_attn: Whether to include a cross-attention layer.
        use_rope: Whether to use rotary positional embeddings in self-attention.
        query_seq_len: Maximum query sequence length.
        rope_theta: Theta parameter for rotary embeddings.
        lora_dim: Intermediate dimension for adaptive normalization modulation.
        use_global_adaln: If True, combines global AdaLN modulation signals.
    """
    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 attn_pdrop: float = 0.1,
                 resid_pdrop: float = 0.1,
                 mlp_pdrop: float = 0.1,
                 use_cross_attn: bool = False,
                 use_rope: bool = False,
                 query_seq_len: int = 128,
                 rope_theta: float = 32,
                 lora_dim: int = 256,
                 use_global_adaln: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.use_cross_attn = use_cross_attn
        self.use_global_adaln = use_global_adaln

        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        self.norm3 = RmsNorm(dim, eps=1e-6) if use_cross_attn else None

        self.self_attn = FlowerAttention(dim=dim, n_heads=heads,
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                                           use_rope=use_rope, max_seq_len=query_seq_len, rope_theta=rope_theta)
        if use_cross_attn:
            self.cross_attn = FlowerCrossAttention(dim=dim, n_heads=heads,
                                                     attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                                                     use_rope=False)
        self.mlp = SwiGlu(dim, dropout=mlp_pdrop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, lora_dim),  # Down-project
            nn.Linear(lora_dim, 6 * dim)  # Up-project to produce 6 modulation signals
        )

    def forward(self, cx: torch.Tensor, c: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                custom_attn_mask: Optional[torch.Tensor] = None,
                custom_cross_attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                global_adaln: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through the FlowBlock.
        
        Args:
            cx: Input tensor for the block (e.g. action latent representations) of shape [B, L, D].
            c: Conditioning tensor (from external encoder).
            context: Optional context tensor for cross-attention.
            custom_attn_mask: Optional attention mask.
            is_causal: If True, uses causal self-attention.
            global_adaln: Optional list of global AdaLN modulation signals.
        
        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = cx.shape
        residual = cx

        # Compute modulation signals.
        modulation = self.adaLN_modulation(c)
        signals = modulation.chunk(6, dim=1)
        if self.use_global_adaln and global_adaln is not None:
            mod_signals = [signals[i] + global_adaln[i] for i in range(6)]
        else:
            mod_signals = signals
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_signals

        # Self-attention block with modulation.
        x_norm = self.norm1(cx)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        x_self = self.self_attn(x_mod, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        x_out = residual + gate_msa.unsqueeze(1) * x_self

        # Optionally apply cross-attention.
        if self.use_cross_attn:
            if context is None:
                raise ValueError("Context is required for cross-attention.")
            x_norm = self.norm2(x_out)
            x_cross = self.cross_attn(x_norm, context, custom_attn_mask=custom_cross_attn_mask)
            x_out = x_out + x_cross

        # MLP block with modulation.
        norm_layer = self.norm3 if self.use_cross_attn else self.norm2
        x_norm = norm_layer(x_out)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod)
        x_final = x_out + gate_mlp.unsqueeze(1) * mlp_out

        return x_final



###############################################################################
# Encoder Classes
###############################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb
    



class SharedAdaLNController(nn.Module):
    """Shared Adaptive Layer Normalization controller for all DiT blocks"""
    def __init__(self, dim, global_conddim, use_cross_attn=False):
        super().__init__()
        # Number of modulation signals needed
        num_mod_signals = 9 if use_cross_attn else 6
        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, num_mod_signals * dim, bias=False),
        )
        self.use_cross_attn = use_cross_attn

        # Zero initialize the final linear layer
        nn.init.zeros_(self.modCX[-1].weight)
        self.use_cross_attn = use_cross_attn

    def forward(self, global_cond):
        mod_signals = self.modCX(global_cond)
        if self.use_cross_attn:
            # Split into 9 parts for cross-attention path
            return mod_signals.chunk(9, dim=-1)
        else:
            # Split into 6 parts for self-attention only path
            return mod_signals.chunk(6, dim=-1)
        



class FreqEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb





class ActionSpaceEmbedderParameter(nn.Module):
    """
    Embeds discrete action indices using direct learnable parameters.
    """
    def __init__(
        self,
        hidden_size,
        max_actions=11,  # 0-10 inclusive
        embedding_size=256,
    ):
        super().__init__()
        # Direct learnable parameters for each action
        self.action_embeddings = nn.Parameter(
            torch.randn(max_actions, embedding_size) * 0.02  # Small initialization
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.max_actions = max_actions
        
    def forward(self, action_indices):
        """
        Convert action indices to embeddings using parameter lookup.
        
        Args:
            action_indices: tensor of shape (batch_size,) containing integers in [0, max_actions-1]
        """
        # Index into the parameter matrix
        embeddings = self.action_embeddings[action_indices]
        
        # Process through MLP
        embeddings = embeddings
        output = self.mlp(embeddings)
        
        return output

    def get_all_embeddings(self):
        """Returns embeddings for all possible actions."""
        return self.mlp(self.action_embeddings)
    



class ZeroEncoder(nn.Module):
    def __init__(self, dit_dim, device):
        super(ZeroEncoder, self).__init__()
        self.dit_dim = dit_dim
        self.device = device

    def forward(self, x):
        return torch.zeros((x.shape[0], self.dit_dim), device=self.device)
    

