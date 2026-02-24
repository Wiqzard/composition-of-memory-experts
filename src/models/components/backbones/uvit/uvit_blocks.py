import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.components.modules.embeddings import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    RotaryEmbedding3D,
    RotaryEmbeddingND,
)
from src.models.components.modules.normalization import RMSNorm as Normalize
from src.models.components.modules.zero_module import zero_module
from src.models.components.transformer.ditv2 import SinusoidalPositionalEmbedding


class EmbedInput(nn.Module):
    """
    Initial downsampling layer for U-ViT.
    One shall replace this with 5/3 DWT, which is fully invertible and may slightly improve performance, according to the Simple Diffusion paper.
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


class ProjectOutput(nn.Module):
    """
    Final upsampling layer for U-ViT.
    One shall replace this with IDWT, which is an inverse operation of DWT.
    """

    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.proj = zero_module(
            nn.ConvTranspose2d(dim, out_channels, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


# pylint: disable-next=invalid-name
def NormalizeWithBias(num_channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """
    Standard ResNet block.
    """

    def __init__(self, channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        assert dropout == 0.0, "Dropout is not supported in ResBlock."
        self.emb_layer = nn.Conv2d(emb_dim, channels * 2, kernel_size=(1, 1))
        self.in_layers = nn.Sequential(
            NormalizeWithBias(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.out_norm = NormalizeWithBias(channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the ResNet block.
        Args:
            x: Input tensor of shape (B, C, H, W).
            emb: Embedding tensor of shape (B, C) or (B, C, H, W).
        Returns:
            Output tensor of shape (B, C, H, W).
        """
        h = self.in_layers(x)
        emb_out = self.emb_layer(emb if emb.dim() == 4 else emb[:, :, None, None])
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return x + h


class NormalizeWithCond(nn.Module):
    """
    Conditioning block for U-ViT, that injects external conditions into the network using FiLM.
    """

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.emb_layer = nn.Linear(emb_dim, dim * 2)
        self.norm = Normalize(dim)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the conditioning block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        scale, shift = self.emb_layer(emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AttentionBlock(nn.Module):
    """
    Simple Attention block for axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        rope: Optional[RotaryEmbeddingND] = None,
    ):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.rope = rope
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)
        self.out = zero_module(nn.Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the attention block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        x = self.norm(x, emb)
        qkv = self.proj(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return x + self.out(x)


class AxialRotaryEmbedding(nn.Module):
    """
    Axial rotary embedding for axial attention.
    Composed of two rotary embeddings for each axis.
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int] | Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        If len(sizes) == 2, each axis corresponds to each dimension.
        If len(sizes) == 3, the first dimension corresponds to the first axis, and the rest corresponds to the second axis.
        This enables to be compatible with the initializations of `.embeddings.RotaryEmbedding2D` and `.embeddings.RotaryEmbedding3D`.
        """
        super().__init__()
        self.ax1 = RotaryEmbedding1D(dim, sizes[0], theta, flatten)
        self.ax2 = (
            RotaryEmbedding1D(dim, sizes[1], theta, flatten)
            if len(sizes) == 2
            else RotaryEmbedding2D(dim, sizes[1:], theta, flatten)
        )


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)


class StandardTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float = 0.0,
        rope: Optional["AxialRotaryEmbedding | RotaryEmbeddingND"] = None,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads."
        self.dim = dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        assert self.heads % self.kv_heads == 0, "heads must be a multiple of kv_heads."

        self.dim_head = dim // heads  # per-Q head dim
        self.dim_kv_total = self.kv_heads * self.dim_head  # total size for K/V projections

        self.rope = rope

        # Pre-norms
        self.norm1 = NormalizeWithCond(dim, emb_dim)
        self.norm2 = NormalizeWithCond(dim, emb_dim)

        # Attention projections
        # Q has H heads; K/V have kv_heads (sharing across Q groups)
        self.q_proj = nn.Linear(dim, heads * self.dim_head, bias=True)
        self.k_proj = nn.Linear(dim, self.dim_kv_total, bias=True)
        self.v_proj = nn.Linear(dim, self.dim_kv_total, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

        # Q/K normalization (per head)
        self.q_norm = Normalize(self.dim_head)
        self.k_norm = Normalize(self.dim_head)

        # Dropouts
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # MLP (4x FFN)
        mlp_dim = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, C)
        emb: ignored (kept for API compatibility)
        """
        # --- Attention ---
        residual = x
        x_norm = self.norm1(x, emb)

        q = self.q_proj(x_norm)  # (B, N, H*Dh)
        k = self.k_proj(x_norm)  # (B, N, kvH*Dh)
        v = self.v_proj(x_norm)  # (B, N, kvH*Dh)

        q = rearrange(q, "b n (h d)    -> b h n d", h=self.heads, d=self.dim_head)
        k = rearrange(k, "b n (kv d)   -> b kv n d", kv=self.kv_heads, d=self.dim_head)
        v = rearrange(v, "b n (kv d)   -> b kv n d", kv=self.kv_heads, d=self.dim_head)

        # Per-head Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to q/k if provided
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Grouped-Query mapping: repeat K/V across groups of Q heads
        if self.kv_heads != self.heads:
            group_size = self.heads // self.kv_heads  # e.g., 4 if H=32, kvH=8
            # repeat_interleave along head dim to map each kv head to a group of q heads
            k = k.repeat_interleave(group_size, dim=1)  # (B, H, N, Dh)
            v = v.repeat_interleave(group_size, dim=1)  # (B, H, N, Dh)

        # Scaled dot-product attention (uses flash/SDPA if available)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        attn = rearrange(attn, "b h n d -> b n (h d)")
        x = residual + self.resid_drop(self.o_proj(attn))

        # --- MLP ---
        residual = x
        x_norm = self.norm2(x, emb)
        x = residual + self.resid_drop(self.mlp(x_norm))
        return x


class StandardTransformerBlockWithMem(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float = 0.0,
        rope: Optional["AxialRotaryEmbedding | RotaryEmbeddingND"] = None,
        kv_heads: Optional[int] = None,
        kv_mem_tokens: int = 0,  # number of learnable KV tokens
        kv_mode: str = "concat",  # "concat" or "cross"
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads."
        self.dim = dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        assert self.heads % self.kv_heads == 0, "heads must be a multiple of kv_heads."

        self.dim_head = dim // heads
        self.dim_kv_total = self.kv_heads * self.dim_head
        self.rope = rope

        # Pre-norms
        self.norm1 = NormalizeWithCond(dim, emb_dim)
        self.norm2 = NormalizeWithCond(dim, emb_dim)

        # Attention projections
        self.q_proj = nn.Linear(dim, heads * self.dim_head, bias=True)
        self.k_proj = nn.Linear(dim, self.dim_kv_total, bias=True)
        self.v_proj = nn.Linear(dim, self.dim_kv_total, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

        # Q/K normalization
        self.q_norm = Normalize(self.dim_head)
        self.k_norm = Normalize(self.dim_head)

        # Dropouts
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # MLP
        mlp_dim = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, bias=True),
        )

        # --- KV memory tokens ---
        self.kv_mem_tokens = kv_mem_tokens
        self.kv_mode = kv_mode
        if kv_mem_tokens > 0:
            self.mem_k = nn.Parameter(torch.randn(1, kv_mem_tokens, self.dim_kv_total))
            self.mem_v = nn.Parameter(torch.randn(1, kv_mem_tokens, self.dim_kv_total))

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Attention ---
        residual = x
        x_norm = self.norm1(x, emb)

        q = self.q_proj(x_norm)  # (B, N, H*Dh)
        k = self.k_proj(x_norm)  # (B, N, kvH*Dh)
        v = self.v_proj(x_norm)  # (B, N, kvH*Dh)

        q = rearrange(q, "b n (h d)  -> b h n d", h=self.heads, d=self.dim_head)
        k = rearrange(k, "b n (kv d) -> b kv n d", kv=self.kv_heads, d=self.dim_head)
        v = rearrange(v, "b n (kv d) -> b kv n d", kv=self.kv_heads, d=self.dim_head)

        # Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if provided
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Expand K/V groups for GQA
        if self.kv_heads != self.heads:
            group_size = self.heads // self.kv_heads
            k = k.repeat_interleave(group_size, dim=1)  # (B, H, N, Dh)
            v = v.repeat_interleave(group_size, dim=1)  # (B, H, N, Dh)

        # ---- Add memory KV tokens ----
        if self.kv_mem_tokens > 0:
            mem_k = self.mem_k.expand(x.size(0), -1, -1)  # (B, M, kvH*Dh)
            mem_v = self.mem_v.expand(x.size(0), -1, -1)

            mem_k = rearrange(mem_k, "b m (kv d) -> b kv m d", kv=self.kv_heads, d=self.dim_head)
            mem_v = rearrange(mem_v, "b m (kv d) -> b kv m d", kv=self.kv_heads, d=self.dim_head)

            if self.kv_heads != self.heads:
                mem_k = mem_k.repeat_interleave(group_size, dim=1)
                mem_v = mem_v.repeat_interleave(group_size, dim=1)

            if self.kv_mode == "concat":
                # Concatenate along sequence length
                k = torch.cat([k, mem_k], dim=2)
                v = torch.cat([v, mem_v], dim=2)

        # Main attention
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        if self.kv_mem_tokens > 0 and self.kv_mode == "cross":
            cross = F.scaled_dot_product_attention(
                q,
                mem_k,
                mem_v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            attn = attn + cross

        attn = rearrange(attn, "b h n d -> b n (h d)")
        x = residual + self.resid_drop(self.o_proj(attn))

        # --- MLP ---
        residual = x
        x_norm = self.norm2(x, emb)
        x = residual + self.resid_drop(self.mlp(x_norm))
        return x


class TransformerBlock2(nn.Module):
    """
    Efficient transformer block with parallel attention + MLP and Query-Key normalization,
    following https://arxiv.org/abs/2302.05442

    Supports axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float,
        rope: Optional[AxialRotaryEmbedding | RotaryEmbeddingND] = None,
    ):
        super().__init__()
        self.rope = rope
        self.norm = NormalizeWithCond(dim, emb_dim)

        self.heads = heads
        dim_head = dim // heads
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=True))

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)

        x = x + self.mlp_out(mlp_h)
        return x


class TransformerBlock(nn.Module):
    """
    Efficient transformer block with parallel attention + MLP and Query-Key normalization,
    following https://arxiv.org/abs/2302.05442

    Supports axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float,
        use_axial: bool = False,
        ax1_len: Optional[int] = None,
        rope: Optional[AxialRotaryEmbedding | RotaryEmbeddingND] = None,
    ):
        super().__init__()
        self.rope = rope.ax2 if (rope is not None and use_axial) else rope
        self.norm = NormalizeWithCond(dim, emb_dim)

        self.heads = heads
        dim_head = dim // heads
        self.use_axial = use_axial
        self.ax1_len = ax1_len
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=True))

        if self.use_axial:
            self.another_attn = AttentionBlock(
                dim, heads, emb_dim, rope.ax1 if rope is not None else None
            )

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        if self.use_axial:
            x, emb = map(
                lambda y: rearrange(y, "b (ax1 ax2) d -> (b ax1) ax2 d", ax1=self.ax1_len),
                (x, emb),
            )
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)

        if self.use_axial:
            ax2_len = x.shape[1]
            x, emb = map(
                lambda y: rearrange(y, "(b ax1) ax2 d -> (b ax2) ax1 d", ax1=self.ax1_len),
                (x, emb),
            )
            x = self.another_attn(x, emb)
            x = rearrange(x, "(b ax2) ax1 d -> (b ax1) ax2 d", ax2=ax2_len)

        x = x + self.mlp_out(mlp_h)

        if self.use_axial:
            x = rearrange(x, "(b ax1) ax2 d -> b (ax1 ax2) d", ax1=self.ax1_len)
        return x


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP block.

    Args:
        dim:            Input/output feature dimension.
        expansion:      MLP expansion factor (e.g., 4.0 like Transformer defaults).
        dropout:        Dropout probability applied after SwiGLU and after output proj.
        bias:           Use bias in linear layers.
        parity_scaling: If True, use inner_dim = floor(expansion * dim * 2/3)
                        to keep parameter count ~parity with GELU MLP.
        zero_init_out:  If True, zero-init the output projection (good for stable residuals).

    Shapes:
        Accepts tensors of shape (..., dim). Returns the same shape.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
        parity_scaling: bool = True,
        zero_init_out: bool = False,
    ):
        super().__init__()
        inner = (
            int(math.floor(dim * expansion * (2.0 / 3.0)))
            if parity_scaling
            else int(math.floor(dim * expansion))
        )
        inner = max(1, inner)

        # Fused projection to produce value & gate in one matmul
        self.fc_in = nn.Linear(dim, 2 * inner, bias=bias)
        self.fc_out = nn.Linear(inner, dim, bias=bias)

        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_out:
            zero_module(self.fc_out)

        self._init_weights()

        self.inner = inner
        self.dim = dim

    def _init_weights(self):
        # Kaiming for input (works well with SiLU), Xavier for output
        nn.init.kaiming_uniform_(self.fc_in.weight, a=math.sqrt(5))
        if self.fc_in.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_in.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc_in.bias, -bound, bound)

        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)
        returns: (..., dim)
        """
        v_and_g = self.fc_in(x)  # (..., 2*inner)
        v, g = v_and_g.split(self.inner, dim=-1)  # (..., inner), (..., inner)
        h = self.drop1(F.silu(g) * v)  # SwiGLU
        out = self.fc_out(h)
        out = self.drop2(out)
        return out


class TransformerBlockKVMem(nn.Module):
    """
    Efficient transformer block with parallel attention + MLP and Query-Key normalization,
    following https://arxiv.org/abs/2302.05442

    Supports axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float,
        use_axial: bool = False,
        ax1_len: Optional[int] = None,
        rope: Optional[AxialRotaryEmbedding | RotaryEmbeddingND] = None,
        num_kv_tokens: Optional[int] = 0,
    ):
        super().__init__()
        self.rope = rope.ax2 if (rope is not None and use_axial) else rope
        self.norm = NormalizeWithCond(dim, emb_dim)

        self.heads = heads
        dim_head = dim // heads
        self.use_axial = use_axial
        self.ax1_len = ax1_len
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=True))

        if self.use_axial:
            self.another_attn = AttentionBlock(
                dim, heads, emb_dim, rope.ax1 if rope is not None else None
            )

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )
        self.num_kv_tokens = num_kv_tokens
        self.k_mem = None
        self.v_mem = None
        if num_kv_tokens > 0:
            # shape: (1, H, M, Dh) so we can expand to batch
            # self.k_mem = nn.Parameter(torch.randn(1, heads, num_kv_tokens, dim_head) * 0.02)
            # self.v_mem = nn.Parameter(torch.randn(1, heads, num_kv_tokens, dim_head) * 0.02)

            self.mem = SwiGLUMLP(
                dim=dim, expansion=4.0, dropout=dropout, bias=True, zero_init_out=True
            )
            # nn.Sequential(
            # nn.Linear(dim, dim * 4, bias=True),
            # nn.SiLU(),
            ## nn.Linear(dim * 4, dim * 4, bias=True),
            ## nn.SiLU(),
            # zero_module(nn.Linear(dim * 4, dim, bias=True)),
            # )
            # self.mem_gate = nn.Parameter(torch.zeros(heads))  # learned in (-1,1) after tanh
            # self.mem_gate = nn.Parameter(torch.zeros(heads))  # learned in (-1,1) after tanh

            # lact
            # self.mem_gate = nn.Sequential(zero_module(nn.Linear(dim, heads)), nn.SiLU())
            # self.mem_gate = nn.Sequential(zero_module(nn.Linear(dim, heads)), nn.Sigmoid())
            self.mem_gate = nn.Sequential(zero_module(nn.Linear(dim, heads)), nn.Tanh())

            # self.mem_attn_out_lora = zero_module(nn.Linear(dim, dim, bias=True))
            self.mem_attn_out_lora = nn.Sequential(
                nn.Linear(dim, 32, bias=True),
                zero_module(nn.Linear(32, dim, bias=False)),
            )

            # self.kv_rope = RotaryEmbedding3D(
            #    dim_head, (500, self.rope.sizes[1], self.rope.sizes[2])
            # )
            # self.pos_enc = SinusoidalPositionalEmbedding(
            #    dim_head, (500, self.rope.sizes[1], self.rope.sizes[2])
            # )

        else:
            self.k_mem = None
            self.v_mem = None

    @staticmethod
    def _rms(t):
        return t.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """

        B, N, C = x.shape
        if self.use_axial:
            x, emb = map(
                lambda y: rearrange(y, "b (ax1 ax2) d -> (b ax1) ax2 d", ax1=self.ax1_len),
                (x, emb),
            )
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.mem is not None:
            q_prior = q
            k_prior = k

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        if self.k_mem is not None:
            # expand to batch: (B, H, M, Dh)
            k_mem = self.k_mem.expand(B, -1, -1, -1)
            v_mem = self.v_mem.expand(B, -1, -1, -1)
            # k_mem = self.kv_rope(k_mem)
            # concat on sequence dimension (n -> n + M)
            # k = torch.cat([k_mem, k], dim=2)
            # v = torch.cat([v_mem, v], dim=2)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)

        # or do cross-attention
        if self.mem is not None:
            f = self.mem(rearrange(q_prior, "b h n d -> b n (h d)")).detach()
            f = rearrange(f, "b n (h d) -> b h n d", h=self.heads)
            # RMS match
            f = f / self._rms(f)
            # f = f * (self._rms(x) / self._rms(f))
            # gate = torch.tanh(self.mem_gate(q)).view(1, self.heads, 1, 1)
            # gate = torch.tanh(self.mem_gate).view(1, self.heads, 1, 1)
            gate = (
                self.mem_gate(_x).permute(0, 2, 1).unsqueeze(-1)
            )  # bs, n, h #.view(1, self.heads, 1, 1)

            # lact
            x = x + gate * f
            # x = gate + (1 - gate) * f
            # x = f

            if getattr(self, "is_training", False):
                self.local_loss = (
                    self.mem(rearrange(k_prior, "b h n d -> b n (h d)"))
                    - rearrange(v, "b h n d -> b n (h d)", h=self.heads)
                ) ** 2
        # else:
        #    x = F.scaled_dot_product_attention(q, k, v)

        if False:  # True:  # self.k_mem is not None and True:  # False:
            x = x + rearrange(
                self.mem(rearrange(q, "b h n d -> b n (h d)")),
                "b n (h d) -> b h n d",
                h=self.heads,
            )

            # x = x + F.scaled_dot_product_attention(q, k_mem, v_mem, attn_mask=None, dropout_p=0.0)

        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)  # + self.mem_attn_out_lora(x)

        if self.use_axial:
            ax2_len = x.shape[1]
            x, emb = map(
                lambda y: rearrange(y, "(b ax1) ax2 d -> (b ax2) ax1 d", ax1=self.ax1_len),
                (x, emb),
            )
            x = self.another_attn(x, emb)
            x = rearrange(x, "(b ax2) ax1 d -> (b ax1) ax2 d", ax2=ax2_len)

        x = x + self.mlp_out(mlp_h)

        if self.use_axial:
            x = rearrange(x, "(b ax1) ax2 d -> b (ax1 ax2) d", ax1=self.ax1_len)
        return x


class StandardTransformerBlock_(nn.Module):
    """
    Standard pre-norm Transformer block:
      LN -> MHA -> residual
      LN -> MLP -> residual

    Keeps:
      - RoPE applied to q and k
      - Per-head q/k normalization (QK-Norm)

    Differences vs your variant:
      - No fused parallel projection
      - No conditional norm; plain LayerNorm
      - No zero-init on output projections
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        rope: Optional["AxialRotaryEmbedding | RotaryEmbeddingND"] = None,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads."
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.rope = rope

        # Pre-norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention projections
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

        # Q/K normalization (per head)
        self.q_norm = Normalize(self.dim_head)
        self.k_norm = Normalize(self.dim_head)

        # Dropouts
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # MLP (GEGLU/GELU would also be fine; sticking to simple 4x FFN)
        mlp_dim = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),
            nn.SiLU(),  # use GELU if you prefer closer to "vanilla"
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, C)
        emb: ignored (kept for API compatibility)
        """
        # --- Attention ---
        residual = x
        x_norm = self.norm1(x)

        q = rearrange(self.q_proj(x_norm), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.k_proj(x_norm), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.v_proj(x_norm), "b n (h d) -> b h n d", h=self.heads)

        # Per-head Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on q/k if provided
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        attn = rearrange(attn, "b h n d -> b n (h d)")
        x = residual + self.resid_drop(self.o_proj(attn))

        # --- MLP ---
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.resid_drop(self.mlp(x_norm))
        return x


class Downsample(nn.Module):
    """
    Downsample block for U-ViT.
    Done by average pooling + conv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable-next=not-callable
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    Upsample block for U-ViT.
    Done by conv + nearest neighbor upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x
