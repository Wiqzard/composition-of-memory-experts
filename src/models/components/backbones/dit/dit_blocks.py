import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp

from src.models.components.modules.embeddings import RotaryEmbeddingND
from src.models.components.modules.zero_module import zero_module


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


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


class Attention(nn.Module):
    """
    Standard MHA with:
      - optional RoPE on q,k
      - per-head LayerNorm on q,k
      - fused SDPA path
      - **mem** residual (SwiGLU MLP on concatenated q) + optional aux loss

    attn_kwargs:
      - causal: bool
      - n_frames: int
      - mem: bool
      - mem_aux_loss: bool (default: False) -> sets self.local_loss
      - mem_detach_q: bool (default: True) -> detach q for mem MLP input
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional["RotaryEmbeddingND"] = None,
        fused_attn: bool = True,
        attn_kwargs: Optional[dict] = None,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        ak = attn_kwargs or {}
        self.causal = ak.get("causal", False)
        self.n_frames = ak.get("n_frames", 1)

        # --- mem extras
        self.mem = ak.get("mem", False)
        self.mem_aux_loss = ak.get("mem_aux_loss", False)
        self.mem_detach_q = ak.get("mem_detach_q", True)
        if self.mem:
            # zero_init_out => starts as a no-op until trained
            self.memory_model = SwiGLUMLP(dim=dim, expansion=4.0, bias=True, zero_init_out=True)
            # (optional) a tiny gate if you want extra safety; comment out if not needed
            # self.mem_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B,H,N,D)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # (B,H,N,D)
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)  # (B,H,N,N)
            if mask is not None:
                inf = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(mask, inf)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v  # (B,H,N,D)

        # ---- mem residual (before output projection)
        if self.mem:
            # q -> (B,N,C)
            q_flat = rearrange(q, "b h n d -> b n (h d)")
            q_flat_in = q_flat  # .detach()
            m_flat = self.memory_model(q_flat_in)  # (B,N,C)
            m = rearrange(m_flat, "b n (h d) -> b h n d", h=self.num_heads)

            # Optional tiny gate if you enabled it in __init__:
            # out = out + torch.tanh(self.mem_gate) * m
            out = out + m

            # match the immediate value stream; detach target to avoid leakage
            v_flat = rearrange(v, "b h n d -> b n (h d)")
            self.local_loss = F.mse_loss(m_flat, v_flat.detach())

        # merge heads, output proj
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionLegacy(nn.Module):
    """
    Adapted from timm.models.vision_transformer,
    to support the use of RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbeddingND] = None,
        fused_attn: bool = True,
        attn_kwargs: Optional[dict] = None,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.causal = attn_kwargs.get("causal", False) if attn_kwargs else False
        self.n_frames = attn_kwargs.get("n_frames", 1) if attn_kwargs else 1

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.fused_attn:
            # pylint: disable-next=not-callable
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                inf = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(mask, inf)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionGQA(nn.Module):
    """
    MHA with optional Grouped-Query Attention (GQA) / Multi-Query Attention (MQA).

    - num_heads:   # of query heads (Q heads)
    - kv_heads:    # of key/value heads (shared across groups of Q heads)
                    kv_heads must divide num_heads.
      Examples:
        kv_heads == num_heads  -> standard MHA
        kv_heads == 1          -> MQA
        kv_heads | num_heads   -> GQA (groups of size num_heads//kv_heads)

    Preserves:
      - RoPE on q,k
      - per-head LayerNorm on q,k
      - fused (sdpa) and manual attention paths
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbeddingND] = None,
        fused_attn: bool = True,
        kv_heads: Optional[int] = None,
        attn_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # --- GQA params
        kv_heads = num_heads if kv_heads is None else kv_heads
        assert num_heads % kv_heads == 0, "num_heads must be a multiple of kv_heads"
        self.kv_heads = kv_heads
        self.group_size = num_heads // kv_heads  # #Q heads that share one K/V head

        # separate projections to allow different #heads for q vs k/v
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        # keep your existing flags
        self.causal = (attn_kwargs or {}).get("causal", False)
        self.n_frames = (attn_kwargs or {}).get("n_frames", 1)
        self.mem = (attn_kwargs or {}).get("mem", False)
        if self.mem:
            self.memory_model = SwiGLUMLP(dim=dim, expansion=4.0, bias=True, zero_init_out=True)

    def _apply_rope_and_norm(self, q, k):
        # q, k: (B, H, N, D)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        return q, k

    def _expand_kv_to_q_heads(self, k, v):
        """
        Repeat K/V heads to match Q heads for SDPA.

        k, v: (B, kvH, N, D) -> (B, H, N, D) by repeating each kv head group_size times.
        """
        if self.kv_heads == self.num_heads:
            return k, v
        # repeat_interleave along head dimension
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        return k, v

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # project to q/k/v with different head counts
        q = (
            self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # (B,H,N,D)
        k = (
            self.k_proj(x).reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # (B,kvH,N,D)
        v = (
            self.v_proj(x).reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # (B,kvH,N,D)

        # per-head norm + RoPE
        q, k = self._apply_rope_and_norm(q, k)

        if self.fused_attn:
            # match Q head count for fused SDPA
            k_exp, v_exp = self._expand_kv_to_q_heads(k, v)  # (B,H,N,D)
            # PyTorch SDPA expects (B, H, N, D)
            out = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=mask,  # should be broadcastable to (B*H, N, N) or (B, H, N, N)
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # (B,H,N,D)
            if self.mem:
                f = self.mem(rearrange(q, "b h n d -> b n (h d)")).detach()
                out = out + f
                if getattr(self, "is_training", False):
                    self.local_loss = (
                        self.mem(rearrange(q, "b h n d -> b n (h d)"))
                        - rearrange(v, "b h n d -> b n (h d)", h=self.heads)
                    ) ** 2

        else:
            raise NotImplementedError("Fused attention is required for DiT models.")
            # manual attention with shared K/V across Q-head groups
            q_scaled = q * self.scale  # (B,H,N,D)

            if self.kv_heads == self.num_heads:
                attn_logits = torch.matmul(q_scaled, k.transpose(-2, -1))  # (B,H,N,N)
                if mask is not None:
                    inf = torch.finfo(attn_logits.dtype).min
                    attn_logits = attn_logits.masked_fill(mask, inf)
                attn = attn_logits.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out = torch.matmul(attn, v)  # (B,H,N,D)
            else:
                # compute attention per Q-head group against its shared K/V head
                B_, H, N_, D = q_scaled.shape
                kvH = self.kv_heads
                G = self.group_size  # H = kvH * G

                # reshape to group queries: (B, kvH, G, N, D)
                qg = q_scaled.view(B_, kvH, G, N_, D)
                # K,V: (B, kvH, N, D) -> add group axis
                k_ = k[:, :, :, :].unsqueeze(2)  # (B, kvH, 1, N, D)
                v_ = v[:, :, :, :].unsqueeze(2)  # (B, kvH, 1, N, D)

                # attn logits: (B, kvH, G, N, N)
                attn_logits = torch.matmul(qg, k_.transpose(-2, -1))
                if mask is not None:
                    inf = torch.finfo(attn_logits.dtype).min
                    # broadcast mask to (B, 1, 1, N, N) or compatible
                    attn_logits = attn_logits.masked_fill(mask, inf)

                attn = attn_logits.softmax(dim=-1)
                attn = self.attn_drop(attn)

                # out_g: (B, kvH, G, N, D)
                out_g = torch.matmul(attn, v_)
                # merge kvH*G back to H
                out = out_g.view(B_, H, N_, D)

        # merge heads
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AdaLayerNorm(nn.Module):
    """
    Adaptive layer norm (AdaLN).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaLN layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class AdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (AdaLN-Zero).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AdaLN-Zero layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return modulate(self.norm(x), shift, scale), gate


class DiTBlock(nn.Module):
    """
    A DiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        **block_kwargs: dict,
    ):
        """
        Args:
            hidden_size: Number of features in the hidden layer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden layer size in the MLP. None to skip the MLP.
            block_kwargs: Additional arguments to pass to the Attention block.
        """
        super().__init__()

        self.norm1 = AdaLayerNormZero(hidden_size)
        kv_heads = block_kwargs.get("kv_heads", None)
        if kv_heads is not None:
            attn_clss = AttentionGQA
        else:
            attn_clss = Attention

        self.attn = attn_clss(
            hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = AdaLayerNormZero(hidden_size)
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.attn.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the DiT block.
        In original implementation, conditioning is uniform across all tokens in the sequence. Here, we extend it to support token-wise conditioning (e.g. noise level can be different for each token).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x, mask=mask)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x


class DITFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass of the DiT final layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x = self.norm_final(x, c)
        x = self.linear(x)
        return x
