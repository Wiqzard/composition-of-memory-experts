import math
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from torch import nn

from src.models.components.backbones.base_backbone import BaseBackbone
from src.models.components.backbones.dit.dit_base import DiTBase, DiTBaseSparse
from src.models.components.backbones.dit.dit_blocks import DiTBlock, DITFinalLayer
from src.models.components.modules.embeddings import (
    RandomDropoutCondEmbedding,
    StochasticTimeEmbedding,
)

# from src.models.components.transformer.ditv2 import DiTBase


try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    # from mamba_ssm import Mamba2

except Exception:
    print("Failed to import mamba_ssm; make sure it is installed correctly.")

from mamba_ssm import Mamba

# from mamba_ssm import Mamba2
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class DiTMamba(nn.Module):
    """
    DiTMamba: windowed DiT with interleaved global processing (full attention or Mamba).

    Core idea
    ---------
    - Input tokens are arranged time-major as in DiTBase: sequence over (t * p) where
      p = num_patches (or 1 for 1D).
    - We process local chunks of k frames at a time, non-overlapping:
        window_size_tokens = k * (num_patches or 1)
      Each local window is run through a stack of DiTBlocks (shared weights by layer index).
    - After every global_every local layers, we run a global layer over ALL tokens at once.
      That global layer can be either:
        * "attn": a DiTBlock over the full sequence (full attention)
        * "mamba": a user-provided module that takes (B, N, C) -> (B, N, C)

    Notes
    -----
    - Handles ragged tail windows (when T % k != 0) without padding; the final shorter
      window is processed as-is.
    - If you need masking, pass mask shaped (B, N) with 1 for valid tokens; it will be
      sliced per-window automatically.
    - Positional embeddings are optional; pass any nn.Module that adds in-place to x
      (signature: pos_emb(x) -> x).

    Args
    ----
    num_patches: int or None
        Number of spatial patches per frame (set None for 1D tokens per frame).
    max_temporal_length: int
        Maximum frames supported (used only for sanity/pos-emb config if provided).
    out_channels: int
        Output channels per token for the diffusion head (doubles internally if learn_sigma=True).
    hidden_size: int
        Token embedding dimension.
    depth: int
        Total number of *local* DiT layers. Global layers are interleaved via global_every.
    num_heads: int
        Attention heads for DiTBlock (used for both local and global attn if selected).
    mlp_ratio: float
        MLP width multiplier in DiTBlock.
    learn_sigma: bool
        If True, output channels are doubled for mean+sigma style heads.
    k: int
        Window size in frames (non-overlapping).
    global_every: int
        Insert one global layer after every global_every local layers. If 0, no globals.
    global_mode: {"attn","mamba"}
        Type of global block. "attn" uses DiTBlock; "mamba" uses mamba_factory.
    mamba_factory: Optional[Callable[[int], nn.Module]]
        Factory returning an nn.Module that maps (B, N, C) -> (B, N, C) with hidden_size features.
        Required if global_mode == "mamba".
    pos_emb: Optional[nn.Module]
        A positional embedding module that can be applied once to (B, N, C) tokens; if provided,
        it is added at the start.
    use_gradient_checkpointing: bool
        Enable PyTorch checkpointing on blocks to save memory.
    kv_heads: Optional[int]
        Forwarded into DiTBlock if you use multi-query/grouped KV setups.
    attn_kwargs: dict
        Extra kwargs for DiTBlock attention (e.g., flash attention flags, memory settings).
    """

    def __init__(
        self,
        *,
        num_patches: Optional[int],
        max_temporal_length: int = 16,
        out_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        k: int = 4,
        global_every: int = 4,
        global_mode: Literal["attn", "mamba"] = "attn",
        mamba_factory: Optional[Callable[[int], nn.Module]] = None,
        pos_emb: Optional[nn.Module] = None,
        use_gradient_checkpointing: bool = False,
        tokens_per_frame: Optional[int] = None,  # legacy alias for num_patches
        kv_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        attn_kwargs = {}

        assert k >= 1, "k (window size in frames) must be >= 1"
        assert depth >= 1, "depth must be >= 1"
        if global_mode == "mamba":
            assert mamba_factory is not None, "Provide mamba_factory when global_mode='mamba'"

        self.num_patches = num_patches
        self.max_temporal_length = max_temporal_length
        self.tokens_per_frame = num_patches if tokens_per_frame is None else tokens_per_frame
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)

        self.k = k
        self.global_every = int(global_every)
        self.global_mode = global_mode
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Local DiT layers (applied per-window)
        self.local_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    kv_heads=kv_heads,
                    attn_kwargs=attn_kwargs,
                )
                for _ in range(depth)
            ]
        )

        # Global blocks interleaved through the stack
        self.global_blocks = nn.ModuleList()
        self.n_mamba_head = 8
        if self.global_every > 0:
            # Number of global insertions = ceil(depth / global_every)
            n_globals = math.ceil(depth / self.global_every)
            for _ in range(n_globals):
                if global_mode == "attn":
                    self.global_blocks.append(
                        DiTBlock(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            kv_heads=kv_heads,
                            attn_kwargs=attn_kwargs,
                        )
                    )
                else:  # "mamba"
                    # self.global_blocks.append(mamba_factory(hidden_size // self.n_mamba_head))
                    self.global_blocks.append(
                        Mamba(
                            d_model=hidden_size // self.n_mamba_head,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        )
                        # Mamba2(
                        #    d_model=hidden_size // self.n_mamba_head,
                        #    d_state=16,
                        #    d_conv=4,
                        #    expand=2,
                        # )
                    )

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

        # Optional positional embedding (applied once at the start)
        self.pos_emb = pos_emb

    def _checkpoint(self, module: nn.Module, *args, **kwargs):
        if self.use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        return module(*args, **kwargs)

    def _iter_windows(
        self, x: torch.Tensor, c: Optional[torch.Tensor], mask: Optional[torch.Tensor]
    ):
        """
        Generator yielding non-overlapping windows along time (frames).
        Shapes:
            x:    (B, N, C)
            c:    (B, N, C) or None
            mask: (B, N) or None
        Yields tuples (xw, cw, mw, start_idx, end_idx) where:
            xw: (B, W_tokens, C), cw same if provided, mw: (B, W_tokens) if provided
            start_idx/end_idx in token space [0, N)
        """
        B, N, Cdim = x.shape
        tpf = self.tokens_per_frame
        assert N % tpf == 0, "Sequence tokens must be divisible by tokens_per_frame (num_patches)."
        T = N // tpf

        w_frames = self.k
        frame_idx = 0
        while frame_idx < T:
            next_frame = min(frame_idx + w_frames, T)
            start_tok = frame_idx * tpf
            end_tok = next_frame * tpf
            xw = x[:, start_tok:end_tok, :]
            cw = c[:, start_tok:end_tok, :] if c is not None else None
            mw = mask[:, start_tok:end_tok] if mask is not None else None
            yield xw, cw, mw, start_tok, end_tok
            frame_idx = next_frame

    def _apply_local_block_once(self, block, x, c, mask):
        B, N, C = x.shape
        tpf = self.tokens_per_frame
        T = N // tpf
        k = self.k
        W = (T + k - 1) // k
        pad_frames = W * k - T
        if pad_frames:
            pad_tokens = pad_frames * tpf
            x = F.pad(x, (0, 0, 0, pad_tokens))
            if c is not None:
                c = F.pad(c, (0, 0, 0, pad_tokens))
            if mask is not None:
                mask = F.pad(mask, (0, pad_tokens), value=0)

        xw = x.view(B, W, k * tpf, C).reshape(B * W, k * tpf, C)
        cw = c.view(B, W, k * tpf, C).reshape(B * W, k * tpf, C) if c is not None else None
        mw = mask.view(B, W, k * tpf).reshape(B * W, k * tpf) if mask is not None else None

        yw = self._checkpoint(block, xw, cw if cw is not None else xw, mw)

        y = yw.view(B, W, k * tpf, C).reshape(B, W * k * tpf, C)
        y = y[:, :N, :]
        return y

    def _apply_local_block_once_scan(
        self,
        block: nn.Module,
        x: torch.Tensor,
        c: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply a single local block to each window independently, then stitch back.
        """
        B, N, Cdim = x.shape
        out_chunks: List[torch.Tensor] = []
        for xw, cw, mw, start, end in self._iter_windows(x, c, mask):
            # DiTBlock signature in your base: block(x, c, mask)
            yw = self._checkpoint(block, xw, cw if cw is not None else xw, mw)
            out_chunks.append((start, end, yw))
        # Stitch back
        y = torch.empty_like(x)
        for start, end, yw in out_chunks:
            y[:, start:end, :] = yw
        return y

    def _apply_global_block_once(
        self,
        block: nn.Module,
        x: torch.Tensor,
        c: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply a single global block to the full token sequence.
        """
        if self.global_mode == "attn":
            return self._checkpoint(block, x, c if c is not None else x, mask)
        else:
            # Mamba-style block: assumes signature y = block(x), optionally ignores c/mask.
            return self._checkpoint(block, x)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, C)
        c: Optional[torch.Tensor] = None,  # (B, N, C) or None
        mask: Optional[torch.Tensor] = None,  # (B, N) or None
    ) -> torch.Tensor:
        """
        Process sequence with windowed local DiT blocks and interleaved global passes.
        """
        B, N, Cdim = x.shape
        assert (
            Cdim == self.hidden_size
        ), f"hidden_size mismatch: got {Cdim}, expected {self.hidden_size}"
        assert (
            N % (self.num_patches or 1) == 0
        ), "N must be divisible by num_patches (or 1 for 1D)."

        # Optional single-shot positional embedding
        if self.pos_emb is not None:
            x = self.pos_emb(x)

        # If no external conditioning c is provided, default to x (common in DiT)
        if c is None:
            c = x

        global_idx = 0
        for i, local_block in enumerate(self.local_blocks, start=1):
            # Local pass on non-overlapping windows of k frames
            x = self._apply_local_block_once(local_block, x, c, mask)

            # Interleave global after every global_every local layers
            if self.global_every > 0 and (i % self.global_every == 0):
                gblock = self.global_blocks[global_idx]
                x = rearrange(x, "b n (h c) -> (b h) n  c", h=self.n_mamba_head)
                x = self._apply_global_block_once(gblock, x, c, mask)
                x = rearrange(x, "(b h) n c -> b n (h c)", h=self.n_mamba_head)
                global_idx += 1

        # If depth is not a multiple of global_every, we may still have one global left by ceil
        # We intentionally do NOT apply trailing globals here; they are already scheduled above.

        # Final projection head
        x = self.final_layer(x, c)
        return x


# ------------------------
# Selective State-Space (per-channel diagonal A, vectors B and C)
# ------------------------


class SelectiveSSM(nn.Module):
    """
    Per-feature diagonal SSM:
        s_t = exp(Δ * A) ⊙ s_{t-1} + B ⊙ x_t
        y_t = ⟨C, s_t⟩
    Shapes:
        x: (B, N, C)
        A,B,C: (C, D)    [C = d_model, D = d_state]
        s: (B, C, D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # Parameters
        self.a_log = nn.Parameter(torch.randn(d_model, d_state) * -0.5)  # (C,D)
        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.02)  # (C,D)
        self.c = nn.Parameter(torch.randn(d_model, d_state) * 0.02)  # (C,D)

        # dt projection (per feature)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        # self.dt_proj = nn.Linear(d_model, d_state, bias=True)
        self.dt_min, self.dt_max = dt_min, dt_max

        # depthwise conv prefilter
        self.filter = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

    def _discretize(self, dt_vec: torch.Tensor):
        """
        dt_vec: (B, N, C) -> dt: (B, 1, C) in [dt_min, dt_max]
        A     : (C, D)    with negative diag via -softplus
        """
        dt = torch.sigmoid(dt_vec) * (self.dt_max - self.dt_min) + self.dt_min  # (B,N,C)
        dt = dt.mean(dim=1, keepdim=True)  # (B,1,C) — share step across time for stability
        A = -F.softplus(self.a_log)  # (C,D), strictly negative
        return dt, A

    @torch.no_grad()
    def _init_state(self, B: int, device, dtype):
        return torch.zeros(B, self.d_model, self.d_state, device=device, dtype=dtype)  # (B,C,D)

    # @torch.compile
    def _scan_forward(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
        """
        x : (B,N,C)
        dt: (B,1,C)
        A : (C,D)
        returns y: (B,N,C)
        """
        B, N, C = x.shape
        D = self.d_state
        assert C == self.d_model, f"d_model mismatch: x has C={C}, module has {self.d_model}"

        # Defensive: if A came in as (D,C) (buggy init), transpose it once
        if A.shape == (D, C):
            A = A.transpose(0, 1)
        assert A.shape == (C, D), f"A must be (C,D), got {tuple(A.shape)}"

        s = self._init_state(B, x.device, x.dtype)  # (B,C,D)
        y = torch.empty(B, N, C, device=x.device, dtype=x.dtype)

        # Broadcasted decay per (B,C,D):
        # dt: (B,1,C) -> (B,C,1)
        dt_bc1 = dt.transpose(1, 2).unsqueeze(-1)  # (B,C,1)
        # A: (C,D) -> (1,C,1,D) -> (B,C,1,D)
        A_b = A.view(1, C, 1, D).expand(B, C, 1, D)  # (B,C,1,D)
        expA = torch.exp(dt_bc1 * A_b).squeeze(2)  # (B,C,D)

        b = self.b
        c = self.c
        # Defensive: ensure (C,D)
        if b.shape == (D, C):
            b = b.transpose(0, 1)
        if c.shape == (D, C):
            c = c.transpose(0, 1)
        b_b = b.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)
        c_b = c.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)

        for t in range(N):
            xt = x[:, t, :].unsqueeze(-1)  # (B,C,1)
            s = expA * s + b_b * xt  # (B,C,D)
            y[:, t, :] = (c_b * s).sum(dim=-1)  # (B,C) -> (B,C)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C=d_model)
        returns: (B, N, C)
        """
        B, N, C = x.shape
        assert C == self.d_model

        # prefilter
        xf = self.filter(x.transpose(1, 2)).transpose(1, 2).contiguous()  # (B,N,C)

        # per-step delta in (0, +) – your choice; this keeps your behavior
        dt_vec = self.dt_proj(xf)  # (B,N,C)
        dt = torch.sigmoid(dt_vec) * (self.dt_max - self.dt_min) + self.dt_min

        # parameters: must be (dim, d_state)
        # A must be strictly negative; you already use -softplus
        A = (-F.softplus(self.a_log)).contiguous()  # (C, D)
        Bp = self.b.contiguous()  # (C, D)
        Cp = self.c.contiguous()  # (C, D)

        # >>> DO NOT transpose u/delta; the fused op expects (B,N,C) <<<
        y_f = selective_scan_fn(
            u=xf.contiguous(),  # (B,N,C)
            delta=dt.contiguous(),  # (B,N,C)
            A=A,
            B=Bp,
            C=Cp,  # each (C,D)
            D=None,  # optional; keep None unless you use it
            z=None,  # optional gating; not used here
            delta_bias=None,  # optional
            delta_softplus=False,  # you already made delta positive
        )

        if not self.bidirectional:
            return y_f

        # bidirectional: run again on time-reversed input, then flip back
        xr = torch.flip(xf, dims=[1]).contiguous()
        dtr = torch.flip(dt, dims=[1]).contiguous()
        y_r = selective_scan_fn(
            u=xr, delta=dtr, A=A, B=Bp, C=Cp, D=None, z=None, delta_bias=None, delta_softplus=False
        )
        return y_f + torch.flip(y_r, dims=[1])

    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,C)
        """
        B, N, C = x.shape
        assert C == self.d_model, f"Expected last dim {self.d_model}, got {C}"

        # depthwise conv prefilter
        xf = self.filter(x.transpose(1, 2)).transpose(1, 2)  # (B,N,C)

        dt_vec = self.dt_proj(xf)  # (B,N,C)
        dt, A = self._discretize(dt_vec)

        y_f = self._scan_forward(xf, dt, A)

        if not self.bidirectional:
            return y_f

        xr = torch.flip(xf, dims=[1])
        y_r = self._scan_forward(xr, dt, A)
        return y_f + torch.flip(y_r, dims=[1])


# ------------------------
# Mamba-style block
# ------------------------
class MambaBlock(nn.Module):
    """
    Mamba-style block with selective SSM and gating.
    (B,N,C) -> (B,N,C)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        d_inner = expand * d_model

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)  # split: u (to SSM) and g (gate)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state, bidirectional=bidirectional)
        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,C)
        """
        residual = x
        x = self.norm(x)
        u_g = self.in_proj(x)  # (B,N,2*d_inner)
        u, g = torch.split(u_g, u_g.shape[-1] // 2, dim=-1)
        g = torch.sigmoid(g)
        v = self.ssm(u)  # (B,N,d_inner)
        z = self.out_proj(g * v)  # gated
        z = self.dropout(z)
        return residual + z


# --- add this somewhere central (once) ---
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        n = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(n + self.eps)


# If you already have SinusoidalPositionalEmbedding in your codebase, reuse it.
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
        super().__init__()
        N = 1
        for s in shape:
            N *= s
        pos = torch.arange(N).float().unsqueeze(1)  # (N,1)
        i = torch.arange(embed_dim).float().unsqueeze(0)  # (1,C)
        div = torch.exp(-math.log(10000.0) * (2 * (i // 2)) / embed_dim)
        pe = torch.zeros(N, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div[:, 0::2])
        pe[:, 1::2] = torch.cos(pos * div[:, 1::2])
        pe = pe.view(1, N, embed_dim)
        if learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer("pe", pe, persistent=False)
        self.learnable = learnable

    def forward(self, x):
        # x: (B,N,C)
        return x + self.pe[:, : x.size(1), :]


def make_mamba(
    d_model: int,
    d_state: int = 16,
    expand: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> nn.Module:
    return MambaBlock(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        dropout=dropout,
        bidirectional=bidirectional,
    )


class DiT3D(BaseBackbone):
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        hidden_size: int,
        patch_size: int,
        variant: str = "full",  # set to "mamba" to enable DiTMamba
        pos_emb_type: str = "learned_1d",
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = False,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        use_causal_mask: bool = False,
        kv_heads: Optional[int] = None,
        mem: bool = False,
        # --- NEW (Mamba interleave) ---
        k: int = 4,  # frames per local window
        global_every: int = 3,  # insert global every N local layers
        global_mode: str = "mamba",  # {"mamba","attn"}
        mamba_d_state: int = 16,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_bidirectional: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.use_causal_mask = use_causal_mask
        self.n_frames = max_tokens

        super().__init__(
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            noise_level_emb_dim=hidden_size,
            use_fourier_noise_embedding=use_fourier_noise_embedding,
            use_causal_mask=use_causal_mask,
            external_cond_dropout=external_cond_dropout,
        )

        # --- CHANGED: handle non-square ---
        channels, H, W = x_shape[:3]
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"H={H}, W={W} must be divisible by patch_size={self.patch_size}"

        self.H, self.W = H, W
        self.Hp, self.Wp = H // self.patch_size, W // self.patch_size  # patches per dim
        self.num_patches = self.Hp * self.Wp
        out_channels = self.patch_size**2 * channels

        # Patch embedder
        self.patch_embedder = PatchEmbed(
            img_size=(H, W),
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
        )

        # --- NEW: choose backbone ---
        # If you want to activate Mamba, set variant="mamba"
        if variant == "mamba":
            # Optional simple 1D positional embedding over N = T * P tokens
            pos_emb = None
            if pos_emb_type in {"learned_1d", "sinusoidal_1d"}:
                pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=hidden_size,
                    shape=(max_tokens * self.num_patches,),
                    learnable=(pos_emb_type == "learned_1d"),
                )

            # Build DiTMamba
            self.dit_base = DiTMamba(
                num_patches=self.num_patches,
                max_temporal_length=max_tokens,
                out_channels=out_channels,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                learn_sigma=False,  # we output pixel patches directly
                k=k,
                global_every=global_every,
                global_mode="mamba" if global_mode == "mamba" else "attn",
                mamba_factory=(
                    (
                        lambda d: make_mamba(
                            d_model=d,
                            d_state=mamba_d_state,
                            expand=mamba_expand,
                            dropout=mamba_dropout,
                            bidirectional=mamba_bidirectional,
                        )
                    )
                    if global_mode == "mamba"
                    else None
                ),
                pos_emb=pos_emb,
                use_gradient_checkpointing=use_gradient_checkpointing,
                kv_heads=kv_heads,
                attn_kwargs={"mem": mem},
            )
        else:
            # Fallback to your original DiTBase/DiTBaseSparse
            base_cls = DiTBaseSparse if variant == "sparse" else DiTBase
            self.dit_base = base_cls(
                num_patches=self.num_patches,
                max_temporal_length=max_tokens,
                out_channels=out_channels,
                variant=variant,
                pos_emb_type=pos_emb_type,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                learn_sigma=False,
                use_gradient_checkpointing=use_gradient_checkpointing,
                kv_heads=kv_heads,
                spatial_grid_shape=(self.Hp, self.Wp),
                attn_kwargs={"mem": mem},
                **kwargs,
            )

        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.hidden_size if self.external_cond_dim else 0

    @staticmethod
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=self.Hp,
            w=self.Wp,
            p=self.patch_size,
            q=self.patch_size,
        )

    def _get_causal_mask(self, q: torch.Tensor) -> torch.Tensor:
        N = q.size(-2)
        assert self.n_frames >= 1
        assert N % self.n_frames == 0, f"N={N} not divisible by n_frames={self.n_frames}"
        tpf = N // self.n_frames
        device = q.device
        pos = torch.arange(N, device=device)
        frame_id = pos // tpf
        allowed = frame_id.unsqueeze(1) >= frame_id.unsqueeze(0)  # (N,N)
        attn_mask_bool = allowed
        return attn_mask_bool

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        noise_levels = noise_levels.to(x.dtype)

        # (B,T,C,H,W) -> patchify -> (B, T*P, C)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        _, num_tokens, _ = x.shape
        if noise_levels.shape[1] != num_tokens:
            emb = self.noise_level_pos_embedding(noise_levels)
            if external_cond is not None:
                emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
            emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)
        else:
            emb = self.noise_level_pos_embedding(noise_levels)
            if external_cond is not None:
                external_emb = self.external_cond_embedding(external_cond, external_cond_mask)
                emb = emb + repeat(external_emb, "b t c -> b (t p) c", p=self.num_patches)

        mask = None
        if self.use_causal_mask:
            mask = self._get_causal_mask(x)

        # Core model
        x = self.dit_base(x, emb, mask)  # (B, T*P, patch_size**2 * C)

        # Unpatchify and restore (B,T,C,H,W)
        x = self.unpatchify(rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches))
        x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)
        return x


if __name__ == "__main__":
    # ---- dummy video batch ----
    B, T = 2, 8  # batch, frames
    C, H, W = 3, 256, 256  # channels, height, width
    x = torch.randn(B, T, C, H, W)  # (B, T, C, H, W)

    # ---- noise levels (frame-wise) ----
    # Your BaseBackbone advertises noise_level_dim=256; give per-frame vectors.
    noise_levels = torch.randn(B, T)  # (B, T, 256)

    # (Optional) external conditioning — set to None if unused
    external_cond = None
    external_cond_mask = None

    # ---- build DiT3D as DiT-B-Mamba ----
    model = DiT3D(
        x_shape=torch.Size([C, H, W]),
        max_tokens=T,  # frames
        external_cond_dim=0,  # not using external cond here
        hidden_size=768,  # DiT-B
        patch_size=16,
        variant="mamba",  # <— enable the Mamba interleave backbone
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        pos_emb_type="learned_1d",
        k=4,  # process 4 frames per local window (non-overlapping)
        global_every=3,  # insert one global Mamba block after every 3 local layers
        global_mode="mamba",
        mamba_d_state=16,
        mamba_expand=2,
        mamba_dropout=0.0,
        mamba_bidirectional=False,
        use_gradient_checkpointing=False,
        use_causal_mask=False,  # set True if you need causal across frames
    )

    model.eval()
    with torch.no_grad():
        y = model(
            x=x,
            noise_levels=noise_levels,  # (B, T, 256) → internally repeated to tokens
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )
    print("output shape:", tuple(y.shape))  # (B, T, C, H, W)  e.g., (2, 8, 3, 256, 256)
#
# from typing import Optional
# import torch
# from torch import nn
# from einops import rearrange, repeat
# from timm.models.vision_transformer import PatchEmbed
# from abc import abstractmethod, ABC
# from typing import Optional
# import torch
# from torch import nn
# from src.models.components.modules.embeddings import (
#    StochasticTimeEmbedding,
#    RandomDropoutCondEmbedding,
# )
# try:
#    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
# except Exception:
#    print('gotchu')
#
# try:
#    #from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
#    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
#    _HAS_FUSED_SCAN = True
# except Exception:
#    selective_scan_fn = None
#    _HAS_FUSED_SCAN = False
#
## from src.models.components.transformer.ditv2 import DiTBase
#
# from src.models.components.backbones.dit.dit_base import DiTBase, DiTBaseSparse
# from src.models.components.backbones.base_backbone import BaseBackbone
#
#
# import math
# from typing import Callable, Optional, Tuple, Literal, Union, List
# import torch
# import torch.nn as nn
#
# from src.models.components.backbones.dit.dit_blocks import DiTBlock, DITFinalLayer
#
# import math
# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DiTMamba(nn.Module):
#    """
#    DiTMamba: windowed DiT with interleaved global processing (full attention or Mamba).
#
#    Core idea
#    ---------
#    - Input tokens are arranged time-major as in DiTBase: sequence over (t * p) where
#      p = num_patches (or 1 for 1D).
#    - We process local chunks of k frames at a time, non-overlapping:
#        window_size_tokens = k * (num_patches or 1)
#      Each local window is run through a stack of DiTBlocks (shared weights by layer index).
#    - After every `global_every` local layers, we run a global layer over ALL tokens at once.
#      That global layer can be either:
#        * "attn": a DiTBlock over the full sequence (full attention)
#        * "mamba": a user-provided module that takes (B, N, C) -> (B, N, C)
#
#    Notes
#    -----
#    - Handles ragged tail windows (when T % k != 0) without padding; the final shorter
#      window is processed as-is.
#    - If you need masking, pass `mask` shaped (B, N) with 1 for valid tokens; it will be
#      sliced per-window automatically.
#    - Positional embeddings are optional; pass any nn.Module that adds in-place to x
#      (signature: pos_emb(x) -> x).
#
#    Args
#    ----
#    num_patches: int or None
#        Number of spatial patches per frame (set None for 1D tokens per frame).
#    max_temporal_length: int
#        Maximum frames supported (used only for sanity/pos-emb config if provided).
#    out_channels: int
#        Output channels per token for the diffusion head (doubles internally if learn_sigma=True).
#    hidden_size: int
#        Token embedding dimension.
#    depth: int
#        Total number of *local* DiT layers. Global layers are interleaved via `global_every`.
#    num_heads: int
#        Attention heads for DiTBlock (used for both local and global attn if selected).
#    mlp_ratio: float
#        MLP width multiplier in DiTBlock.
#    learn_sigma: bool
#        If True, output channels are doubled for mean+sigma style heads.
#    k: int
#        Window size in frames (non-overlapping).
#    global_every: int
#        Insert one global layer after every `global_every` local layers. If 0, no globals.
#    global_mode: {"attn","mamba"}
#        Type of global block. "attn" uses DiTBlock; "mamba" uses `mamba_factory`.
#    mamba_factory: Optional[Callable[[int], nn.Module]]
#        Factory returning an nn.Module that maps (B, N, C) -> (B, N, C) with `hidden_size` features.
#        Required if global_mode == "mamba".
#    pos_emb: Optional[nn.Module]
#        A positional embedding module that can be applied once to (B, N, C) tokens; if provided,
#        it is added at the start.
#    use_gradient_checkpointing: bool
#        Enable PyTorch checkpointing on blocks to save memory.
#    kv_heads: Optional[int]
#        Forwarded into DiTBlock if you use multi-query/grouped KV setups.
#    attn_kwargs: dict
#        Extra kwargs for DiTBlock attention (e.g., flash attention flags, memory settings).
#    """
#
#    def __init__(
#        self,
#        *,
#        num_patches: Optional[int],
#        max_temporal_length: int = 16,
#        out_channels: int = 4,
#        hidden_size: int = 1152,
#        depth: int = 24,
#        num_heads: int = 16,
#        mlp_ratio: float = 4.0,
#        learn_sigma: bool = True,
#        k: int = 4,
#        global_every: int = 4,
#        global_mode: Literal["attn", "mamba"] = "attn",
#        mamba_factory: Optional[Callable[[int], nn.Module]] = None,
#        pos_emb: Optional[nn.Module] = None,
#        use_gradient_checkpointing: bool = False,
#        tokens_per_frame: Optional[int] = None,  # legacy alias for num_patches
#        kv_heads: Optional[int] = None,
#        **kwargs,
#    ):
#        super().__init__()
#        attn_kwargs = {}
#
#        assert k >= 1, "k (window size in frames) must be >= 1"
#        assert depth >= 1, "depth must be >= 1"
#        if global_mode == "mamba":
#            assert mamba_factory is not None, "Provide mamba_factory when global_mode='mamba'"
#
#        self.num_patches = num_patches
#        self.max_temporal_length = max_temporal_length
#        self.tokens_per_frame = num_patches if tokens_per_frame is None else tokens_per_frame
#        self.hidden_size = hidden_size
#        self.learn_sigma = learn_sigma
#        self.out_channels = out_channels * (2 if learn_sigma else 1)
#
#        self.k = k
#        self.global_every = int(global_every)
#        self.global_mode = global_mode
#        self.use_gradient_checkpointing = use_gradient_checkpointing
#
#        # Local DiT layers (applied per-window)
#        self.local_blocks = nn.ModuleList(
#            [
#                DiTBlock(
#                    hidden_size=hidden_size,
#                    num_heads=num_heads,
#                    mlp_ratio=mlp_ratio,
#                    kv_heads=kv_heads,
#                    attn_kwargs=attn_kwargs,
#                )
#                for _ in range(depth)
#            ]
#        )
#
#        # Global blocks interleaved through the stack
#        self.global_blocks = nn.ModuleList()
#        if self.global_every > 0:
#            # Number of global insertions = ceil(depth / global_every)
#            n_globals = math.ceil(depth / self.global_every)
#            for _ in range(n_globals):
#                if global_mode == "attn":
#                    self.global_blocks.append(
#                        DiTBlock(
#                            hidden_size=hidden_size,
#                            num_heads=num_heads,
#                            mlp_ratio=mlp_ratio,
#                            kv_heads=kv_heads,
#                            attn_kwargs=attn_kwargs,
#                        )
#                    )
#                else:  # "mamba"
#                    self.global_blocks.append(mamba_factory(hidden_size))
#
#        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)
#
#        # Optional positional embedding (applied once at the start)
#        self.pos_emb = pos_emb
#
#    def _checkpoint(self, module: nn.Module, *args, **kwargs):
#        if self.use_gradient_checkpointing:
#            from torch.utils.checkpoint import checkpoint
#
#            return checkpoint(module, *args, use_reentrant=False, **kwargs)
#        return module(*args, **kwargs)
#
#    def _iter_windows(
#        self, x: torch.Tensor, c: Optional[torch.Tensor], mask: Optional[torch.Tensor]
#    ):
#        """
#        Generator yielding non-overlapping windows along time (frames).
#        Shapes:
#            x:    (B, N, C)
#            c:    (B, N, C) or None
#            mask: (B, N) or None
#        Yields tuples (xw, cw, mw, start_idx, end_idx) where:
#            xw: (B, W_tokens, C), cw same if provided, mw: (B, W_tokens) if provided
#            start_idx/end_idx in token space [0, N)
#        """
#        B, N, Cdim = x.shape
#        tpf = self.tokens_per_frame
#        assert N % tpf == 0, "Sequence tokens must be divisible by tokens_per_frame (num_patches)."
#        T = N // tpf
#
#        w_frames = self.k
#        frame_idx = 0
#        while frame_idx < T:
#            next_frame = min(frame_idx + w_frames, T)
#            start_tok = frame_idx * tpf
#            end_tok = next_frame * tpf
#            xw = x[:, start_tok:end_tok, :]
#            cw = c[:, start_tok:end_tok, :] if c is not None else None
#            mw = mask[:, start_tok:end_tok] if mask is not None else None
#            yield xw, cw, mw, start_tok, end_tok
#            frame_idx = next_frame
#
#    def _apply_local_block_once(self, block, x, c, mask):
#        B, N, C = x.shape
#        tpf = self.tokens_per_frame
#        T = N // tpf
#        k = self.k
#        W = (T + k - 1) // k
#        pad_frames = W * k - T
#        if pad_frames:
#            pad_tokens = pad_frames * tpf
#            x = F.pad(x, (0,0,0,pad_tokens))
#            if c is not None: c = F.pad(c, (0,0,0,pad_tokens))
#            if mask is not None: mask = F.pad(mask, (0,pad_tokens), value=0)
#
#        xw = x.view(B, W, k*tpf, C).reshape(B*W, k*tpf, C)
#        cw = c.view(B, W, k*tpf, C).reshape(B*W, k*tpf, C) if c is not None else None
#        mw = mask.view(B, W, k*tpf).reshape(B*W, k*tpf) if mask is not None else None
#
#        yw = self._checkpoint(block, xw, cw if cw is not None else xw, mw)
#
#        y = yw.view(B, W, k*tpf, C).reshape(B, W*k*tpf, C)
#        y = y[:, :N, :]
#        return y
#
#    def _apply_local_block_once_scan(
#        self,
#        block: nn.Module,
#        x: torch.Tensor,
#        c: Optional[torch.Tensor],
#        mask: Optional[torch.Tensor],
#    ) -> torch.Tensor:
#        """
#        Apply a single local block to each window independently, then stitch back.
#        """
#        B, N, Cdim = x.shape
#        out_chunks: List[torch.Tensor] = []
#        for xw, cw, mw, start, end in self._iter_windows(x, c, mask):
#            # DiTBlock signature in your base: block(x, c, mask)
#            yw = self._checkpoint(block, xw, cw if cw is not None else xw, mw)
#            out_chunks.append((start, end, yw))
#        # Stitch back
#        y = torch.empty_like(x)
#        for start, end, yw in out_chunks:
#            y[:, start:end, :] = yw
#        return y
#
#    def _apply_global_block_once(
#        self,
#        block: nn.Module,
#        x: torch.Tensor,
#        c: Optional[torch.Tensor],
#        mask: Optional[torch.Tensor],
#    ) -> torch.Tensor:
#        """
#        Apply a single global block to the full token sequence.
#        """
#        if self.global_mode == "attn":
#            return self._checkpoint(block, x, c if c is not None else x, mask)
#        else:
#            # Mamba-style block: assumes signature y = block(x), optionally ignores c/mask.
#            return self._checkpoint(block, x)
#
#    def forward(
#        self,
#        x: torch.Tensor,  # (B, N, C)
#        c: Optional[torch.Tensor] = None,  # (B, N, C) or None
#        mask: Optional[torch.Tensor] = None,  # (B, N) or None
#    ) -> torch.Tensor:
#        """
#        Process sequence with windowed local DiT blocks and interleaved global passes.
#        """
#        B, N, Cdim = x.shape
#        assert (
#            Cdim == self.hidden_size
#        ), f"hidden_size mismatch: got {Cdim}, expected {self.hidden_size}"
#        assert (
#            N % (self.num_patches or 1) == 0
#        ), "N must be divisible by num_patches (or 1 for 1D)."
#
#        # Optional single-shot positional embedding
#        if self.pos_emb is not None:
#            x = self.pos_emb(x)
#
#        # If no external conditioning c is provided, default to x (common in DiT)
#        if c is None:
#            c = x
#
#        global_idx = 0
#        for i, local_block in enumerate(self.local_blocks, start=1):
#            # Local pass on non-overlapping windows of k frames
#            x = self._apply_local_block_once(local_block, x, c, mask)
#
#            # Interleave global after every `global_every` local layers
#            if self.global_every > 0 and (i % self.global_every == 0):
#                gblock = self.global_blocks[global_idx]
#                x = self._apply_global_block_once(gblock, x, c, mask)
#                global_idx += 1
#
#        # If depth is not a multiple of global_every, we may still have one global left by ceil
#        # We intentionally do NOT apply trailing globals here; they are already scheduled above.
#
#        # Final projection head
#        x = self.final_layer(x, c)
#        return x
#
#
#
## ------------------------
## Selective State-Space (per-channel diagonal A, vectors B and C)
## ------------------------
# class SelectiveSSM(nn.Module):
#    """
#    Per-feature diagonal SSM with fused selective scan (Mamba-style).
#        s_t = exp(Δ * A) ⊙ s_{t-1} + B ⊙ x_t
#        y_t = ⟨C, s_t⟩
#    Shapes:
#        x: (B, N, C)
#        A,B,C: (C, D)    [C = d_model, D = d_state]
#        returns: (B, N, C)
#    """
#    def __init__(
#        self,
#        d_model: int,
#        d_state: int = 16,
#        dt_min: float = 1e-3,
#        dt_max: float = 0.1,
#        bidirectional: bool = False,
#    ):
#        super().__init__()
#        self.d_model = d_model
#        self.d_state = d_state
#        self.bidirectional = bidirectional
#
#        # Parameters (same as before)
#        self.a_log = nn.Parameter(torch.randn(d_model, d_state) * -0.5)  # (C,D)
#        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.02)      # (C,D)
#        self.c = nn.Parameter(torch.randn(d_model, d_state) * 0.02)      # (C,D)
#
#        # dt projection (per feature)
#        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
#        self.dt_min, self.dt_max = dt_min, dt_max
#
#        # depthwise conv prefilter stays
#        self.filter = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
#
#    def _discretize(self, dt_vec: torch.Tensor):
#        """
#        dt_vec: (B, N, C) -> dt in [dt_min, dt_max]
#        A: (C,D) with negative diag via -softplus
#        """
#        dt = torch.sigmoid(dt_vec) * (self.dt_max - self.dt_min) + self.dt_min  # (B,N,C)
#        A = -F.softplus(self.a_log)  # (C,D), strictly negative
#        return dt, A
#
#    def _scan_forward_ref(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
#        """
#        Supports both time-invariant dt (B,1,C) and per-timestep dt (B,N,C).
#        x : (B,N,C)
#        dt: (B,1,C) or (B,N,C)
#        A : (C,D)
#        """
#        B, N, C = x.shape
#        D = self.d_state
#        s = torch.zeros(B, C, D, device=x.device, dtype=x.dtype)
#        y = torch.empty(B, N, C, device=x.device, dtype=x.dtype)
#
#        # Build expA with optional time axis at dim=2
#        # dt: (B,1,C)->(B,C,1,1) or (B,N,C)->(B,C,N,1)
#        dt_bc1 = dt.transpose(1, 2).unsqueeze(-1)        # (B,C,1,1) or (B,C,N,1)
#        A_b    = A.view(1, C, 1, D).expand(B, C, 1, D)   # (B,C,1,D)
#        expA   = torch.exp(dt_bc1 * A_b)                 # (B,C,1,D) or (B,C,N,D)
#
#        b_b = self.b.unsqueeze(0).expand(B, -1, -1)      # (B,C,D)
#        c_b = self.c.unsqueeze(0).expand(B, -1, -1)      # (B,C,D)
#
#        time_varying = expA.size(2) != 1                 # True if (B,C,N,D)
#
#        if time_varying:
#            for t in range(N):
#                expA_t = expA[:, :, t, :]                # (B,C,D)
#                xt     = x[:, t, :].unsqueeze(-1)        # (B,C,1)
#                s      = expA_t * s + b_b * xt           # (B,C,D)
#                y[:, t, :] = (c_b * s).sum(dim=-1)       # (B,C)
#        else:
#            expA_s = expA.squeeze(2)                     # (B,C,D)
#            for t in range(N):
#                xt     = x[:, t, :].unsqueeze(-1)        # (B,C,1)
#                s      = expA_s * s + b_b * xt           # (B,C,D)
#                y[:, t, :] = (c_b * s).sum(dim=-1)
#
#        return y
#
#    #def _scan_forward_ref(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
#    #    """Reference slow Python loop (your original), used as fallback."""
#    #    B, N, C = x.shape
#    #    D = self.d_state
#    #    s = torch.zeros(B, C, D, device=x.device, dtype=x.dtype)
#    #    y = torch.empty(B, N, C, device=x.device, dtype=x.dtype)
#
#    #    # broadcast precompute
#    #    dt_bc1 = dt.transpose(1, 2).unsqueeze(-1)  # (B,C,1)
#    #    A_b = A.view(1, C, 1, D).expand(B, C, 1, D)  # (B,C,1,D)
#    #    expA = torch.exp(dt_bc1 * A_b).squeeze(2)  # (B,C,D)
#
#    #    b_b = self.b.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)
#    #    c_b = self.c.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)
#
#    #    for t in range(N):
#    #        xt = x[:, t, :].unsqueeze(-1)  # (B,C,1)
#    #        s = expA * s + b_b * xt       # (B,C,D)
#    #        y[:, t, :] = (c_b * s).sum(dim=-1)
#    #    return y
#
#    def _scan_forward_fused(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
#        """
#        Fused selective scan via mamba-ssm.
#        x  : (B, N, dim)   dim = d_inner
#        dt : (B, N, dim)   (or (B,1,dim) / (B,N,1) — broadcastable)
#        A  : (dim, d_state) or (d_state, dim)
#        self.b, self.c : (dim, d_state) or (d_state, dim)
#        """
#        B, N, dim = x.shape
#        D = self.d_state
#
#        # Ensure dt has time dimension; allow (B,1,dim) and broadcast
#        if dt.shape[-1] == 1 and dim != 1:
#            dt = dt.expand(B, N, dim)
#
#        # Coerce A, B, C to (dim, D)
#        A_ = A
#        Bp = self.b
#        Cp = self.c
#
#        # Defensive transposes to the required (dim, D) layout
#        if A_.shape == (D, dim):
#            A_ = A_.transpose(0, 1).contiguous()
#        elif A_.shape != (dim, D):
#            raise RuntimeError(f"A has shape {tuple(A_.shape)}; expected (dim={dim}, dstate={D}) or (dstate, dim)")
#
#        if Bp.shape == (D, dim):
#            Bp = Bp.transpose(0, 1).contiguous()
#        elif Bp.shape != (dim, D):
#            raise RuntimeError(f"B has shape {tuple(Bp.shape)}; expected (dim={dim}, dstate={D}) or (dstate, dim)")
#
#        if Cp.shape == (D, dim):
#            Cp = Cp.transpose(0, 1).contiguous()
#        elif Cp.shape != (dim, D):
#            raise RuntimeError(f"C has shape {tuple(Cp.shape)}; expected (dim={dim}, dstate={D}) or (dstate, dim)")
#
#        # Contiguity helps the kernel
#        x  = x.contiguous()
#        dt = dt.contiguous()
#        A_ = A_.contiguous()
#        Bp = Bp.contiguous()
#        Cp = Cp.contiguous()
#
#        # Call fused op (no extra D/z needed)
#        y = selective_scan_fn(
#            x,               # u: (B, N, dim)
#            dt,              # delta: (B, N, dim)
#            A_,              # (dim, D)
#            Bp,              # (dim, D)
#            Cp,              # (dim, D)
#            D=None, z=None, delta_bias=None, delta_softplus=False,
#        )
#        return y
#
#    #def _scan_forward_fused(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
#    #    """
#    #    Fused selective scan path via mamba-ssm.
#    #    Expect shapes:
#    #        x  : (B, N, C)       -> treat C as features (D_model)
#    #        dt : (B, N, C)       -> per-timestep delta
#    #        A  : (C, D_state)
#    #        B,C: (C, D_state)
#    #    Returns:
#    #        y: (B, N, C)
#    #    """
#    #    # The interface expects (B, N, D_inner) inputs and per-channel params.
#    #    # We map our symbols to the common API: selective_scan_fn(u, delta, A, B, C, D=None, z=None, ...)
#    #    # Fold the D_state dimension inside the op; selective_scan_fn handles the recurrence internally.
#
#    #    # Reshape params to contiguous memory (defensive)
#    #    A = A.contiguous()                  # (C, D)
#    #    Bp = self.b.contiguous()            # (C, D)
#    #    Cp = self.c.contiguous()            # (C, D)
#
#    #    # selective_scan_fn contracts over the state dim internally.
#    #    # Use keyword args to be robust across versions.
#    #    y = selective_scan_fn(
#    #        x,                      # u: (B, N, C)   (the "input" features)
#    #        dt,                     # delta: (B, N, C)
#    #        A,                      # A: (C, D)
#    #        Bp,                     # B: (C, D)
#    #        Cp,                     # C: (C, D)
#    #        D=None,                 # optional skip/identity, not used here
#    #        z=None,                 # optional extra gate, we already apply gate outside
#    #        delta_bias=None,
#    #        delta_softplus=False,   # we already squashed dt into [dt_min, dt_max]
#    #    )
#    #    return y
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        """
#        x: (B,N,C)
#        """
#        B, N, C = x.shape
#        assert C == self.d_model
#
#        # depthwise conv prefilter (same as before)
#        xf = self.filter(x.transpose(1, 2)).transpose(1, 2)  # (B,N,C)
#
#        dt_vec = self.dt_proj(xf)  # (B,N,C)
#        dt, A = self._discretize(dt_vec)
#
#        if _HAS_FUSED_SCAN:
#            y_f = self._scan_forward_fused(xf, dt, A)
#        else:
#            # fallback to slow ref
#            y_f = self._scan_forward_ref(xf, dt, A)
#
#        if not self.bidirectional:
#            return y_f
#
#        # simple bi-directional: scan on reversed sequence and add
#        xr = torch.flip(xf, dims=[1])
#        if _HAS_FUSED_SCAN:
#            y_r = self._scan_forward_fused(xr, dt, A)
#        else:
#            y_r = self._scan_forward_ref(xr, dt, A)
#        return y_f + torch.flip(y_r, dims=[1])
#
#
##if not _HAS_FUSED_SCAN:
##    try:
##        # Compiles only the slow Python fallback path
##        SelectiveSSM.forward = torch.compile(SelectiveSSM.forward, mode="max-autotune")
##    except Exception:
##        pass
#
#
#
##class SelectiveSSM(nn.Module):
##    """
##    Per-feature diagonal SSM:
##        s_t = exp(Δ * A) ⊙ s_{t-1} + B ⊙ x_t
##        y_t = ⟨C, s_t⟩
##    Shapes:
##        x: (B, N, C)
##        A,B,C: (C, D)    [C = d_model, D = d_state]
##        s: (B, C, D)
##    """
##
##    def __init__(
##        self,
##        d_model: int,
##        d_state: int = 16,
##        dt_min: float = 1e-3,
##        dt_max: float = 0.1,
##        bidirectional: bool = False,
##    ):
##        super().__init__()
##        self.d_model = d_model
##        self.d_state = d_state
##        self.bidirectional = bidirectional
##
##        # Parameters
##        self.a_log = nn.Parameter(torch.randn(d_model, d_state) * -0.5)  # (C,D)
##        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.02)  # (C,D)
##        self.c = nn.Parameter(torch.randn(d_model, d_state) * 0.02)  # (C,D)
##
##        # dt projection (per feature)
##        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
##        self.dt_min, self.dt_max = dt_min, dt_max
##
##        # depthwise conv prefilter
##        self.filter = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
##
##    def _discretize(self, dt_vec: torch.Tensor):
##        """
##        dt_vec: (B, N, C) -> dt: (B, 1, C) in [dt_min, dt_max]
##        A     : (C, D)    with negative diag via -softplus
##        """
##        dt = torch.sigmoid(dt_vec) * (self.dt_max - self.dt_min) + self.dt_min  # (B,N,C)
##        dt = dt.mean(dim=1, keepdim=True)  # (B,1,C) — share step across time for stability
##        A = -F.softplus(self.a_log)  # (C,D), strictly negative
##        return dt, A
##
##    @torch.no_grad()
##    def _init_state(self, B: int, device, dtype):
##        return torch.zeros(B, self.d_model, self.d_state, device=device, dtype=dtype)  # (B,C,D)
##
##    def _scan_forward(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor):
##        """
##        x : (B,N,C)
##        dt: (B,1,C)
##        A : (C,D)
##        returns y: (B,N,C)
##        """
##        B, N, C = x.shape
##        D = self.d_state
##        assert C == self.d_model, f"d_model mismatch: x has C={C}, module has {self.d_model}"
##
##        # Defensive: if A came in as (D,C) (buggy init), transpose it once
##        if A.shape == (D, C):
##            A = A.transpose(0, 1)
##        assert A.shape == (C, D), f"A must be (C,D), got {tuple(A.shape)}"
##
##        s = self._init_state(B, x.device, x.dtype)  # (B,C,D)
##        y = torch.empty(B, N, C, device=x.device, dtype=x.dtype)
##
##        # Broadcasted decay per (B,C,D):
##        # dt: (B,1,C) -> (B,C,1)
##        dt_bc1 = dt.transpose(1, 2).unsqueeze(-1)  # (B,C,1)
##        # A: (C,D) -> (1,C,1,D) -> (B,C,1,D)
##        A_b = A.view(1, C, 1, D).expand(B, C, 1, D)  # (B,C,1,D)
##        expA = torch.exp(dt_bc1 * A_b).squeeze(2)  # (B,C,D)
##
##        b = self.b
##        c = self.c
##        # Defensive: ensure (C,D)
##        if b.shape == (D, C):
##            b = b.transpose(0, 1)
##        if c.shape == (D, C):
##            c = c.transpose(0, 1)
##        b_b = b.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)
##        c_b = c.unsqueeze(0).expand(B, -1, -1)  # (B,C,D)
##
##        for t in range(N):
##            xt = x[:, t, :].unsqueeze(-1)  # (B,C,1)
##            s = expA * s + b_b * xt  # (B,C,D)
##            y[:, t, :] = (c_b * s).sum(dim=-1)  # (B,C) -> (B,C)
##
##        return y
##
##    def forward(self, x: torch.Tensor) -> torch.Tensor:
##        """
##        x: (B,N,C)
##        """
##        B, N, C = x.shape
##        assert C == self.d_model, f"Expected last dim {self.d_model}, got {C}"
##
##        # depthwise conv prefilter
##        xf = self.filter(x.transpose(1, 2)).transpose(1, 2)  # (B,N,C)
##
##        dt_vec = self.dt_proj(xf)  # (B,N,C)
##        dt, A = self._discretize(dt_vec)
##
##        y_f = self._scan_forward(xf, dt, A)
##
##        if not self.bidirectional:
##            return y_f
##
##        xr = torch.flip(xf, dims=[1])
##        y_r = self._scan_forward(xr, dt, A)
##        return y_f + torch.flip(y_r, dims=[1])
#
#
## ------------------------
## Mamba-style block
## ------------------------
# class MambaBlock(nn.Module):
#    def __init__(
#        self,
#        d_model: int,
#        d_state: int = 16,
#        expand: int = 2,
#        dropout: float = 0.0,
#        bidirectional: bool = False,
#    ):
#        super().__init__()
#        self.d_model = d_model
#        d_inner = expand * d_model
#
#        self.norm = RMSNorm(d_model)
#        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)
#        # this line is unchanged, but now SelectiveSSM is fused internally
#        self.ssm = SelectiveSSM(d_inner, d_state=d_state, bidirectional=bidirectional)
#        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        residual = x
#        x = self.norm(x)
#        u_g = self.in_proj(x)                 # (B,N,2*d_inner)
#        u, g = torch.split(u_g, u_g.shape[-1] // 2, dim=-1)
#        v = self.ssm(u)                       # (B,N,d_inner) — now fused
#        z = self.out_proj(torch.sigmoid(g) * v)
#        return residual + self.dropout(z)
#
# class MambaBlock(nn.Module):
#    def __init__(
#        self,
#        d_model: int,
#        d_state: int = 16,
#        expand: int = 2,
#        dropout: float = 0.0,
#        bidirectional: bool = False,
#    ):
#        super().__init__()
#        self.d_model = d_model
#        d_inner = expand * d_model
#
#        self.norm = RMSNorm(d_model)
#        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)
#        # this line is unchanged, but now SelectiveSSM is fused internally
#        self.ssm = SelectiveSSM(d_inner, d_state=d_state, bidirectional=bidirectional)
#        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        residual = x
#        x = self.norm(x)
#        u_g = self.in_proj(x)                 # (B,N,2*d_inner)
#        u, g = torch.split(u_g, u_g.shape[-1] // 2, dim=-1)
#        v = self.ssm(u)                       # (B,N,d_inner) — now fused
#        z = self.out_proj(torch.sigmoid(g) * v)
#        return residual + self.dropout(z)
#
##class MambaBlock(nn.Module):
##    """
##    Mamba-style block with selective SSM and gating.
##    (B,N,C) -> (B,N,C)
##    """
##
##    def __init__(
##        self,
##        d_model: int,
##        d_state: int = 16,
##        expand: int = 2,
##        dropout: float = 0.0,
##        bidirectional: bool = False,
##    ):
##        super().__init__()
##        self.d_model = d_model
##        self.expand = expand
##        d_inner = expand * d_model
##
##        self.norm = RMSNorm(d_model)
##        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)  # split: u (to SSM) and g (gate)
##        self.ssm = SelectiveSSM(d_inner, d_state=d_state, bidirectional=bidirectional)
##        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
##        self.dropout = nn.Dropout(dropout)
##
##    def forward(self, x: torch.Tensor) -> torch.Tensor:
##        """
##        x: (B,N,C)
##        """
##        residual = x
##        x = self.norm(x)
##        u_g = self.in_proj(x)  # (B,N,2*d_inner)
##        u, g = torch.split(u_g, u_g.shape[-1] // 2, dim=-1)
##        g = torch.sigmoid(g)
##        v = self.ssm(u)  # (B,N,d_inner)
##        z = self.out_proj(g * v)  # gated
##        z = self.dropout(z)
##        return residual + z
#
#
## --- add this somewhere central (once) ---
# class RMSNorm(nn.Module):
#    def __init__(self, d: int, eps: float = 1e-6):
#        super().__init__()
#        self.eps = eps
#        self.weight = nn.Parameter(torch.ones(d))
#    def forward(self, x):
#        n = x.pow(2).mean(dim=-1, keepdim=True)
#        return self.weight * x * torch.rsqrt(n + self.eps)
#
## If you already have SinusoidalPositionalEmbedding in your codebase, reuse it.
# class SinusoidalPositionalEmbedding(nn.Module):
#    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
#        super().__init__()
#        N = 1
#        for s in shape: N *= s
#        pos = torch.arange(N).float().unsqueeze(1)  # (N,1)
#        i = torch.arange(embed_dim).float().unsqueeze(0)  # (1,C)
#        div = torch.exp(-math.log(10000.0) * (2 * (i // 2)) / embed_dim)
#        pe = torch.zeros(N, embed_dim)
#        pe[:, 0::2] = torch.sin(pos * div[:, 0::2])
#        pe[:, 1::2] = torch.cos(pos * div[:, 1::2])
#        pe = pe.view(1, N, embed_dim)
#        if learnable:
#            self.pe = nn.Parameter(pe)
#        else:
#            self.register_buffer("pe", pe, persistent=False)
#        self.learnable = learnable
#    def forward(self, x):
#        # x: (B,N,C)
#        return x + self.pe[:, : x.size(1), :]
#
#
# def make_mamba(
#    d_model: int,
#    d_state: int = 16,
#    expand: int = 2,
#    dropout: float = 0.0,
#    bidirectional: bool = False,
# ) -> nn.Module:
#    return MambaBlock(
#        d_model=d_model,
#        d_state=d_state,
#        expand=expand,
#        dropout=dropout,
#        bidirectional=bidirectional,
#    )
#
# class DiT3D(BaseBackbone):
#    def __init__(
#        self,
#        x_shape: torch.Size,
#        max_tokens: int,
#        external_cond_dim: int,
#        hidden_size: int,
#        patch_size: int,
#        variant: str = "full",              # set to "mamba" to enable DiTMamba
#        pos_emb_type: str = "learned_1d",
#        depth: int = 28,
#        num_heads: int = 16,
#        mlp_ratio: float = 4.0,
#        use_gradient_checkpointing: bool = False,
#        use_fourier_noise_embedding: bool = False,
#        external_cond_dropout: float = 0.0,
#        use_causal_mask: bool = False,
#        kv_heads: Optional[int] = None,
#        mem: bool = False,
#        # --- NEW (Mamba interleave) ---
#        k: int = 4,                         # frames per local window
#        global_every: int = 3,              # insert global every N local layers
#        global_mode: str = "mamba",         # {"mamba","attn"}
#        mamba_d_state: int = 16,
#        mamba_expand: int = 2,
#        mamba_dropout: float = 0.0,
#        mamba_bidirectional: bool = False,
#        **kwargs,
#    ):
#        self.hidden_size = hidden_size
#        self.patch_size = patch_size
#        self.use_causal_mask = use_causal_mask
#        self.n_frames = max_tokens
#
#        super().__init__(
#            x_shape=x_shape,
#            max_tokens=max_tokens,
#            external_cond_dim=external_cond_dim,
#            noise_level_emb_dim=hidden_size,
#            use_fourier_noise_embedding=use_fourier_noise_embedding,
#            use_causal_mask=use_causal_mask,
#            external_cond_dropout=external_cond_dropout,
#        )
#
#        # --- CHANGED: handle non-square ---
#        channels, H, W = x_shape[:3]
#        assert (H % self.patch_size == 0 and W % self.patch_size == 0), \
#            f"H={H}, W={W} must be divisible by patch_size={self.patch_size}"
#
#        self.H, self.W = H, W
#        self.Hp, self.Wp = H // self.patch_size, W // self.patch_size  # patches per dim
#        self.num_patches = self.Hp * self.Wp
#        out_channels = self.patch_size**2 * channels
#
#        # Patch embedder
#        self.patch_embedder = PatchEmbed(
#            img_size=(H, W),
#            patch_size=self.patch_size,
#            in_chans=self.in_channels,
#            embed_dim=self.hidden_size,
#            bias=True,
#        )
#
#        # --- NEW: choose backbone ---
#        # If you want to activate Mamba, set variant="mamba"
#        if variant == "mamba":
#            # Optional simple 1D positional embedding over N = T * P tokens
#            pos_emb = None
#            if pos_emb_type in {"learned_1d", "sinusoidal_1d"}:
#                pos_emb = SinusoidalPositionalEmbedding(
#                    embed_dim=hidden_size,
#                    shape=(max_tokens * self.num_patches,),
#                    learnable=(pos_emb_type == "learned_1d"),
#                )
#
#            # Build DiTMamba
#            self.dit_base = DiTMamba(
#                num_patches=self.num_patches,
#                max_temporal_length=max_tokens,
#                out_channels=out_channels,
#                hidden_size=hidden_size,
#                depth=depth,
#                num_heads=num_heads,
#                mlp_ratio=mlp_ratio,
#                learn_sigma=False,                        # we output pixel patches directly
#                k=k,
#                global_every=global_every,
#                global_mode="mamba" if global_mode == "mamba" else "attn",
#                mamba_factory=(lambda d: make_mamba(
#                    d_model=d,
#                    d_state=mamba_d_state,
#                    expand=mamba_expand,
#                    dropout=mamba_dropout,
#                    bidirectional=mamba_bidirectional,
#                )) if global_mode == "mamba" else None,
#                pos_emb=pos_emb,
#                use_gradient_checkpointing=use_gradient_checkpointing,
#                kv_heads=kv_heads,
#                attn_kwargs={"mem": mem},
#            )
#        else:
#            # Fallback to your original DiTBase/DiTBaseSparse
#            base_cls = DiTBaseSparse if variant == "sparse" else DiTBase
#            self.dit_base = base_cls(
#                num_patches=self.num_patches,
#                max_temporal_length=max_tokens,
#                out_channels=out_channels,
#                variant=variant,
#                pos_emb_type=pos_emb_type,
#                hidden_size=hidden_size,
#                depth=depth,
#                num_heads=num_heads,
#                mlp_ratio=mlp_ratio,
#                learn_sigma=False,
#                use_gradient_checkpointing=use_gradient_checkpointing,
#                kv_heads=kv_heads,
#                spatial_grid_shape=(self.Hp, self.Wp),
#                attn_kwargs={"mem": mem},
#                **kwargs,
#            )
#
#        self.initialize_weights()
#
#    @property
#    def in_channels(self) -> int:
#        return self.x_shape[0]
#
#    @property
#    def noise_level_dim(self) -> int:
#        return 256
#
#    @property
#    def noise_level_emb_dim(self) -> int:
#        return self.hidden_size
#
#    @property
#    def external_cond_emb_dim(self) -> int:
#        return self.hidden_size if self.external_cond_dim else 0
#
#    @staticmethod
#    def _patch_embedder_init(embedder: PatchEmbed) -> None:
#        w = embedder.proj.weight.data
#        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#        nn.init.zeros_(embedder.proj.bias)
#
#    def initialize_weights(self) -> None:
#        self._patch_embedder_init(self.patch_embedder)
#        def _mlp_init(module: nn.Module) -> None:
#            if isinstance(module, nn.Linear):
#                nn.init.normal_(module.weight, std=0.02)
#                if module.bias is not None:
#                    nn.init.zeros_(module.bias)
#        self.noise_level_pos_embedding.apply(_mlp_init)
#        if self.external_cond_embedding is not None:
#            self.external_cond_embedding.apply(_mlp_init)
#
#    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
#        return rearrange(
#            x, "b (h w) (p q c) -> b (h p) (w q) c",
#            h=self.Hp, w=self.Wp, p=self.patch_size, q=self.patch_size,
#        )
#
#    def _get_causal_mask(self, q: torch.Tensor) -> torch.Tensor:
#        N = q.size(-2)
#        assert self.n_frames >= 1
#        assert N % self.n_frames == 0, f"N={N} not divisible by n_frames={self.n_frames}"
#        tpf = N // self.n_frames
#        device = q.device
#        pos = torch.arange(N, device=device)
#        frame_id = pos // tpf
#        allowed = frame_id.unsqueeze(1) >= frame_id.unsqueeze(0)  # (N,N)
#        attn_mask_bool = allowed
#        return attn_mask_bool
#
#    def forward(
#        self,
#        x: torch.Tensor,
#        noise_levels: torch.Tensor,
#        external_cond: Optional[torch.Tensor] = None,
#        external_cond_mask: Optional[torch.Tensor] = None,
#        **kwargs,
#    ) -> torch.Tensor:
#        input_batch_size = x.shape[0]
#        noise_levels = noise_levels.to(x.dtype)
#
#        # (B,T,C,H,W) -> patchify -> (B, T*P, C)
#        x = rearrange(x, "b t c h w -> (b t) c h w")
#        x = self.patch_embedder(x)
#        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)
#
#        _, num_tokens, _ = x.shape
#        if noise_levels.shape[1] != num_tokens:
#            emb = self.noise_level_pos_embedding(noise_levels)
#            if external_cond is not None:
#                emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
#            emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)
#        else:
#            emb = self.noise_level_pos_embedding(noise_levels)
#            if external_cond is not None:
#                external_emb = self.external_cond_embedding(external_cond, external_cond_mask)
#                emb = emb + repeat(external_emb, "b t c -> b (t p) c", p=self.num_patches)
#
#        mask = None
#        if self.use_causal_mask:
#            mask = self._get_causal_mask(x)
#
#        # Core model
#        x = self.dit_base(x, emb, mask)  # (B, T*P, patch_size**2 * C)
#
#        # Unpatchify and restore (B,T,C,H,W)
#        x = self.unpatchify(rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches))
#        x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)
#        return x
#
# if __name__ == "__main__":
#    # ---- dummy video batch ----
#    B, T = 2, 8                   # batch, frames
#    C, H, W = 3, 256, 256         # channels, height, width
#    x = torch.randn(B, T, C, H, W)  # (B, T, C, H, W)
#
#    # ---- noise levels (frame-wise) ----
#    # Your BaseBackbone advertises noise_level_dim=256; give per-frame vectors.
#    noise_levels = torch.randn(B, T)  # (B, T, 256)
#
#    # (Optional) external conditioning — set to None if unused
#    external_cond = None
#    external_cond_mask = None
#
#    # ---- build DiT3D as DiT-B-Mamba ----
#    model = DiT3D(
#        x_shape=torch.Size([C, H, W]),
#        max_tokens=T,                 # frames
#        external_cond_dim=0,          # not using external cond here
#        hidden_size=768,              # DiT-B
#        patch_size=16,
#        variant="mamba",              # <— enable the Mamba interleave backbone
#        depth=12, num_heads=12, mlp_ratio=4.0,
#        pos_emb_type="learned_1d",
#        k=4,                          # process 4 frames per local window (non-overlapping)
#        global_every=3,               # insert one global Mamba block after every 3 local layers
#        global_mode="mamba",
#        mamba_d_state=16, mamba_expand=2, mamba_dropout=0.0, mamba_bidirectional=False,
#        use_gradient_checkpointing=False,
#        use_causal_mask=False,        # set True if you need causal across frames
#    )
#
#    model.eval()
#    with torch.no_grad():
#        y = model(
#            x=x,
#            noise_levels=noise_levels,          # (B, T, 256) → internally repeated to tokens
#            external_cond=external_cond,
#            external_cond_mask=external_cond_mask,
#        )
#    print("output shape:", tuple(y.shape))       # (B, T, C, H, W)  e.g., (2, 8, 3, 256, 256)
