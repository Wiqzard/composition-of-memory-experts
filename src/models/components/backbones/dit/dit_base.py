"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
Extended to support:
- Temporal sequence modeling
- 1D input additionally to 2D spatial input
- Token-wise conditioning
"""

import math
import torch
import torch.nn.functional as F
import math
from typing import Literal, Optional, Tuple, Callable, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from src.models.components.modules.embeddings import RotaryEmbedding3D

from src.models.components.backbones.dit.dit_blocks import (
    DiTBlock,
    DITFinalLayer,
)

# from src.models.components.transformer.ditv2_blocks import (
#    DiTBlock,
#    DITFinalLayer,
# )


Variant = Literal["full", "factorized_encoder", "factorized_attention"]
PosEmb = Literal[
    "learned_1d", "sinusoidal_1d", "sinusoidal_3d", "sinusoidal_factorized", "rope_3d"
]


def rearrange_contiguous_many(
    tensors: Tuple[torch.Tensor, ...], *args, **kwargs
) -> Tuple[torch.Tensor, ...]:
    return tuple(rearrange(t, *args, **kwargs).contiguous() for t in tensors)


class DiTBase(nn.Module):
    """
    A DiT base model.
    """

    def __init__(
        self,
        num_patches: Optional[int] = None,
        max_temporal_length: int = 16,
        out_channels: int = 4,
        variant: Variant = "full",
        pos_emb_type: PosEmb = "learned_1d",
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_gradient_checkpointing: bool = False,
        kv_heads: Optional[int] = None,
        spatial_grid_shape: Optional[Tuple[int, int]] = None,
        attn_kwargs: Optional[dict] = {},
    ):
        """
        Args:
            num_patches: Number of patches in the image, None for 1D inputs.
            max_temporal_length: Maximum length of the temporal sequence.
            variant: Variant of the DiT model to use.
                - "full": process all tokens at once
                - "factorized_encoder": alternate between spatial transformer blocks and temporal transformer blocks
                - "factorized_attention": decompose the multi-head attention in the transformer block, compute spatial self-attention and then temporal self-attention
            pos_emb_type: Type of positional embedding to use.
                - "learned_1d": learned 1D positional embeddings
                - "sinusoidal_1d": sinusoidal 1D positional embeddings
                - "sinusoidal_3d": sinusoidal 3D positional embeddings
                - "sinusoidal_factorized": sinusoidal 2D positional embeddings for spatial and 1D for temporal
                - "rope_3d": rope 3D positional embeddings
        """
        super().__init__()
        self._check_args(num_patches, variant, pos_emb_type)
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.num_patches = num_patches
        self.max_temporal_length = max_temporal_length
        self.max_tokens = self.max_temporal_length * (num_patches or 1)
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.variant = variant
        self.pos_emb_type = pos_emb_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.spatial_grid_shape = spatial_grid_shape

        match self.pos_emb_type:
            case "learned_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                    learnable=True,
                )
            case "sinusoidal_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                )
            case "sinusoidal_3d":
                H, W = self.spatial_grid_hw
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length, H, W),
                )

            case "sinusoidal_factorized":
                H, W = self.spatial_grid_hw
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(H, W),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )

            case "rope_3d":
                H, W = self.spatial_grid_hw
                rope = RotaryEmbedding3D(
                    dim=self.hidden_size // num_heads,
                    sizes=(self.max_temporal_length, H, W),
                )

        mem = attn_kwargs.get("mem", False)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(mlp_ratio if self.variant != "factorized_attention" else None),
                    rope=rope if self.pos_emb_type == "rope_3d" else None,
                    kv_heads=kv_heads,
                    attn_kwargs=attn_kwargs if i in (0, 4) and mem else {},
                )
                for i in range(depth)
            ]
        )
        print(2000 * mem)
        self.temporal_blocks = (
            nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if self.is_factorized
            else None
        )

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

    @property
    def is_factorized(self) -> bool:
        return self.variant in {"factorized_encoder", "factorized_attention"}

    @property
    def is_pos_emb_absolute_once(self) -> bool:
        return self.pos_emb_type in {"learned_1d", "sinusoidal_1d", "sinusoidal_3d"}

    @property
    def is_pos_emb_absolute_factorized(self) -> bool:
        return self.pos_emb_type == "sinusoidal_factorized"

    @property
    def spatial_grid_hw(self) -> Optional[Tuple[int, int]]:
        if self.num_patches is None:
            return None
        if self.spatial_grid_shape is not None:
            H, W = self.spatial_grid_shape
            assert (
                H * W == self.num_patches
            ), f"spatial_grid_shape={H,W} must multiply to num_patches={self.num_patches}"
            return H, W
        # Back-compat: allow square as a default, but demand shape if non-square
        S = int(self.num_patches**0.5)
        assert S * S == self.num_patches, (
            "num_patches is not a perfect square. "
            "Pass spatial_grid_shape=(H, W) to use non-square inputs."
        )
        return S, S

    # @property
    # def spatial_grid_size(self) -> Optional[int]:
    #    if self.num_patches is None:
    #        return None
    #    grid_size = int(self.num_patches**0.5)
    #    assert (
    #        grid_size * grid_size == self.num_patches
    #    ), "num_patches must be a square number"
    #    return grid_size

    @staticmethod
    def _check_args(num_patches: Optional[int], variant: Variant, pos_emb_type: PosEmb):
        # if variant not in {"full", "factorized_encoder", "factorized_attention"}:
        #    raise ValueError(f"Unknown variant {variant}")
        if pos_emb_type not in {
            "learned_1d",
            "sinusoidal_1d",
            "sinusoidal_3d",
            "sinusoidal_factorized",
            "rope_3d",
        }:
            raise ValueError(f"Unknown positional embedding type {pos_emb_type}")
        if num_patches is None:
            assert variant == "full", "For 1D inputs, factorized variants are not supported"
            assert pos_emb_type in {
                "learned_1d",
                "sinusoidal_1d",
            }, "For 1D inputs, only 1D positional embeddings are supported"

        # if pos_emb_type == "rope_3d":
        #    assert variant == "full", "Rope3D is only supported with full variant"

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        x_img = None
        if x.size(1) > self.max_tokens:
            if not self.training or self.num_patches is None:
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
                )

            else:  # image-video joint training
                video_end = self.max_temporal_length * self.num_patches
                x, x_img, c, c_img = (
                    x[:, :video_end],
                    x[:, video_end:],
                    c[:, :video_end],
                    c[:, video_end:],
                )
                x_img, c_img = rearrange_contiguous_many(
                    (x_img, c_img), "b (t p) c -> (b t) p c", p=self.num_patches
                )  # as if they are sequences of length 1

        seq_batch_size = x.size(0)
        img_batch_size = x_img.size(0) if x_img is not None else None

        seq_states = {"x": x, "c": c, "batch_size": seq_batch_size}
        img_states = (
            {"x": x_img, "c": c_img, "batch_size": img_batch_size} if x_img is not None else None
        )

        def execute_in_parallel(
            fn: Callable[
                [torch.Tensor, torch.Tensor, int],
                Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            ],
        ):
            """execute a function in parallel on the sequence and image tensors"""
            seq_result = fn(seq_states["x"], seq_states["c"], seq_states["batch_size"])
            if isinstance(seq_result, tuple):
                seq_states["x"], seq_states["c"] = seq_result
            else:
                seq_states["x"] = seq_result
            if img_states is not None:
                img_result = fn(img_states["x"], img_states["c"], img_states["batch_size"])
                if isinstance(img_result, tuple):
                    img_states["x"], img_states["c"] = img_result
                else:
                    img_states["x"] = img_result

        if self.is_pos_emb_absolute_once:
            execute_in_parallel(lambda x, c, batch_size: self.pos_emb(x))
        if self.is_pos_emb_absolute_factorized and not self.is_factorized:

            def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.spatial_pos_emb(x)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                x = self.temporal_pos_emb(x)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            execute_in_parallel(add_pos_emb)

        if self.is_factorized:
            execute_in_parallel(
                lambda x, c, batch_size: rearrange_contiguous_many(
                    (x, c), "b (t p) c -> (b t) p c", p=self.num_patches
                )
            )
            if self.is_pos_emb_absolute_factorized:
                execute_in_parallel(lambda x, c, batch_size: self.spatial_pos_emb(x))

        for i, (block, temporal_block) in enumerate(
            zip(self.blocks, self.temporal_blocks or [None for _ in range(self.depth)])
        ):
            execute_in_parallel(lambda x, c, batch_size: self.checkpoint(block, x, c, mask))

            if self.is_factorized:
                execute_in_parallel(
                    lambda x, c, batch_size: rearrange_contiguous_many(
                        (x, c), "(b t) p c -> (b p) t c", b=batch_size
                    )
                )
                if i == 0 and self.pos_emb_type == "sinusoidal_factorized":
                    execute_in_parallel(lambda x, c, batch_size: self.temporal_pos_emb(x))
                execute_in_parallel(lambda x, c, batch_size: self.checkpoint(temporal_block, x, c))
                execute_in_parallel(
                    lambda x, c, batch_size: rearrange_contiguous_many(
                        (x, c), "(b p) t c -> (b t) p c", b=batch_size
                    )
                )
        if self.is_factorized:
            execute_in_parallel(
                lambda x, c, batch_size: rearrange_contiguous_many(
                    (x, c), "(b t) p c -> b (t p) c", b=batch_size
                )
            )

        execute_in_parallel(lambda x, c, batch_size: self.final_layer(x, c))

        x = seq_states["x"]
        x_img = img_states["x"] if img_states is not None else None
        if x_img is not None:
            x_img = rearrange(x_img, "(b t) p c -> b (t p) c", b=seq_batch_size)
            x = torch.cat([x, x_img], dim=1)
        return x


def make_sliding_helpers(T, P, w, overlap, device, dtype):
    stride = w - overlap
    # number of windows to cover T (ceil), pad if needed
    n_win = math.ceil((max(T - w, 0)) / stride) + 1
    T_needed = (n_win - 1) * stride + w
    pad_T = max(0, T_needed - T)

    # time indices for each (window, offset)
    start = torch.arange(n_win, device=device) * stride  # (n_win,)
    offs = torch.arange(w, device=device)  # (w,)
    idx = start[:, None] + offs[None, :]  # (n_win, w) in [0, T_needed-1]
    idx_flat = idx.reshape(-1)  # (n_win*w,)
    return stride, n_win, T_needed, pad_T, idx, idx_flat


def to_local_overlapping(x, w, overlap, num_patches):
    # x: (B, T*P, C)  ->  (B*n_win, w*P, C)
    B, TP, C = x.shape
    P = num_patches
    assert TP % P == 0, "T*P must be divisible by P"
    T = TP // P

    stride, n_win, T_needed, pad_T, idx, _ = make_sliding_helpers(
        T, P, w, overlap, x.device, x.dtype
    )

    x4 = rearrange(x, "b (t p) c -> b t p c", p=P)  # (B, T, P, C)
    if pad_T:
        pad = torch.zeros(B, pad_T, P, C, device=x.device, dtype=x.dtype)
        x4 = torch.cat([x4, pad], dim=1)  # (B, T_needed, P, C)

    # unfold along time: (B, T_needed, P, C) -> (B, n_win, w, P, C)
    x_unf = x4.unfold(dimension=1, size=w, step=stride)  # (B, n_win, w, P, C)
    # flatten windows into batch, and (w,P) into tokens
    x_loc = rearrange(x_unf, "b n w p c -> (b n) (w p) c")
    return x_loc, (T, T_needed, pad_T, n_win, w, P, stride)


def to_global_overlapping(x_loc, meta):
    # x_loc: (B*n_win, w*P, C)  ->  (B, T*P, C)
    T, T_needed, pad_T, n_win, w, P, stride = meta
    Bn, WP, C = x_loc.shape
    assert WP == w * P
    B = Bn // n_win

    # reshape back to (B, n_win, w, P, C)
    x_unf = rearrange(x_loc, "(b n) (w p) c -> b n w p c", b=B, n=n_win, w=w, p=P)

    # overlap-add with averaging
    out = torch.zeros(B, T_needed, P, C, device=x_loc.device, dtype=x_loc.dtype)
    cnt = torch.zeros(B, T_needed, P, 1, device=x_loc.device, dtype=x_loc.dtype)

    # time indices per (window, offset)
    start = torch.arange(n_win, device=x_loc.device) * stride  # (n_win,)
    offs = torch.arange(w, device=x_loc.device)  # (w,)
    idx = (start[:, None] + offs[None, :]).reshape(-1)  # (n_win*w,)

    # flatten (n_win, w) → (n_win*w), then scatter-add into time dim
    src = x_unf.reshape(B, n_win * w, P, C)  # (B, n_win*w, P, C)
    idx_b = idx.view(1, -1, 1, 1).expand(B, -1, P, C)  # (B, n_win*w, P, C)

    out.scatter_add_(dim=1, index=idx_b, src=src)
    cnt.scatter_add_(dim=1, index=idx_b[..., :1], src=torch.ones_like(src[..., :1]))

    out = out / cnt.clamp_min(1)

    # drop padding, reshape back to (B, T*P, C)
    if T_needed != T:
        out = out[:, :T]
    x_global = rearrange(out, "b t p c -> b (t p) c")
    return x_global


def _sliding_window_block_batched(x, c, block, window_size: int, stride: int):
    """
    Sliding-window wrapper that batches all windows through a full-attention block once.

    x, c: (B, N, C)
    block: callable (x_chunk, c_chunk) -> (B, chunk_len, C)
    """
    B, N, C = x.shape
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be > 0")
    if window_size >= N:
        return block(x, c)

    # Number of windows to cover the sequence (last window is forced to include tail)
    num_wins = 1 + max(0, math.ceil((N - window_size) / stride))
    last_start = (num_wins - 1) * stride
    need = last_start + window_size
    pad_len = max(0, need - N)

    if pad_len:
        x = F.pad(x, (0, 0, 0, pad_len))  # pad along N
        c = F.pad(c, (0, 0, 0, pad_len))
    N_pad = x.size(1)

    # Build batched sliding windows with unfold (on the time/token axis)
    # (B, num_wins, W, C)
    x_wins = x.unfold(dimension=1, size=window_size, step=stride)  # (B, num_wins, W, C)
    c_wins = c.unfold(dimension=1, size=window_size, step=stride)

    # Collapse batch & windows -> run block once
    Bw = B * num_wins
    x_in = x_wins.contiguous().view(Bw, window_size, C)
    c_in = c_wins.contiguous().view(Bw, window_size, C)

    #y_wins = block(x_in, c_in)  # (B*num_wins, W, C)
    #y_wins = y_wins.view(B, num_wins, window_size, C)
    y_wins = block(x_in, c_in)                 # (B*num_wins, W, C)
    y_wins = y_wins.to(x.dtype)                # ensure same dtype as accumulators
    y_wins = y_wins.view(B, num_wins, window_size, C)
    # Triangular weights for smooth stitching
    w = torch.linspace(0, 1, window_size, device=x.device, dtype=x.dtype)
    w = torch.minimum(w, 1 - w) * 2.0
    w = w.clamp_min(1e-6).view(1, 1, window_size, 1)  # (1,1,W,1) -> broadcast

    # Prepare overlap-add via scatter_add_
    acc = torch.zeros(B, N_pad, C, device=x.device, dtype=x.dtype)
    wsum = torch.zeros(B, N_pad, 1, device=x.device, dtype=x.dtype)

    # Indices of each window along the N axis: shape (num_wins, W)
    starts = torch.arange(num_wins, device=x.device) * stride
    offsets = torch.arange(window_size, device=x.device)
    idx = (starts[:, None] + offsets[None, :]).view(1, num_wins, window_size)  # (1, num_wins, W)
    idx = idx.expand(B, -1, -1)  # (B, num_wins, W)

    # Flatten for scatter
    idx_flat = idx.reshape(B, -1)  # (B, num_wins*W)

    # Weighted outputs: (B, num_wins, W, C) -> (B, num_wins*W, C)
    yw_flat = (y_wins * w).reshape(B, -1, C)
    w_flat = w.expand(B, num_wins, window_size, 1).reshape(B, -1, 1)

    # Scatter-add along N dimension
    acc.scatter_add_(dim=1, index=idx_flat.unsqueeze(-1).expand_as(yw_flat), src=yw_flat)
    wsum.scatter_add_(dim=1, index=idx_flat.unsqueeze(-1).expand_as(w_flat), src=w_flat)

    y = acc / wsum.clamp_min(1e-6)

    # Crop to original length
    if pad_len:
        y = y[:, :N, :]

    return y


class DiTBaseSparse(DiTBase):
    def __init__(
        self,
        *args,
        global_attention_layer_idx: Optional[Tuple[int]] = [],
        local_attention_window_size: int = 16,
        local_attention_overlap: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_attention_layer_idx = global_attention_layer_idx or ()
        self.local_attention_window_size = local_attention_window_size
        self.local_attention_overlap = local_attention_overlap

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        seq_batch_size = x.size(0)

        if self.is_pos_emb_absolute_factorized and not self.is_factorized:

            def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.spatial_pos_emb(x)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                x = self.temporal_pos_emb(x)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            x = add_pos_emb(x, None, seq_batch_size)

        for i, block in enumerate(self.blocks):
            if i in self.global_attention_layer_idx:
                x = block(x, c)
            else:
                x = _sliding_window_block_batched(
                    x,
                    c,
                    block,
                    self.local_attention_window_size * self.num_patches,
                    self.local_attention_overlap * self.num_patches,
                )
                # x = block(x, c)

        #        overlap = 5 # frames
        #        to_local  = lambda x: rearrange(x, "b ((t_local t_group) p) c -> (b t_group) (t_local p) c",
        #                                        p=self.num_patches, t_local=self.local_attention_window_size, t_group=self.max_temporal_length // self.local_attention_window_size)
        #        to_global = lambda x: rearrange(x, "(b t_group) (t_local p) c -> b ((t_local t_group) p) c",
        #                                        p=self.num_patches, t_local=self.local_attention_window_size, t_group=self.max_temporal_length // self.local_attention_window_size)
        #
        #        x = to_local(x)
        #        c = to_local(c)
        #
        #        for i, block in enumerate(self.blocks):
        #            if i in self.global_attention_layer_idx:
        #                x = to_global(x)
        #                c = to_global(c)
        #                x = block(x, c)
        #                x = to_local(x)
        #                c = to_local(c)
        #            else:
        #                x = block(x, c)
        #

        x = self.final_layer(x, c)
        return x


class DiTBaseCausal(nn.Module):
    """
    A DiT base model.
    """

    def __init__(
        self,
        num_patches: Optional[int] = None,
        max_temporal_length: int = 16,
        out_channels: int = 4,
        variant: Variant = "full",
        pos_emb_type: PosEmb = "learned_1d",
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_gradient_checkpointing: bool = False,
        kv_heads: Optional[int] = None,
        spatial_grid_shape: Optional[Tuple[int, int]] = None,
        num_denoiser_layers: Optional[int] = 0,
    ):
        """
        Args:
            num_patches: Number of patches in the image, None for 1D inputs.
            max_temporal_length: Maximum length of the temporal sequence.
            variant: Variant of the DiT model to use.
                - "full": process all tokens at once
                - "factorized_encoder": alternate between spatial transformer blocks and temporal transformer blocks
                - "factorized_attention": decompose the multi-head attention in the transformer block, compute spatial self-attention and then temporal self-attention
            pos_emb_type: Type of positional embedding to use.
                - "learned_1d": learned 1D positional embeddings
                - "sinusoidal_1d": sinusoidal 1D positional embeddings
                - "sinusoidal_3d": sinusoidal 3D positional embeddings
                - "sinusoidal_factorized": sinusoidal 2D positional embeddings for spatial and 1D for temporal
                - "rope_3d": rope 3D positional embeddings
        """
        super().__init__()
        self._check_args(num_patches, variant, pos_emb_type)
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.num_patches = num_patches
        self.max_temporal_length = max_temporal_length
        self.max_tokens = self.max_temporal_length * (num_patches or 1)
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.variant = variant
        self.pos_emb_type = pos_emb_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.spatial_grid_shape = spatial_grid_shape
        self.num_denoiser_layers = num_denoiser_layers

        match self.pos_emb_type:
            case "learned_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                    learnable=True,
                )
            case "sinusoidal_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                )
            case "sinusoidal_3d":
                H, W = self.spatial_grid_hw
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length, H, W),
                )

            case "sinusoidal_factorized":
                H, W = self.spatial_grid_hw
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(H, W),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )

            case "rope_3d":
                H, W = self.spatial_grid_hw
                rope = RotaryEmbedding3D(
                    dim=self.hidden_size // num_heads,
                    sizes=(self.max_temporal_length, H, W),
                )

        attn_kwargs = {}
        depth = self.depth + self.num_denoiser_layers

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(mlp_ratio if self.variant != "factorized_attention" else None),
                    rope=rope if self.pos_emb_type == "rope_3d" else None,
                    kv_heads=kv_heads,
                    attn_kwargs=attn_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.temporal_blocks = (
            nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if self.is_factorized
            else None
        )

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

    @property
    def is_factorized(self) -> bool:
        return self.variant in {"factorized_encoder", "factorized_attention"}

    @property
    def is_pos_emb_absolute_once(self) -> bool:
        return self.pos_emb_type in {"learned_1d", "sinusoidal_1d", "sinusoidal_3d"}

    @property
    def is_pos_emb_absolute_factorized(self) -> bool:
        return self.pos_emb_type == "sinusoidal_factorized"

    @property
    def spatial_grid_hw(self) -> Optional[Tuple[int, int]]:
        if self.num_patches is None:
            return None
        if self.spatial_grid_shape is not None:
            H, W = self.spatial_grid_shape
            assert (
                H * W == self.num_patches
            ), f"spatial_grid_shape={H,W} must multiply to num_patches={self.num_patches}"
            return H, W
        # Back-compat: allow square as a default, but demand shape if non-square
        S = int(self.num_patches**0.5)
        assert S * S == self.num_patches, (
            "num_patches is not a perfect square. "
            "Pass spatial_grid_shape=(H, W) to use non-square inputs."
        )
        return S, S

    # @property
    # def spatial_grid_size(self) -> Optional[int]:
    #    if self.num_patches is None:
    #        return None
    #    grid_size = int(self.num_patches**0.5)
    #    assert (
    #        grid_size * grid_size == self.num_patches
    #    ), "num_patches must be a square number"
    #    return grid_size

    @staticmethod
    def _check_args(num_patches: Optional[int], variant: Variant, pos_emb_type: PosEmb):
        if variant not in {"full", "factorized_encoder", "factorized_attention"}:
            raise ValueError(f"Unknown variant {variant}")
        if pos_emb_type not in {
            "learned_1d",
            "sinusoidal_1d",
            "sinusoidal_3d",
            "sinusoidal_factorized",
            "rope_3d",
        }:
            raise ValueError(f"Unknown positional embedding type {pos_emb_type}")
        if num_patches is None:
            assert variant == "full", "For 1D inputs, factorized variants are not supported"
            assert pos_emb_type in {
                "learned_1d",
                "sinusoidal_1d",
            }, "For 1D inputs, only 1D positional embeddings are supported"

        if pos_emb_type == "rope_3d":
            assert variant == "full", "Rope3D is only supported with full variant"

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self,
        x_clean: torch.Tensor,
        x: torch.Tensor,
        cond_emb: torch.Tensor,
        ext_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        x_img = None
        if x.size(1) > self.max_tokens:
            if not self.training or self.num_patches is None:
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
                )

            else:  # image-video joint training
                video_end = self.max_temporal_length * self.num_patches
                x, x_img, c, c_img = (
                    x[:, :video_end],
                    x[:, video_end:],
                    c[:, :video_end],
                    c[:, video_end:],
                )
                x_img, c_img = rearrange_contiguous_many(
                    (x_img, c_img), "b (t p) c -> (b t) p c", p=self.num_patches
                )  # as if they are sequences of length 1

        seq_batch_size = x.size(0)
        img_batch_size = x_img.size(0) if x_img is not None else None

        seq_states = {"x": x_clean, "c": ext_emb, "batch_size": seq_batch_size}

        img_states = (
            {"x": x_img, "c": c_img, "batch_size": img_batch_size} if x_img is not None else None
        )

        def execute_in_parallel(
            fn: Callable[
                [torch.Tensor, torch.Tensor, int],
                Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            ],
        ):
            """execute a function in parallel on the sequence and image tensors"""
            seq_result = fn(seq_states["x"], seq_states["c"], seq_states["batch_size"])
            if isinstance(seq_result, tuple):
                seq_states["x"], seq_states["c"] = seq_result
            else:
                seq_states["x"] = seq_result
            if img_states is not None:
                img_result = fn(img_states["x"], img_states["c"], img_states["batch_size"])
                if isinstance(img_result, tuple):
                    img_states["x"], img_states["c"] = img_result
                else:
                    img_states["x"] = img_result

        if self.is_pos_emb_absolute_once:
            execute_in_parallel(lambda x, c, batch_size: self.pos_emb(x))
        if self.is_pos_emb_absolute_factorized and not self.is_factorized:

            def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.spatial_pos_emb(x)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                x = self.temporal_pos_emb(x)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            execute_in_parallel(add_pos_emb)

        if self.is_factorized:
            execute_in_parallel(
                lambda x, c, batch_size: rearrange_contiguous_many(
                    (x, c), "b (t p) c -> (b t) p c", p=self.num_patches
                )
            )
            if self.is_pos_emb_absolute_factorized:
                execute_in_parallel(lambda x, c, batch_size: self.spatial_pos_emb(x))

        z = x_clean
        for i, (block, temporal_block) in enumerate(
            zip(
                self.blocks,
                self.temporal_blocks
                or [None for _ in range(self.depth + self.num_denoiser_layers)],
            )
        ):
            if i < self.depth:
                z = block(z, ext_emb, mask)

            elif i == self.depth:
                x = rearrange(x, "b (t p) c -> b t p c", p=self.num_patches)
                z = rearrange(z, "b (t p) c -> b t p c", p=self.num_patches)
                x_clean = rearrange(x_clean, "b (t p) c -> b t p c", p=self.num_patches)

                # pairs = torch.stack([z, x_clean.detach(), x], dim=2)  # (B, T, 2, P, C)
                pairs = torch.stack([z, x], dim=2)  # (B, T, 2, P, C)
                x = rearrange(pairs, "b t n p c -> (b t) (n p) c")

                cond_emb_expanded = rearrange(
                    cond_emb, "b (p t) c -> (b t) p c", p=self.num_patches
                )
                n = pairs.size(2)
                cond_emb_expanded = cond_emb_expanded.repeat_interleave(n, dim=1)  # (B*T, 2P, C)

                x = self.checkpoint(block, x, cond_emb_expanded)
            else:
                x = self.checkpoint(block, x, cond_emb_expanded)

        x = x[:, -self.num_patches :]  # keep only the "noisy" tokens
        x = rearrange(x, "(b t) p c -> b (t p) c", b=seq_batch_size)
        x = self.final_layer(x, cond_emb)

        # x = seq_states["x"]
        # x_img = img_states["x"] if img_states is not None else None
        # if x_img is not None:
        #    x_img = rearrange(x_img, "(b t) p c -> b (t p) c", b=seq_batch_size)
        #    x = torch.cat([x, x_img], dim=1)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
        super().__init__()
        if learnable:
            max_tokens = np.prod(shape)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, max_tokens, embed_dim).normal_(std=0.02),
                requires_grad=True,
            )

        else:
            self.register_buffer(
                "pos_emb",
                torch.from_numpy(get_nd_sincos_pos_embed(embed_dim, shape)).float().unsqueeze(0),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len]


def get_nd_sincos_pos_embed(
    embed_dim: int,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Get n-dimensional sinusoidal positional embeddings.
    Args:
        embed_dim: Embedding dimension.
        shape: Shape of the input tensor.
    Returns:
        Positional embeddings with shape (shape_flattened, embed_dim).
    """
    assert embed_dim % (2 * len(shape)) == 0
    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape])
    grid = np.stack(grid, axis=0)  # (ndim, *shape)
    return np.concatenate(
        [
            get_1d_sincos_pos_embed_from_grid(embed_dim // len(shape), grid[i])
            for i in range(len(shape))
        ],
        axis=1,
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Args:
        embed_dim: Embedding dimension.
        pos: Position tensor of shape (...).
    Returns:
        Positional embeddings with shape (-1, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
