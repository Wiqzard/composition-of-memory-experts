"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
Extended to support:
- Temporal sequence modeling
- 1D input additionally to 2D spatial input
- Token-wise conditioning
"""

from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from src.models.components.modules.embeddings import RotaryEmbedding3D
from src.models.components.transformer.ditv2_blocks import DiTBlock, DITFinalLayer

Variant = Literal["full", "factorized_encoder", "factorized_attention"]
PosEmb = Literal[
    "learned_1d", "sinusoidal_1d", "sinusoidal_3d", "sinusoidal_factorized", "rope_3d"
]


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
        use_causal_mask: bool = False,
        n_frames: int = 1,
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
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )
            case "sinusoidal_factorized":
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.spatial_grid_size, self.spatial_grid_size),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )
            case "rope_3d":
                rope = RotaryEmbedding3D(
                    dim=self.hidden_size // num_heads,
                    sizes=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(mlp_ratio if self.variant != "factorized_attention" else None),
                    rope=rope if self.pos_emb_type == "rope_3d" else None,
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
    def spatial_grid_size(self) -> Optional[int]:
        if self.num_patches is None:
            return None
        grid_size = int(self.num_patches**0.5)
        assert grid_size * grid_size == self.num_patches, "num_patches must be a square number"
        return grid_size

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
        self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the DiTBase model (simplified, single-token-set version).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        # If you still need to check input length vs. max_tokens:
        if x.size(1) > self.max_tokens:
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
            )

        batch_size = x.size(0)

        # 1) Absolute positional embeddings if needed
        if self.is_pos_emb_absolute_once:
            x = self.pos_emb(x)

        # 2) Factorized absolute positional embeddings (non-factorized case)
        if self.is_pos_emb_absolute_factorized and not self.is_factorized:
            # Factor out space, then time, then refold:
            # (b (t p) c) -> (b t) p c -> (b p) t c -> ...
            def add_pos_emb(x: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.spatial_pos_emb(x)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                x = self.temporal_pos_emb(x)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            x = add_pos_emb(x, batch_size)

        # 3) Factorize the sequence (if required)
        if self.is_factorized:
            x, c = rearrange_contiguous_many((x, c), "b (t p) c -> (b t) p c", p=self.num_patches)
            if self.is_pos_emb_absolute_factorized:
                x = self.spatial_pos_emb(x)

        # 4) Transformer Blocks
        for i, (block, temporal_block) in enumerate(
            zip(self.blocks, self.temporal_blocks or [None for _ in range(self.depth)])
        ):
            # Spatial block
            x = self.checkpoint(block, x, c, mask=mask)

            if self.is_factorized:
                # Switch space <-> time
                x, c = rearrange_contiguous_many((x, c), "(b t) p c -> (b p) t c", b=batch_size)
                # Possibly apply temporal pos emb the first time
                if i == 0 and self.pos_emb_type == "sinusoidal_factorized":
                    x = self.temporal_pos_emb(x)
                # Temporal block
                x = self.checkpoint(temporal_block, x, c)
                # Switch back
                x, c = rearrange_contiguous_many((x, c), "(b p) t c -> (b t) p c", b=batch_size)

        # 5) Un-factorize if needed
        if self.is_factorized:
            x, c = rearrange_contiguous_many((x, c), "(b t) p c -> b (t p) c", b=batch_size)

        # 6) Final layer
        x = self.final_layer(x, c)

        return x
