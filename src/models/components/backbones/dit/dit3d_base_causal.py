from abc import ABC, abstractmethod
from typing import Optional

import torch
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from torch import nn

from src.models.components.backbones.base_backbone import BaseBackbone
from src.models.components.backbones.dit.dit_base import DiTBase, DiTBaseCausal
from src.models.components.modules.embeddings import (
    RandomDropoutCondEmbedding,
    StochasticTimeEmbedding,
)

# from src.models.components.transformer.ditv2 import DiTBase


class DiT3D(BaseBackbone):
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        hidden_size: int,
        patch_size: int,
        variant: str = "full",
        pos_emb_type: str = "learned_1d",
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = False,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        use_causal_mask: bool = False,
        kv_heads: Optional[int] = None,
        num_denoiser_layers: Optional[int] = 0,
    ):
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.use_causal_mask = use_causal_mask
        self.n_frames = max_tokens
        self.num_denoiser_layers = num_denoiser_layers

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

        # timm supports tuple img_size
        self.patch_embedder = PatchEmbed(
            img_size=(H, W),  # CHANGED
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
        )
        # self.patch_embedder_clean = PatchEmbed(
        #    img_size=(H, W),  # CHANGED
        #    patch_size=self.patch_size,
        #    in_chans=self.in_channels,
        #    embed_dim=self.hidden_size,
        #    bias=True,
        # )

        self.external_cond_embedding_total = RandomDropoutCondEmbedding(
            self.external_cond_dim * max_tokens,
            self.external_cond_emb_dim,
            dropout_prob=self.external_cond_dropout,
        )

        self.dit_base = DiTBaseCausal(
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
            # --- NEW: tell DiTBase the spatial patch grid ---
            spatial_grid_shape=(
                self.Hp,
                self.Wp,
            ),  # requires the DiTBase change I gave you
            num_denoiser_layers=num_denoiser_layers,
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
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, patch_size**2 * C)
        Returns:
            (B, H, W, C)
        """
        # --- CHANGED: use Hp, Wp (non-square safe) ---
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=self.Hp,  # number of patches along H
            w=self.Wp,  # number of patches along W
            p=self.patch_size,
            q=self.patch_size,
        )

    def _get_causal_mask(self, q: torch.Tensor) -> torch.Tensor:
        """
        Framewise causal mask (query can attend to its own frame and all past frames).
        Returns a boolean mask of shape (N, N) where True = MASK OUT (disallow).
        """
        # q, k: (B, H, N, D)
        N = q.size(-2)
        assert self.n_frames >= 1, "n_frames must be >= 1"
        assert (
            N % self.n_frames == 0
        ), f"Sequence length N={N} must be divisible by n_frames={self.n_frames}"

        tpf = N // self.n_frames  # tokens per frame
        device = q.device

        # Frame index per token position [0..n_frames-1]
        pos = torch.arange(N, device=device)
        frame_id = pos // tpf  # (N,)

        # allow if key_frame <= query_frame
        # allowed[i, j] = True means j is visible to i
        allowed = frame_id.unsqueeze(1) >= frame_id.unsqueeze(0)  # (N, N)

        # SDPA expects True = disallowed (masked)
        # AAattn_mask_bool = ~allowed  # (N, N)
        attn_mask_bool = allowed  # (N, N)
        return attn_mask_bool

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        x_clean: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        num_frames = x.shape[1]
        noise_levels = noise_levels.to(x.dtype)
        # Merge (B, T) into one dimension for patch embedding
        x_clean = rearrange(x_clean, "b t c h w -> (b t) c h w")
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # x_clean = self.patch_embedder_clean(x_clean)
        x_clean = self.patch_embedder(x_clean)
        x = self.patch_embedder(x)

        x_clean = rearrange(x_clean, "(b t) p c -> b (t p) c", b=input_batch_size)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        # We check if we have patch-wise noise levels or frame wise noise levels
        emb = self.noise_level_pos_embedding(noise_levels)
        if external_cond is not None:
            ext_emb = self.external_cond_embedding(external_cond, external_cond_mask)
            ext_emb_total = self.external_cond_embedding_total(
                rearrange(external_cond, "b t c -> b 1 (t c)"), external_cond_mask
            )
            emb = emb + ext_emb

        cond_emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)
        ext_emb = repeat(ext_emb, "b t c -> b (t p) c", p=self.num_patches)

        ### TEMPORARY
        ext_emb_total = repeat(ext_emb_total, "b t c -> b (t p) c", p=self.num_patches)
        x_clean[:, : self.num_patches] = x_clean[:, : self.num_patches] + ext_emb_total

        # Pass to DiTBase
        mask = self._get_causal_mask(x_clean)
        x = self.dit_base(
            x_clean=x_clean, x=x, cond_emb=cond_emb, ext_emb=ext_emb, mask=mask
        )  # (B, N, C)

        # Unpatchify
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B*T, H, W, C)

        # Reshape back to (B, T, ...)
        x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)  # (B, T, C, H, W)
        return x
