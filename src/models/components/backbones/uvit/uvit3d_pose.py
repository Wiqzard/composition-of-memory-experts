from typing import Optional, Tuple

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.components.modules.embeddings import RandomDropoutPatchEmbed

from .uvit3d import UViT3D


class UViT3DPose(UViT3D):
    """
    U-ViT with pose embedding.
    """

    def __init__(
        self,
        channels,
        emb_dim,
        patch_size,
        block_types,
        block_dropouts,
        num_updown_blocks,
        num_mid_blocks,
        num_heads,
        pos_emb_type,
        use_checkpointing,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        external_cond_dropout: float,
        use_causal_mask=True,
        use_fourier_noise_embedding=False,
    ):
        self.conditioning_dropout = external_cond_dropout
        self.external_cond_dim = external_cond_dim
        self.x_shape = x_shape
        # self.external_cond_emb_dim = emb_dim  # assuming shared embedding dim

        super().__init__(
            channels=channels,
            emb_dim=emb_dim,
            patch_size=patch_size,
            block_types=block_types,
            block_dropouts=block_dropouts,
            num_updown_blocks=num_updown_blocks,
            num_mid_blocks=num_mid_blocks,
            num_heads=num_heads,
            pos_emb_type=pos_emb_type,
            use_checkpointing=use_checkpointing,
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=external_cond_dim,
            use_causal_mask=use_causal_mask,
            use_fourier_noise_embedding=use_fourier_noise_embedding,
        )

        self.external_cond_embedding = self._build_external_cond_embedding()

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return RandomDropoutPatchEmbed(
            dropout_prob=self.conditioning_dropout,
            img_size=self.x_shape[1],
            patch_size=self.patch_size,
            in_chans=self.external_cond_dim,
            embed_dim=self.external_cond_emb_dim,
            bias=True,
            flatten=False,
        )

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        x, emb = map(
            lambda y: rearrange(y, "(b t) c h w -> b (t h w) c", t=self.temporal_length),
            (x, emb),
        )
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        return x, emb

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone, with pose conditioning.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C', H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for U-ViT3DPose model."

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                # pylint: disable-next=not-callable
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.num_levels)
        ]
        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, embs[i_level], i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, embs[-1], self.num_levels - 1)

        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[i_level], i_level, is_up=True)

        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
