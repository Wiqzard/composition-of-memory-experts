from abc import abstractmethod, ABC
from typing import Optional
import torch
from torch import nn
from src.models.components.modules.embeddings import (
    StochasticTimeEmbedding,
    RandomDropoutCondEmbedding,
)


class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        noise_level_emb_dim: int,
        use_fourier_noise_embedding: bool = False,
        external_cond_dropout: float = 0.0,
        use_causal_mask: bool = False,
    ):
        super().__init__()

        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        self.external_cond_dropout = external_cond_dropout

        # Store the embedding dimension for noise levels
        self._noise_level_emb_dim = noise_level_emb_dim

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=use_fourier_noise_embedding,
        )

        self.external_cond_embedding = self._build_external_cond_embedding()

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return (
            RandomDropoutCondEmbedding(
                self.external_cond_dim,
                self.external_cond_emb_dim,
                dropout_prob=self.external_cond_dropout,
            )
            if self.external_cond_dim
            else None
        )

    @property
    def noise_level_dim(self) -> int:
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def external_cond_emb_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
