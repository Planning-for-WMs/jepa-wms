import math

import torch
import torch.nn as nn


class LatentActionEncoder(nn.Module):
    """Transformer-based encoder that compresses chunks of primitive actions into latent macro-actions.

    Takes a variable-length sequence of primitive actions between waypoints and produces
    a single latent macro-action vector via CLS token extraction.

    Reference: "Hierarchical World Models" (arxiv 2604.03208) — action encoder A_ψ.
    """

    def __init__(
        self,
        action_dim,
        latent_action_dim,
        max_chunk_size,
        encoder_depth=2,
        encoder_heads=4,
        encoder_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_action_dim = latent_action_dim
        self.max_chunk_size = max_chunk_size
        self.encoder_dim = encoder_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim) * 0.02)
        self.action_embed = nn.Linear(action_dim, encoder_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_chunk_size + 1, encoder_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, latent_action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, action_chunk, chunk_lengths=None):
        """
        Args:
            action_chunk: (B, max_chunk_size, action_dim) — padded action sequence
            chunk_lengths: (B,) — actual length of each chunk (for padding mask). None = no masking.

        Returns:
            latent_action: (B, latent_action_dim)
        """
        B, T, _ = action_chunk.shape

        x = self.action_embed(action_chunk)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : T + 1]

        src_key_padding_mask = None
        if chunk_lengths is not None:
            # CLS token (position 0) is never masked; action positions masked if >= chunk_length
            positions = torch.arange(T + 1, device=action_chunk.device).unsqueeze(0)
            # True = masked (ignored) in PyTorch convention
            src_key_padding_mask = positions >= (chunk_lengths.unsqueeze(1) + 1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        latent_action = self.head(x[:, 0])

        return latent_action
