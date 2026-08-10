"""The "shallow ViT" radiomics encoder (paper Fig. 4, p.15: "Radiomics
Encoder (1 block)"): a learnable CLS token + learned per-patch position
embeddings, a linear input projection from raw per-patch radiomics
features to the transformer's embedding dim, one standard pre-norm
transformer encoder block, and a final LayerNorm. Output is the CLS
token's final hidden state -- the "Radiomics CLS Token" in Fig. 4.

MaskedRadiomicsPretrainer wraps the encoder for BERT/MAE-style
self-supervised pretraining on an external CT corpus (LUNA16) that has no
survival labels: mask a fraction of patch tokens, replace them with a
learnable [MASK] embedding, and train the encoder + a small linear decoder
to reconstruct the masked patches' original (standardized) feature
vectors from context.
"""

import torch
import torch.nn as nn


class RadiomicsTransformerEncoder(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int = 128, n_patches: int = 36, n_heads: int = 4, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor, mask_embed: torch.Tensor | None = None, masked_positions: torch.Tensor | None = None) -> dict:
        """tokens: (B, n_patches, feature_dim). If masked_positions (B, n_patches)
        bool is given, those patch positions are replaced with mask_embed
        before the transformer (pretraining path); otherwise all patches
        are used as-is (fine-tuning / inference path).
        """
        x = self.input_proj(tokens)  # (B, n_patches, embed_dim)
        if masked_positions is not None:
            x = torch.where(masked_positions.unsqueeze(-1), mask_embed.to(x.dtype), x)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_patches, embed_dim)
        x = x + self.pos_embed[:, : x.shape[1]]

        out = self.encoder(x)
        out = self.norm(out)
        return {"cls": out[:, 0], "patches": out[:, 1:]}


class MaskedRadiomicsPretrainer(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int = 128, n_patches: int = 36, mask_ratio: float = 0.4, **encoder_kwargs):
        super().__init__()
        self.encoder = RadiomicsTransformerEncoder(feature_dim, embed_dim, n_patches, **encoder_kwargs)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_embed, std=0.02)
        self.decoder = nn.Linear(embed_dim, feature_dim)
        self.mask_ratio = mask_ratio
        self.n_patches = n_patches

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction_loss_per_masked_token_mse_terms handled by caller
        -- here just returns (predicted, target, masked_positions))."""
        B, n_patches, _ = tokens.shape
        n_mask = max(1, int(round(n_patches * self.mask_ratio)))

        masked_positions = torch.zeros(B, n_patches, dtype=torch.bool, device=tokens.device)
        for b in range(B):
            idx = torch.randperm(n_patches, device=tokens.device)[:n_mask]
            masked_positions[b, idx] = True

        out = self.encoder(tokens, mask_embed=self.mask_embed, masked_positions=masked_positions)
        predicted = self.decoder(out["patches"])  # (B, n_patches, feature_dim)
        return predicted, masked_positions
