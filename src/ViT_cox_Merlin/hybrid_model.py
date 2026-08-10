"""The full two-branch architecture from paper Fig. 4 / eq. 2-3, adapted
per project instructions:
  - Image branch: DINOv3 CLS token (384-dim, precomputed/cached) instead
    of the paper's from-scratch 12-block pixel ViT -- "the CLS token is
    the feature vector."
  - Radiomics branch: RadiomicsTransformerEncoder (1 block), optionally
    initialized from LUNA16 masked-token pretraining.
  - Each branch: r_img = beta . phi(CLS_img), r_rad = gamma . phi(CLS_rad)
    (eq. 2), phi = ReLU.
  - Mixed risk: F = (1-alpha) r_img + alpha r_rad (eq. 3).
  - Separate small projection heads map each branch's CLS token into a
    shared space for the NT-Xent contrastive loss that aligns modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiomics_transformer import RadiomicsTransformerEncoder


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class HybridViTCox(nn.Module):
    def __init__(self, image_dim: int, radiomics_feature_dim: int, radiomics_embed_dim: int = 128, n_patches: int = 36, proj_dim: int = 64):
        super().__init__()
        self.radiomics_encoder = RadiomicsTransformerEncoder(radiomics_feature_dim, radiomics_embed_dim, n_patches)

        self.beta = nn.Linear(image_dim, 1, bias=False)
        self.gamma = nn.Linear(radiomics_embed_dim, 1, bias=False)

        self.image_proj_head = ProjectionHead(image_dim, proj_dim)
        self.radiomics_proj_head = ProjectionHead(radiomics_embed_dim, proj_dim)

    def forward(self, image_cls: torch.Tensor, radiomics_tokens: torch.Tensor, alpha: float) -> dict:
        rad_out = self.radiomics_encoder(radiomics_tokens)
        rad_cls = rad_out["cls"]

        r_img = self.beta(F.relu(image_cls)).squeeze(-1)
        r_rad = self.gamma(F.relu(rad_cls)).squeeze(-1)
        risk = (1 - alpha) * r_img + alpha * r_rad

        z_img = self.image_proj_head(image_cls)
        z_rad = self.radiomics_proj_head(rad_cls)

        return {"risk": risk, "r_img": r_img, "r_rad": r_rad, "z_img": z_img, "z_rad": z_rad}


def nt_xent_loss(z_img: torch.Tensor, z_rad: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:

    B = z_img.shape[0]
    if B < 2:
        return torch.zeros((), device=z_img.device, dtype=z_img.dtype)

    sim = z_img @ z_rad.T / temperature  # (B, B), sim[n, m] = sim(z_img_n, z_rad_m)
    labels = torch.arange(B, device=z_img.device)
    return F.cross_entropy(sim, labels)
