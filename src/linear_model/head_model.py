
import torch
import torch.nn as nn


class LinearCoxHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x)).squeeze(-1)


class TwoBranchCoxHead(nn.Module):
 

    def __init__(self, radiomics_dim: int, image_dim: int, radiomics_dropout: float = 0.5, image_dropout: float = 0.1):
        super().__init__()
        self.radiomics_dim = radiomics_dim
        self.image_dim = image_dim
        self.radiomics_dropout = nn.Dropout(radiomics_dropout) if radiomics_dropout > 0 else nn.Identity()
        self.image_dropout = nn.Dropout(image_dropout) if image_dropout > 0 else nn.Identity()
        self.radiomics_linear = nn.Linear(radiomics_dim, 1, bias=False)
        self.image_linear = nn.Linear(image_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_radiomics = x[:, : self.radiomics_dim]
        x_image = x[:, self.radiomics_dim :]
        risk_radiomics = self.radiomics_linear(self.radiomics_dropout(x_radiomics))
        risk_image = self.image_linear(self.image_dropout(x_image))
        return (risk_radiomics + risk_image).squeeze(-1)


def cox_ph_loss(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
 
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum_risk = torch.logcumsumexp(risk, dim=0)
    n_events = event.sum()
    if n_events == 0:
        return risk.sum() * 0.0

    return -torch.sum((risk - log_cumsum_risk) * event) / (n_events + eps)
