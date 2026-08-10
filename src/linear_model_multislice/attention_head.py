import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolCoxHead(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(input_dim, attn_dim)
        self.U = nn.Linear(input_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Linear(input_dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: (N, S, D) padded slice tokens. mask: (N, S) 1=real, 0=pad.
        Returns risk (N,).
        """
        gated = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))  # (N,S,attn)
        scores = self.w(self.dropout(gated)).squeeze(-1)                    # (N,S)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=1)                               # (N,S)
        z = torch.bmm(alpha.unsqueeze(1), tokens).squeeze(1)               # (N,D)
        return self.beta(z).squeeze(-1)

    def attention_weights(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))
        scores = self.w(gated).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        return torch.softmax(scores, dim=1)
