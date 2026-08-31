import torch
from torch import nn


class Encoder(nn.Module):
    """Projects (batch, inp_seq_len, n_features) -> (batch, inp_seq_len, n_embd)."""

    def __init__(self, features, n_embd, inp_seq_len, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(len(features), n_embd)
        self.pos = nn.Parameter(torch.zeros(1, inp_seq_len, n_embd))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(x) + self.pos)
