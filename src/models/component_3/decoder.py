from torch import nn


class Decoder(nn.Module):
    """(batch, n_embd) -> (batch, horizon, n_targets)."""

    def __init__(self, targets, n_embd, horizon, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.n_targets = len(targets)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, horizon * self.n_targets),
        )

    def forward(self, latent):
        return self.mlp(latent).view(-1, self.horizon, self.n_targets)
