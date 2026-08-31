from torch import nn


class LatentSpace(nn.Module):
    """Self-attends over (batch, inp_seq_len, n_embd), pools to (batch, n_embd)."""

    def __init__(self, n_embd, n_head, n_layer=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)

    def forward(self, x):
        return self.encoder(x).mean(dim=1)
