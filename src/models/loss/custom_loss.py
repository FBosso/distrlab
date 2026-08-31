from torch import nn


class ExampleLoss(nn.Module):
    """Plain MSE. Swap for whatever your model actually needs."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)
