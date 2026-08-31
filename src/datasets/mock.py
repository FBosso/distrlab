import math

import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    """Sine-mixture toy series: no file, no download, actually learnable."""

    def __init__(self, features, targets, inp_seq_len, horizon, num_samples=512, train_split=0.8, seed=42):
        self.attrs = sorted({f["name"] for f in features} | {t["name"] for t in targets})
        self.window = inp_seq_len + horizon
        self.num_samples = num_samples
        gen = torch.Generator().manual_seed(seed)
        self.data = [{a: self._series(gen) for a in self.attrs} for _ in range(num_samples)]

    def _series(self, gen, n_waves=3):
        t = torch.arange(self.window, dtype=torch.float32)
        y = torch.randn(self.window, generator=gen) * 0.05
        for _ in range(n_waves):
            freq = torch.empty(1).uniform_(0.02, 0.2, generator=gen)
            amp = torch.empty(1).uniform_(0.5, 1.5, generator=gen)
            phase = torch.empty(1).uniform_(0, 2 * math.pi, generator=gen)
            y += amp * torch.sin(2 * math.pi * freq * t + phase)
        return y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
