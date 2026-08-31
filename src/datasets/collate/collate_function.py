import torch


class CollateFn:
    """{attr: tensor(window)} items -> (x, y). x = features' first inp_seq_len steps,
    y = targets' next horizon steps."""

    def __init__(self, features, targets, inp_seq_len, horizon):
        self.features = features
        self.targets = targets
        self.inp_seq_len = inp_seq_len
        self.horizon = horizon

    def __call__(self, batch):
        x = torch.stack([
            torch.stack([item[f["name"]][:self.inp_seq_len] for f in self.features], dim=-1)
            for item in batch
        ])
        y = torch.stack([
            torch.stack([item[t["name"]][self.inp_seq_len:self.inp_seq_len + self.horizon] for t in self.targets], dim=-1)
            for item in batch
        ])
        return x, y
