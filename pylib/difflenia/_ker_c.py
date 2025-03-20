import torch


def ker_c(x, r, w, b):
    return (b * torch.exp(-(((x.unsqueeze(-1) - r) / w) ** 2) / 2)).sum(-1)
