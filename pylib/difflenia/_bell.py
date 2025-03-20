import torch


def bell(x, m, s):
    return torch.exp(-(((x - m) / s) ** 2) / 2)
