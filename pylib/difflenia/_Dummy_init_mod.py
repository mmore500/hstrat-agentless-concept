import torch


class Dummy_init_mod(torch.nn.Module):
    def __init__(self, init):
        torch.nn.Module.__init__(self)
        self.register_parameter("init", torch.nn.Parameter(init))
