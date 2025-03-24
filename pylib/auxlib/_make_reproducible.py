import random

import numpy as np
import torch


def make_reproducible(seed=1):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
