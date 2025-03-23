# adapted from
# https://developmentalsystems.org/sensorimotor-lenia
import torch

field_func = {
    0: lambda n, m, s: torch.max(
        torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s**2)
    )
    ** 4
    * 2
    - 1,  # polynomial (quad4)
    1: lambda n, m, s: torch.exp(-((n - m) ** 2) / (2 * s**2) - 1e-3) * 2
    - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1,  # step (stpz)
    3: lambda n, m, s: -torch.clamp(n - m, 0, 1) * s,  # food eating kernl
}
