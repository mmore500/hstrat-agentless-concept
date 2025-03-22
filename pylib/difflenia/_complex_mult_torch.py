import torch


def complex_mult_torch(X, Y):
    """Computes the complex multiplication in Pytorch when the tensor last dimension is 2: 0 is the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, "Last dimension must be 2"
    return torch.stack(
        (
            X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
            X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0],
        ),
        dim=-1,
    )
