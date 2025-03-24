import numpy as np


# adapted from https://stackoverflow.com/a/76369149/17332200
def reverse_bits(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    dtype = np.asanyarray(x).dtype
    return np.flip(
        np.packbits(np.flip(np.unpackbits(x.view(np.uint8)))).view(dtype)
    )
