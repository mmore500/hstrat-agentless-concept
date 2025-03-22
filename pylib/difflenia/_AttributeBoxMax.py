import numpy as np

from ..auxlib._get_largest_coord import get_largest_coord


class AttributeBoxMax:
    def __init__(self, size):
        self._size = size

    def __call__(self, coordinate, frame, prev_frame):
        x, y = coordinate
        grid = np.concatenate(
            [
                prev_frame.numpy().repeat(2, 2),
                prev_frame.numpy(),
            ],
            axis=2,
        ).sum(-1)
        return get_largest_coord(grid, (x, y), self._size)
