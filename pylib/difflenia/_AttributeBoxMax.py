import numpy as np

from ..auxlib._get_largest_coord import get_largest_coord


class AttributeBoxMax:
    def __init__(self, size):
        self._size = size

    def __call__(self, coordinate, frame, prev_frame):
        x, y = coordinate
        grid = prev_frame.detach().cpu().numpy()
        assert len(grid.shape) == 2
        assert not np.isnan(np.sum(grid.ravel()))
        return get_largest_coord(grid, (x, y), self._size)
