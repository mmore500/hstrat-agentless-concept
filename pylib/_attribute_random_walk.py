import random
import typing

import numpy as np


def attribute_random_walk(
    coordinate: typing.Tuple[int, int],
    frame: np.array,
    prev_frame: np.array,
) -> typing.Tuple[int, int]:
    max_x, max_y = prev_frame.shape
    x, y = coordinate

    x += random.randint(-1, 1)
    y += random.randint(-1, 1)

    x %= max_x
    y %= max_y

    return x, y
