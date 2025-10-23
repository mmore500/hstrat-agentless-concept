import typing

import numpy as np

from .auxlib._downstream_tilted_site_selection import (
    downstream_tilted_site_selection,
)
from .auxlib._randomize_nth_bit import randomize_nth_bit


def build_hstrat_surfaces(
    frames: typing.List[typing.Any],
    dims: typing.Tuple[int, int],
    surface_transform: typing.Callable,
    dstream_algo: str,
    dstream_S: int,
) -> typing.List[np.ndarray]:
    """Trace a path backwards from end coordinate.

    Parameters
    ----------
    coordinate : Tuple[int, int]
        Starting coordinate.

    frames : List[typing.Any]
        List of frames to traverse, in chronological order.

    attributor : Callable
        Function to determine the next coordinate.

    Returns
    -------
    List[Tuple[int, int]]
        List of coordinates traversed, ending with end coordinate.
    """
    if dstream_algo != "dstream.tilted_algo" or dstream_S != 64:
        raise NotImplementedError

    surface = np.random.randint(0, 2**dstream_S, size=dims, dtype=np.uint64)
    surfaces = []
    for i, frame in enumerate(frames):
        surfaces.append(surface)
        surface = surface_transform(surface, frame)
        T = dstream_S + i
        k = downstream_tilted_site_selection(dstream_S, T)
        surface = randomize_nth_bit(surface, k)

    return surfaces
