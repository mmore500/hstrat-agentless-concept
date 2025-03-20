import itertools as it
import typing


def build_path(
    coordinate: typing.Tuple[int, int],
    frames: typing.List[typing.Any],
    attributor: typing.Callable,
) -> typing.List[typing.Tuple[int, int]]:
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
    path = [coordinate]
    for frame, prev_frame in it.pairwise(reversed(frames)):
        coordinate = attributor(coordinate, frame, prev_frame)
        path.append(coordinate)
    return reversed(path)
