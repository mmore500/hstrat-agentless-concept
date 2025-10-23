import itertools as it
import typing


def make_seed_coordinates(
    *,
    x_coords: typing.List[int],
    y_coords: typing.List[int],
) -> typing.List[typing.Tuple[int, int]]:

    seed_coordinates = it.product(
        y_coords,
        x_coords,
    )
    return [*map(tuple, map(reversed, seed_coordinates))]
