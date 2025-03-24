import typing


def dewrap_path(
    path: typing.List[typing.Tuple[int, int]],
    xmax: typing.Optional[int],
    ymax: typing.Optional[int],
) -> typing.List[typing.List[typing.Tuple[int, int, int]]]:
    """Detect path steps where a wraparound event occurs and "unwrap" modulo."""
    path = [*path]
    if not path:
        return path

    dewrap_coordinates = [[(0, *path[0])]]
    for raw_x, raw_y in path[1:]:
        prev_t, prev_x, prev_y = dewrap_coordinates[-1][-1]

        if xmax is not None:
            fix_x = min(
                (raw_x, raw_x + xmax, raw_x - xmax),
                key=lambda x: abs(x - prev_x),
            )
        else:
            fix_x = raw_x

        if ymax is not None:
            fix_y = min(
                (raw_y, raw_y + ymax, raw_y - ymax),
                key=lambda y: abs(y - prev_y),
            )
        else:
            fix_y = raw_y

        dewrap_coordinates[-1].append((prev_t + 1, fix_x, fix_y))

        if (fix_x, fix_y) != (raw_x, raw_y):
            dewrap_coordinates.append([(prev_t + 1, raw_x, raw_y)])

    return dewrap_coordinates
