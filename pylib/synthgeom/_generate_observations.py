import numpy as np


def generate_observations(
    SX,
    SY,
    *,
    num_frames=1000,
    crea_func=np.ones,
    crea_size=10,
):
    min_x = SX // 8
    min_y = SY // 8

    max_x = SX - SX // 8
    max_y = SY - SY // 8

    x, y = min_x, min_y
    frames = []
    for __ in range(num_frames):

        if y == min_y and x < max_x:
            x += 1
        elif x == max_x and y < max_y:
            y += 1
        elif y == max_y and x > min_x:
            x -= 1
        elif x == min_x and y > min_y:
            y -= 1
        else:
            assert False

        frame = np.zeros((SX, SY))
        frame[
            y - crea_size : y + crea_size, x - crea_size : x + crea_size
        ] = crea_func((2 * crea_size, 2 * crea_size))

        frames.append(frame)

    return frames
