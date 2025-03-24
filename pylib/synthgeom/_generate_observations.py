import numpy as np


def generate_observations(
    SX,
    SY,
    *,
    crea_func=np.ones,
    crea_size=10,
    mirror_backtrack=False,
    num_frames=1000,
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

    half_idx = num_frames // 2
    if mirror_backtrack:
        assert len(frames[-half_idx - 1 :: -1]) == len(frames[half_idx:])
        for forward_frame, backward_frame in zip(
            frames[half_idx:], frames[-half_idx - 1 :: -1]
        ):
            forward_frame += backward_frame * (forward_frame == 0.0)

    return frames
