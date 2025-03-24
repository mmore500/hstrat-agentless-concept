import typing

import numpy as np
import pandas as pd

from ._downstream_tilted_site_selection import downstream_tilted_site_selection
from ._make_hstrat_surface_dataframe import make_hstrat_surface_dataframe
from ._randomize_nth_bit import randomize_nth_bit


def make_dstream_validation_dataframe(
    differentia_override: typing.Callable,
    validator: str,
    S: int,
    T: int,
    dstream_algo: str,
) -> pd.DataFrame:

    if S != 64 or dstream_algo != "dstream.tilted_algo":
        raise NotImplementedError
    surfaces = np.empty(shape=(2, 2), dtype=np.uint64)
    for Tbar in range(T):
        surfaces = randomize_nth_bit(
            surfaces,
            downstream_tilted_site_selection(S, Tbar),
            rand_val=differentia_override(Tbar),
        )

    df = make_hstrat_surface_dataframe(
        surfaces,
        T,
        surfaces,
        dstream_algo=dstream_algo,
    )
    df["downstream_validate_exploded"] = validator
    df["downstream_validate_unpacked"] = f"pl.col('dstream_T') == {T}"

    return df
