import numpy as np
import pandas as pd


def make_hstrat_surface_dataframe(
    surfaces: np.ndarray,
    T: int,
    values: np.ndarray,
    dstream_algo: str,
) -> pd.DataFrame:
    """Create a DataFrame from a list of surfaces and values.

    Parameters
    ----------
    surfaces : np.ndarray
        Array of surfaces.

    values : np.ndarray
        Array of values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'surface' and 'value'.
    """
    row_indices, col_indices = np.indices(surfaces.shape)
    return pd.DataFrame(
        {
            "data_hex": [
                f"{T:016x}{surface:016x}" for surface in surfaces.ravel()
            ],
            "dstream_algo": dstream_algo,
            "dstream_storage_bitoffset": 64,
            "dstream_storage_bitwidth": 64,
            "dstream_T_bitoffset": 0,
            "dstream_T_bitwidth": 64,
            "dstream_S": 64,
            "col_index": col_indices.ravel(),
            "row_index": row_indices.ravel(),
            "value": values.ravel(),
        },
    )
