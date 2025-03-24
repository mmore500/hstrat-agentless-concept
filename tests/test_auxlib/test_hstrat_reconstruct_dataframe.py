import pandas as pd

from pylib.auxlib._hstrat_reconstruct_dataframe import (
    hstrat_reconstruct_dataframe,
)


def test_hstrat_reconstruct_dataframe():
    genome_df = pd.DataFrame(
        {
            "awoo": ["bar", "baz"],
            "dstream_algo": ["dstream.steady_algo", "dstream.steady_algo"],
            "downstream_version": ["1.0.1", "1.0.1"],
            "data_hex": ["080001030702050406", "0b0001030702050906"],
            "dstream_storage_bitoffset": [8, 8],
            "dstream_storage_bitwidth": [64, 64],
            "dstream_T_bitoffset": [0, 0],
            "dstream_T_bitwidth": [8, 8],
            "dstream_S": [8, 8],
        },
    )
    phylo_df = hstrat_reconstruct_dataframe(genome_df)
    assert len(phylo_df) == len(genome_df)
    assert "ancestor_id" in phylo_df.columns
    assert "id" in phylo_df.columns
